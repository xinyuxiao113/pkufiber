import torch, copy, numpy as np, matplotlib.pyplot as plt
import torch.nn.functional as F, torch.nn.init as init, torch.nn as nn
from typing import Union, Tuple

from pkufiber.op import get_beta2, get_beta1
from pkufiber.core import TorchSignal, TorchTime, TorchInput
from pkufiber.dsp.op import dispersion_kernel, dconv, nconv
from pkufiber.dsp.layers import MLP, Parameter


class FDBP(nn.Module):
    """
    Digital Back Propagation (DBP) with hyper-network for optical communication systems.

    Attributes:
        Nmodes: Number of signal polarization modes (1 or 2).
        step: Number of DBP steps.
        dtaps: Size of the dispersion kernel.
        ntaps: Size of the nonlinear filter.
        d_share: Whether to share the dispersion kernel across steps.
        n_share: Whether to share the nonlinear filter across steps.
        dz: Step size for DBP in meters.
        gamma: Nonlinear coefficient.
        D: Dispersion parameter.
        Fc: Center frequency.
        Fi: Initial frequency.
        Fs: Sampling rate.
        overlaps: Overlap size for each step.

    Example FDBP:
        net = FDBP(2, 5, 1001, 101)
        signal = TorchSignal(val=torch.randn(1, 10000, 2).to(torch.complex64), t=TorchTime(0, 0, 2))
        task_info = torch.rand(1,4)
        y = net(signal, task_info)
        print(y)
    """

    def __init__(
        self,
        Nmodes: int,
        step: int,
        dtaps: int,
        ntaps: int,
        d_share: bool = True,
        n_share: bool = True,
        d_train: bool = False,
        L: float = 2000e3,
        gamma: float = 0.0016567,
        D: float = 16.5,
        Fs: float = 160e9,
        Fc: float = 299792458 / 1550e-9,
        Fi: float = 299792458 / 1550e-9,
    ):
        super(FDBP, self).__init__()
        self.Nmodes = Nmodes
        self.step = step
        self.dtaps = dtaps
        self.ntaps = ntaps
        self.d_share = d_share
        self.n_share = n_share
        self.d_train = d_train
        self.dz = L / step  # Step size in meters
        self.gamma = gamma
        self.D = D
        self.Fc = Fc
        self.Fi = Fi
        self.Fs = Fs
        self.overlaps = step * ((dtaps - 1) + (ntaps - 1)) // 2

        d_num = 1 if d_share else self.step
        n_num = 1 if n_share else self.step

        beta2 = (
            get_beta2(self.D, self.Fc) / 1e3
        )  # Second-order dispersion coefficient [s^2/m]
        beta1 = (
            get_beta1(self.D, self.Fc, self.Fi) / 1e3
        )  # First-order dispersion coefficient [s/m]
        # print(Fs, beta2, beta1)
        Dkernel_init = dispersion_kernel(
            -self.dz, self.dtaps, Fs, beta2, beta1, domain="time"
        ).to(torch.complex64)

        self.Dkernel_real = nn.ParameterList(
            [
                nn.Parameter(Dkernel_init.real, requires_grad=d_train)
                for _ in range(d_num)
            ]
        )
        self.Dkernel_imag = nn.ParameterList(
            [
                nn.Parameter(Dkernel_init.imag, requires_grad=d_train)
                for _ in range(d_num)
            ]
        )
        self.Nkernel = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        self.Nmodes, self.Nmodes, self.ntaps, dtype=torch.float32
                    )
                )
                for _ in range(n_num)
            ]
        )  # Nonlinear kernel [Nmodes, Nmodes, ntaps]

    def update_Dkernel(self, Fi: float, Fs: Union[torch.Tensor, float]):
        """
        Update the dispersion kernel.

        Args:
            Fi: Initial frequency as a torch.Tensor.
            Fs: Sampling rate as a torch.Tensor or float.
        """
        beta2 = get_beta2(self.D, self.Fc) / 1e3
        beta1 = get_beta1(self.D, self.Fc, Fi) / 1e3
        Dkernel = dispersion_kernel(
            -self.dz, self.dtaps, Fs, beta2, beta1, domain="time"
        )
        for i in range(len(self.Dkernel_real)):
            self.Dkernel_real[i].data = Dkernel.real
            self.Dkernel_imag[i].data = Dkernel.imag

    def forward(self, signal: TorchSignal, task_info: torch.Tensor) -> TorchSignal:
        """
        Forward pass for DBP with hyper-network.

        Input:
            signal: TorchSignal with val shape [B, L, Nmodes].  Complex64 type.
            task_info: torch.Tensor with shape [B, 4]. Contains [P, Fi, Fs, Nch] with units [dBm, Hz, Hz, 1].

        Output:
            TorchSignal with val shape [B, L - C, Nmodes], where C = steps * (dtaps - 1 + ntaps - 1).
        """
        assert signal.val.dtype == torch.complex64, "Input signal must be complex type."
        x = signal.val  # [batch, L*sps, Nmodes]
        t = copy.deepcopy(signal.t)  # [start, stop, sps]
        batch = x.shape[0]

        P = 1e-3 * 10 ** (task_info[:, 0] / 10) / self.Nmodes  # Power [W]

        for i in range(self.step):
            # Linear step
            Dk = self.Dkernel_real[min(i, len(self.Dkernel_real) - 1)].expand(
                x.shape[0], self.dtaps
            ) + 1j * self.Dkernel_imag[min(i, len(self.Dkernel_imag) - 1)].expand(
                x.shape[0], self.dtaps
            )
            x = dconv(x, Dk, stride=1)  # [batch, L*sps - dtaps + 1, Nmodes]
            t.conv1d_t(self.dtaps, stride=1)

            # Nonlinear step
            start, stop = t.start, t.stop
            t.conv1d_t(self.ntaps, stride=1)
            phi = nconv(
                torch.abs(x) ** 2,
                self.Nkernel[min(i, len(self.Nkernel) - 1)].expand(
                    x.shape[0], self.Nmodes, self.Nmodes, self.ntaps
                ),
                1,
            )
            x = x[:, t.start - start : t.stop - stop + x.shape[1]] * torch.exp(
                1j * phi * self.gamma * P[:, None, None] * self.dz
            )  # [batch, L*sps - dtaps + 1, Nmodes]

        return TorchSignal(x, t)
    
    def rmps(self) -> int:
        """
        Return the number of parameters in the model.
        """
        from pkufiber.dsp.nonlinear_compensation.rmps import rmps_fdbp
        return rmps_fdbp(self.dtaps, self.ntaps, self.step)
    
    
if __name__ == "__main__":
    device = 'cuda:0'
    net = FDBP(2, 5, 1001, 101)
    signal = TorchSignal(
        val=torch.randn(1, 10000, 2).to(torch.complex64), t=TorchTime(0, 0, 2)
    )
    task_info = torch.rand(1, 4)

    signal = signal.to(device)
    task_info = task_info.to(device)
    net = net.to(device)
    
    import time 
    t0 = time.time()
    y = net(signal, task_info)
    t1 = time.time()
    print('time:', t1 - t0)

    print(y)
