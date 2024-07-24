"""
    DSP module.
    DSP = DBP + ADF.
    DBP: Digital Back Propagation.
    ADF: Adaptive Decision Feedback.
    ADFCell: ADF cell.
    ADF: ADF module.
    LDBP: Learnable DBP module.
"""

import torch, copy, numpy as np, matplotlib.pyplot as plt
import torch.nn.functional as F, torch.nn.init as init, torch.nn as nn
from typing import Union, Tuple

from pkufiber.core import TorchSignal, TorchTime, TorchInput
from pkufiber.op import get_beta2, get_beta1
from pkufiber.dsp.op import dispersion_kernel, dconv, nconv
from pkufiber.dsp.layers import MLP


class MetaDBP(nn.Module):
    """
    Meta Digital Back Propagation (MetaDBP) with a hyper-network for optical communication systems.

    Attributes:
        dz: Step size for DBP in meters.
        step: Number of DBP steps.
        dtaps: Size of the dispersion kernel.
        ntaps: Size of the nonlinear filter.
        Nmodes: Number of signal polarization modes (1 or 2).
        task_dim: Dimension of the task input to the hyper-network.
        task_hidden_dim: Hidden size of the MLP used in Meta-DBP.
        gamma: Nonlinear coefficient.
        D: Dispersion parameter.
        Fc: Center frequency.
        Fi: Intermediate frequency.
        Fs: Sampling rate.
        overlaps: Overlap size for each step.

    Example MetaDBP:
        net = MetaDBP(2, 5, 1001, 101)
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
        task_dim=1,
        task_hidden_dim=100,
        L: float = 2000e3,
        gamma: float = 0.0016567,
        D: float = 16.5,
        Fs: float = 160e9,
        Fc: float = 299792458 / 1550e-9,
        Fi: float = 299792458 / 1550e-9,
    ):
        super(MetaDBP, self).__init__()
        self.dz = L / step  # Step size in meters
        self.step = step
        self.dtaps = dtaps
        self.ntaps = ntaps
        self.Nmodes = Nmodes
        self.task_dim = task_dim
        self.task_hidden_dim = task_hidden_dim
        self.task_mlp = MLP(task_dim, self.task_hidden_dim, self.Nmodes**2)

        self.gamma = gamma  # Nonlinear coefficient [1/W/m]
        self.D = D
        self.Fc = Fc
        self.Fi = Fi
        self.Fs = Fs
        self.overlaps = step * ((dtaps - 1) + (ntaps - 1)) // 2

    def set_ntaps(self, ntaps: int):
        """
        Set the number of taps for the nonlinear filter.

        Args:
            ntaps: Integer specifying the new size of the nonlinear filter.
        """
        self.ntaps = ntaps
        self.overlaps = self.step * ((self.dtaps - 1) + (self.ntaps - 1)) // 2

    def get_D_kernel(
        self, Fi: torch.Tensor, Fs: Union[torch.Tensor, float]
    ) -> torch.Tensor:
        """
        Calculate the dispersion kernel.

        Args:
            Fi: Initial frequency as a torch.Tensor.
            Fs: Sampling rate as a torch.Tensor or float.

        Returns:
            Dispersion kernel as a complex64 torch.Tensor.
        """
        beta2 = (
            get_beta2(self.D, self.Fc) / 1e3
        )  # Second-order dispersion coefficient [s^2/m]
        beta1 = (
            get_beta1(self.D, self.Fc, self.Fi) / 1e3
        )  # First-order dispersion coefficient [s/m]
        return dispersion_kernel(
            -self.dz, self.dtaps, Fs, beta2, beta1, domain="time"
        ).to(torch.complex64)

    def get_pos_enc(
        self, task_info: torch.Tensor, taps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate positional encoding for the task information.

        Args:
            task_info: torch.Tensor with shape [batch, 4] containing [P, Fi, Fs, Nch].
            taps: Integer specifying the number of taps.

        Returns:
            Tuple containing expanded positional encoding and sampling period.
        """
        scale = 1 / 200
        batch = task_info.shape[0]
        Fs = task_info[:, 2] / 160e9
        Ts = (1 / Fs).view(batch, 1, 1).expand(batch, taps, 1)  # Sampling period
        pos_enc = torch.abs(torch.arange(-(taps // 2), taps // 2 + 1)).to(
            task_info.device
        )  # Positional encoding
        expand_pos = pos_enc.expand(batch, taps).view(batch, taps, 1) * scale
        return expand_pos, Ts

    def get_N_kernel(self, task_info: torch.Tensor) -> torch.Tensor:
        """
        Calculate the nonlinear kernel.

        Args:
            task_info: torch.Tensor with shape [batch, 4] containing [P, Fi, Fs, Nch].

        Returns:
            Nonlinear kernel as a torch.Tensor.
        """
        batch = task_info.shape[0]
        pos_enc, Ts = self.get_pos_enc(task_info, self.ntaps)
        k1 = Ts**2 * self.task_mlp(pos_enc * Ts**2) * torch.exp(-pos_enc * Ts**2)
        Nkernel = k1.permute(0, 2, 1).reshape(
            batch, self.Nmodes, self.Nmodes, self.ntaps
        )

        return Nkernel

    def forward(self, signal: TorchSignal, task_info: torch.Tensor) -> TorchSignal:
        """
        Forward pass for the MetaDBP network.

        Args:
            signal: TorchSignal with val shape [B, L, Nmodes].
            task_info: torch.Tensor with shape [B, 4] containing [P, Fi, Fs, Nch].

        Returns:
            TorchSignal with val shape [B, L - C, Nmodes], where C = steps * (dtaps - 1 + ntaps - 1).
        """
        assert signal.val.dtype == torch.complex64, "Input signal must be complex64."
        x = signal.val  # [batch, L*sps, Nmodes]
        t = copy.deepcopy(signal.t)  # [start, stop, sps]
        P = 1e-3 * 10 ** (task_info[:, 0] / 10) / self.Nmodes  # Power [W]
        Dkernel = self.get_D_kernel(task_info[:, 1], task_info[:, 2]).to(
            signal.val.device
        )  # Dispersion kernel
        Nkernel = self.get_N_kernel(task_info)  # Nonlinear kernel

        for i in range(self.step):
            # Linear step
            x = dconv(x, Dkernel, stride=1)  # Convolution with dispersion kernel
            t.conv1d_t(self.dtaps, stride=1)

            # Nonlinear step
            start, stop = t.start, t.stop
            t.conv1d_t(self.ntaps, stride=1)
            phi = nconv(torch.abs(x) ** 2, Nkernel, 1)  # Nonlinear phase shift
            x = x[:, t.start - start : t.stop - stop + x.shape[1]] * torch.exp(
                1j * phi * self.gamma * P[:, None, None] * self.dz
            )

        return TorchSignal(x, t)


if __name__ == "__main__":

    net = MetaDBP(2, 5, 1001, 101)
    signal = TorchSignal(
        val=torch.randn(1, 10000, 2).to(torch.complex64), t=TorchTime(0, 0, 2)
    )
    task_info = torch.rand(1, 4)
    import time 
    t0 = time.time()
    y = net(signal, task_info)
    t1 = time.time()
    print('time:', t1 - t0)
    print(y)
