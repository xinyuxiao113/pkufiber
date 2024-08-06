'''
Pure Frequency-domain Digital Back Propagation. 
'''

import torch, copy, numpy as np, matplotlib.pyplot as plt
import torch.nn.functional as F, torch.nn.init as init, torch.nn as nn
from typing import Union, Tuple

from pkufiber.op import get_beta2, get_beta1
from pkufiber.core import TorchSignal, TorchTime, TorchInput
from pkufiber.dsp.op import dispersion_kernel, dconv, nconv
from pkufiber.dsp.layers import MLP, Parameter, ComplexLinear, ComplexConv1d


class FreqDBP(nn.Module):
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
        M: int = 41,
        rho: float = 1,
        overlaps: int=100,
        
        L: float = 2000e3,
        gamma: float = 0.0016567,
        D: float = 16.5,
        Fs: float = 160e9,
        Fc: float = 299792458 / 1550e-9,
        Fi: float = 299792458 / 1550e-9,
    ):
        super(FreqDBP, self).__init__()
        self.Nmodes = Nmodes
        self.step = step
        self.M = M 
        self.rho = rho
        self.index = self.get_index()
        self.dz = L / step  # Step size in meters
        self.gamma = gamma
        self.D = D
        self.Fc = Fc
        self.Fi = Fi
        self.Fs = Fs
        self.overlaps = overlaps
        self.beta2 = get_beta2(self.D, self.Fc) / 1e3            # Second-order dispersion coefficient [s^2/m]
        self.beta1 = get_beta1(self.D, self.Fc, self.Fi) / 1e3   # First-order dispersion coefficient [s/m]
        self.fc = ComplexLinear(len(self.index), 1)
    
    def get_index(self):
        S = []
        for n1 in range(-(self.M//2), self.M//2):
            for n2 in range(-(self.M//2), self.M//2):
                if abs(n1*n2) <= self.rho * (self.M//2):
                    S.append((n1, n2))
        return S
    
    def nonlinear_step(self, x_freq: torch.Tensor, P: torch.Tensor):
        features = []
        for (n1, n2) in self.index:
            A = torch.roll(x_freq, shifts=n1, dims=1)  * torch.roll(x_freq, shifts=n1+n2, dims=1).conj()
            features.append((A + A.roll(1, dims=-1)) * torch.roll(x_freq, shifts=n2, dims=1))
    
        E = torch.stack(features, dim=-1) # [batch, L,  Nmodes, L^2]
        delta = self.fc(E).squeeze(-1)
        return x_freq + delta * P[:,None,None]



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
        x = signal.val                 # [batch, L*sps, Nmodes]
        x_freq = torch.fft.fft(x, dim=1)  # [batch, L*sps, Nmodes]
        t = copy.deepcopy(signal.t)  # [start, stop, sps]
        P = 1e-3 * 10 ** (task_info[:, 0] / 10) / self.Nmodes  # Power [W]
        Dkernel = dispersion_kernel(-self.dz, x.shape[1], self.Fs, self.beta2, self.beta1, domain="freq").to(torch.complex64) 
        Dkernel = Dkernel.to(x.device)

        for i in range(self.step):
            # Linear step
            x_freq = x_freq * Dkernel[...,None] 
            # Nonlinear step
            x_freq = self.nonlinear_step(x_freq, P)
        
        x = torch.fft.ifft(x_freq, dim=1)[:,(self.overlaps//2):x.shape[1]-(self.overlaps//2),:]
        t = TorchTime(self.overlaps//2, -(self.overlaps//2), t.sps)
        return TorchSignal(x, t)


if __name__ == "__main__":
    device = 'cuda:0'
    net = FreqDBP(2, 5, M=41, rho=1)
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
