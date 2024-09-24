"""
    NN equalizer.
"""

import torch.nn as nn, torch, numpy as np, torch
from typing import Union, List, Tuple, Optional

from pkufiber.core import TorchSignal, TorchTime
from pkufiber.dsp.layers import ComplexLinear, ComplexConv1d
from pkufiber.dsp.nonlinear_compensation.op import triplets, trip_op, get_power
from pkufiber.dsp.nonlinear_compensation.pbc.features import TripletFeatures, IndexType
from pkufiber.dsp.nonlinear_compensation.pbc.pbc import EqPBC, EqAMPBCstep





class EqFrePBC(nn.Module):
    '''
    PBC step.
    M: int, default=41, the length of the filter.
    rho: float, default=1, the parameter of the filter.
    index_type: str, default="reduce-1", ['full', 'reduce-1', 'reduce-2']
    pol_share: bool, default=False, whether the two polarizations share the same filter.

    '''
    def __init__(self, M: int = 41, rho: float = -1, overlaps: int = 40, strides: int = -1):
        assert overlaps % 2 == 0 and M % 2 == 1, "overlaps should be even and M should be odd."
        super(EqFrePBC, self).__init__()
        self.M = M
        self.rho = rho if rho > 0 else M/2
        self.index = self.get_index()
        self.fc = ComplexLinear(len(self.index), 1)
        self.overlaps = overlaps
        self.strides = strides

    def get_index(self):
        S = []
        for n1 in range(-(self.M//2), self.M//2):
            for n2 in range(-(self.M//2), self.M//2):
                if abs(n1*n2) <= self.rho * (self.M//2):
                    S.append((n1, n2))
        return S

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, L, Nmodes]
        task_info: [batch, 4]
        --> [batch, L - M + 1, Nmodes]
        """
        batch, L, Nmodes = x.shape
        P = get_power(task_info, Nmodes, x.device)
        x0 = x
        x = torch.fft.fft(x, dim=1)  # [batch, L, Nmodes]
        
        E3 = []
        for (n1, n2) in self.index:
            A = torch.roll(x, shifts=n1, dims=1)  * torch.roll(x, shifts=n1+n2, dims=1).conj()
            E3.append((A + A.roll(1, dims=-1)) * torch.roll(x, shifts=n2, dims=1))
    
        E = torch.stack(E3, dim=-1) # [batch, L,  Nmodes, L^2]
        delta = self.fc(E).squeeze(-1)
        delta = torch.fft.ifft(delta, dim=1) 

        # x0 + delta * P[:,None,None] = ifft(x + delta * P[:,None,None])
        return (x0 + delta * P[:,None,None])[:,(self.overlaps//2):L-(self.overlaps//2),:]

    def rmps(self, strides=-1) -> int:
        from pkufiber.dsp.nonlinear_compensation.rmps import rmps_edc, rmps_fft
        if strides == -1: strides = self.strides
        if strides == -1: strides = self.overlaps + 1

        FFT_size = strides + self.overlaps
        return (4*len(self.index)*3*FFT_size + rmps_fft(FFT_size)*2)/strides



class EqFreAMPBC(nn.Module):
    '''
    PBC step.
    M: int, default=41, the length of the filter.
    rho: float, default=1, the parameter of the filter.
    index_type: str, default="reduce-1", ['full', 'reduce-1', 'reduce-2']
    pol_share: bool, default=False, whether the two polarizations share the same filter.

    '''
    def __init__(self, M: int = 41, rho: float = -1, overlaps: int = 40, strides: int = -1):
        assert overlaps % 2 == 0 and M % 2 == 1, "overlaps should be even and M should be odd."
        super(EqFreAMPBC, self).__init__()
        self.M = M
        self.rho = rho if rho > 0 else M/2
        self.index = self.get_index()
        self.pbc = EqAMPBCstep(M=self.M, rho=self.rho, fwm_share=False, decision=False, discard=False)
        self.overlaps = overlaps
        self.strides = strides

    def get_index(self):
        S = []
        for n1 in range(-(self.M//2), self.M//2):
            for n2 in range(-(self.M//2), self.M//2):
                if abs(n1*n2) <= self.rho * (self.M//2):
                    S.append((n1, n2))
        return S

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, L, Nmodes]
        task_info: [batch, 4]
        --> [batch, L - M + 1, Nmodes]
        """
        batch, L, Nmodes = x.shape
        P = get_power(task_info, Nmodes, x.device)
        x0 = x
        x = torch.fft.fft(x, dim=1)             # [batch, L, Nmodes]
        delta = 0.01*self.pbc(x, task_info)          # [batch, L, Nmodes]
        delta = torch.fft.ifft(delta, dim=1)    # [batch, L, Nmodes]
        return (x0 + delta * P[:,None,None])[:,(self.overlaps//2):L-(self.overlaps//2),:]

    def rmps(self, strides=-1) -> int:
        from pkufiber.dsp.nonlinear_compensation.rmps import rmps_edc, rmps_fft
        if strides == -1: strides = self.strides
        if strides == -1: strides = self.overlaps + 1

        FFT_size = strides + self.overlaps
        return (4*len(self.index)*3*FFT_size + rmps_fft(FFT_size)*2)/strides


class EqConvAMPBC(nn.Module):
    '''
    PBC step.
    M: int, default=41, the length of the filter.
    rho: float, default=1, the parameter of the filter.
    index_type: str, default="reduce-1", ['full', 'reduce-1', 'reduce-2']
    pol_share: bool, default=False, whether the two polarizations share the same filter.

    '''
    def __init__(self, M: int = 41, rho: float = -1, overlaps: int = 40, strides: int = -1, kernel_size: int=11):
        assert overlaps % 2 == 0 and M % 2 == 1, "overlaps should be even and M should be odd."
        super(EqConvAMPBC, self).__init__()
        self.M = M
        self.rho = rho if rho > 0 else M/2
        self.index = self.get_index()
        self.pbc = EqAMPBCstep(M=self.M, rho=self.rho, fwm_share=False, decision=False, discard=False)
        self.overlaps = overlaps
        self.strides = strides

        self.emb = ComplexConv1d(2,2, kernel_size=kernel_size, padding=kernel_size//2, padding_mode='circular')
        self.dec = ComplexConv1d(2,2, kernel_size=kernel_size, padding=kernel_size//2, padding_mode='circular')

    def get_index(self):
        S = []
        for n1 in range(-(self.M//2), self.M//2):
            for n2 in range(-(self.M//2), self.M//2):
                if abs(n1*n2) <= self.rho * (self.M//2):
                    S.append((n1, n2))
        return S

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, L, Nmodes]
        task_info: [batch, 4]
        --> [batch, L - M + 1, Nmodes]
        """
        batch, L, Nmodes = x.shape
        P = get_power(task_info, Nmodes, x.device)
        x0 = x
        x = self.emb(x.transpose(1,2)).transpose(1,2)            # [batch, L, Nmodes]
        delta = 0.01*self.pbc(x, task_info)                      # [batch, L, Nmodes]
        delta = self.dec(delta.transpose(1,2)).transpose(1,2)    # [batch, L, Nmodes]
        return (x0 + delta * P[:,None,None])[:,(self.overlaps//2):L-(self.overlaps//2),:]

    def rmps(self, strides=-1) -> int:
        from pkufiber.dsp.nonlinear_compensation.rmps import rmps_edc, rmps_fft
        if strides == -1: strides = self.strides
        if strides == -1: strides = self.overlaps + 1

        FFT_size = strides + self.overlaps
        return (4*len(self.index)*3*FFT_size + rmps_fft(FFT_size)*2)/strides


                
if __name__ == "__main__":
    M = 41
    Nmodes = 2
    batch = 2
    x = torch.randn(batch, 100, Nmodes) + 0j
    task_info = torch.randn(batch, 4)
    eq = EqConvAMPBC(M)
    y = eq(x, task_info)
    print(y.shape)  # [batch, L, Nmodes]