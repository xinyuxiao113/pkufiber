'''
DBP sps = 1:  after CDC + ADF.
'''

import torch.nn as nn, torch, numpy as np, torch
from typing import Union, List, Tuple, Optional

from pkufiber.core import TorchSignal, TorchTime
from pkufiber.dsp.layers import ComplexLinear, ComplexConv1d
from pkufiber.dsp.nonlinear_compensation.op import triplets, trip_op, get_power
from pkufiber.dsp.op import dispersion_kernel, dconv, nconv
from pkufiber.op import get_beta2, get_beta1
from pkufiber.dsp.nonlinear_compensation.pbc.features import TripletFeatures, IndexType

'''
每个模型都需要 M, overlaps参数
'''

class NonlinearFilter(nn.Module):
    
    def __init__(self, Nmodes, step, ntaps, share, gamma=0.0016567, L=2000e3):  
        super(NonlinearFilter, self).__init__()
        self.Nmodes = Nmodes
        self.L = L
        self.step = step
        self.ntaps = ntaps
        self.gamma = gamma

        n_num = 1 if share else self.step
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

    def forward(self, x, task_info, i):
        '''
        [B, M, Nmodes] -> [B, M - ntaps + 1, Nmodes]
        '''
        P = 1e-3 * 10 ** (task_info[:, 0] / 10) / self.Nmodes  # Power [W]
        phi = nconv(torch.abs(x) ** 2,
                self.Nkernel[min(i, len(self.Nkernel) - 1)].expand(
                    x.shape[0], self.Nmodes, self.Nmodes, self.ntaps
                ),
                1,)
        x = x[:, self.ntaps//2:x.shape[1] - (self.ntaps//2)] * torch.exp(1j*phi*self.gamma*P[:,None,None]*self.L / self.step)
        return x
        



class EqDBP_test(nn.Module):
    """
    PBC equalizer.
    M: int, default=41, the length of the filter.
    rho: float, default=1, the parameter of the filter.
    index_type: str, default="reduce-1", ['full', 'reduce-1', 'reduce-2']
    pol_share: bool, default=False, whether the two polarizations share the same filter.

    """

    def __init__(self, Nmodes, step, dtaps=2001, ntaps=401, n_share=True, Fs=80e9, D=16.5, Fc: float = 299792458 / 1550e-9, Fi: float = 299792458 / 1550e-9, L=2000e3, gamma=0.0016567):
        super(EqDBP_test, self).__init__()
        self.Nmodes = Nmodes
        self.step = step
        self.Fs = Fs
        self.D = D 
        self.Fc = Fc
        self.Fi = Fi
        self.L = L
        self.gamma = gamma
        self.dtaps = dtaps 
        self.ntaps = ntaps
        self.overlaps = ((dtaps - 1)*2 + ntaps - 1) * step 
        self.beta2 = get_beta2(self.D, self.Fc) / 1e3            # Second-order dispersion coefficient [s^2/m]
        self.beta1 = get_beta1(self.D, self.Fc, self.Fi) / 1e3   # First-order dispersion coefficient [s/m]

        self.nonlinear = NonlinearFilter(Nmodes, step, ntaps, n_share, L=L, gamma=gamma)


    def disp_freq(self, x, dz, dtaps):
        kernel = dispersion_kernel(dz, x.shape[1], self.Fs, self.beta2, self.beta1, domain="freq").to(torch.complex64) 
        kernel = kernel.to(x.device)
        x = torch.fft.ifft(torch.fft.fft(x, dim=1) * kernel[...,None], dim=1)
        return x[:, (dtaps//2):-(dtaps//2),:]



    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        """
        Input:
            signal:  [batch, M, Nmodes]
            task_info: torch.Tensor or None. [B, 4] ot None.    [P,Fi,Fs,Nch]
        Output:
            Tensor. [batch, Nmodes]
        """
        Nmodes = x.shape[-1]
        if Nmodes == 1 and self.pol_num == 2:
            raise ValueError(
                "Nmodes=1 and pol_num=2 is not a good choise, please set pol_share=True."
            )
        
        #  step 1: add dispersion 
        x = self.disp_freq(x, self.L, (self.dtaps-1)*self.step+1)

        # step 2: DBP on sps = 1
        for i in range(self.step):
            x = self.disp_freq(x, -self.L / self.step, self.dtaps)
            x = self.nonlinear(x, task_info, i)

        return x


    def rmps(self) -> int:
        '''
        real mulitplication times per sample.
        x 4: 4 real multiplications in each complex multiplication.
        '''
        # return TripletFeatures(self.M, self.rho, 'full').hdim * 3 * 4
        raise NotImplementedError


if __name__ == "__main__":
    x = torch.randn(5, 40000, 2) + 1j
    task_info = torch.randn(5, 4)
    model = EqDBP_test(Nmodes=2, step=5)
    y = model(x, task_info)
    print(y.shape)
    # print(model.rmps())
    print(model.overlaps)
    print(model)
    print(model.__class__.__name__)