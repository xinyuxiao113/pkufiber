"""
    NN equalizer.
"""

import torch.nn as nn, torch, numpy as np, torch
from typing import Union, List, Tuple, Optional

from pkufiber.core import TorchSignal, TorchTime
from pkufiber.dsp.layers import ComplexLinear, ComplexConv1d
from pkufiber.dsp.nonlinear_compensation.op import triplets, trip_op, get_power
from pkufiber.dsp.nonlinear_compensation.pbc.features import TripletFeatures, IndexType, SoFeatures
import pkufiber.dsp.nonlinear_compensation.pbc as pbc

class EqSoPBC(nn.Module):
    """
    PBC equalizer.
    M: int, default=41, the length of the filter.
    rho: float, default=1, the parameter of the filter.
    index_type: str, default="reduce-1", ['full', 'reduce-1', 'reduce-2']
    pol_share: bool, default=False, whether the two polarizations share the same filter.

    """

    def __init__(self, Nmodes:int=2, M2:int=11, rho2:float=1.0,  pol_share: bool=False, decision=False, fo_pbc:str='EqAMPBC', fo_info:dict={}):
        super(EqSoPBC, self).__init__()
        self.M = fo_info['M']
        self.overlaps = self.M - 1
        self.pbc = getattr(pbc, fo_pbc)(**fo_info)
        self.M2 = M2
        assert self.M >= self.M2, "M should be larger than M2."
        self.features2 = SoFeatures(M2, Nmodes, rho2, decision)
        self.pol_num = 1 if pol_share else 2
        self.nn = nn.ModuleList(
            [
                ComplexLinear(self.features2.hdim, 1, bias=False)
                for _ in range(self.pol_num)
            ]
        )


    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        """
        Input:
            signal:  [batch, M, Nmodes]
            task_info: torch.Tensor or None. [B, 4] ot None.    [P,Fi,Fs,Nch]
        Output:
            Tensor. [batch, Nmodes]
            Nmodes = 1:
                O_{b,k,i} = gamma P0^{3/2} * sum_{m,n} E_{b, k+n, i} E_{b, k+m+n, i}^* E_{b, k+m, i} C_{m,n}
            Nmodes = 2:
                O_{b,k,i} = gamma P0^{3/2} * sum_{m,n} (E_{b, k+n, i} E_{b, k+m+n, i}^* +  E_{b, k+n, -i} E_{b, k+m+n, -i}^*) E_{b, k+m, i} C_{m,n}
        """
        Nmodes = x.shape[-1]
        if Nmodes == 1 and self.pol_num == 2:
            raise ValueError(
                "Nmodes=1 and pol_num=2 is not a good choise, please set pol_share=True."
            )
        P = get_power(task_info, Nmodes, x.device)
        p = x.shape[1] // 2
        
        features2 = self.features2(x[:,p-(self.M2//2):p+(self.M2//2)+1,:])      # [batch, Nmodes, h2]

        E2 = [
            self.nn[min(i, self.pol_num - 1)](features2[..., i, :])
            for i in range(Nmodes)
        ]  # [batch, 1]
        E2 = torch.cat(E2, dim=-1)  # [batch, Nmodes]

        return self.pbc(x, task_info)  + 0.01 * E2 * P[:, None]**2
         # [batch, Nmodes]

    def rmps(self) -> int:

        return self.pbc.rmps() + self.features2.rmps() + 4*self.features2.hdim 
    
if __name__ == '__main__':
    M = 41
    Nmodes = 2
    batch = 10
    x = torch.randn(batch, M, Nmodes) + 1j * torch.randn(batch, M, Nmodes)
    task_info = torch.randn(batch, 4)
    eq = EqSoPBC(Nmodes, M2=11, rho2=1.0, fo_pbc='EqAMPBC', fo_info={'M':41, 'rho':1.0})
    # eq = eqPBC_step(M, 1)
    y = eq(x, task_info)
    print(y.shape)
