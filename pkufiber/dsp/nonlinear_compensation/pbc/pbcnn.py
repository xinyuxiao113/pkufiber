"""
    NN equalizer.
"""

import torch.nn as nn, torch, numpy as np, torch
from typing import Union, List, Tuple, Optional

from pkufiber.core import TorchSignal, TorchTime
from pkufiber.dsp.layers import ComplexLinear, ComplexConv1d, CReLU, CLeakyReLU, Clinear
from pkufiber.dsp.nonlinear_compensation.op import triplets, trip_op, get_power

from pkufiber.dsp.nonlinear_compensation.pbc.features import TripletFeatures, IndexType

act_dict = {
    'CReLU': CReLU,
    'CLeakyReLU': CLeakyReLU,
    'Clinear': Clinear,
}



class EqPBCNN(nn.Module):
    """
    PBC equalizer.
    M: int, default=41, the length of the filter.
    rho: float, default=1, the parameter of the filter.
    index_type: str, default="reduce-1", ['full', 'reduce-1', 'reduce-2']
    pol_share: bool, default=False, whether the two polarizations share the same filter.

    """

    def __init__(self, M: int = 41, rho: float=1., hidden_sizes:list=[2, 10], index_type: Union[IndexType, str]="reduce-1", pol_share: bool=False, decision=False, act='CLeakyReLU'):
        super(EqPBCNN, self).__init__()
        assert M % 2 == 1, "M must be odd."
        self.features = TripletFeatures(M, rho, index_type, decision)
        self.pol_num = 1 if pol_share else 2
        self.overlaps = M - 1
        self.hidden_sizes = hidden_sizes   


        activation = act_dict[act]
        self.nn = nn.ModuleList(
            [
                nn.Sequential(
                    ComplexLinear(self.features.hdim,hidden_sizes[0] , bias=False, zero_init=False),
                    activation(),
                    ComplexLinear(hidden_sizes[0], hidden_sizes[1], bias=False, zero_init=False),
                    activation(),
                    ComplexLinear(hidden_sizes[1], 1, bias=False, zero_init=False)
                    )
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
        features = self.features.nonlinear_features(x)  # [batch, Nmodes, len(S)]
        E = [
            self.nn[min(i, self.pol_num - 1)](features[..., i, :])
            for i in range(Nmodes)
        ]  # [batch, 1]
        E = torch.cat(E, dim=-1)  # [batch, Nmodes]
        return x[:, self.features.M // 2, :] + E * P[:, None]  # [batch, Nmodes]

    def rmps(self) -> int:
        nn_cmps =  self.features.hdim*self.hidden_sizes[0] + self.hidden_sizes[0]*self.hidden_sizes[1] + self.hidden_sizes[1]*1
        return self.features.rmps() + 4 * nn_cmps


if __name__ == "__main__":
    M = 41
    Nmodes = 2
    batch = 10
    x = torch.randn(batch, M, Nmodes) + 1j * torch.randn(batch, M, Nmodes)
    task_info = torch.randn(batch, 4)
    eq = EqPBCNN(M=M)
    y = eq(x, task_info)
    print(y.shape)