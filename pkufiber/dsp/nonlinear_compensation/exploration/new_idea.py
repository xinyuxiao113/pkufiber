import torch
import torch.nn as nn
from pkufiber.dsp.nonlinear_compensation.nneq import EqCNNBiLSTM 
from pkufiber.dsp.nonlinear_compensation.fno.fno import EqFno
from pkufiber.dsp.nonlinear_compensation.pbc import EqAMPBC
from pkufiber.dsp.nonlinear_compensation.pbc.features import TripletFeatures
from pkufiber.dsp.layers import ComplexConv1d, ComplexLinear
from pkufiber.dsp.nonlinear_compensation.op import triplets, trip_op, get_power
 

class EqAMPBCaddConv(nn.Module):
    def __init__(self, M: int=41, rho: float=4.0, Nmodes=2, fwm_share: bool=False, decision=False):
        super(EqAMPBCaddConv, self).__init__()
        self.M = M
        self.overlaps = M - 1
        self.pbc = EqAMPBC(M, rho, fwm_share, decision)
        self.nn = ComplexConv1d(in_channels=Nmodes, out_channels=Nmodes, kernel_size=M, init='zeros')

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        '''
            x: [batch, L, Nmodes]
        '''
        c = x.shape[1] // 2
        return (self.pbc(x[:, c - self.pbc.M // 2 : c + self.pbc.M // 2 + 1, :], task_info) 
                + self.nn(x.transpose(1,2)).squeeze() )


class EqSoNN(nn.Module):
    def __init__(self, M: int=41, rho: float=4.0, hdim:int=10):
        super(EqSoNN, self).__init__()
        self.pbc = EqAMPBC(M, rho)
        self.features = TripletFeatures(M, rho)
        self.f1 = ComplexLinear(in_features=self.features.hdim, out_features=hdim)
        self.f2 = ComplexLinear(in_features=self.features.hdim, out_features=hdim)

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        # x [batch, L, Nmodes]
        P = get_power(task_info, x.shape[-1], x.device)  # [batch]
 
        fo = self.features.nonlinear_features(x)  # [batch, Nmodes, fo_hdim]
        A = self.f1(fo)                               # [batch, Nmodes, hdim]
        B = self.f2(x.transpose(1,2))                 # [batch, Nmodes, hdim]
        return self.pbc(x, task_info) + torch.sum(A*B*B.conj() + A*B*B, dim=-1)*P[:,None]**2

if __name__ == "__main__":
    print('hello')


