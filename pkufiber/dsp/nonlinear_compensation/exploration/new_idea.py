import torch
import numpy as np
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
    

    def rmps(self) -> int:
        from pkufiber.dsp.nonlinear_compensation.rmps import rmps_edc
        return self.pbc.rmps() + rmps_edc(self.M)*2


class EqSoNN(nn.Module):
    def __init__(self, M: int=41, rho: float=4.0, hdim:int=10, Nmodes=2):
        super(EqSoNN, self).__init__()
        self.M = M 
        self.overlaps = M - 1  
        self.pbc = EqAMPBC(M, rho)
        self.features = TripletFeatures(M, rho)
        self.hdim = hdim
        self.f1 = ComplexConv1d(in_channels=Nmodes, out_channels=hdim, kernel_size=self.features.hdim)
        self.f2 = ComplexConv1d(in_channels=Nmodes, out_channels=hdim, kernel_size=M)

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        # x [batch, L, Nmodes]
        xt = x.transpose(1,2)                                                  # [batch, Nmodes, L]
        P = get_power(task_info, x.shape[-1], x.device)                        # [batch]
        fo = self.features.nonlinear_features(x)                               # [batch, Nmodes, hdim]
        A = torch.cat([self.f1(fo), self.f1(fo.flip(dims=(1,)))], dim=-1)      # [batch, hdim, Nmodes]
        B = torch.cat([self.f2(xt), self.f2(xt.flip(dims=(1,)))], dim=-1)      # [batch, hdim, Nmodes]
        return self.pbc(x, task_info) + 1e-4/np.sqrt(self.hdim) * torch.sum(A*B*B.conj() + A.conj()*B*B, dim=1)*P[:,None]**2
    
    def rmps(self) -> int:
        raise NotImplementedError
        # return self.pbc.rmps() + self.features.rmps() + 2*self.hdim

if __name__ == "__main__":
    print('hello')


