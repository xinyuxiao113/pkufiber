import torch
import torch.nn as nn
from pkufiber.dsp.nonlinear_compensation.nneq import EqCNNBiLSTM 
from pkufiber.dsp.nonlinear_compensation.fno.fno import EqFno
from pkufiber.dsp.nonlinear_compensation.pbc import EqAMPBC


class EqAMPBCaddNN(nn.Module):
    def __init__(self, pbc_info, nn_info):
        super(EqAMPBCaddNN, self).__init__()
        self.pbc = EqAMPBC(**pbc_info)
        self.nn = EqCNNBiLSTM(**nn_info)
        self.M = max(self.pbc.M, self.nn.M)
        self.overlaps = self.M - 1

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        c = x.shape[1] // 2

        return (
            self.pbc(x[:, c - self.pbc.M // 2 : c + self.pbc.M // 2 + 1, :], task_info)
            + self.nn(x[:, c - self.nn.M // 2 : c + self.nn.M // 2 + 1, :], task_info)
            - x[:, x.shape[1] // 2, :]
        )
    
    def rmps(self) -> int:
        return self.pbc.rmps() + self.nn.rmps()

class EqAMPBCaddFNO(nn.Module):
    def __init__(self, pbc_info, nn_info):
        super(EqAMPBCaddFNO, self).__init__()
        self.pbc = EqAMPBC(**pbc_info)
        self.nn = EqFno(**nn_info)
        self.M = max(self.pbc.M, self.nn.M)
        self.overlaps = self.M - 1

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        c = x.shape[1] // 2

        return (
            self.pbc(x[:, c - self.pbc.M // 2 : c + self.pbc.M // 2 + 1, :], task_info)
            + self.nn(x[:, c - self.nn.M // 2 : c + self.nn.M // 2 + 1, :], task_info)
            - x[:, x.shape[1] // 2, :]
        )


if __name__ == "__main__":
    pbc_info = {"M": 41, "rho": 1, "fwm_share": False}
    nn_info = {"M": 41, "Nmodes": 2, "hidden_size": 226, "res_net": True}
    model = EqAMPBCaddNN(pbc_info, nn_info)
    print(model)
