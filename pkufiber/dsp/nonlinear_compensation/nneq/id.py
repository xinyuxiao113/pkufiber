import torch
import torch.nn as nn
from pkufiber.dsp.nonlinear_compensation.op import get_power


class EqID(nn.Module):
    """
    Complex [B, M, Nmodes] -> Complex [B, Nmodes]

    Parameters:
    M (int): Length of the input sequence.
    Nmodes (int): Number of modes (default=2).
    """
    M: int 
    Nmodes: int

    def __init__(self, M: int, Nmodes=2):
        super(EqID, self).__init__()
        self.M = M

    def forward(self, x, task_info):

        return x[:, self.M // 2, :]
