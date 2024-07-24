import torch
import torch.nn as nn
from pkufiber.dsp.nonlinear_compensation.op import get_power
from neuralop.models import FNO


class EqFno(nn.Module):

    def __init__(self, M: int=41, Nmodes:int=2, fno_layers: int=4, fno_modes: int=16, fno_channels: int=16, lifting_channels: int=64, projection_channels: int=64):
        super(EqFno, self).__init__()
        assert M % 2 == 1, 'M must be odd'
        # [B, in_channels, L]   ->   [B, out_channels, L]
        self.fno = FNO(n_modes=(fno_modes,), hidden_channels=fno_channels, 
                       lifting_channels=lifting_channels, projection_channels=projection_channels,
                       in_channels=2*Nmodes, out_channels=2*Nmodes, n_layers=fno_layers) 
        self.Nmodes = Nmodes 
        self.M = M
        self.overlaps = M - 1
    
    def forward(self, x, task_info):
        '''
        x: [batch, L, Nmodes]   task_info: [batch, 4]
        --> [batch, L - M + 1, Nmodes]
        '''
        x0 = x
        P = get_power(task_info, self.Nmodes, x.device)  # [batch]
        x = x * torch.sqrt(P[:, None, None])  # [batch, L, Nmodes]
        x = torch.cat([x.real, x.imag], dim=-1).transpose(1,2)  # [B, Nmodes*2, L]
        x = self.fno(x) # [B, Nmodes*2, L]
        x = x.transpose(1,2)  # [B, L, Nmodes*2]
        x = x[...,0:self.Nmodes] + x[...,self.Nmodes:2*self.Nmodes]*1j  # [B, L, Nmodes]  complex
        x = x / torch.sqrt(P[:, None, None])  # [batch, L, Nmodes]
        return (x0 + x)[:,self.M//2:-(self.M//2),:]  # [batch, L - M + 1, Nmodes]
    
if __name__ == '__main__':
    x = torch.randn(5, 100, 2) + 1j*torch.randn(5, 100, 2)
    task_info = torch.randn(5, 4)
    model = EqFno(41, 2, 16, 64)
    y = model(x, task_info)
    print(y.shape)  # torch.Size([2, 41, 2])