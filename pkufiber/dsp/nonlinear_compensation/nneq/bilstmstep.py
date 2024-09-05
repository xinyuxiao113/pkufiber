import torch
import torch.nn as nn
from pkufiber.dsp.nonlinear_compensation.op import get_power


class EqBiLSTMstep(nn.Module):
    """
        Complex [B, M, Nmodes] -> Complex [B, Nmodes]

        Parameters:
        M (int): Length of the input sequence.
        Nmodes (int): Number of modes (default=2).
        hidden_size (int): Hidden size for LSTM (default=226).
        res_net (bool): Whether to use residual network (default=True).
    """
    def __init__(self, M: int=41, Nmodes=2, hidden_size=226, overlaps=20, res_net=True):
        super(EqBiLSTMstep, self).__init__()
        self.M = M
        self.overlaps = overlaps
        self.hidden_size = hidden_size 
        self.res_net = res_net
        self.Nmodes = Nmodes
        self.lstm = nn.LSTM(
            input_size=Nmodes * 2,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dense = nn.Linear(2 * hidden_size, Nmodes * 2, bias=False)
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.001)  # Adjust the mean and std as needed

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        P = get_power(task_info, self.Nmodes, x.device)
        x = x * torch.sqrt(P[:, None, None])  # [batch, M, Nmodes]
        x0 = x[:,(self.overlaps//2):-(self.overlaps//2),:]               
        x = torch.cat(
            [x.real, x.imag], dim=-1
        )  # Complex [B, M, Nmodes]  -> float [B, M, Nmodes*2]

        x, _ = self.lstm(x)  # float [B, M, Nmodes*2]  -> float [B, M, hidden_size*2]

        x = x[:,(self.overlaps//2):-(self.overlaps//2),:] 

        x = self.dense(x)    # float [B, M - overlaps, hidden_size*2] -> float [B, M - overlaps,  Nmodes*2]

        # convert to complex
        x = x.view(x.shape[0], x.shape[1], self.Nmodes, 2)  # [B, M - overlaps, Nmodes * 2] -> [B, M - overlaps, Nmodes, 2]
        x = x[..., 0] + (1j) * x[..., 1]  # float [B, M - overlaps,  Nmodes, 2] -> complex [B, M - overlaps, Nmodes]
        if self.res_net:
            x = x + x0
        x = x / torch.sqrt(P[:, None, None])  # [batch, M - overlaps,  Nmodes]
        return x

    def rmps(self) -> int:
        '''
        Performance versus Complexity Study of Neural Network Equalizers in Coherent Optical Systems
        '''
        ns = self.M 
        nh = self.hidden_size
        ni = 4
        no = 2
        return 2*ns*nh*(4*ni + 4*nh + 3 + no)//(ns - self.overlaps)