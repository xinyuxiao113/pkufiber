import torch
import torch.nn as nn
from pkufiber.dsp.nonlinear_compensation.op import get_power


class EqBiLSTM(nn.Module):
    """
        Complex [B, M, Nmodes] -> Complex [B, Nmodes]

        Parameters:
        M (int): Length of the input sequence.
        Nmodes (int): Number of modes (default=2).
        hidden_size (int): Hidden size for LSTM (default=226).
        res_net (bool): Whether to use residual network (default=True).
    """
    def __init__(self, M: int=41, Nmodes=2, hidden_size=226, res_net=True):
        super(EqBiLSTM, self).__init__()
        self.M = M
        self.res_net = res_net
        self.Nmodes = Nmodes
        self.lstm = nn.LSTM(
            input_size=Nmodes * 2,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(2 * hidden_size * M, Nmodes * 2, bias=False)
        # nn.init.normal_(self.dense.weight, mean=0.0, std=0.01)  # Adjust the mean and std as needed

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        P = get_power(task_info, self.Nmodes, x.device)
        x = x * torch.sqrt(P[:, None, None])  # [batch, M, Nmodes]
        x0 = x[:, self.M // 2, :]  # Complex [B, M, Nmodes]  -> complex [B, Nmodes]
        x = torch.cat(
            [x.real, x.imag], dim=-1
        )  # Complex [B, M, Nmodes]  -> float [B, M, Nmodes*2]

        x, _ = self.lstm(x)  # float [B, M, Nmodes*2]  -> float [B, M, hidden_size*2]
        x = self.flatten(x)
        x = self.dense(x)  # float [B, M*hidden_size*2] -> float [B, Nmodes*2]

        # convert to complex
        x = x.view(-1, self.Nmodes, 2)  # float [B, Nmodes*2] -> float [B, Nmodes, 2]
        x = x[..., 0] + (1j) * x[..., 1]  # float [B, Nmodes, 2] -> complex [B, Nmodes]
        if self.res_net:
            x = x + x0
        x = x / torch.sqrt(P[:, None])  # [batch, Nmodes]
        return x
