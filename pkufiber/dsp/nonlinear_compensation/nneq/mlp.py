import torch
import torch.nn as nn
from pkufiber.dsp.nonlinear_compensation.op import get_power


class EqMLP(nn.Module):
    """
    Complex [B, M, Nmodes] -> Complex [B, Nmodes]

    Parameters:
    M (int): Length of the input sequence.
    Nmodes (int): Number of modes (default=2).
    widths (list): Widths of the hidden layers (default=[149, 132, 596]).
    res_net (bool): Whether to use residual network (default=True).
    """
    M: int 
    Nmodes: int
    widths: list[int]
    res_net: bool

    def __init__(self, M: int, Nmodes=2, widths=[149, 132, 596], res_net=True):
        super(EqMLP, self).__init__()
        self.M = M
        self.overlaps = M - 1
        
        self.res_net = res_net
        self.Nmodes = Nmodes
        self.flatten = nn.Flatten()
        self.widths = widths
        self.fc_layers = nn.ModuleList()

        self.fc_layers.append(nn.Linear(M * Nmodes * 2, widths[0], bias=False))
        for i in range(len(widths) - 1):
            self.fc_layers.append(nn.Linear(widths[i], widths[i + 1], bias=False))
        self.fc_out = nn.Linear(widths[-1], Nmodes * 2, bias=False)
        nn.init.normal_(
            self.fc_out.weight, mean=0.0, std=0
        )  # Adjust the mean and std as needed
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        P = get_power(task_info, self.Nmodes, x.device)
        x = x * torch.sqrt(P[:, None, None])  # [batch, M, Nmodes]
        x0 = x[:, self.M // 2, :]
        x = torch.cat(
            [x.real, x.imag], dim=-1
        )  # Complex [B, M, Nmodes]  -> float [B, M, Nmodes*2]
        x = self.flatten(x)  # float [B, M, Nmodes*2] -> float [B, M*Nmodes*2]

        for fc in self.fc_layers:
            x = self.act(fc(x))  # float [B, *] -> float [B, widths[i]]
        x = self.fc_out(x)  # float [B, widths[-1]] -> float [B, Nmodes*2]

        # convert to complex
        x = x.view(-1, self.Nmodes, 2)  # float [B, Nmodes*2] -> float [B, Nmodes, 2]
        x = x[..., 0] + (1j) * x[..., 1]  # float [B, Nmodes, 2] -> complex [B, Nmodes]
        if self.res_net:
            x = x + x0
        x = x / torch.sqrt(P[:, None])  # [batch, Nmodes]
        return x
