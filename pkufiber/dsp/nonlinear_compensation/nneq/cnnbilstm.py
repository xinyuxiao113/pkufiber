import torch
import torch.nn as nn
from pkufiber.dsp.nonlinear_compensation.op import get_power

class EqCNNBiLSTM(nn.Module):
    """
    Complex [B, M, Nmodes] -> Complex [B, Nmodes]

    Parameters:
    M (int): Length of the input sequence.
    Nmodes (int): Number of modes (default=2).
    channels (int): Number of channels for Conv1d (default=244).
    kernel_size (int): Kernel size for Conv1d (default=10).
    hidden_size (int): Hidden size for LSTM (default=113).
    num_layers (int): Number of LSTM layers (default=1).
    res_net (bool): Whether to use residual network (default=True).
    """

    def __init__(
        self,
        M: int = 41,
        Nmodes: int = 2,
        channels: int = 64,
        kernel_size: int = 11,
        hidden_size: int = 40,
        num_layers: int = 1,
        res_net: bool = True,
    ):
        super(EqCNNBiLSTM, self).__init__()
        self.M = M
        self.overlaps = M - 1
        self.res_net = res_net
        self.Nmodes = Nmodes
        self.channels = channels
        self.kernel_size = kernel_size  
        self.hidden_size = hidden_size  

        self.conv1d = nn.Conv1d(
            in_channels=2 * Nmodes, 
            out_channels=channels, 
            kernel_size=kernel_size
        )
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(
            2 * hidden_size * (M - kernel_size + 1), Nmodes * 2, bias=False
        )
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.0)  # Adjust as needed

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        # Get power and scale input
        P = get_power(task_info, self.Nmodes, x.device)  # [batch]
        x = x * torch.sqrt(P[:, None, None])  # [batch, M, Nmodes]
        x0 = x[:, self.M // 2, :]

        # Convert to real and imaginary parts
        x = torch.cat([x.real, x.imag], dim=-1)  # [B, M, Nmodes*2]
        x = x.permute(0, 2, 1)  # [B, Nmodes*2, M]

        # Apply convolution and activation
        x = self.conv1d(x)  # [B, channels, M-kernel_size+1]
        x = self.leaky_relu(x)  # [B, channels, M-kernel_size+1]

        # Reorder dimensions for LSTM
        x = x.permute(0, 2, 1)  # [B, M-kernel_size+1, channels]

        # Apply LSTM
        x, _ = self.lstm(x)  # [B, M-kernel_size+1, hidden_size*2]

        # Flatten and apply dense layer
        x = self.flatten(x)  # [B, M*hidden_size*2]
        x = self.dense(x)  # [B, Nmodes*2]

        # Convert back to complex
        x = x.view(-1, self.Nmodes, 2)  # [B, Nmodes, 2]
        x = x[..., 0] + (1j) * x[..., 1]  # [B, Nmodes]

        # Apply residual connection if enabled
        if self.res_net:
            x = x0 +  x

        x = x / torch.sqrt(P[:, None])  # [batch, Nmodes]
        return x
    
    def rmps(self) -> int:
        '''
        Performance versus Complexity Study of Neural Network Equalizers in Coherent Optical Systems
        '''
        ni = 4
        nf = self.channels
        nk = self.kernel_size
        ns = self.M
        nh = self.hidden_size
        no = 2
        return ni*nf*nk*(ns-nk+1) + (ns - nk + 1)*2*nh*(4*nf+4*nh+3+no)