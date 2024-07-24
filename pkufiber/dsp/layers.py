"""
    This file contains the implementation of complex-valued layers.
"""

import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.init as init


class Id(nn.Module):
    def __init__(self):
        super(Id, self).__init__()

    def forward(self, x):
        return x


class StepFunction(nn.Module):
    def __init__(self, start: int = -8, stop: int = 8):
        super(StepFunction, self).__init__()
        num = stop - start + 1
        self.start = start
        self.stop = stop
        self.function_values = nn.Parameter(torch.ones(num))

    def forward(self, x):
        x = torch.minimum(
            torch.maximum(x, torch.tensor(self.start)), torch.tensor(self.stop)
        )
        idx = (x - self.start).to(torch.int)
        return self.function_values[idx]


class MLP(nn.Module):
    """
    Multi-layer perceptron.

    R^3 -> R^1
    """

    def __init__(self, input_size=4, hidden_size=100, output_size=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 50)
        self.fc3 = nn.Linear(50, 2)
        self.fc4 = nn.Linear(2, output_size, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return 0.1 * x


class Parameter(nn.Module):

    def __init__(self, output_size=401):
        super(Parameter, self).__init__()

        self.parameter = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        assert x.ndim <= 2
        if x.ndim == 1:
            return self.parameter
        else:
            batch = x.shape[0]
            return torch.stack([self.parameter] * batch, dim=0)


class CReLU(nn.Module):
    def forward(self, x):
        return F.relu(x.real) + F.relu(x.imag) * (1j)
    
class CLeakyReLU(nn.Module):
    def forward(self, x):
        return F.leaky_relu(x.real) + F.leaky_relu(x.imag) * (1j)
    
class Clinear(nn.Module):
    def forward(self, x):
        return x


class ComplexLinear(nn.Module):
    """
    Complex linear layer.
    """

    def __init__(self, in_features, out_features, zero_init=True, **kwargs):
        super(ComplexLinear, self).__init__()
        self.real = nn.Linear(in_features, out_features, **kwargs)
        self.imag = nn.Linear(in_features, out_features, **kwargs)

        # 初始化最后一层的权重为零
        if zero_init:
            init.zeros_(self.real.weight)
            init.zeros_(self.imag.weight)

    def forward(self, x):
        return torch.complex(
            self.real(x.real) - self.imag(x.imag), self.imag(x.real) + self.real(x.imag)
        )


class ComplexMLP(nn.Module):
    """
    Complex MLP.
    """

    def __init__(self, input_size=4, hidden_size=100, output_size=401):
        super(ComplexMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 20)
        self.fc3 = nn.Linear(20, output_size)

        # 初始化最后一层的权重为零
        init.zeros_(self.fc3.weight)
        init.zeros_(self.fc3.bias)
        # init.zeros_(self.fc2.bias)
        # init.zeros_(self.fc1.bias)

    def forward(self, x):
        # x = F.normalize(x, dim=0, p=2)    # [batch, task_dim]
        x = x / torch.tensor([1, 1.9e14, 8e10, 10]).to(x.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ComplexLSTM(nn.Module):
    """
    Complex LSTM layer.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, **kwargs):
        super(ComplexLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_directions = 1
        self.LSTM = nn.LSTM(
            input_size * 2, hidden_size * 2, num_layers, batch_first=True, **kwargs
        )

    def forward(self, x, state):
        """
        input: x, state
            x: [batch, L, Ci] complex
            state: ([depth, batch, Co*2], [depth, batch, Co*2])
        output:
            y, state
            y:  [batch, L, Ch]

        """
        # x: [batch, L, Ci]
        x = torch.cat([x.real, x.imag], dim=-1)  # [batch, L, 2*Ci]
        x, state = self.LSTM(x, state)  # [batch, L, 2*Ch]
        xs = torch.chunk(x, 2, dim=-1)
        x = torch.complex(xs[0], xs[1])
        return x, state

    def init_carry(self, batch, device):
        h_zeros = torch.zeros(
            self.num_layers,
            batch,
            self.hidden_size * 2,
            dtype=torch.float32,
            device=device,
        )
        c_zeros = torch.zeros(
            self.num_layers,
            batch,
            self.hidden_size * 2,
            dtype=torch.float32,
            device=device,
        )
        hx = (h_zeros, c_zeros)
        return hx


class ComplexGRU(nn.Module):
    """
    Complex  GRU layer.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, **kwargs):
        super(ComplexGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_directions = 1
        self.GRU = nn.GRU(
            input_size * 2, hidden_size * 2, num_layers, batch_first=True, **kwargs
        )

    def forward(self, x, state):
        """
        input: x, state
            x: [batch, L, Ci] complex
            state: ([depth, batch, Co*2], [depth, batch, Co*2])
        output:
            y, state
            y:  [batch, L, Ch]

        """
        # x: [batch, L, Ci]
        x = torch.cat([x.real, x.imag], dim=-1)  # [batch, L, 2*Ci]
        x, state = self.GRU(x, state)  # [batch, L, 2*Ch]
        xs = torch.chunk(x, 2, dim=-1)
        x = torch.complex(xs[0], xs[1])
        return x, state

    def init_carry(self, batch, device):
        h_zeros = torch.zeros(
            self.num_layers,
            batch,
            self.hidden_size * 2,
            dtype=torch.float32,
            device=device,
        )
        return h_zeros


class ComplexConv1d(nn.Module):
    """
    Complex 1D convolution layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        init="xavier",
    ):
        super(ComplexConv1d, self).__init__()
        self.conv1d_r = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )
        self.conv1d_i = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )
        if init == "xavier":
            nn.init.xavier_normal_(self.conv1d_r.weight)
            nn.init.xavier_normal_(self.conv1d_i.weight)
        elif init == "zeros":
            nn.init.zeros_(self.conv1d_r.weight)
            nn.init.zeros_(self.conv1d_i.weight)
        elif init == "central":
            nn.init.constant_(self.conv1d_r.weight, 0.5)
            nn.init.constant_(self.conv1d_i.weight, 0.5)
        else:
            raise ValueError("invalid init type {}".format(init))

    def forward(self, x):
        x_r = self.conv1d_r(x.real) - self.conv1d_i(x.imag)
        x_i = self.conv1d_r(x.imag) + self.conv1d_i(x.real)
        return x_r + 1j * x_i


def complex_weight_composition(wr, wi):
    """
    composition ComplexLinear weight to real net.
    Input: wr: [k, N], wi: [k, N]
    return: [2*k, 2*N]
    """
    w1 = torch.cat([wr, -wi], dim=1)
    w2 = torch.cat([wi, wr], dim=1)
    return torch.cat([w1, w2], dim=0)


def complex_bias_composition(br, bi):
    """
    composition ComplexLinear bias to real net.
    Input: br: [N],bi: [N]
    return: [2*N]
    """
    return torch.cat([br - bi, br + bi], dim=0)


if __name__ == "main":

    class CLinear(torch.nn.Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

            self.nn = torch.nn.Linear(2 * in_features, 2 * out_features, bias=bias)

        def forward(self, x):
            x = torch.cat([x.real, x.imag], dim=-1)
            x = self.nn(x)
            return x[..., : self.out_features] + 1j * x[..., self.out_features :]

    net1 = ComplexLinear(3, 5, bias=True)
    net2 = CLinear(3, 5, bias=True)

    weight1 = complex_weight_composition(net1.real.weight.data, net1.imag.weight.data)
    net2.nn.weight.data = weight1

    bias1 = complex_bias_composition(net1.real.bias.data, net1.imag.bias.data)
    net2.nn.bias.data = bias1

    x = torch.randn(10, 3) + 1j * torch.randn(10, 3)
    print(torch.sum(torch.abs(net1(x) - net2(x))))
