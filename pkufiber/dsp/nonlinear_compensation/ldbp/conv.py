import torch, copy, numpy as np, matplotlib.pyplot as plt
import torch.nn.functional as F, torch.nn.init as init, torch.nn as nn

from pkufiber.dsp.layers import ComplexConv1d
from pkufiber.core import TorchSignal, TorchTime, TorchInput


class downsamp(nn.Module):
    """
    Downsample module.
    A simple replacement of ADF.
    """

    def __init__(self, taps=32, Nmodes=1, batch_size=None, sps=2, init="zeros"):
        super(downsamp, self).__init__()
        self.taps = taps
        self.Nmodes = Nmodes
        self.conv = ComplexConv1d(
            Nmodes, Nmodes, self.taps, stride=sps, padding=0, bias=False, init="zeros"
        )
        self.overlaps = (self.taps - 1) // 2

    def forward(self, signal: TorchSignal):
        x = signal.val  # [batch, L*sps, Nmodes]
        sps = signal.t.sps
        t = copy.deepcopy(signal.t)
        t.conv1d_t(self.taps, sps)
        y = self.conv(x.permute([0, 2, 1]))  # [batch, Nmodes, L - taps + 1]
        y = y.permute([0, 2, 1])  # [batch, L - taps + 1, Nmodes]
        return TorchSignal(y, t)

    def detach_state(self):
        pass
