"""
常用的核心类
Core classes for TorchDSP.
1. TorchTime: signal time information.
2. TorchSignal: signal information.
3. TorchInput: data structure for training neural network.
"""

from typing import Union
import torch, numpy as np
import argparse
import json


class WdmSig(object):
    """
    WDM signal from transmitter.
    """

    def __init__(
        self,
        signal: torch.Tensor = torch.zeros(1),
        symb: torch.Tensor = torch.zeros(1),
        pulse: torch.Tensor = torch.zeros(1),
        sps: int = 2,
        Rs: float = 32e9,
        freqspace: float = 50e9,
        config: dict = {},
    ):
        self.signal = signal
        self.symb = symb
        self.pulse = pulse
        self.sps = sps
        self.Rs = Rs
        self.freqspace = freqspace

    def __repr__(self):
        return f"WdmSig(signal=tensor {self.signal.shape}, symb={self.symb.shape}, pulse=tensor {self.pulse.shape}), sps={self.sps}, Rs={self.Rs}, freqspace={self.freqspace}"


def dict_type(s):
    try:
        return json.loads(s)
    except ValueError:
        raise argparse.ArgumentTypeError("Not a valid dictionary string: '%s'" % s)


class TorchTime:
    """
    Signal time information.
    Attributes:
        start (int): start time.
        stop (int): stop time.
        sps (int): samples per symbol.
    """

    def __init__(self, start: int, stop: int, sps: int):
        self.start = start
        self.stop = stop
        self.sps = sps

    def __repr__(self):
        return f"TorchTime(start={self.start}, stop={self.stop}, sps={self.sps})"

    def conv1d_t(self, taps, stride, rtap=None, mode="valid"):
        """
        Transformation of time information after 1d convolution.
        Input:
            taps: length of convolution kernel.
            stride: stride of convolution.
            rtap: right tap of convolution kernel.
            mode: 'valid', 'same', 'full'.
        Output:
            None
        """
        assert (
            self.sps >= stride
        ), f"sps of input SigTime must be >= stride: {stride}, got {self.sps} instead"
        if rtap is None:
            rtap = (taps - 1) // 2
        delay = -(-(rtap + 1) // stride) - 1
        if mode == "full":
            tslice = (
                -delay * stride,
                taps - stride * (rtap + 1),
            )  # TODO: think more about this
        elif mode == "same":
            tslice = (0, 0)
        elif mode == "valid":
            tslice = (delay * stride, (delay + 1) * stride - taps)
        else:
            raise ValueError("invalid mode {}".format(mode))
        self.start = (self.start + tslice[0]) // stride
        self.stop = (self.stop + tslice[1]) // stride
        self.sps = self.sps // stride
        return self


class TorchSignal:
    """
    Signal information.
    Attributes:
        val (torch.Tensor): signal value. [batch, L, Nmodes]
        t (TorchTime): signal time information.
    """

    def __init__(self, val=torch.zeros(1), t=TorchTime(0, 0, 2), dtype=torch.complex64):
        if isinstance(val, np.ndarray):
            self.val = torch.tensor(val).to(dtype)  # [batch, L, Nmodes]
        elif isinstance(val, torch.Tensor):
            self.val = val
        else:
            raise ValueError("val must be np.ndarray or torch.Tensor")
        self.t = t

    def __repr__(self):
        return f"TorchSignal(val: tensor with {self.val.shape}, {self.val.device}, t:{self.t})"

    def to(self, dtype):
        """
        Change device or type of signal value.
        Input:
        (1) change device: 'cpu', 'cuda:0'
        (2) change type: torch.float32, torch.complex64
        """
        self.val = self.val.to(dtype)
        return self

    def init_time(self):
        """
        Realign time information to initial state.
        """
        self.t = TorchTime(0, 0, self.t.sps)
        return self

    def get_slice(self, L, i):
        """
        Obtain a signal slice starting from the ith symbol, spanning L symbols while reset the time information.   periodic index
        """
        idx = (torch.arange(L * self.t.sps) + i * self.t.sps) % self.val.shape[1]
        return TorchSignal(val=self.val[:, idx], t=TorchTime(0, 0, self.t.sps))


class TorchInput:
    """
    TorchInput is a data structure for training neural network.
    Attributes:
        signal_input (TorchSignal): input signal.
        signal_output (TorchSignal): output signal.
        task_info (torch.Tensor): task information includes [P, Fi, Fs, Nch]  with unit [dBm, Hz, Hz, 1]. shape: [batch, 4].
    """

    def __init__(
        self,
        signal_input: TorchSignal,
        signal_output: TorchSignal,
        task_info: torch.Tensor = torch.ones(4),
    ):
        self.signal_input = signal_input.to(torch.complex64)
        self.signal_output = signal_output.to(torch.complex64)
        self.task_info = task_info.to(torch.float32)

    def __repr__(self):
        return f"TorchInput(signal_input: {self.signal_input.__repr__()},\n           signal_output: {self.signal_output.__repr__()},\n           task_info: {self.task_info.shape}, {self.task_info.device})"

    def to(self, dtype):
        """
        Change device or type of signal value.
        Input:
        (1) change device: 'cpu', 'cuda:0'
        (2) change type: torch.float32, torch.complex64
        """
        self.signal_input = self.signal_input.to(dtype)
        self.signal_target = self.signal_output.to(dtype)
        self.task_info = self.task_info.to(dtype)
        return self

    def get_data(self, L, idx):
        """
        Obtain a signal slice starting from the idx symbol, spanning L symbols.
        """
        signal_input = self.signal_input.get_slice(L, idx)
        signal_output = self.signal_output.get_slice(L, idx)
        return TorchInput(signal_input, signal_output, self.task_info)
