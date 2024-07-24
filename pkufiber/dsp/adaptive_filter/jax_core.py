from flax import struct
import scipy.constants as const
import jax, optax, jax.random as rd, jax.numpy as jnp, flax.linen as nn, numpy as np, matplotlib.pyplot as plt
from typing import Any, NamedTuple, Iterable, Callable, Optional
from collections import namedtuple
from functools import partial

Array = Any
DeviceArray = Any


DataInput = namedtuple("DataInput", ["y", "x", "w0", "a"])


class parameters:
    """
    Basic class to be used as a struct of parameters
    """

    pass


@struct.dataclass
class JaxTime:
    start: int = struct.field(pytree_node=False)
    stop: int = struct.field(pytree_node=False)
    sps: int = struct.field(pytree_node=False)


@struct.dataclass
class JaxSignal:
    val: Array  # value. [Nfft,Nmodes]  or [batch,Nfft,Nmodes]
    t: JaxTime = struct.field(pytree_node=False)  # (start,stop,sps)
    Fs: float  # sampling rate  [Hz]
    power: float = 0  # power [dBm]
    Fi: float = 0  # input frequency [Hz]
    Nch: int = 1  # number of channels


def wrap_signal(dataset):
    """
    Input:
        dataset: a dataset from optical_flax.datasets
    Output:
        y,x: MySignal
    """
    sps = dataset.a["sps"]
    Fs = dataset.a["samplerate"]
    power = dataset.a["lpdbm"]
    Fi = dataset.a["carrier_frequency"]
    Nch = dataset.a["channels"]
    x = JaxSignal(dataset.x, JaxTime(0, 0, 1), Fs / sps, power, Fi, Nch)
    y = JaxSignal(dataset.y, JaxTime(0, 0, sps), Fs, power, Fi, Nch)
    return y, x


def align_signal(x, t: JaxTime) -> tuple:
    """
    Align signal.
    Input:
        x, t
    Output:
        two array with same shape and aligned time index.
    """
    L = x.shape[-2]
    return x[..., t.start : L + t.stop, :]


def get_MySignal(val, a):
    return JaxSignal(
        val=val,
        t=JaxTime(0, 0, a["sps"]),
        Fs=a["samplerate"],
        power=a["power"],
        Fi=a["carrier_frequency"],
        Nch=a["Nch"],
    )


def conv1d_t(t, taps, rtap, stride, mode):
    assert (
        t.sps >= stride
    ), f"sps of input SigTime must be >= stride: {stride}, got {t.sps} instead"
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
    return JaxTime(
        (t.start + tslice[0]) // stride, (t.stop + tslice[1]) // stride, t.sps // stride
    )


def conv1d_slicer(taps, rtap=None, stride=1, mode="valid"):
    def slicer(signal: JaxSignal) -> JaxSignal:
        x = signal.val
        xt = signal.t
        yt = conv1d_t(xt, taps, rtap, stride, mode)
        D = xt.sps // yt.sps
        zt = JaxTime(yt.start * D, yt.stop * D, xt.sps)
        x = x[zt.start - xt.start : x.shape[0] + zt.stop - xt.stop]
        return JaxSignal(
            val=x, Fs=signal.Fs, t=zt, power=signal.power, Fi=signal.Fi, Nch=signal.Nch
        )

    return slicer
