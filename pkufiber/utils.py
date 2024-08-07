"""
一些常用的工具函数
"""

import torch, numpy as np, time, matplotlib.pyplot as plt
from functools import wraps, partial
import torch.fft
import scipy.special as special
from typing import Tuple, Union


def calc_time(f):
    """
    Decorator for calculating the time cost of a function.
    """

    @wraps(f)
    def _f(*args, **kwargs):
        t0 = time.time()
        y = f(*args, **kwargs)
        t1 = time.time()
        print(f" {f.__name__} complete, time cost(s):{t1-t0}")
        return y

    return _f


def qfactor(ber):
    """
    Calculate Q^2 value from bit error rate.
    Input:
        ber: bit error rate.
    Output:
        Q^2 value.
    """
    return 20 * np.log10(
        np.sqrt(2) * np.maximum(special.erfcinv(np.minimum(2 * ber, 0.999)), 0.0)
    )


def show_symb(
    sig: Union[torch.Tensor, np.ndarray],
    symb: Union[torch.Tensor, np.ndarray],
    s: float = 10,
    figsize: tuple = (8, 4),
    title: str = "title",
):
    """
    Show the scatter plot of the symbols.
    Input:
        sig:  L2 = 1    [Nsymb, Nmodes]
        symb: L2 = 1    [Nsymb, Nmodes]
        s: size of the dots.
        figsize: size of the figure.
        title: title of the figure.
    Output:
        ax: axes of the figure.
    """
    if isinstance(sig, torch.Tensor):
        sig = sig.detach().cpu().numpy()
        symb = symb.detach().cpu().numpy()  # type: ignore
    Nmodes = sig.shape[-1]
    symb_set = np.unique(symb)

    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=200)
    fig.suptitle(title)

    for p in range(Nmodes):
        for sym in symb_set:

            z = sig[..., p][symb[..., p] == sym]
            ax[p].scatter(z.real, z.imag, s=s, marker='.')  # type: ignore

    return ax
