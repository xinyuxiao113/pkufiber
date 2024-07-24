"""
在内部常用的算子，函数等。
"""

import numpy as np, torch, scipy.constants as const
from functools import partial
from torch.fft import fft, ifft, fftfreq, fftshift
from typing import Union, Tuple


def get_omega(Fs: Union[int, float, torch.Tensor], Nfft: int) -> torch.Tensor:
    """
    get signal fft angular frequency.
    Input:
        Fs: sampling frequency. [Hz], float, int or [batch,]
        Nfft: number of sampling points.
    Output:
        omega:jnp.Array [Nfft,]
    """
    if isinstance(Fs, (int, float)):
        return 2 * np.pi * Fs * fftfreq(Nfft)
    else:
        return 2 * torch.pi * Fs[:, None] * fftfreq(Nfft)[None, :].to(Fs.device)


def get_beta1(
    D: float, Fc: float, Fi: Union[torch.Tensor, float]
) -> Union[torch.Tensor, float]:
    """
    Calculate beta1.
    Input:
        D:[ps/nm/km]    Fc: [Hz]   Fi: [Hz]
    Output:
        beta1:    [s/km]
    """
    beta2 = get_beta2(D, Fc)  # [s^2/km]
    beta1 = 2 * np.pi * (Fi - Fc) * beta2  # [s/km]
    return beta1


def get_beta2(D: float, Fc: float) -> float:
    """
    Input:
        D: dispersion coeff. [ps/nm/km]    Fc: central frequency. [Hz]
    Output:
        beta2:    [ps*s/nm]=[s^2/km]
    """
    c_kms = const.c / 1e3  # speed of light (vacuum) in km/s
    lamb = c_kms / Fc  # [km]
    beta2 = -(D * lamb**2) / (2 * np.pi * c_kms)  # [ps*s/nm]=[s^2/km]
    return beta2


def dispersion_op(
    h: float, Edim: int, beta2: float, alpha: float, Nfft: int, Fs: float
) -> torch.Tensor:
    """
    Input:
        h: dispersion distance. [km],
        Edim: 2 or 3,
        beta2: [s^2/km]
        alpha: [dB/km],
        Nfft: number of sampling points.
        Fs: sampling rate. [Hz]
    Output:
        (Nfft,1) or (1,Nfft,1)
    """
    omega = get_omega(Fs, Nfft)  # (Nfft,), [Hz]
    kernel = torch.exp(-(alpha / 2) * (h) - 1j * (beta2 / 2) * (omega**2) * (h))
    if Edim == 3:
        return kernel[None, :, None]
    elif Edim == 2:
        return kernel[:, None]
    else:
        raise (ValueError)


def mean_power(x: torch.Tensor) -> torch.Tensor:
    """
    Caculate average power of signal.
    Input:
        x: Array.
    Output:
        scaler.
    """
    return torch.mean(torch.abs(x) ** 2)


def frame(x: torch.Tensor, flen: int, fstep: int, fnum: int = -1) -> torch.Tensor:
    """
        generate circular frame from Array x.
    Input:
        x: Arrays about to be framed with shape (B, *dims)
        flen: frame length.
        fstep: step size of frame.
        fnum: steps which frame moved. If fnum==None, then fnum --> 1 + (N - flen) // fstep
    Output:
        A extend array with shape (fnum, flen, *dims)
    """
    N = x.shape[0]

    if fnum == -1:
        fnum = 1 + (N - flen) // fstep

    ind = (np.arange(flen)[None, :] + fstep * np.arange(fnum)[:, None]) % N
    return x[ind, ...]


def circsum(a: torch.Tensor, N: int) -> torch.Tensor:
    """
    Transform a 1D array a to a N length array.
    Input:
        a: 1D Array.
        N: a integer.
    Output:
        d: 1D array with length N.

    d[k] = sum_{i=0}^{+infty} a[k+i*N]
    """
    b = frame(a, N, N)
    t = b.shape[0] * N
    c = a[t::]
    d = torch.sum(b, dim=0)
    d[0 : len(c)] = d[0 : len(c)] + c
    return d


def conv_circ(signal: torch.Tensor, ker: torch.Tensor, dim=0) -> torch.Tensor:
    """
    N-size circular convolution.

    Input:
        signal: real Nd tensor with shape (N,)
        ker: real 1D tensor with shape (N,).
    Output:
        signal conv_N ker.
    """
    ker_shape = [1] * signal.dim()
    ker_shape[dim] = -1
    ker = ker.reshape(ker_shape)
    result = torch.fft.ifft(
        torch.fft.fft(signal, dim=dim) * torch.fft.fft(ker, dim=dim), dim=dim
    )

    if not signal.is_complex() and not ker.is_complex():
        result = result.real

    return result


def circFilter(h: torch.Tensor, x: torch.Tensor, dim=0) -> torch.Tensor:
    """
    1D Circular convolution. fft version.
    h: 1D kernel.     x: Nd signal
    """
    k = len(h) // 2
    h_ = circsum(h, x.shape[dim])
    h_ = torch.roll(h_, -k)
    return conv_circ(x, h_, dim=dim)
