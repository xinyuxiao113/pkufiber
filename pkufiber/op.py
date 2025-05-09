"""
在内部常用的算子，函数等。
"""

import numpy as np, torch, scipy.constants as const
from functools import partial
from torch.fft import fft, ifft, fftfreq, fftshift
from typing import Union, Tuple
import torch
import torch.nn.functional as F


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

def frame_on_dim(x: torch.Tensor, flen: int, fstep: int, fnum: int = -1, dim: int=0) -> torch.Tensor:
    '''

    '''
    x = x.transpose(0, dim)               # (Bd, ..., B0,  ...)
    x = frame(x, flen, fstep, fnum)       # (fnum, flen, ..., B0, ...)  
    dims = list(range(x.dim()))
    dims.append(dims.pop(1))                 
    x = x.permute(dims)                   # (fnum, ..., B0, ..., flen)  
    x = x.transpose(0, dim)
    return x




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




def stft_on_dimension(signal, n_fft, hop_length, win_length, dim=-1):
    """
    对张量的指定维度进行 STFT，并自动处理填充问题。
    :param signal: 输入信号张量（复数类型），形状可以是 [batch, ..., L]，L是指定维度的长度
    :param n_fft: FFT 点数
    :param hop_length: 窗口滑动步长
    :param win_length: 窗口长度
    :param dim: 指定要进行 STFT 的维度
    :return: 计算后的 STFT 结果和填充的信号长度
    """
    original_shape = signal.shape
    signal = signal.transpose(dim, -1)  # 将指定维度移到最后
    batch_shape = signal.shape[:-1]  # 获取批量维度形状
    L = signal.shape[-1]
    
    # 计算所需的填充长度
    pad_amount = n_fft - hop_length  # 根据 hop_length 和 n_fft 的关系来决定填充
    padded_signal = F.pad(signal, (0, pad_amount))  # 在信号末尾进行填充

    # 计算STFT，首先将其他维度展平为2D张量
    flattened_signal = padded_signal.reshape(-1, padded_signal.shape[-1])
    # window_fun = torch.hamming_window(win_length, device=signal.device)
    window_fun = torch.ones(win_length, device=signal.device)
    stft_result = torch.stft(flattened_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True, window=window_fun, center=False)

    # 恢复批量维度
    stft_result = stft_result.view(*batch_shape, *stft_result.shape[-2:])

    return stft_result, padded_signal.shape[-1], original_shape, dim

def istft_on_dimension(stft_result, original_length, n_fft, hop_length, win_length, original_shape, dim=-1):
    """
    对 STFT 结果进行逆变换，并裁剪到原始长度，同时处理指定维度。
    :param stft_result: STFT 结果
    :param original_length: 原始信号的长度
    :param n_fft: FFT 点数
    :param hop_length: 窗口滑动步长
    :param win_length: 窗口长度
    :param original_shape: 原始信号的形状
    :param dim: 指定的 ISTFT 操作维度
    :return: 逆变换后的信号，裁剪至原始长度
    """
    batch_shape = stft_result.shape[:-2]  # 获取批量维度形状
    
    # 将stft_result展平为2D张量
    flattened_stft = stft_result.reshape(-1, stft_result.shape[-2], stft_result.shape[-1])
    
    # 进行逆STFT
    # window_fun = torch.hamming_window(win_length, device=stft_result.device)
    window_fun = torch.ones(win_length, device=stft_result.device)
    reconstructed_signal = torch.istft(flattened_stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length, length=original_length, return_complex=True, window=window_fun, center=False)

    # 恢复批量维度并裁剪长度
    reconstructed_signal = reconstructed_signal.view(*batch_shape, original_length)

    # 将指定的维度移回原来的位置
    reconstructed_signal = reconstructed_signal.transpose(dim, -1).reshape(original_shape)

    return reconstructed_signal



def periodic_padding(tensor, pad_width, dim):
    """
    使用 F.pad 实现张量的周期填充。

    参数:
    - tensor: 输入张量。
    - pad_width: 填充宽度（两端填充的长度）。
    - dim: 指定填充的维度。

    返回:
    - 填充后的张量。
    """
    if pad_width == 0:
        return tensor

    # 将指定维度移动到最后一维，以方便 F.pad 操作
    tensor = tensor.transpose(dim, -1)
    
    # 提取两端的填充值
    left_padding = tensor[..., -pad_width:]  # 右端的数据用于填充左侧
    right_padding = tensor[..., :pad_width]  # 左端的数据用于填充右侧
    
    # 拼接两端并使用 F.pad 进行填充
    tensor = torch.cat([left_padding, tensor, right_padding], dim=-1)
    
    # 将维度还原到原始顺序
    tensor = tensor.transpose(dim, -1)
    
    return tensor
