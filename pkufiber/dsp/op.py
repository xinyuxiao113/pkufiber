"""
    Baseline algorithms for digital signal processing.
        CDC: Chromatic dispersion compensation.
        DBP: Digital back propagation.
"""

import torch, numpy as np
from torch.fft import fft, fftfreq, fftshift, ifft
from typing import Union
import torch, numpy as np, torch.nn.functional as F, time
import torch.fft
from functools import partial, wraps
from commpy.modulation import QAMModem

from pkufiber.op import get_omega, frame, circsum, conv_circ, circFilter


def dispersion_kernel(
    dz: float,
    dtaps: int,
    Fs: Union[torch.Tensor, float, int],
    beta2: float = -2.1044895291667417e-26,
    beta1: Union[torch.Tensor, float, int] = torch.zeros(1),
    domain="time",
) -> torch.Tensor:
    """
    Dispersion kernel in time domain or frequency domain.
    Input:
        dz: Dispersion distance.              [m]
        dtaps: length of kernel.
        Fs: Sampling rate of signal.          [Hz]         [batch] or float
        beta2: 2 order  dispersion coeff.     [s^2/m]
        beta1: 1 order dispersion coeff.      [s/m]        [batch] or float
        domain: 'time' or 'freq'
    Output:
        h:jnp.array. (dtaps,)
        h is symmetric: jnp.flip(h) = h.
    """
    if type(Fs) == float or type(Fs) == int:
        Fs = torch.tensor([Fs])
    if type(beta1) == float or type(beta1) == int:
        beta1 = torch.tensor([beta1])
    assert isinstance(Fs, torch.Tensor) and isinstance(
        beta1, torch.Tensor
    ), "Fs and beta1 should be torch.Tensor."

    omega = get_omega(Fs, dtaps)  # (batch, dtaps)
    beta1 = beta1.to(omega.device)  # (batch, 1)
    kernel = torch.exp(
        -1j * beta1[:, None] * omega * dz - 1j * (beta2 / 2) * (omega**2) * dz
    )  # beta1: (batch,)

    if domain == "time":
        return torch.fft.fftshift(torch.fft.ifft(kernel, axis=-1), axis=-1)
    elif domain == "freq":
        return kernel
    else:
        raise (ValueError)


def lin_op(
    E: torch.Tensor,
    z: float,
    dz: float,
    Fs: torch.Tensor,
    beta2: float = -2.1044895291667417e-26,
    beta1: float = 0,
) -> torch.Tensor:
    """
    Linear operator with time domain convolution.
    Input:
        E: E.val  [Nfft,Nmodes] or [batch, Nfft,Nmodes]
        z: operator start position.  [m]
        dz: operator distance.      [m]
        Fs: samplerate, [Hz].
        dtaps: kernel shape.
    Output:
        E: E.val [Nfft, Nmodes]
    """
    Nfft = E.shape[-2]
    kernel = dispersion_kernel(
        dz, Nfft, Fs, beta2, beta1, domain="freq"
    )  # [batch, Nfft]
    kernel = kernel[..., None]  # [batch, Nfft, 1]
    x = ifft(fft(E, dim=-2) * kernel, dim=-2)

    if E.ndim == 2:
        x = x.squeeze(0)
    return x


def exp_integral(
    z: float, alpha: float = 4.605170185988092e-05, span_length: float = 80e3
) -> np.ndarray:
    """
       Optical Power integral along z.
    Input:
        z: Distance [m]
        alpha: [/m]
        span_length: [m]
    Output:
        exp_integral(z) = int_{0}^{z} P(z) dz
        where P(z) = exp( -alpha *(z % Lspan))
    """
    k = z // span_length
    z0 = z % span_length

    return (
        k * (1 - np.exp(-alpha * span_length)) / alpha
        + (1 - np.exp(-alpha * z0)) / alpha
    )


def leff(
    z: float, dz: float, alpha: float = 4.605170185988092e-05, span_length: float = 80e3
):
    """
       Optical Power integral along z.
    Input:
        z: Distance [m]
        dz: step length. [m]
        alpha: [/m]
        span_length: [m]
    Output:
        exp_integral(z) = int_{z}^{z + dz} P(z) dz
        where P(z) = exp( -alpha *(z % Lspan))
    """
    return exp_integral(z + dz) - exp_integral(z)


def nonlin_op(E: torch.Tensor, z: float, dz: float, gamma=0.0016567) -> torch.Tensor:
    """
    NonLinear operator.
    Input:
        E: E.val  [Nfft,Nmodes] or [batch, Nfft,Nmodes]
        z: operator start position.
        dz: operator distance.
        gamma: nonlinear coeff.  [/W/m]
    Output:
        E: E.val [Nfft, Nmodes]
    """
    phi = gamma * leff(z, dz) * torch.sum(torch.abs(E) ** 2, dim=-1)[..., None]
    x = torch.exp(-(1j) * phi) * E  # type: ignore
    return x


def decision(const: torch.Tensor, v: torch.Tensor, stopgrad=True):
    """
    simple symbol decision based on Euclidean distance
    Input:
        const: [Nconst]   v: [batch, Nmodes]
    Output:
        d: [batch, Nmodes]
    """

    d = const[torch.argmin(torch.abs(const[:, None, None] - v[None, ...]), dim=0)]
    return d


def vmapconv1d(input, filter, stride):
    """
    [batch, L] x [batch, k] -> [batch, L - k +1]

    output[i,n] = sum_{j} input[i, n-j] * filter[i, j]

    Valid convolution along axis=1, vecterize in batch dimension axis=0.
    Input:
        input: [batch, L]
        filter: [batch, k]
        stride: int
    Output:
        [batch, L - k +1]
    """
    filter = filter[:, None, :]  # [batch, 1, k]
    return F.conv1d(input, filter, stride=stride, groups=input.shape[0])


def dconv(input, filter, stride):
    """
    [batch, L, Nmodes] x [batch, k] -> [batch, L - k +1, Nmodes]


        output[i,n,m] = sum_{j} input[i, k-1+n-j, m] * filter[i, j]

        Dispersion convolution.
        Input:
            input: [batch, L, Nmodes]
            filter: [batch, k]
            stride: int
        Output:
            [batch, L - k +1, Nmodes]
    """

    return torch.vmap(vmapconv1d, in_dims=(-1, None, None), out_dims=-1)(
        input, torch.flip(filter, dims=(-1,)), stride
    )


def conv1d(input, filter, stride):
    """
    [Ci, L] x [Co, Ci, ntaps] -> [Co, L - k +1]    or     [batch, Ci, L] x [Co, Ci, ntaps] -> [batch, Co, L - k +1]

    output[b, n, l] = sum_{m, j} input[b, m, l + j] * filter[n, m, j]

    Multi-channel convolution.
    Input:
        input:  [Ci, L] or [batch, Ci, L]
        filter: [Co, Ci, ntaps]
        stride: int
    Output:
        [Co, L] or [batch, Co, L - k +1]
    """
    return F.conv1d(input, filter, stride=stride)


def nconv(input, filter, stride):
    """
    [batch, L, Ci] x [batch, Co, Ci, ntaps] -> [batch, L - k +1, Co]

    Nonlinear operator convolution.
    Input:
        input:  [batch, L, Ci]
        filter: [batch, Co, Ci, ntaps]
        stride: int
    Output:
        [batch, L - k +1, Co]
    """
    return torch.vmap(conv1d, in_dims=(0, 0, None), out_dims=0)(
        input.permute([0, 2, 1]), filter, stride
    ).permute([0, 2, 1])


if __name__ == "__main__":

    print("hello")
