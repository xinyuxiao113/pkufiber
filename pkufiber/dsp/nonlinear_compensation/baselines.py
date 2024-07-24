import torch, numpy as np
from torch.fft import fft, fftfreq, fftshift, ifft
from pkufiber.dsp.op import lin_op, nonlin_op, leff


def cdc(
    E: torch.Tensor,
    Fs: torch.Tensor,
    length: float,
    beta2: float = -2.1044895291667417e-26,
    beta1: float = 0,
) -> torch.Tensor:
    """
        CD compensatoin.
    Input:
        E: digital signal.   [batch, Nfft,Nmodes]
        Fs: samplerate. [Hz]   [batch,]
        length >0, dz > 0: [m]
        beta2: 2 order  dispersion coeff.     [s^2/m]
        beta1: 1 order dispersion coeff.      [s/m]
    """
    E = lin_op(E, length, -length, Fs, beta2, beta1)
    return E


def dbp(
    E: torch.Tensor,
    length: float,
    dz: float,
    Fs: torch.Tensor,
    power_dbm: torch.Tensor,
    beta2: float = -2.1044895291667417e-26,
    beta1: float = 0,
    gamma: float = 0.0016567,
    order=1,
) -> torch.Tensor:
    """
        Digital back propagation.
    Input:
        E: digital signal.          [Nfft,Nmodes] or [batch, Nfft,Nmodes]
        Fs: samplerate, [Hz]            [1]       or [batch, 1]
        length >0, dz > 0: [m]
        dz: step size. [m]
        beta2: 2 order  dispersion coeff.     [s^2/m]
        beta1: 1 order dispersion coeff.      [s/m]
        power_dbm: power of each channel. [dBm]    per channel per mode power = 1e-3*10**(power_dbm/10)/Nmodes  [W].
    """
    Nmodes = E.shape[-1]
    if Nmodes == 2:
        gamma = 8 / 9 * gamma
    scale_param = 1e-3 * 10 ** (power_dbm / 10) / Nmodes  # [batch]
    scale_param = scale_param.reshape([-1] + [1] * (E.ndim - 1))
    E = E * torch.sqrt(scale_param)
    K = int(length / dz)
    z = length

    if order == 1:
        for i in range(K):
            E = lin_op(E, z, -dz, Fs, beta2, beta1)
            E = nonlin_op(E, z, -dz, gamma)
            z = z - dz
    elif order == 2:
        E = lin_op(E, z, -dz / 2, Fs, beta2, beta1)
        for i in range(K - 1):
            E = nonlin_op(E, z, -dz, gamma)
            E = lin_op(E, z, -dz, Fs, beta2, beta1)
            z = z - dz
        E = nonlin_op(E, z, -dz, gamma)
        E = lin_op(E, z, -dz / 2, Fs, beta2, beta1)
        z = z - dz
    else:
        raise (ValueError)

    return E / torch.sqrt(scale_param)


if __name__ == "__main__":

    print("hello")
