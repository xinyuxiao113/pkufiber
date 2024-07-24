import numpy as np
import torch
import scipy.constants as const
from functools import partial
from typing import Tuple
from torch.fft import fft, ifft

from pkufiber.utils import calc_time
from pkufiber.op import get_omega, get_beta2, dispersion_op
from pkufiber.core import WdmSig


def choose_dz(
    freqspace: float,
    Lspan: float,
    Pch_dBm: float,
    Nch: int,
    beta2: float,
    gamma: float,
    dz_max: float = 0.5,
    eps: float = 1e-2,
) -> float:
    """
    Input:
        freqspace:[Hz]
        Lspan: [km]
        Pch_dBm:[dBm]
        beta2:[s^2/km]
        gamma:[/W/km]
        dz_max:[km]
    Output:
        dz [km]
    """

    Bw = freqspace * Nch  # [Hz]
    power = 1e-3 * 10 ** (Pch_dBm / 10) * Nch  # [W]
    dz = eps * 200 / (np.abs(beta2) * Bw**2 * gamma * power * Lspan)
    dz = 2 ** (int(np.log2(dz)))
    return min(dz_max, dz)


def su2(alpha, phi):
    """
    Generate a SU2 matrix.
    Input:
        alpha: [rad]   phi: [rad]
    Output:
        [2,2] tensor
    """
    return torch.tensor(
        [
            [torch.cos(alpha), torch.sin(alpha) * torch.exp(1j * phi)],
            [-torch.sin(alpha) * torch.exp(-1j * phi), torch.cos(alpha)],
        ]
    )


def delay(h, dbeta1, Nfft, Fs):
    """
    delay operator.
    Input:
        h:[km]  dbeta1:[s/km]   Fs:[Hz]
    Output:
        T(omega): [2,2,Nfft]
    """

    omega = get_omega(Fs, Nfft)
    return torch.tensor(
        [
            [torch.exp(-1j * omega * dbeta1 / 2 * h), torch.zeros(Nfft)],
            [torch.zeros(Nfft), torch.exp(1j * omega * dbeta1 / 2 * h)],
        ]
    )


def scatter_matrix(h, dbeta1, Nfft, Fs, z):
    # [2,2,Nfft]

    alpha = torch.rand() * 2 * torch.pi
    phi = torch.rand() * 2 * torch.pi
    R = su2(alpha, phi)  # [2,2]
    T = delay(h, dbeta1, Nfft, Fs)  # [2,2,Nfft]
    return R[..., None] * T


def pmd(E, T):
    """
    E: [batch, Nfft, Nmodes] or [Nfft, Nmodes],  Nmodes = 2
    T: [2,2,Nfft]
    """
    if E.ndim == 3:
        return torch.einsum("bnp,ptn->bnt", E, T)
    elif E.ndim == 2:
        return torch.einsum("np, ptn->nt", E, T)
    else:
        raise (ValueError)


def edfa(
    Ei: torch.Tensor,
    Fs: float = 100e9,
    G: float = 20,
    NF: float = 4.5,
    Fc: float = 193.1e12,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Simple EDFA model

    :param Ei: input signal field [nparray]
    :param Fs: sampling frequency [Hz][scalar]
    :param G: gain [dB][scalar, default: 20 dB]
    :param NF: EDFA noise figure [dB][scalar, default: 4.5 dB]
    :param Fc: optical center frequency [Hz][scalar, default: 193.1e12 Hz]

    :return: amplified noisy optical signal [nparray]
    """
    # assert G > 0, 'EDFA gain should be a positive scalar'
    # assert NF >= 3, 'The minimal EDFA noise figure is 3 dB'

    NF_lin = 10 ** (NF / 10)
    G_lin = 10 ** (G / 10)
    nsp = G_lin * NF_lin / (2 * (G_lin - 1))
    N_ase = (G_lin - 1) * nsp * const.h * Fc
    p_noise = N_ase * Fs
    noise = torch.randn(Ei.shape, dtype=torch.complex64, device=device) * np.sqrt(
        p_noise
    )
    return Ei * np.sqrt(G_lin) + noise


@calc_time
def fiber_transmission(
    tx_data: WdmSig,
    seed: int,
    Ltotal: float = 2000,
    Lspan: float = 80,
    hz: float = 0.5,
    alpha: float = 0.2,
    D: float = 16.5,
    gamma: float = 1.6567,
    Fc: float = 193.1e12,
    amp: str = "edfa",
    NF: float = 4.5,
    order: int = 2,
    openPMD: bool = False,
    Dpmd: float = 3,
    Lcorr: float = 0.1,
    device: str = "cuda:0",
) -> Tuple[WdmSig, dict]:
    """
    Manakov model split-step Fourier (symmetric, dual-pol.)

    Parameters:
    ----------
    tx_data: WdmSig
        Transmitted WDM signal.
    seed: int
        Random seed.
    Ltotal: float, optional
        Total fiber length in km. Default is 2000 km.
    Lspan: float, optional
        Span length in km. Default is 80 km.
    hz: float, optional
        Step-size for the split-step Fourier method in km. Default is 0.5 km.
    alpha: float, optional
        Fiber attenuation parameter in dB/km. Default is 0.2 dB/km.
    D: float, optional
        Chromatic dispersion parameter in ps/nm/km. Default is 16.5 ps/nm/km.
    gamma: float, optional
        Fiber nonlinear parameter in 1/W/km. Default is 1.6567 1/W/km.
    Fc: float, optional
        Carrier frequency in Hz. Default is 193.1e12 Hz.
    amp: str, optional
        Amplifier type, either 'edfa', 'ideal', or 'None'. Default is 'edfa'.
    NF: float, optional
        EDFA noise figure in dB. Default is 4.5 dB.
    order: int, optional
        SSFM order, either 1 or 2. Default is 2.
    openPMD: bool, optional
        Whether to use PMD effect. Default is False.
    Dpmd: float, optional
        PMD coefficient in ps/sqrt(km). Default is 3.
    Lcorr: float, optional
        Fiber correlation length in km. Default is 0.1 km.
    device: str, optional
        Device to perform computation on. Default is 'cuda:0'.

    Returns:
    -------
    Tuple[WdmSig, dict]
        A tuple containing the transmitted WDM signal and a configuration dictionary.

    Note:
    -----
    The function simulates the Manakov equation:
    E_z = -1/2*alpha*E + j beta2/2 E_tt - j gamma |E|^2E
    """

    # Set random seed
    torch.manual_seed(seed)

    # Channel parameters
    Ei = tx_data.signal.to(device)
    Fs = tx_data.Rs * tx_data.sps
    Bandwidth = Fs
    alpha0 = alpha / (10 * np.log10(np.exp(1)))  # [1]
    beta2 = get_beta2(D, Fc)  # [ps*s/nm]=[s^2/km]
    Nfft = Ei.shape[-2]
    dbeta1 = Dpmd / np.sqrt(2 * Lcorr) * 1e-12  # [s/km]

    # Linear Operator
    linOperator = partial(
        dispersion_op, Edim=Ei.ndim, beta2=beta2, alpha=alpha0, Nfft=Nfft, Fs=Fs
    )
    LinOp_hz = linOperator(hz).to(device)
    LinOp_half_dz = linOperator(hz / 2).to(device)

    # Amplifier
    myEDFA = partial(edfa, Fs=Bandwidth, G=alpha * Lspan, NF=NF, Fc=Fc, device=device)

    # Nonlinear coeff.
    Gamma = 8 / 9 * gamma if Ei.shape[-1] == 2 else gamma

    # Calculate step length  in one span.
    Nspans = int(np.floor(Ltotal / Lspan))
    Nsteps = int(Lspan / hz)

    z = 0

    # one order scheme
    if order == 1:
        for i in range(Nspans):
            for j in range(Nsteps):
                # linear step  (frequency domain)
                Ei = fft(Ei, axis=-2)
                Ei = Ei * LinOp_hz

                # PMD step  (frequency domain)
                if openPMD:
                    Ti = scatter_matrix(hz, dbeta1, Nfft, Fs, z).to(device)
                    Ei = pmd(Ei, Ti)
                Ei = ifft(Ei, axis=-2)

                # nonlinear step  (time domain)
                Ei = Ei * torch.exp(
                    -1j * Gamma * torch.sum(Ei * torch.conj(Ei), dim=-1)[..., None] * hz
                )

            if amp == "edfa":
                Ei = myEDFA(Ei)
            elif amp == "ideal":
                Ei = Ei * np.exp(alpha0 / 2 * Nsteps * hz)
            elif amp is None:
                Ei = Ei

    elif order == 2:
        Ei = fft(Ei, axis=-2)
        for i in range(Nspans):
            for j in range(Nsteps):
                # First linear step (frequency domain)
                Ei = Ei * LinOp_half_dz

                # PMD step  (frequency domain)
                if openPMD:
                    Ti = scatter_matrix(hz, dbeta1, Nfft, Fs, z)
                    Ei = pmd(Ei, Ti)

                # Nonlinear step (time domain) Ei [batch, Nfft, Nmodes]
                Ei = ifft(Ei, axis=-2)
                Ei = Ei * torch.exp(
                    -1j * Gamma * torch.sum(Ei * torch.conj(Ei), dim=-1)[..., None] * hz
                )

                # Second linear step (frequency domain)
                Ei = fft(Ei, axis=-2)
                Ei = Ei * LinOp_half_dz

                z = z + hz

            if amp == "edfa":
                Ei = myEDFA(Ei)
            elif amp == "ideal":
                Ei = Ei * np.exp(alpha0 / 2 * Nsteps * hz)
            elif amp is None:
                Ei = Ei
        Ei = ifft(Ei, axis=-2)
    else:
        raise (ValueError)
    config = {
        "seed": seed,
        "Ltotal": Ltotal,
        "Lspan": Lspan,
        "hz": hz,
        "alpha": alpha,
        "D": D,
        "gamma": gamma,
        "Fc": Fc,
        "amp": amp,
        "NF": NF,
        "order": order,
        "openPMD": openPMD,
        "Dpmd": Dpmd,
        "Lcorr": Lcorr,
        "unit info": "Ltotal:[km]  Lspan:[km]  hz:[km]  alpha:[dB/km]  D:[ps/nm/km]  Fc:[Hz]  gamma:[1/W/km]  Dpmd:[s/sqrt(km)]  Lcorr:[km] NF:[dB]",
    }
    return (
        WdmSig(
            signal=Ei.to("cpu"),
            symb=tx_data.symb,
            pulse=tx_data.pulse,
            sps=tx_data.sps,
            Rs=tx_data.Rs,
            freqspace=tx_data.freqspace,
        ),
        config,
    )


if __name__ == "__main__":
    import pkufiber.simulation as sml

    tx_data, tx_config = sml.wdm_transmitter(
        123,
        batch=10,
        M=16,
        Nbits=400000,
        sps=16,
        Nch=5,
        Nmodes=1,
        Rs=36,
        freqspace=50,
        Pch_dBm=0,
        Ai=1,
        Vpi=2,
        Vb=-2,
        Ntaps=4096,
        roll=0.1,
        pulse_type="rc",
        device="cuda:0",
    )
    trans_data, fiber_config = sml.fiber_transmission(
        tx_data,
        seed=123,
        Ltotal=80,
        Lspan=80,
        hz=0.5,
        alpha=0.2,
        D=16,
        gamma=1.3,
        Fc=193.1e12,
        amp="edfa",
        NF=4.5,
        order=1,
        openPMD=False,
        Dpmd=3,
        Lcorr=0.1,
        device="cuda:0",
    )
    print(trans_data)
