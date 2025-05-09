import torch
import numpy as np
import warnings
from torch.fft import fft
from typing import Tuple,Union

from pkufiber.op import get_omega, mean_power, circFilter
from pkufiber.utils import qfactor, calc_time
from pkufiber.core import WdmSig

from pkufiber.simulation.transmitter import local_oscillator, QAM


def downsampling(x0, rate):
    """
    Input:
        x0: [batch, Nftt, Nmodes] or [Nfft, Nmodes]
        rate: int
    Output:
        y: [batch, Nftt//rate, Nmodes] or [Nfft//rate, Nmodes]
    """
    if x0.ndim == 2:
        y = x0[::rate, :]
    elif x0.ndim == 3:
        y = x0[:, ::rate, :]
    else:
        raise (ValueError)
    return y


def balanced_pd(E1, E2, R=1):
    """
    Balanced photodetector (BPD)

    :param E1: input field [nparray]
    :param E2: input field [nparray]
    :param R: photodiode responsivity [A/W][scalar, default: 1 A/W]

    :return: balanced photocurrent
    """
    # assert R > 0, 'PD responsivity should be a positive scalar'
    assert E1.shape == E2.shape, "E1 and E2 need to have the same size"
    i1 = R * E1 * torch.conj(E1)
    i2 = R * E2 * torch.conj(E2)
    return i1 - i2


def hybrid_2x4_90deg(E1, E2):
    """
    Optical 2 x 4 90° hybrid

    :param E1: input signal field [nparray]
    :param E2: input LO field [nparray]

    :return: hybrid outputs
    """
    assert E1.shape == E2.shape, "E1 and E2 need to have the same size"

    # optical hybrid transfer matrix
    T = torch.tensor(
        [
            [1 / 2, 1j / 2, 1j / 2, -1 / 2],
            [1j / 2, -1 / 2, 1 / 2, 1j / 2],
            [1j / 2, 1 / 2, -1j / 2, -1 / 2],
            [-1 / 2, 1j / 2, -1 / 2, 1j / 2],
        ]
    ).to(E1.device)

    Ei = torch.stack([E1, torch.zeros_like(E1), torch.zeros_like(E1), E2])  # [4, N]

    Eo = T @ Ei.to(torch.complex64)

    return Eo


def coherent_receiver(Es, Elo, Rd=1):
    """
    Single polarization coherent optical front-end

    :param Es: input signal field [nparray]
    :param Elo: input LO field [nparray]
    :param Rd: photodiode resposivity [scalar]

    :return: downconverted signal after balanced detection
    """
    assert Es.shape == Elo.shape, "Es and Elo need to have the same size"

    # optical 2 x 4 90° hybrid
    Eo = hybrid_2x4_90deg(Es, Elo)

    # balanced photodetection
    sI = balanced_pd(Eo[1, :], Eo[0, :], Rd)
    sQ = balanced_pd(Eo[2, :], Eo[3, :], Rd)

    return sI + 1j * sQ


def frequency_cut_receiver(
    E: torch.Tensor, chid: int, sps_in: int, sps_out: int, Fs: float, freqspace: float
) -> torch.Tensor:
    """
    Get single channel information from WDM signal.
    Input:
        E: 1D array. (Nfft,Nmodes) or (batch, Nfft, Nmodes)
        chid: channel id.  -Nch/2,  ..., 0, 1,..., Nch/2
        sps_in: sps of input signal.
        sps_out: sps of output signal.
        Nch: number of channels.
        Fs: sampling rate.
        freqspace: frequency space between channels.
    Output:
        Eout: single channel signal. (Nfft,Nmodes)
    """
    assert sps_in % sps_out == 0
    k = chid
    Nfft = E.shape[-2]
    omega = get_omega(Fs, Nfft)
    f = omega / (2 * np.pi)  # [Nfft]
    if E.ndim == 2:
        H = (torch.abs(f - k * freqspace) < freqspace / 2)[:, None]
    elif E.ndim == 3:
        H = (torch.abs(f - k * freqspace) < freqspace / 2)[None, :, None]
    else:
        raise (ValueError)

    x0 = torch.fft.ifft(
        torch.roll(
            fft(E, axis=-2) * H.to(E.device), -k * int(freqspace / Fs * Nfft), dims=-2
        ),
        dim=-2,
    )
    Eout = downsampling(x0, sps_in // sps_out)

    return Eout


def map_symbols_to_indices(X, Y = QAM(16).const()):

    # 扩展X的维度，使其能够与Y进行广播运算
    X_expanded = X.unsqueeze(-1)  # [B, T, 1]
    
    # 计算X中每个符号与Y中符号的距离，并找到最小距离的索引
    distances = torch.abs(X_expanded - Y)  # [B, T, N]
    indices = torch.argmin(distances, dim=-1)  # [B, T]
    
    return indices


def nearst_symb(y, constSymb: torch.Tensor = QAM(16).const().to(torch.complex64)):  # type: ignore
    """
    y: normaized symbols.
    ConstSymb: [M] normalized constellation symbols.
    """
    power = mean_power(y)
    if power > 2 or power < 0.5:
        warnings.warn("The power of y is not near to 1. Do you normalize y?")
    constSymb = constSymb.to(y.device)
    const = constSymb.reshape([1] * len(y.shape) + [-1])  # type: ignore
    y = y.unsqueeze(-1)
    k = torch.argmin(torch.abs(y - const), dim=-1)
    return constSymb[k]


def ser(y, truth, M=16, eps=1e-3):
    """
    Calculate SER.
    Input:
        y: normalized symbols. [batch, Nsymb, Nmodes] or [Nsymb, Nmodes]
        truth: normalized symbols. [batch, Nsymb, Nmodes] or [Nsymb, Nmodes]
        M: modulation order.
        eps: error threshold.
    """
    const_symbol = QAM(16).const().to(torch.complex64)
    z = nearst_symb(y, const_symbol)
    er = torch.abs(z - truth) > eps
    return torch.mean(er * 1.0, dim=-2).to("cpu").numpy(), z


def ber(y: torch.Tensor, truth: torch.Tensor, M=16):
    """
    Calculate BER.
    Input:
        y:  normalized symbols.    [Nsymb,Nmodes]  or [batch, Nsymb,Nmodes]   L2(y) ~ 1
        truth: normalized symbols. [Nsymb, Nmodes] or [batch, Nsymb,Nmodes]   L2(truth) ~ 1
    return:
        metric, [Nbits, Nmodes]
    """
    assert y.ndim >= 2
    assert y.shape == truth.shape

    def getpower(x):
        return torch.mean(torch.abs(x) ** 2, dim=-2)

    def SNR_fn(y, x):
        return 10.0 * torch.log10(getpower(x) / getpower(x - y))

    ser_value, z = ser(y, truth, M=M)
    mod = QAM(M)
    dim = z.ndim - 2
    br = mod.demodulate(z * np.sqrt(mod.Es), dim=dim)
    bt = mod.demodulate(truth * np.sqrt(mod.Es), dim=dim)
    ber_value = torch.mean((br != bt) * 1.0, dim=-2)

    return {
        "BER": ber_value.cpu().numpy(),
        "SER": ser_value,
        "Qsq": qfactor(ber_value.cpu().numpy()),
        "SNR": SNR_fn(y, truth).cpu().detach().numpy(),
    }


def qfactor_all(Rx: Union[torch.Tensor, np.ndarray], Tx: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Q^2 factor for all modes.
    Input:
        Rx: [Nfft, Nmodes]
        Tx: [Nfft, Nmodes]
    Output:
        Q: [Nmodes]
    """
    if isinstance(Rx, np.ndarray):
        Rx = torch.tensor(Rx)
    if isinstance(Tx, np.ndarray):
        Tx = torch.tensor(Tx)
    ber_value = ber(Rx, Tx)['BER']
    Q = qfactor(np.mean(ber_value))
    return Q



def qfactor_path(
    Rx: torch.Tensor, Tx: torch.Tensor, Ntest: int = 10000, stride: int = 10000
) -> list:
    """
    Calculate Q^2 factor along the path.
    Input:
        Rx: [Nfft, Nmodes]
        Tx: [Nfft, Nmodes]
        Ntest: number of symbols for test.
        stride: stride for test.
    Output:
        Q: [N]
    """
    Q = []
    for t in np.arange(0, Rx.shape[-2] - Ntest, stride):
        Q.append(
            np.mean(
                ber(Rx[t : t + Ntest], Tx[t : t + Ntest])[
                    "Qsq"
                ]
            )
        )
    return Q


@calc_time
def wdm_receiver(
    wdmsig: WdmSig,
    seed: int,
    chid: int,
    rx_sps: int,
    FO: float = 0,
    lw: float = 0,
    phi_lo: float = 0,
    Plo_dBm: float = 10,
    method: str = "frequency cut",
    device: str = "cuda:0",
) -> Tuple[WdmSig, dict]:
    """
    WDM Receiver: Processes the received WDM signal and performs down-conversion, filtering, and demodulation.

    Parameters:
    ----------
    wdmsig: WdmSig
        The received WDM signal.
    seed: int
        Random seed.
    chid: int
        Channel ID. Range from -Nch/2 to Nch/2.
    rx_sps: int
        Receiver samples per symbol.
    FO: float, optional
        Frequency offset in Hz. Default is 0 Hz.
    lw: float, optional
        Linewidth of local oscillator in Hz. Default is 0 Hz.
    phi_lo: float, optional
        Initial phase of the local oscillator in radians. Default is 0 radians.
    Plo_dBm: float, optional
        Power of the local oscillator in dBm. Default is 10 dBm.
    method: str, optional
        Method for processing. Either 'frequency cut' or 'filtering'. Default is 'frequency cut'.
    device: str, optional
        Device to perform computation on. Default is 'cuda:0'.

    Returns:
    -------
    Tuple[WdmSig, dict]
        A tuple containing the processed WDM signal and a configuration dictionary.
    """

    torch.manual_seed(seed)
    trans_data = wdmsig.signal
    assert trans_data.shape[-1] == 1 or trans_data.shape[-1] == 2

    dims = trans_data.ndim
    batch = trans_data.shape[0]
    N = trans_data.shape[-2]
    Ta = 1 / wdmsig.Rs / wdmsig.sps
    freq = chid * wdmsig.freqspace
    sigWDM = trans_data.to(device)  # [batch, Nfft, Nmodes] or [Nfft, Nmodes]

    sigLO, phi_pn_lo = local_oscillator(
        batch, Ta, FO, lw, phi_lo, freq, N, Plo_dBm, device=device
    )

    if method == "frequency cut":

        sigRx2 = frequency_cut_receiver(
            sigWDM, chid, wdmsig.sps, rx_sps, 1 / Ta, wdmsig.freqspace
        )
        sigRx2 = sigRx2 * torch.exp(-1j * phi_pn_lo[:, :: (wdmsig.sps // rx_sps), None])

    elif method == "filtering":

        # step 1: coherent receiver
        CR1 = torch.vmap(coherent_receiver, in_dims=(-1, None), out_dims=-1)
        if dims == 2:
            sigRx1 = CR1(sigWDM, sigLO)  # [Nfft, Nmodes], [Nfft]
        elif dims == 3:
            CR2 = torch.vmap(CR1, in_dims=(0, 0), out_dims=0)
            sigRx1 = CR2(sigWDM, sigLO)  # [batch, Nfft, Nmodes], [batch, Nfft]
        else:
            print("dims:", dims)
            raise ValueError("dims should be 2 or 3.")

        filter1 = torch.vmap(circFilter, in_dims=(None, -1), out_dims=-1)
        if dims == 2:
            sigRx2 = filter1(wdmsig.pulse.to(device), sigRx1)
        elif dims == 3:
            filter2 = torch.vmap(filter1, in_dims=(None, 0), out_dims=0)
            sigRx2 = filter2(wdmsig.pulse.to(device), sigRx1)
        else:
            raise ValueError("dims should be 2 or 3.")

        sigRx2 = downsampling(sigRx2, wdmsig.sps // rx_sps)
    else:
        raise ValueError("method should be 'frequency cut' or 'filtering' ")

    # step 3: normalization and resampling # TODO: 可以优化！
    sigRx = sigRx2 / torch.sqrt(mean_power(sigRx2))
    config = {
        "seed": seed,
        "chid": chid,
        "rx_sps": rx_sps,
        "FO": FO,
        "lw": lw,
        "phi_lo": phi_lo,
        "Plo_dBm": Plo_dBm,
        "method": method,
    }

    Nch = wdmsig.symb.shape[-2]
    idx = Nch // 2 + chid
    return (
        WdmSig(
            signal=sigRx.to("cpu"),
            symb=wdmsig.symb[:, :, idx : idx + 1, :],
            pulse=wdmsig.pulse,
            sps=rx_sps,
            Rs=wdmsig.Rs,
            freqspace=wdmsig.freqspace,
        ),
        config,
    )


def test():
    from pkufiber.simulation.transmitter import wdm_transmitter
    from pkufiber.simulation.channel import fiber_transmission

    tx_data, tx_config = wdm_transmitter(
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
    trans_data, fiber_config = fiber_transmission(
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
    rx_data, rx_config = wdm_receiver(
        trans_data,
        seed=123,
        chid=0,
        rx_sps=2,
        FO=0,
        lw=0,
        phi_lo=0,
        Plo_dBm=10,
        method="frequency cut",
        device="cuda:0",
    )

    print(rx_data)


if __name__ == "__main__":
    test()
