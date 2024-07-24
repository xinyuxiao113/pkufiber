import numpy as np
import torch
from commpy.modulation import QAMModem
from typing import Tuple
from commpy.filters import rrcosfilter, rcosfilter
from pkufiber.utils import calc_time
from pkufiber.op import circFilter, mean_power
from pkufiber.core import WdmSig


class QAM(QAMModem):
    """
    QAM Modulation format.
    """

    def __init__(self, M, reorder_as_gray=True):
        super(QAM, self).__init__(M)
        self.constellation_torch = torch.tensor(self.constellation)
        self.Es = torch.tensor(self.Es)

    def bit2symbol(self, bits):
        N = len(bits)
        pow = 2 ** torch.arange(N - 1, -1, -1)
        idx = torch.sum(pow * bits)
        return self.constellation_torch[idx]

    def demodulate(self, symbs: torch.Tensor, dim=1):
        """
        demodulate for the last axis. dim >= 0
        """
        # symbs: [*]
        assert dim >= 0
        shape = [1] * (symbs.ndim + 1)
        shape[dim + 1] = -1
        shape_out = list(symbs.shape)
        shape_out[dim] = -1
        idx = torch.argmin(
            torch.abs(
                symbs.unsqueeze(dim + 1)
                - self.constellation_torch.reshape(shape).to(symbs.device)
            ),
            dim=dim + 1,
        )  # [batch, Nsymb] or [Nsymb]
        pow = (
            (2 ** torch.arange(self.num_bits_symbol - 1, -1, -1))
            .reshape(shape)
            .to(symbs.device)
        )
        bits = (idx.unsqueeze(dim + 1) // pow) % 2  # [*, 4]
        return bits.reshape(shape_out)

    def modulate(self, bits: torch.Tensor, dim=0):
        constellation = self.constellation_torch.to(bits.device)
        pow = 2 ** torch.arange(self.num_bits_symbol - 1, -1, -1).to(
            bits.device
        )  # [num_bits_symbol]
        bits_batch = bits.unfold(
            dim, self.num_bits_symbol, self.num_bits_symbol
        )  # [*, num_bits_symbol]
        idx = torch.sum(bits_batch * pow, dim=-1)  # [*]
        return constellation[idx]

    def const(self) -> torch.Tensor:
        return self.constellation_torch / np.sqrt(self.Es)


def choose_sps(Nch: int, freqspace: float, Rs: float) -> int:
    """
    Choose transmitter sps according to frequence space (freqspace) and baudrate (Rs).
    Input:
        Nch: Number of channels.
        freqspace: Frequency space. [Hz]
        Rs: Symbol rate. [Hz]
    Output:
        Transmitter samples per symbol.
    """
    power = int(np.log2(Nch * freqspace / Rs)) + 1
    return 2**power


def mzm(Ai, Vπ, u, Vb):
    """
    MZM modulator

    :param Vπ: Vπ-voltage
    :param Vb: bias voltage
    :param u:  modulator's driving signal (real-valued)
    :param Ai: amplitude of the input CW carrier

    :return Ao: output optical signal
    """
    π = np.pi
    Ao = Ai * torch.cos(0.5 / Vπ * (u + Vb) * π)

    return Ao


def iqm(Ai, u, Vπ, VbI, VbQ):
    """
    IQ modulator

    :param Vπ: MZM Vπ-voltage
    :param VbI: in-phase MZM bias voltage
    :param VbQ: quadrature MZM bias voltage
    :param u:  modulator's driving signal (complex-valued baseband)
    :param Ai: amplitude of the input CW carrier

    :return Ao: output optical signal
    """
    Ao = mzm(Ai / np.sqrt(2), Vπ, u.real, VbI) + 1j * mzm(
        Ai / np.sqrt(2), Vπ, u.imag, VbQ
    )

    return Ao


def pulse_shape(pulseType, SpS=2, N=1024, alpha=0.1, Ts=1):
    """
    Generate pulse shaping filters

    :param pulseType: 'rect','nrz','rrc'
    :param SpS: samples per symbol
    :param N: number of filter coefficients
    :param alpha: RRC rolloff factor
    :param Ts: symbol period

    :return filterCoeffs: normalized filter coefficients

    pulse: [*,x1,x2,...,x2,x1]
    """
    assert N % 2 == 0
    fa = (1 / Ts) * SpS  # sample rate

    t = np.linspace(-2, 2, SpS)
    Te = 1

    if pulseType == "rect":
        filterCoeffs = np.concatenate(
            (np.zeros(int(SpS / 2)), np.ones(SpS), np.zeros(int(SpS / 2)))
        )
    elif pulseType == "nrz":
        filterCoeffs = np.convolve(
            np.ones(SpS), 2 / (np.sqrt(np.pi) * Te) * np.exp(-(t**2) / Te), mode="full"
        )
    elif pulseType == "rrc":
        tindex, filterCoeffs = rrcosfilter(N, alpha, Ts, fa)
    else:
        tindex, filterCoeffs = rcosfilter(N, alpha, Ts, fa)

    return filterCoeffs / np.sqrt(np.sum(filterCoeffs**2))


def upsample(tensor, dim, N):
    """
    Upsample a tensor along a specified dimension.
    Input:
        tensor: input tensor.
        dim: dimension to upsample.
        N: upsampling factor.
    """
    original_shape = list(tensor.shape)
    # 新的形状，把指定的维度乘以N
    new_shape = original_shape.copy()
    new_shape[dim] *= N
    # 创建一个新的全零张量
    result = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)

    # 使用切片将原始数据填充到新张量中
    slices_original = [slice(None) for _ in original_shape]
    slices_result = [slice(None) for _ in new_shape]
    slices_result[dim] = slice(None, None, N)

    result[tuple(slices_result)] = tensor[tuple(slices_original)]

    return result


def phase_noise(batch, lw, Nsamples, Ts, dtype=torch.float64):
    """
    Generate phase noise.
    Input:
        batch: number of signals.
        lw: linewidth. [Hz]
        Nsamples; number of samples.
        Ts: sample time. [s]
    Output:
        phase noise. (batch, Nsamples)
    """
    sigma2 = 2 * np.pi * lw * Ts
    phi = torch.randn(batch, Nsamples, dtype=dtype) * np.sqrt(sigma2)

    return torch.cumsum(phi, dim=1)


def local_oscillator(batch, Ta, FO, lw, phi_lo, freq, N, Plo_dBm, device="cuda:0"):
    """
    Input:
        batch: batch size.
        Ta: sample time. [s]
        FO: frequencey offset. [Hz]
        lw: linewidth.    [Hz]
        phi_lo: init phase error.  [rad]
        freq: frequency. [Hz]
        N: signal length.
        Plo_dBm: power [dBm]
    Output:
        sigLO: [batch, Nfft]
        phi_noise: [batch, Nfft]
    """

    Plo = 10 ** (Plo_dBm / 10) * 1e-3  # power in W
    Delta_f = freq + FO  # downshift of the channel to be demodulated

    # generate LO field
    t = torch.arange(0, N).to(torch.float64) * Ta
    pn = phase_noise(batch, lw, N, Ta)
    phi = phi_lo + 2 * torch.pi * Delta_f * t[None, :] + pn
    phi_noise = phi_lo + 2 * torch.pi * FO * t[None, :] + pn
    sigLO = np.sqrt(Plo) * torch.exp(1j * phi)

    return sigLO.to(device), phi_noise.to(device)


def freq_grid(Nch: int, freqspace: float) -> torch.Tensor:
    """
        generate a frequency grid.
    Input:
        Nch: number of channels.
        freqspace: frequency space.
    Output:
        jax.Array.
    """
    freqGrid = (
        torch.arange(-int(Nch / 2), int(Nch / 2) + 1, 1).to(torch.float64) * freqspace
    )
    if (Nch % 2) == 0:
        freqGrid += freqspace / 2

    return freqGrid


def wdm_base(Nfft: int, fa: float, freqGrid: torch.Tensor) -> torch.Tensor:
    """
        Generate a WDM waves.
    Input:
        Nfft: number of samples.
        fa: sampling rate. [hz]
        freqGrid: frequen
    Output:
        jax.Array with shape [Nfft, Nch].
    """
    t = torch.arange(0, Nfft).to(torch.float64)
    assert t.dtype == torch.float64
    # [Nsymb*SpS, Nch]
    wdm_wave = torch.exp(1j * 2 * torch.pi * (freqGrid[None, :] / fa) * t[:, None])
    assert wdm_wave.dtype == torch.complex128
    return wdm_wave


def wdm_merge(
    sigWDM: torch.Tensor, Fs: float, Nch: int, freqspace: float
) -> torch.Tensor:
    """
        Multiplex all WDM channel signals to a single WDM signal.
    Input:
        sigWDM: Signals for all WDM channels with shape (batch, Nfft, Nch, Nmodes) or [Nfft,Nch,Nmodes]
        Fs: float, sampling rate.
        Nch: number of channels.
        freqspace: frequency space.
    Output:
        The Single WDM signal with shape [batch, Nfft, Nmdoes] or [Nfft, Nmdoes].
    """
    # sigWDM.shape  [batch, Nfft,Nch,Nmodes] or [Nfft,Nch,Nmodes]
    Nfft = sigWDM.shape[-3]
    freqGrid = torch.arange(-int(Nch / 2), int(Nch / 2) + 1, 1) * freqspace

    wdm_wave = wdm_base(Nfft, Fs, freqGrid)  # [Nfft, Nch]
    if sigWDM.ndim == 4:
        wdm_wave = wdm_wave[None, ..., None]
    elif sigWDM.ndim == 3:
        wdm_wave = wdm_wave[..., None]
    else:
        raise (ValueError)

    wdm_wave = wdm_wave.to(sigWDM.device)
    x = torch.sum(sigWDM * wdm_wave, dim=-2)
    return x  # [batch, Nfft, Nmdoes] or [Nfft, Nmdoes]


@calc_time
def wdm_transmitter(
    seed: int,
    batch: int,
    M: int,
    Nbits: int,
    sps: int,
    Nch: int,
    Nmodes: int,
    Rs: float,
    freqspace: float,
    Pch_dBm: float = 0,
    Ai: float = 1,
    Vpi: float = 2,
    Vb: float = -2,
    Ntaps: int = 4096,
    roll: float = 0.1,
    pulse_type: str = "rc",
    device: str = "cuda:0",
) -> Tuple[WdmSig, dict]:
    """
    WDM transmitter: Generates a complex baseband waveform representing a WDM signal with arbitrary number of carriers

    Parameters:
    ----------
    seed: int
        Random seed.
    batch: int
        Number of signals.
    M: int
        QAM order.
    Nbits: int
        Total number of bits per carrier.
    sps: int
        Samples per symbol.
    Nch: int
        Number of WDM channels.
    Nmodes: int
        Number of polarization modes.
    Rs: float
        Symbol rate in GHz.
    freqspace: float
        Frequency spacing in GHz.
    Pch_dBm: float, optional
        Signal power per channel in dBm. Default is 0.
    Ai: float, optional
        Input CW carrier amplitude. Default is 1.
    Vpi: float, optional
        MZM Vπ-voltage. Default is 2.
    Vb: float, optional
        MZM bias voltage. Default is -2.
    Ntaps: int, optional
        Number of filter taps. Default is 4096.
    roll: float, optional
        RRC rolloff factor. Default is 0.1.
    pulse_type: str, optional
        Pulse shape type. Default is 'rc'.
    device: str, optional
        Device. Default is 'cuda:0'.

    Returns:
    -------
    Tuple[WdmSig, dict]
        WdmSig: WDM signal with pulse shape and configuration.
        symbs: Transmitted symbols with shape [batch, Nsymb, Nch, Nmodes].
    """
    torch.manual_seed(seed)
    # Verify sampling theorem
    Rs = Rs * 1e9  # symbol rate [Hz]
    freqspace = freqspace * 1e9  # frequency space [Hz]

    fa = Rs * sps
    fc = Nch / 2 * freqspace
    print(
        "Sample rate fa: %g, Cut off frequency fc: %g, fa > 2fc: %s"
        % (fa, fc, fa > 2 * fc)
    )
    if fa < 2 * fc:
        print("sampling thm does not hold!")
        raise (ValueError)

    # modulation scheme
    mod = QAM(M=M)
    Pch = 10 ** (Pch_dBm / 10) * 1e-3

    # pulse shape
    pulse = pulse_shape(pulse_type, sps, N=Ntaps, alpha=roll)
    pulse = torch.tensor(pulse, dtype=torch.complex128).to(device)

    x = torch.randint(0, 2, (batch, Nbits, Nch, Nmodes), device=device)
    symbs = mod.modulate(x, dim=1)
    x = symbs / np.sqrt(mod.Es)  # [batch, Nsymb, Nch, Nmodes]
    x = upsample(x, 1, sps)
    x = circFilter(pulse, x, dim=1)
    x = iqm(Ai, 0.5 * x, Vpi, Vb, Vb)
    # [batch, Nsymb*sps, Nch, Nmodes]
    x = np.sqrt(Pch / Nmodes) * x / torch.sqrt(mean_power(x))
    x = wdm_merge(x, Rs * sps, Nch, freqspace)  # [batch, Nsymb*sps, Nmodes]

    x = x.to("cpu")
    symbs = symbs.to("cpu")
    config = {
        "seed": seed,
        "batch": batch,
        "M": 16,
        "Nbits": Nbits,
        "sps": sps,
        "Nch": Nch,
        "Nmodes": Nmodes,
        "Rs": Rs,
        "freqspace": freqspace,
        "Pch_dBm": Pch_dBm,
        "Ai": Ai,
        "Vpi": Vpi,
        "Vb": Vb,
        "Ntaps": Ntaps,
        "roll": roll,
        "pulse_type": pulse_type,
        "shape_info": "sigWDM:[batch, Nsymb*SpS, Nch, Nmodes],  SymbTx:[batch, Nsymb, Nch, Nmodes]",
        "pulse": pulse,
        "Fc": 299792458 / 1550e-9,
    }

    return (
        WdmSig(signal=x, symb=symbs, pulse=pulse, sps=sps, Rs=Rs, freqspace=freqspace),
        config,
    )


if __name__ == "__main__":

    device = "cuda:0"
    tx_data, tx_config = wdm_transmitter(
        123,
        batch=10,
        M=16,
        Nbits=400000,
        sps=16,
        Nch=5,
        Nmodes=1,
        Rs=20,
        freqspace=50,
        Pch_dBm=0,
        Ai=1,
        Vpi=2,
        Vb=-2,
        Ntaps=4096,
        roll=0.1,
        pulse_type="rc",
        device=device,
    )
    print(tx_data)
