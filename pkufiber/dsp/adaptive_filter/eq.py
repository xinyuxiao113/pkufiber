import jax, jax.random as rd, jax.numpy as jnp, flax.linen as nn, numpy as np, matplotlib.pyplot as plt, jax.lax as lax
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
from flax.core import freeze, unfreeze, lift
from flax.linen.initializers import lecun_normal
from tqdm import tqdm
from collections import namedtuple
from functools import partial

from .jax_op import scan, frame
from .jax_core import JaxSignal, JaxTime, conv1d_t
import pkufiber.dsp.adaptive_filter.jax_adf as af



class MimoAf(nn.Module):
    taps: int = 32
    rtap: Any = None
    train: Any = False  # 'train' or 'test'
    mimofn: Any = af.ddlms
    learnable: bool = True
    mimokwargs: Any = freeze({})
    mimoinitargs: Any = freeze({})

    @nn.compact
    def __call__(
        self, signal: JaxSignal, truth: JaxSignal, update_state: bool, return_weights: bool=False
    ) -> Union[JaxSignal, Tuple]:

        ## parameters
        if self.learnable:
            if self.mimofn == af.ddlms:
                eta_w = self.param("eta_w", lambda *_: jnp.arctanh(1 / 2**6))
                eta_f = self.param("eta_f", lambda *_: jnp.log(1 / 2**7))
                eta_s = self.param("eta_s", lambda *_: jnp.log(1 / 2**11))
                eta_b = self.param("eta_b", lambda *_: jnp.log(1 / 2**11))
                beta = self.param("beta", lambda *_: jnp.log(1 / 2**11))
                lr_w = jax.nn.tanh(eta_w)
                lr_f = jnp.exp(eta_f)
                lr_s = jnp.exp(eta_s)
                lr_b = jnp.exp(eta_b)
                beta_ = jnp.exp(beta)
                mimo_fn = partial(
                    self.mimofn, lr_w=lr_w, lr_f=lr_f, lr_s=lr_s, lr_b=lr_b, beta=beta_
                )
            elif self.mimofn == af.rde:
                eta = self.param("eta", lambda *_: jnp.arctanh(1 / 2**15))
                lr = jax.nn.tanh(eta)
                mimo_fn = partial(self.mimofn, lr=lr)
            elif self.mimofn == af.lms:
                eta = self.param("eta", lambda *_: jnp.arctanh(1e-4))
                lr = jax.nn.tanh(eta)
                mimo_fn = partial(self.mimofn, lr=lr)
            elif self.mimofn == af.mucma:
                eta = self.param("eta", lambda *_: jnp.arctanh(1e-4))
                beta = self.param("beta", lambda *_: jnp.arctanh(0.999))
                lr = jax.nn.tanh(eta)
                mimo_fn = partial(
                    self.mimofn,
                    lr=lr,
                    dims=signal.val.shape[-1],
                    beta=jax.nn.tanh(beta),
                )
            else:
                raise (NotImplementedError)

        else:
            mimo_fn = self.mimofn

        x = signal.val
        dims = x.shape[-1]
        sps = signal.t.sps
        t = conv1d_t(signal.t, self.taps, self.rtap, sps, "valid")
        x = frame(x, self.taps, sps)

        mimo_init, mimo_update, mimo_apply = mimo_fn(
            train=self.train, **self.mimokwargs
        )
        is_init = self.has_variable(
            "state", "mimoaf"
        )  # 这个是为了初始化时，不更新carry.
        state = self.variable(
            "state",
            "mimoaf",
            lambda *_: (0, mimo_init(dims=dims, taps=self.taps, **self.mimoinitargs)),
            (),
        )
        if truth is not None:
            truth = truth.val[t.start : truth.val.shape[-2] + t.stop]

        af_step, af_stats = state.value
        af_step, (af_stats, (af_weights, _)) = af.iterate(
            mimo_update, af_step, af_stats, x, truth
        )
        y = mimo_apply(af_weights, x)  # 所有的 symbol 都过了之后才运用filter

        if update_state:  # TODO
            state.value = (af_step, af_stats)  # type: ignore

        if return_weights:
            return signal.replace(val=y, t=t, Fs=signal.Fs / sps), af_weights # type: ignore
        else:
            return signal.replace(val=y, t=t, Fs=signal.Fs / sps)  # type: ignore


# @partial(jax.jit, backend="cpu", static_argnums=(2, 3, 4, 5, 6))
def mimoaf(Rx, Tx, taps=32, sps=2, lead_symbols=2000, lr=(1 / 2**6, 1 / 2**7), return_weights=False):
    """
    Input:
        Rx: jax.Array, (Nsymb * sps, Nmodes)
        Tx: jax.Array, (Nsymb, Nmodes)
        taps: int, number of taps
        sps: int, samples per symbol
        lead_symbols: int, number of symbols used to train the filter
        lr: list, learning rate for weight and frequency offset
    Output:
        z: jax.Array, (Nsymb, Nmodes)  or   (z, weights)
    """
    signal = JaxSignal(val=Rx, t=JaxTime(0, 0, sps), Fs=0)
    truth = JaxSignal(val=Tx, t=JaxTime(0, 0, 1), Fs=0)
    model = MimoAf(
        taps=taps,
        train=lambda n: n < lead_symbols,
        mimofn=af.ddlms,
        learnable=False,
        mimokwargs={"lr_w": lr[0], "lr_f": lr[1], "lr_b": 0},
    )
    z, state = model.init_with_output(jax.random.PRNGKey(0), signal, truth, True, return_weights)
    return z


def mimoaf_lms(Rx, Tx, taps=32, sps=2, lead_symbols=2000, lr=1e-4, return_weights=False):
    """
    Input:
        Rx: jax.Array, (Nsymb * sps, Nmodes)
        Tx: jax.Array, (Nsymb, Nmodes)
        taps: int, number of taps
        sps: int, samples per symbol
        lead_symbols: int, number of symbols used to train the filter
        lr: list, learning rate for weight and frequency offset
    Output:
        z: jax.Array, (Nsymb, Nmodes)  or   (z, weights)
    """
    signal = JaxSignal(val=Rx, t=JaxTime(0, 0, sps), Fs=0)
    truth = JaxSignal(val=Tx, t=JaxTime(0, 0, 1), Fs=0)
    model = MimoAf(
        taps=taps,
        train=lambda n: n < lead_symbols,
        mimofn=af.lms,
        learnable=False,
        mimokwargs={"lr": lr},
    )
    z, state = model.init_with_output(jax.random.PRNGKey(0), signal, truth, True, return_weights)
    return z




def cpr(Rx, Tx, N=61, constSymb=None, carry=None, lead_symbols=2000):
    """
    Carrier phase recovery (CPR) for single mode.
    This algrithm can back propagation.
    Much faster on cpu.s

    Input:
    ----------
    Rx : complex-valued ndarray
        Received constellation symbols. (Nsymb, Nmodes)
    Tx : complex-valued ndarray
    N  : int, Half of the 2*N+1 average window.

    Output:
    -------
    Eo, theta, carry
    """
    Nsymb = Rx.shape[0]
    Nmodes = Rx.shape[1]
    pilotInd = jnp.arange(lead_symbols, dtype=int)

    if constSymb == None:
        constSymb = jnp.unique(Tx)

    def D(x):
        d = jnp.abs(x - constSymb) ** 2
        idx = jnp.argmin(d)
        return constSymb[idx]

    def GD(carry, data):
        """
        (N,), (), ()  x  (),(),() --> (N,), () x ()
        phi = [phi_k, phi_{k-1}, ..., phi_{k-N+1}]
        mode: True--use decision,  False--Use pilot.
        """
        phi, varphi, iter = carry
        xk, yk, mode = data
        pk = yk * jnp.exp(1j * varphi)
        dk = D(pk) * mode + xk * (1 - mode)
        phik = jnp.angle(dk / pk) + varphi
        phi = jnp.roll(phi, 1)
        phi = phi.at[0].set(phik)

        varphi = (iter <= N) * jnp.mean(phik) + (iter > N) * jnp.mean(phi)

        iter = iter + 1
        return (phi, varphi, iter), varphi

    GD_vmap = jax.vmap(
        GD, in_axes=-1, out_axes=-1
    )  # (N,Nmodes), (Nmodes), (Nmodes)  x (Nmodes),(Nmodes),(Nmodes) --> (N,Nmodes), (Nmodes) x (Nmodes)
    if carry == None:
        phi = jnp.zeros([N, Nmodes])
        varphi = jnp.mean(phi, axis=0)
        iter = jnp.zeros(Nmodes, dtype=int)
        carry = (phi, varphi, iter)
    mode = jnp.ones([Nsymb, Nmodes]).at[pilotInd].set(0)
    data = (Tx, Rx, mode)
    theta = scan(GD_vmap, carry, data)[1]  # carry, phis
    output = Rx * jnp.exp(1j * theta)

    return output, theta, carry


@partial(jax.jit, static_argnums=(2,3))
def bps(Rx, constSymb, N, B):
    """
    Blind phase search (BPS) algorithm.
    This algorithm can not back propagation. (piece wise constant, grad = 0)
    Input:
    ----------
    Ei : complex-valued ndarray
        Received constellation symbols. (Nsymb, Nmodes)
    constSymb : complex-valued ndarray
        Complex-valued constellation.  (M,)
    N : int
        Half of the 2*N+1 average window.
    B : int
        number of test phases.

    Output:
    -------
    Rx_output, theta, carry

    """
    phi = jnp.arange(0, B) * (jnp.pi / 2) / B - jnp.pi / 4  # test phases  (B,)

    def Lk(phi, x, constSymb):
        """
        () x () x (M,) -->  ()
        """
        return jnp.min(jnp.abs(x * jnp.exp(1j * phi) - constSymb) ** 2)

    L_phi = jax.vmap(
        Lk, in_axes=(0, None, None), out_axes=0
    )  # (B,) x () x (M,) -->  (B,)
    L_phi_x = jax.vmap(
        L_phi, in_axes=(None, 0, None), out_axes=1
    )  # (B,) x (Nsymb) x (M,)  --> (B,Nsymb)
    L_final = jax.vmap(
        L_phi_x, in_axes=(None, -1, None), out_axes=-1
    )  # (B,) x (Nsymb, Nmodes) x (M,)  --> (B, Nsymb, Nmodes)

    Ls = L_final(phi, Rx, constSymb)  # (B,Nsymb,Nmodes)
    convolve_ = jax.vmap(
        partial(jnp.convolve, mode="same"), in_axes=(-1, None), out_axes=-1
    )  # (Nsymb,Nmodes) x (2N+1,) --> (Nsymb, Nmodes)
    convolve = jax.vmap(
        convolve_, in_axes=(0, None), out_axes=0
    )  # (B, Nsymb,Nmodes) x (2N+1,) --> (B, Nsymb, Nmodes)
    score = convolve(Ls, jnp.ones(2 * N + 1))  # (B, Nsymb, Nmodes)
    onehot = jax.nn.one_hot(jnp.argmin(score, axis=0), B).transpose(
        [2, 0, 1]
    )  # (B, Nsymb, Nmodes)
    phase = jnp.sum(onehot * phi[:, None, None], axis=0)
    theta = jnp.unwrap(phase, axis=0, period=np.pi / 2)

    return Rx * jnp.exp(1j * theta), theta, None


def ddpll(
    Rx, Tx, Kv=0.1, constSymb=None, carry=None, lead_symbols=2000, k0=1, k1=-1, k2=1
):
    """
        Decision-directed Phase-locked Loop (DDPLL) algorithm.
        This algrithm can back propagation.
        # This algorithm is much faster on cpu !!
    Input:
        Ei : complex-valued ndarray. Received constellation symbols.
        Kv : float scalar. Loop filter gain.
        constSymb : complex-valued ndarra. Complex-valued ideal constellation symbols.
        symbTx : complex-valued ndarray. Transmitted symbol sequence.
        carry: initial phase estimate.
        k0,k1,k2: momentum parameters.
        pilotInd : int ndarray. Indexes of pilot-symbol locations.
    Output:
        Eo, theta, carry

    References
    [1] H. Meyer, Digital Communication Receivers: Synchronization, Channel
    estimation, and Signal Processing, Wiley 1998. Section 5.8 and 5.9.

    """
    Nsymb = Rx.shape[0]
    Nmodes = Rx.shape[1]
    pilotInd = np.arange(lead_symbols, dtype=int)
    if constSymb == None:
        constSymb = jnp.unique(Tx)
    # if symbTx == None:
    #     symbTx = jnp.zeros(Ei.shape)
    #     mode = jnp.ones([Nsymb,Nmodes]).at[pilotInd].set(0)
    # else:
    mode = jnp.ones([Nsymb, Nmodes]).at[pilotInd].set(0)

    # Loop filter coefficients  [1,k2,k2]
    a1b = jnp.array([k0, k1, k2])

    def Lk(phi, y, constSymb, target, mode):
        """
        mode: pilot=False,  decision=True
        """
        return jnp.min(
            jnp.abs(y * jnp.exp(1j * phi) - constSymb) ** 2
        ) * mode + jnp.abs(y * jnp.exp(1j * phi) - target) ** 2 * (1 - mode)

    g = jax.grad(Lk, argnums=0)

    def GD(carry, data):
        """
        (3,), ()  x (),(),() --> (3,), () x ()
        """
        u, phi = carry
        xk, yk, mode = data

        gk = g(phi, yk, constSymb, xk, mode)
        u = u.at[1].set(u[2])
        u = u.at[2].set(gk)
        u = u.at[0].set(jnp.sum(a1b * u))
        phip = phi
        phi = phi - Kv * u[0]
        return (u, phi), phip
        # return jax.lax.stop_gradient((u, phi)),  jax.lax.stop_gradient(phip)

    GD_vmap = jax.vmap(
        GD, in_axes=-1, out_axes=-1
    )  # (3,Nmodes), (Nmodes)  x (Nmodes),(Nmodes),(Nmodes) --> (3,Nmodes), (Nmodes) x (Nmodes)

    if carry == None:
        u = jnp.zeros([3, Nmodes])  # [u_f, u_d1, u_d]
        phi = jnp.zeros(Nmodes)
        carry = (u, phi)

    mode = jnp.ones([Nsymb, Nmodes]).at[pilotInd].set(0)
    data = (Tx, Rx, mode)  # (Nsymb, Nmodes), (Nsymb, Nmodes), (Nsymb, Nmodes)
    carry, theta = scan(GD_vmap, carry, data)  # carry, phis
    theta = jnp.unwrap(theta, axis=0, period=np.pi / 2)
    return Rx * jnp.exp(1j * theta), theta, carry


def foeddpll(
    Rx,
    Tx,
    Kf=0.01,
    Kn=0.1,
    constSymb=None,
    carry=None,
    w0=0,
    p0=0,
    k0=1,
    k1=-1,
    k2=1,
    lead_symbols=2000,
):
    """
        FOEddpll. remove two scale phase rotation!
     Input:
        Ei : complex-valued ndarray. Received constellation symbols.
        Kf : float scalar. learning rate for frequency offset (FO).
        Kn : float scalar. learning rate for phase rotation .
        constSymb : complex-valued ndarray. Complex-valued ideal constellation symbols.
        symbTx : complex-valued ndarray. Transmitted symbol sequence.
        carry: state.
        w0,p0: intial FO, phase estimate.
        k0,k1,k2: momentum parameters.
        pilotInd : int ndarray. Indexes of pilot-symbol locations.

    Output:
        Eo, theta, carry
    """

    Nsymb = Rx.shape[0]
    Nmodes = Rx.shape[1]
    pilotInd = np.arange(lead_symbols, dtype=int)
    mode = jnp.ones([Nsymb, Nmodes]).at[pilotInd].set(0)

    # Loop filter coefficients  [1,k2,k2]
    a1b = jnp.array([k0, k1, k2])

    def Lk(phi, y, constSymb, target, mode):
        """
        mode: pilot=False,  decision=True
        """
        return jnp.min(
            jnp.abs(y * jnp.exp(-1j * phi) - constSymb) ** 2
        ) * mode + jnp.abs(y * jnp.exp(-1j * phi) - target) ** 2 * (1 - mode)

    g = jax.grad(Lk, argnums=0)

    def GD(carry, data):
        """
        (3,), (), ()  x (),(),() --> (3,), () x ()
        """
        u, w, phi = carry
        xk, yk, mode = data

        phip = phi + w
        gk = g(phip, yk, constSymb, xk, mode)

        u = u.at[1].set(u[2])
        u = u.at[2].set(gk)
        u = u.at[0].set(jnp.sum(a1b * u))

        phi = phip - Kn * u[0]
        w = w - Kf * u[0]
        return (u, w, phi), (phip, w)
        # return jax.lax.stop_gradient((u, phi)),  jax.lax.stop_gradient(phip)

    GD_vmap = jax.vmap(
        GD, in_axes=-1, out_axes=-1
    )  # (3,Nmodes), (Nmodes),(Nmodes)  x (Nmodes),(Nmodes),(Nmodes) --> (3,Nmodes), (Nmodes) x (Nmodes)

    if carry == None:
        u = jnp.zeros([3, Nmodes])  # [u_f, u_d1, u_d]
        phi = jnp.ones(Nmodes) * (p0 % (2 * np.pi))
        w = jnp.ones(Nmodes) * (w0 % (2 * np.pi))
        carry = (u, w, phi)

    mode = jnp.ones([Nsymb, Nmodes]).at[pilotInd].set(0)
    data = (Tx, Rx, mode)  # (Nsymb, Nmodes), (Nsymb, Nmodes), (Nsymb, Nmodes)
    carry, (theta, w) = scan(GD_vmap, carry, data)  # carry, phis
    theta = jnp.unwrap(theta, axis=0, period=np.pi / 2)
    return Rx * jnp.exp(1j * theta), (theta, w), carry
