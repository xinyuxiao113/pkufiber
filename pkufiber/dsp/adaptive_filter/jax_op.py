import jax.numpy as jnp
import jax, optax
import numpy as np
from jax.numpy.fft import fft, fftfreq, fftshift
from jax import lax, jit, vmap, numpy as jnp, device_put
from functools import partial

Array = jax.Array


def circsum(a: Array, N: int) -> Array:
    """
    Transform a 1D array a to a N length array.
    Input:
        a: 1D Array.
        N: a integer.
    Output:
        d: 1D array with length N.

    d[k] = sum_{i=0}^{+infty} a[k+i*N]
    """
    b = frame(a, N, N)  # [*,N]
    t = b.shape[0] * N
    c = a[t::]
    d = jnp.sum(b, axis=0)
    d = d.at[0 : c.size].add(c)
    return d


def l2(x: Array) -> Array:
    """
    Caculate average L2 Norm of x.
    Input:
        x: Array.
    Output:
        scaler.
    """
    return jnp.sqrt(jnp.mean(jnp.abs(x) ** 2))


def relative_l2(X: Array, Y: Array) -> Array:
    """
    Relative L2 norm of X,Y.
    """
    return l2(X - Y) / l2(X)


def mirror(x: Array, axis: int = 0) -> Array:
    """
    Mirror operator for an array at any axis. Use for ifft.
    Input:
        x=[x0,x1,...,x_{N-1}]
    Output:
        Px=[x0,x_{N-1},x_{N-2},...,x_2]
    """
    return jnp.roll(jnp.flip(x, axis), 1, axis)


# 使用 jnp.fft.ifft 会造成误差  #BUG: fft(ifft) != id, 在float64下可以看出区别
def ifft(x: Array, axis: int = -1) -> Array:
    """
        Same as jnp.fft.ifft
        #BUG: Use this approach instead of jnp.fft.ifft can not handle 64 precision. Already fixed!
        #  jnp.fft.ifft(jnp.fft.fft(x)) - x is  about 1e-8.
    Input:
        x: array.
        axis: ifft axis.
    Output:
        ifft(x): ifft Array of x on the specified axis.
    """
    x = jnp.fft.fft(x, axis=axis)
    N = x.shape[axis]
    return 1 / N * mirror(x, axis=axis)


def firFilter(h: Array, x: Array) -> Array:
    """
    1D zeros pad convolutoin. Equive to jnp.convolve(mode='same')

    Input:
        h: 1D convolution kernel. (K,)   K=2k or  2k+1 .  center of h is ar h[k+1].
        x: 1D signal. (N,)
    Output:
        y: output filtered signal. (N,)
        y[i] = sum_{j=0}^{K} h[j]x[i-k-j]   x use zero padding.
    """
    N = h.size
    x = jnp.pad(x, (0, int(N / 2)), "constant")
    y = convolve(h, x)[0 : x.size]

    return y[int(N / 2) : y.size]


def circFilter(h: Array, x: Array) -> Array:
    """
        1D Circular convolution. overlap_and_save version.
    Input:
        h: (K,)  K = 2k or 2k + 1. center of h is ar h[k+1].
        x: (N,)
    Output:
        z: (N,)
        z[i] = sum_{j=0}^{K} h[j]x[i-k-j]   x use circular padding.
    """
    k = h.shape[0] // 2  # h[k] is the center of h(t).
    N = x.shape[0]
    y = convolve(x, h)  # [N+2k-1]  or  [N+2k]
    z = circsum(y, N)
    z = jnp.roll(z, -k)
    return z


def circFilter_(h: Array, x: Array) -> Array:
    """
    1D Circular convolution. fft version.
    """
    k = h.size // 2
    h_ = circsum(h, x.size)
    h_ = jnp.roll(h_, -k)
    return conv_circ(h_, x)


def frame_gen(x: Array, flen: int, fstep: int, fnum: int = -1):
    """
        generate circular frame from Array x.
    Input:
        x: Arrays about to be framed with shape (B, *dims)
        flen: frame length.
        fstep: step size of frame.
        fnum: steps which frame moved.
    Output:
        A generator. Each step output a data with size (flen, *dims)
    """
    s = np.arange(flen)
    N = x.shape[0]

    if fnum == -1:
        fnum = 1 + (N - flen) // fstep

    for i in range(fnum):
        yield x[(s + i * fstep) % N]


def frame(x: Array, flen: int, fstep: int, fnum: int = -1) -> Array:
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


def conv_circ(signal: Array, ker: Array) -> Array:
    """
    N-size circular convolution.

    Input:
        signal: real 1D array with shape (N,)
        ker: real 1D array with shape (N,).
    Output:
        signal conv_N  ker.
    """
    return jnp.fft.ifft(jnp.fft.fft(signal) * jnp.fft.fft(ker))


def corr_circ(x: Array, y: Array) -> Array:  # 不交换
    """
    N-size correlation.
    Input:
        x: 1D array with shape (N,)
        y: 1D array with shape (N,)
    Output:
        z: 1D array with shape (N,)
        z[n] = sum_{i=1}_{N} x[i] y[i - n] = sum_{i=1}_{N} x[i + n]y[i]
    """
    return conv_circ(x, jnp.roll(jnp.flip(y), 1))


def auto_rho(x: Array, y: Array) -> Array:
    """
        auto-correlation coeff.
    Input:
        x: Array 1. (N,)
        y: Array 2. (N,)
    Output:
        Correlated coefficients of x,y.(N,)
    """
    N = len(x)
    Ex = jnp.mean(x)
    Ey = jnp.mean(y)
    Vx = jnp.var(x)
    Vy = jnp.var(y)
    return (corr_circ(x, y) / N - Ex * Ey) / jnp.sqrt(Vx * Vy)


def exp_integral(
    z: float, alpha: float = 4.605170185988092e-05, span_length: float = 80e3
) -> jax.Array:
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
        k * (1 - jnp.exp(-alpha * span_length)) / alpha
        + (1 - jnp.exp(-alpha * z0)) / alpha
    )


def leff(
    z: float, dz: float, alpha: float = 4.605170185988092e-05, span_length: float = 80e3
) -> jax.Array:
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


def get_omega(Fs: float, Nfft: int) -> Array:
    """
    get signal fft angular frequency.
    Input:
        Fs: sampling frequency. [Hz]
        Nfft: number of sampling points.
    Output:
        omega:jnp.Array [Nfft,]
    """
    return 2 * np.pi * Fs * fftfreq(Nfft)


def dispersion_kernel(
    dz: float,
    dtaps: int,
    Fs: int,
    beta2: float = -2.1044895291667417e-26,
    beta1: float = 0,
    domain="time",
) -> Array:
    """
    Dispersion kernel in time domain or frequency domain.

    Input:
        dz: Dispersion distance.              [m]
        dtaps: length of kernel.
        Fs: Sampling rate of signal.          [Hz]
        beta2: 2 order  dispersion coeff.     [s^2/m]
        beta1: 1 order dispersion coeff.      [s/m]
        domain: 'time' or 'freq'
    Output:
        h:jnp.array. (dtaps,)
        h is symmetric: jnp.flip(h) = h.
    """
    omega = get_omega(Fs, dtaps)
    kernel = jnp.exp(-1j * beta1 * omega * dz - 1j * (beta2 / 2) * (omega**2) * dz)

    if domain == "time":
        return fftshift(ifft(kernel))
    elif domain == "freq":
        return kernel
    else:
        raise (ValueError)


def dispersion_time_kernel(
    dz: float,
    dtaps: int,
    Fs: int,
    beta2: float = -2.1044895291667417e-26,
    beta1=0,
    domain="time",
    steps=8,
):
    Dtaps = (dtaps - 1) * steps + 1
    k = (Dtaps - dtaps) // 2
    h = dispersion_kernel(dz, Dtaps, Fs, beta2, beta1, domain)[k : Dtaps - k]

    return h


def pretrain_kernel(
    dz: float,
    dtaps: int,
    Fs: int,
    beta2: float = -2.1044895291667417e-26,
    beta1: float = 0.0,
    domain: str = "time",
    steps: int = 8,
    pre_train_steps: int = 200,
) -> jax.Array:
    Dtaps = (dtaps - 1) * steps + 1
    k = (Dtaps - dtaps) // 2
    h = dispersion_kernel(dz, Dtaps, Fs, beta2, beta1, domain)[k : Dtaps - k]
    target = dispersion_kernel(dz * steps, Dtaps, Fs, beta2, beta1, domain)
    tx = optax.rmsprop(learning_rate=1e-4)
    opt_state = tx.init(h)

    def Conv(h):
        x = h
        for i in range(1, steps):
            x = convolve(x, h)
        return x

    def Loss(h):
        return jnp.sum(jnp.abs(Conv(h) - target) ** 2)

    @jax.jit
    def update(params, opt_state):
        loss_value, grads = jax.value_and_grad(Loss)(params)
        grads = jax.tree_map(lambda x: jnp.conj(x), grads)
        uptdates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, uptdates)
        return params, opt_state, loss_value

    if steps == 1:
        return h

    for i in range(pre_train_steps):
        h, opt_state, loss_value = update(h, opt_state)

    return h


def isfloat(x):
    return issubclass(x.dtype.type, np.floating)


def iscomplex(x):
    return issubclass(x.dtype.type, np.complexfloating)


def scan(
    f, init, xs, length=None, reverse=False, unroll=1, jit_device=None, jit_backend=None
):
    """
    "BUG: ``lax.scan`` is known to cause memory leaks when not called within a jitted function"
    "https://github.com/google/jax/issues/3158#issuecomment-631851006"
    "https://github.com/google/jax/pull/5029/commits/977c9c40efa378d1321a7dd8c712af528939ed5f"
    "https://github.com/google/jax/pull/5029"
    "NOTE": ``scan`` runs much slower on GPU than CPU if loop iterations are small (GPU IO bottleneck?)
    "https://github.com/google/jax/issues/2491"
    "https://github.com/google/jax/pull/3076"
    """

    @partial(jit, static_argnums=(0, 3, 4, 5), device=jit_device, backend=jit_backend)
    def _scan(f, init, xs, length, reverse, unroll):
        return lax.scan(f, init, xs, length=length, reverse=reverse, unroll=unroll)

    return _scan(f, init, xs, length, reverse, unroll)


def conv1d_lax(signal, kernel, mode="SAME"):
    return _conv1d_lax(signal, kernel, mode)


@partial(jit, static_argnums=(2,))
def _conv1d_lax(signal, kernel, mode):
    """
    CPU impl. is insanely slow for large kernels, jaxlib-cuda (i.e. cudnn's GPU impl.)
    is highly recommended
    see https://github.com/google/jax/issues/5227#issuecomment-748386278
    """
    x = device_put(signal)
    h = device_put(kernel)

    if x.shape[0] < h.shape[0]:
        x, h = h, x

    mode = mode.upper()

    if mode == "FULL":  # lax.conv_general_dilated has no such mode by default
        pads = h.shape[0] - 1
        mode = [(pads, pads)]
    elif (
        mode == "SAME"
    ):  # lax.conv_general_dilated use Matlab convention on even-tap kernel in its own SAME mode
        lpads = h.shape[0] - 1 - (h.shape[0] - 1) // 2
        hpads = h.shape[0] - 1 - h.shape[0] // 2
        mode = [(lpads, hpads)]
    else:  # VALID mode is fine
        pass

    x = x[jnp.newaxis, :, jnp.newaxis]
    h = h[::-1, jnp.newaxis, jnp.newaxis]
    dn = lax.conv_dimension_numbers(x.shape, h.shape, ("NWC", "WIO", "NWC"))

    # lax.conv_general_dilated runs much slower than numpy.convolve on CPU_device
    x = lax.conv_general_dilated(
        x,  # lhs = image tensor
        h,  # rhs = conv kernel tensor
        (1,),  # window strides
        mode,  # padding mode
        (1,),  # lhs/image dilation
        (1,),  # rhs/kernel dilation
        dn,
    )  # dimension_numbers = lhs, rhs, out dimension permu

    return x[0, :, 0]


# TODO apply lru_cache?
def _largest_prime_factor(n):
    """brute-force finding of greatest prime factor of integer number."""
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n


def _fft_size_factor(x, gpf, cond=lambda _: True):
    """calculates the integer number exceeding parameter x and containing
    only the prime factors not exceeding gpf, and statisfy extra condition (optional)
    """
    if x <= 0:
        raise ValueError("The input value for factor is not positive.")
    x = int(x) + 1

    if gpf > 1:
        while _largest_prime_factor(x) > gpf or not cond(x):
            x += 1

    return x


def conv1d_oa_fftsize(
    signal_length, kernel_length, oa_factor=8, max_fft_prime_factor=5
):
    target_fft_size = kernel_length * oa_factor
    if target_fft_size < signal_length:
        fft_size = _fft_size_factor(target_fft_size, max_fft_prime_factor)
    else:
        fft_size = _fft_size_factor(
            max(signal_length, kernel_length), max_fft_prime_factor
        )

    return fft_size


def _conv1d_fft_oa_same(signal, kernel, fft_size):
    signal = device_put(signal)
    kernel = device_put(kernel)

    kernel_length = kernel.shape[-1]  # kernel/filter length

    signal = _conv1d_fft_oa_full(signal, kernel, fft_size)

    signal = signal[(kernel_length - 1) // 2 : -(kernel_length // 2)]

    return signal


def _conv1d_fft_oa_valid(signal, kernel, fft_size):
    signal = device_put(signal)
    kernel = device_put(kernel)

    kernel_length = kernel.shape[-1]  # kernel/filter length

    signal = _conv1d_fft_oa_full(signal, kernel, fft_size)

    signal = signal[kernel_length - 1 : signal.shape[-1] - kernel_length + 1]

    return signal


def _conv1d_fft_oa_full(signal, kernel, fft_size):
    """fast 1d convolute underpinned by FFT and overlap-and-add operations"""
    if isfloat(signal) and isfloat(kernel):
        fft = jnp.fft.rfft
        ifft = jnp.fft.irfft
    else:
        fft = jnp.fft.fft
        ifft = jnp.fft.ifft

    signal = device_put(signal)
    kernel = device_put(kernel)

    signal_length = signal.shape[-1]
    kernel_length = kernel.shape[-1]

    output_length = signal_length + kernel_length - 1
    frame_length = fft_size - kernel_length + 1

    frames = -(-signal_length // frame_length)

    signal = jnp.pad(signal, [0, frames * frame_length - signal_length])
    signal = jnp.reshape(signal, [-1, frame_length])

    signal = ifft(fft(signal, fft_size) * fft(kernel, fft_size), fft_size)
    signal = overlap_and_add(signal, frame_length)

    signal = signal[:output_length]

    return signal


@partial(jit, static_argnums=(2, 3))
def _conv1d_fft_oa(signal, kernel, fft_size, mode):
    if mode == "same":
        signal = _conv1d_fft_oa_same(signal, kernel, fft_size)
    elif mode == "full":
        signal = _conv1d_fft_oa_full(signal, kernel, fft_size)
    elif mode == "valid":
        signal = _conv1d_fft_oa_valid(signal, kernel, fft_size)
    else:
        raise ValueError("invalid mode %s" % mode)
    return signal.real if isfloat(signal) and isfloat(kernel) else signal


def conv1d_fft_oa(signal, kernel, fft_size=None, oa_factor=10, mode="SAME"):
    mode = mode.lower()
    if fft_size is None:
        signal_length = signal.shape[-1]
        kernel_length = kernel.shape[-1]
        fft_size = conv1d_oa_fftsize(signal_length, kernel_length, oa_factor=oa_factor)

    return _conv1d_fft_oa(signal, kernel, fft_size, mode)


@partial(jit, static_argnums=(1,))
def overlap_and_add(array, frame_step):
    array_shape = array.shape
    frame_length = array_shape[1]
    frames = array_shape[0]

    # Compute output length.
    output_length = frame_length + frame_step * (frames - 1)

    # If frame_length is equal to frame_step, there's no overlap so just
    # reshape the tensor.
    if frame_step == frame_length:
        return jnp.reshape(array, (output_length,))

    # Compute the number of segments, per frame.
    segments = -(-frame_length // frame_step)
    paddings = [[0, segments], [0, segments * frame_step - frame_length]]
    array = jnp.pad(array, paddings)

    # Reshape
    array = jnp.reshape(array, [frames + segments, segments, frame_step])

    array = jnp.transpose(array, [1, 0, 2])

    shape = [(frames + segments) * segments, frame_step]
    array = jnp.reshape(array, shape)

    array = array[..., : (frames + segments - 1) * segments, :]

    shape = [segments, (frames + segments - 1), frame_step]
    array = jnp.reshape(array, shape)

    # Now, reduce over the columns, to achieve the desired sum.
    array = jnp.sum(array, axis=0)

    # Flatten the array.
    shape = [(frames + segments - 1) * frame_step]
    array = jnp.reshape(array, shape)

    # Truncate to final length.
    array = array[:output_length]

    return array


def fftconvolve(x, h, mode="full"):
    x = jnp.atleast_1d(x) * 1.0
    h = jnp.atleast_1d(h) * 1.0

    mode = mode.lower()

    if x.shape[0] < h.shape[0]:
        tmp = x
        x = h
        h = tmp

    T = h.shape[0]
    N = x.shape[0] + T - 1

    y = _fftconvolve(x, h)

    if mode == "full":
        return y
    elif mode == "same":
        return y[(T - 1) // 2 : N - T // 2]
    elif mode == "valid":
        return y[T - 1 : N - T + 1]
    else:
        raise ValueError("invalid mode " "%s" "" % mode)


@jit
def _fftconvolve(x, h):
    if isfloat(x) and isfloat(h):
        fft = jnp.fft.rfft
        ifft = jnp.fft.irfft
    else:
        fft = jnp.fft.fft
        ifft = jnp.fft.ifft

    out_length = x.shape[0] + h.shape[0] - 1
    n = _fft_size_factor(out_length, 5)
    y = ifft(fft(x, n) * fft(h, n), n)
    y = y[:out_length]
    return y


def fftconvolve2(x, h, mode="full"):
    # TODO: add float support
    x = jnp.atleast_2d(x)
    h = jnp.atleast_2d(h)

    mode = mode.lower()

    T0 = h.shape[0]
    T1 = h.shape[1]
    N0 = x.shape[0] + T0 - 1
    N1 = x.shape[1] + T1 - 1

    y = _fftconvolve2(x, h)

    if mode == "full":
        return y
    elif mode == "same":
        return y[(T0 - 1) // 2 : N0 - T0 // 2, (T1 - 1) // 2 : N1 - T1 // 2]
    elif mode == "valid":
        return y[T0 - 1 : N0 - T0 + 1, T1 - 1 : N1 - T1 + 1]
    else:
        raise ValueError("invalid mode " "%s" "" % mode)


@jit
def _fftconvolve2(x, h):
    fft = jnp.fft.fft2
    ifft = jnp.fft.ifft2
    out_shape = [x.shape[0] + h.shape[0] - 1, x.shape[1] + h.shape[1] - 1]
    fft_shape = [_fft_size_factor(out_shape[0], 5), _fft_size_factor(out_shape[1], 5)]
    hpad = jnp.pad(h, [[0, fft_shape[0] - h.shape[0]], [0, fft_shape[1] - h.shape[1]]])
    xpad = jnp.pad(x, [[0, fft_shape[0] - x.shape[0]], [0, fft_shape[1] - x.shape[1]]])
    y = ifft(fft(xpad) * fft(hpad))
    y = y[: out_shape[0], : out_shape[1]]
    return y.real if isfloat(x) and isfloat(h) else y


def _convolve(a, v, mode, method):
    a = jnp.atleast_1d(a) + 0.0
    v = jnp.atleast_1d(v) + 0.0
    method = method.lower()

    if a.shape[0] < v.shape[0]:
        a, v = v, a

    if method == "auto":
        method = 0 if v.shape[0] < 3 else 1
    elif method == "direct":
        method = 0
    elif method == "fft":
        method = 1
    else:
        raise ValueError("invalid method")

    if method == 0:
        # jnp.convolve does not support complex value yet, but is slightly faster than conv1d_lax on float inputs
        conv = jnp.convolve if isfloat(a) and isfloat(v) else conv1d_lax
    else:
        # simple switch tested not bad on my cpu/gpu. TODO fine tune by interacting with overlap-add factor
        conv = (
            conv1d_fft_oa
            if a.shape[0] >= 500 and a.shape[0] / v.shape[0] >= 50
            else fftconvolve
        )

    return conv(a, v, mode=mode)


def convolve(a, v, mode="full", method="auto"):
    return _convolve(a, v, mode, method)
