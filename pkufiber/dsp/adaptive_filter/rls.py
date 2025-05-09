# DD-RLS: MIMO
from pkufiber.dsp.adaptive_filter.jax_op import frame
from pkufiber.dsp.adaptive_filter.jax_adf import iterate
from pkufiber.dsp.adaptive_filter.jax_core import JaxSignal, JaxTime, conv1d_t
from pkufiber.dsp.adaptive_filter.jax_adf import adaptive_filter, mimoinitializer, partial, AdaptiveFilter, Union, Array, Schedule, QAM, make_schedule, decision
import jax.numpy as jnp
import jax

def apply_filter(w, u):
    '''
    w: [dim, dim, taps]   dim_out x dim_in x taps
    u: [dim, taps]        dim_in x taps
    output: [dim]         dim_out
    '''
    dims = u.shape[0]
    taps = u.shape[1]
    u = u.reshape(-1)                 # [taps*dims]       u = [u_x, u_y]
    w = w.reshape(dims, taps*dims)    # [dims, taps*dims] w = [[w_xx, w_xy], [w_yx, w_yy]]
    v = w.conj() @ u                  # [dims]            v = [v_x, v_y]

    return v


@partial(adaptive_filter, trainable=True)
def ddrls(
    lambda_: float = 0.99,
    delta: float = 1e-3,
    train: Union[bool, Schedule] = False,
    eps: float = 1e-9,      
    const: Array = QAM(16).const(),
) -> AdaptiveFilter:
    """Decision-Directed Recursive Least Squares adaptive equalizer
    
    形状说明：
    - u: (dims, taps)                输入信号缓存
    - d: (dims,)                     期望输出
    - w: (dims, dims, taps)          滤波器权重
    - P: (dims, taps*dims, taps*dims)    逆相关矩阵
    """
    const = jnp.asarray(const)
    train = make_schedule(train)

    def init(taps=32, dims=2, dtype=jnp.complex64, mimoinit="zeros"):
        # w0: (dims, dims, taps)
        w0 = mimoinitializer(taps, dims, dtype, mimoinit)
        w0 = w0.reshape(dims, dims, taps)
        
        # P0: (dims, dims*taps, dims*taps)
        P0 = jnp.tile(jnp.eye(dims * taps, dtype=dtype) / delta, (dims, 1, 1))
        return (w0, P0)

    def update(i, state, inp):
        w, P = state  # w: (dims, dims, taps), P: (dims, dims*taps, dims*taps)
        u, x = inp    # u: (dims, taps), x: (dims,)
        taps = u.shape[-1]
        dims = u.shape[0]
        
        # 前向滤波
        v = apply_filter(w, u) # (dims,)
        
        # 判决
        d = jnp.where(train(i), x, decision(const, v))  # (dims,)
        # d = x        

        # 计算误差
        e = d - v  # (dims,)
        
        # 构造扩展的输入向量
        u_ext = u.reshape(-1)  # (taps*dims,)
        
        # RLS更新
        # 对每个输出维度分别更新

        #  P: (dims, taps*dims, taps*dims), u_ext: (taps*dims,)

        # 计算增益 k
        k = jnp.zeros((dims, taps*dims), dtype=jnp.complex64)
        for j in range(dims):
            denominator = lambda_ + u_ext.conj().T @ P[j] @ u_ext + eps
            k = k.at[j].set(P[j] @ u_ext / denominator)
        
        # 更新 P
        for j in range(dims):
            P_j_updated = (P[j] - jnp.outer(k[j], u_ext.conj() @ P[j])) / lambda_
            P = P.at[j].set(P_j_updated)
        
        # 更新滤波器系数
        k_reshaped = k.reshape(dims, dims, taps)
        w = w + k_reshaped * e.conj().reshape(dims, 1, 1)
        
        # 计算代价函数
        l = jnp.sum(jnp.abs(e) ** 2)
        
        out = ((w, P), (l, d))
        state = (w, P)
        
        return state, out

    def apply(ws, yf):
        '''
        ws: [L, dim, dim, taps]
        yf: [L, dims, taps]
        output: [L, dim]
        '''
        return jax.vmap(apply_filter)(ws, yf)
    
    return AdaptiveFilter(init, update, apply)



def dd_rls(Rx, Tx, taps=8, sps=2, lead_symbols=2000, lambda_=0.999, delta=1e-3, eps=1e-6):
    '''
    Rx: [Nsymb * sps, dims]
    Tx: [Nsymb, dims]
    taps: int, number of taps
    sps: int, samples per symbol
    lead_symbols: int, number of symbols used to train the filter
    lambda_: float, forgetting factor
    delta: float, initial inverse correlation matrix
    eps: float, epsilon
    '''
    signal = JaxSignal(val=jnp.array(Rx.numpy()), t=JaxTime(0, 0, 2), Fs=0) 
    truth = JaxSignal(val=jnp.array(Tx.numpy()), t=JaxTime(0, 0, 1), Fs=0)
    dims = Rx.shape[1]
    adf = ddrls(lambda_=lambda_, delta=delta, eps=eps, train=lambda n: n < lead_symbols)
    state = adf.init_fn(taps=taps, dims=dims)
    u_input = frame(signal.val, taps, sps)  # [L, taps, dims]
    u_input = u_input.transpose(0,2,1)     # [L, dims, taps]
    t = conv1d_t(signal.t, taps, None, sps, "valid")
    truth = truth.val[t.start: truth.val.shape[-2] + t.stop]
    af_step, (af_stats, (af_weights, info)) = iterate(adf.update_fn, 0, state, u_input, truth)
    y = adf.eval_fn(af_weights[0], u_input)
    return y, truth, af_weights, info
