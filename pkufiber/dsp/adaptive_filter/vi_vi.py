# V&V algorithm for 16QAM
import jax.numpy as jnp
import jax
from pkufiber.dsp.adaptive_filter.jax_adf import adaptive_filter, mimoinitializer, partial, AdaptiveFilter, Union, Array, Schedule, QAM, make_schedule, decision
from pkufiber.dsp.adaptive_filter.jax_op import frame

def vi_vi_qpsk(symbs: jax.Array, taps=15) -> jax.Array:
    '''
    QPSK的V&V相位恢复算法
    Args:
        symbs: [L, dims] 输入符号
        taps: 滑动窗口大小
    Returns:
        phase_corrected_symbs: [L, dims] 相位校正后的符号
    '''
    # 4次方运算消除QPSK调制
    s2 = symbs**2  
    s4 = s2**2     # [L, dims]
    
    # 创建归一化的滑动窗口kernel
    kernel = jnp.ones(taps) / taps
    
    # 使用卷积计算滑动平均
    # mode='same' 确保输出长度与输入相同
    s4_avg = jnp.convolve(s4, kernel, mode='same')
    
    # 计算平均相位，需要除以4
    phi = jnp.angle(s4_avg) / 4
    
    # 相位校正
    corrected_symbs = symbs * jnp.exp(-1j * phi)
    
    return corrected_symbs * jnp.exp(-1j * jnp.pi/4)

    

def vi_vi_qam16_block(symbs: jax.Array) -> jax.Array:
    '''
    QAM16的V&V相位恢复算法: 拆分QPSK--Laser Linewidth Tolerance for 16-QAM Coherent Optical Systems Using QPSK Partitioning
    Args:
        symbs: [taps] 输入符号
        taps: 滑动窗口大小
    '''
    # 分类
    Es = 1
    theta_rot = jnp.pi/4 - jnp.arctan(1/3)
    modules = jnp.abs(symbs)**2
    S1 = (modules < 6/Es) + (modules > 14/Es)
    Sxo = (modules >= 6/Es) * (modules <= 14/Es)

    # 计算和选择
    s = symbs / jnp.abs(symbs)             # normalization
    S1_sum = jnp.sum(s**4 * S1)
    Sxo_add = (s * jnp.exp(1j * theta_rot))**4 * Sxo
    Sxo_sub = (s * jnp.exp(-1j * theta_rot))**4 * Sxo
    # 在 Sxo_add, Sxo_sub中选择离S1_sum最近的
    Sxo_select = jnp.where(jnp.abs(Sxo_add - S1_sum) <= jnp.abs(Sxo_sub - S1_sum), Sxo_add, Sxo_sub) * Sxo

    phi = jnp.angle(jnp.sum(Sxo_select + S1_sum))/4

    return phi


def vi_vi_qam16(symbs, taps=15):
    '''
    QAM16的V&V相位恢复算法
    Args:
        symbs: [L] 输入符号
        taps: 滑动窗口大小
    '''
    x = frame(symbs, taps, 1)        # [L - taps + 1,  taps]
    x = x.reshape(x.shape[0], -1)         # [L - taps + 1, taps*dims]
    phi = jax.vmap(vi_vi_qam16_block)(x)

    theta = jnp.unwrap(phi, axis=0, period=jnp.pi / 2)


    return symbs[taps//2: -taps//2+1] * jnp.exp(-1j * theta) * jnp.exp(-1j * jnp.pi/4)


def vi_vi_qam16_dual(symbs, taps=15):
    '''
    QAM16的V&V相位恢复算法: 双通道

    symbs: [L, dims]
    taps: 滑动窗口大小
    '''
    return jax.vmap(vi_vi_qam16, in_axes=(-1, None), out_axes=-1)(symbs, taps)