import numpy as np, jax.numpy as jnp, jax
from commpy.modulation import QAMModem
from commpy.filters import rrcosfilter, rcosfilter
from functools import partial
from .jax_op import circFilter, frame, circFilter_


class QAM(QAMModem):
    """
    QAM Modulation format.
    """

    def __init__(self, M, reorder_as_gray=True):
        super(QAM, self).__init__(M)
        self.constellation_jnp = jnp.array(self.constellation)
        self.Es = jnp.array(self.Es)

    def bit2symbol(self, bits):
        N = len(bits)
        pow = 2 ** jnp.arange(N - 1, -1, -1)
        idx = jnp.sum(pow * bits)
        return self.constellation_jnp[idx]

    def modulate(self, bits):
        bits_batch = frame(bits, self.num_bits_symbol, self.num_bits_symbol)
        symbol_batch = jax.vmap(self.bit2symbol)(bits_batch)
        return symbol_batch

    def const(self):
        return self.constellation / np.sqrt(self.Es)
