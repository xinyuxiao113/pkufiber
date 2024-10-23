import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

with HiddenPrints():
    import jax 
    jax.devices()

import pkufiber.data as data
import pkufiber.dsp as dsp
import pkufiber.simulation as simulation
import pkufiber.core as core
import pkufiber.utils as utils
import pkufiber.op as op

from .core import TorchInput, TorchSignal, TorchTime
from .utils import show_symb
from .op import get_beta2, get_omega

from .dsp.nonlinear_compensation.op import get_power, estimate_dtaps
from .dsp.nonlinear_compensation.loss import mse, adaptive_ber

from .simulation.receiver import ser, ber, qfactor, qfactor_path, qfactor_all
from .simulation.transmitter import QAM



__all__ = [
    "data",
    "dsp",
    "simulation",
    "core",
    "utils",
    "op",
    "ser",
    "ber",
    "qfactor",
    "qfactor_all",
    "qfactor_path",
    "show_symb",
    "get_beta2",
    "get_omega",
    "get_power",
    "QAM",
    "TorchInput",
    "TorchSignal",
    "TorchTime",
    "mse",
    "adaptive_ber",
]
