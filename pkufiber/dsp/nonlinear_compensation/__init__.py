# pkufiber/dsp/nonlinear_compensation/__init__.py
from . import rmps

from .exploration.pbc_nn import EqAMPBCaddNN, EqAMPBCaddFNO
from .exploration.new_idea import EqSoNN, EqAMPBCaddConv
from .exploration.eqdbp import EqDBP
from .exploration.eqdbp_test import EqDBP_test
from .exploration.eqpbcdbp import EqPbcDBP, EqAMPbcDBP
from .exploration.eqfreqdbp import EqFreqPbcDBP, EqFreqAMPbcDBP
from .exploration.freqtimepbc import EqFreqTimePBC


# Train Dispersion filter.
from .exploration.eqdbp_trainD import EqDBP_trainD, DispersionFilter
from .exploration.eqpbcdbp_trainD import EqPbcDBP_trainD, EqAMPbcDBP_trainD
from .exploration.eqfreqdbp_trainD import EqFreqPbcDBP_trainD, EqFreqAMPbcDBP_trainD
from .exploration.eqsnsedbp import EqSNSEDBP

from .baselines import cdc, dbp
from .pbc import IndexType, EqFrePBC, EqStftPBC,EqStftAMPBC, EqFreAMPBC, EqConvAMPBC
from .pbc.stftpbc import EqStftSnsePBC
from .pbc import EqPBC, EqAMPBC, EqAMPBCstep, EqPBCstep, MultiStepAMPBC, MultiStepPBC, EqPBCNN, EqSoPBC
from .ldbp import FDBP, MetaDBP, downsamp, FreqPbcDBP, PbcDBP, AMPbcDBP, FreqAMPbcDBP, FDBP_trainD
from .nneq import EqCNNBiLSTM, EqBiLSTM, EqMLP, EqID, EqBiLSTMClass, EqBiLSTMstep
from .fno.fno import EqFno


__all__ = [
    "EqStftAMPBC",
    "EqFrePBC",
    "EqAMPBCaddConv",
    "EqSoNN",
    "EqAMPBCaddNN",
    "EqAMPBCaddFNO",
    "cdc",
    "dbp",
    "EqPBC",
    "EqAMPBC",
    "EqAMPBCstep",
    "EqPBCstep",
    "MultiStepAMPBC",
    "MultiStepPBC",
    "FDBP",
    "MetaDBP",
    "downsamp",
    "EqCNNBiLSTM",
    "EqBiLSTM",
    "EqMLP",
    "EqID",
    "EqFno",
    "IndexType",
    "FreqPbcDBP",
    "PbcDBP",
]
