'''
DBP sps = 1:  after CDC + ADF.
'''

import torch.nn as nn, torch, numpy as np, torch
from typing import Union, List, Tuple, Optional
from pkufiber.dsp.nonlinear_compensation.exploration.eqdbp_trainD import EqDBP_trainD, DispersionFilter

D = DispersionFilter(25, 401, 80e3, d_train=True, d_share=False, Fs=160e9)
D = D.cuda()
D.train_filter(lr=3e-4)
