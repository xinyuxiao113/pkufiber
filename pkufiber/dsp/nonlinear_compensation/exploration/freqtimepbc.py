import torch
import torch.nn as nn
from pkufiber.dsp.nonlinear_compensation.nneq import EqCNNBiLSTM 
from pkufiber.dsp.nonlinear_compensation.fno.fno import EqFno
from pkufiber.dsp.nonlinear_compensation.pbc import EqAMPBCstep, EqStftPBC
 


class EqFreqTimePBC(nn.Module):
    
    def __init__(self, ampbc_info, stftpbc_info):
        super(EqFreqTimePBC, self).__init__()
        assert ampbc_info['M'] == stftpbc_info['overlaps'] + 1
        self.ampbc = EqAMPBCstep(**ampbc_info)
        self.stftpbc = EqStftPBC(**stftpbc_info)
        self.overlaps = ampbc_info['M'] - 1

    def forward(self, x, task_info):

        return (self.ampbc(x, task_info) + self.stftpbc(x, task_info))/2
    
    def rmps(self):
        return self.ampbc.rmps() + self.stftpbc.rmps()

if __name__ == '__main__':
    net = EqStftPBC()
    ampbc_info = {'M': 41, 'rho': 1}
    stftpbc_info = {'M': 41, 'rho': 1, 'overlaps': 40, 'strides': 161}
    net = EqFreqTimePBC(ampbc_info, stftpbc_info)

    x = torch.rand(5,1000,2) + 1j 
    task = torch.rand(5, 4)

    print(net(x, task).shape)