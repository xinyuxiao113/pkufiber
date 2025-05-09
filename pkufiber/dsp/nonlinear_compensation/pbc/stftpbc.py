"""
    NN equalizer.
"""

import torch.nn as nn, torch, numpy as np, torch
from typing import Union, List, Tuple, Optional

from pkufiber.core import TorchSignal, TorchTime
from pkufiber.dsp.layers import ComplexLinear, ComplexConv1d
from pkufiber.dsp.nonlinear_compensation.op import triplets, trip_op, get_power
from pkufiber.dsp.nonlinear_compensation.pbc.features import TripletFeatures, IndexType
from pkufiber.dsp.nonlinear_compensation.pbc.pbc import EqPBC, EqAMPBCstep, EqPBCstep
from pkufiber.op import stft_on_dimension, istft_on_dimension, periodic_padding
from pkufiber.dsp.nonlinear_compensation.filter import SnseFilter


class EqStftPBC(nn.Module):
    '''
    PBC step.
    M: int, default=41, the length of the filter.
    rho: float, default=1, the parameter of the filter.
    index_type: str, default="reduce-1", ['full', 'reduce-1', 'reduce-2']
    pol_share: bool, default=False, whether the two polarizations share the same filter.

    '''
    def __init__(self, M: int = 41, rho: float = -1, overlaps: int = 40, strides: int = -1):
        assert overlaps % 2 == 0 and M % 2 == 1, "overlaps should be even and M should be odd."
        super(EqStftPBC, self).__init__()
        self.M = M
        self.rho = rho if rho > 0 else M/2
        self.index = self.get_index()
        self.fc = ComplexLinear(len(self.index), 1)
        self.overlaps = overlaps
        self.strides = strides
        self.n_fft = self.overlaps + strides
        self.hop_length = strides if strides > 0 else overlaps // 2  # 设置hop_length
        

    def get_index(self):
        S = []
        for n1 in range(-(self.M//2), self.M//2):
            for n2 in range(-(self.M//2), self.M//2):
                if abs(n1*n2) <= self.rho * (self.M//2):
                    S.append((n1, n2))
        return S

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, L, Nmodes]
        task_info: [batch, 4]
        --> [batch, L - overlaps, Nmodes]
        """
        batch, L, Nmodes = x.shape
        P = get_power(task_info, Nmodes, x.device)
        x0 = x

        # STFT: 对x应用STFT
        x_stft,_,_,_ = stft_on_dimension(x, self.n_fft, self.hop_length, self.n_fft, dim=1)  # [batch, Nmodes, n_fft, steps]
        E3 = []
        for (n1, n2) in self.index:
            A = torch.roll(x_stft, shifts=n1, dims=2) * torch.roll(x_stft, shifts=n1 + n2, dims=2).conj()
            E3.append((A + A.roll(1, dims=1)) * torch.roll(x_stft, shifts=n2, dims=2))

        E = torch.stack(E3, dim=-1)  # [batch, Nmodes, n_fft, steps, len(index)]
        delta = self.fc(E).squeeze(-1)   # [batch, Nmodes, n_fft, steps]

        # ISTFT: 对delta进行逆STFT
        delta_istft = istft_on_dimension(delta, x0.shape[1], self.n_fft, self.hop_length, self.n_fft, x0.shape, dim=1)  # [batch, L, Nmodes]

        # 组合原始信号和补偿后的信号
        output = (x0 + delta_istft * P[:, None, None])

        # 截取中心部分以匹配原始信号长度
        return output[:, (self.overlaps//2):L-(self.overlaps//2), :]

    def rmps(self) -> int:
        from pkufiber.dsp.nonlinear_compensation.rmps import rmps_edc, rmps_fft
        
        FFT_size = self.strides + self.overlaps
        return (4*len(self.index)*3*FFT_size + rmps_fft(FFT_size)*2)/self.strides


class EqStftAMPBC(nn.Module):
    '''
    PBC step.
    M: int, default=41, the length of the filter.
    rho: float, default=1, the parameter of the filter.
    index_type: str, default="reduce-1", ['full', 'reduce-1', 'reduce-2']
    pol_share: bool, default=False, whether the two polarizations share the same filter.

    '''
    def __init__(self, M: int = 41, rho: float = -1, overlaps: int = 40, strides: int = -1):
        assert overlaps % 2 == 0 and M % 2 == 1, "overlaps should be even and M should be odd."
        super(EqStftAMPBC, self).__init__()
        self.M = M
        self.rho = rho if rho > 0 else M/2
        self.index = self.get_index()
        # self.pbc = EqPBCstep(M=self.M, rho=self.rho)
        self.pbc = EqAMPBCstep(M=self.M, rho=self.rho, fwm_share=False, decision=False, discard=False)
        self.overlaps = overlaps
        self.strides = strides
        self.n_fft = self.overlaps + strides
        self.hop_length = strides if strides > 0 else overlaps // 2  # 设置hop_length
        

    def get_index(self):
        S = []
        for n1 in range(-(self.M//2), self.M//2):
            for n2 in range(-(self.M//2), self.M//2):
                if abs(n1*n2) <= self.rho * (self.M//2):
                    S.append((n1, n2))
        return S

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, L, Nmodes]
        task_info: [batch, 4]
        --> [batch, L - overlaps, Nmodes]
        """
        batch, L, Nmodes = x.shape
        P = get_power(task_info, Nmodes, x.device)
        x0 = x

        # STFT: 对x应用STFT
        x_stft,_,_,_ = stft_on_dimension(x, self.n_fft, self.hop_length, self.n_fft, dim=1)  # [batch, Nmodes, n_fft, steps]
        steps, n_fft = x_stft.shape[-1], x_stft.shape[-2]
        x_stft = x_stft.permute([0, 3, 2, 1])   # [batch, steps, n_fft, Nmodes]
        x_stft = x_stft.reshape(-1, x_stft.shape[-2], x_stft.shape[-1]) # [batch*steps, n_fft, Nmodes]
        x_stft = self.pbc(x_stft, task_info.repeat(steps, 1)) # [batch*steps, n_fft, Nmodes]
        x_stft = x_stft.reshape(batch, steps, n_fft, Nmodes) # [batch, steps, n_fft, Nmodes]
        x_stft = x_stft.permute([0, 3, 2, 1])   # [batch, Nmodes, n_fft, steps]

        x_istft = istft_on_dimension(x_stft, x0.shape[1], self.n_fft, self.hop_length, self.n_fft, x0.shape, dim=1)  # [batch, L, Nmodes]
        # 截取中心部分以匹配原始信号长度
        return x_istft[:, (self.overlaps//2):L-(self.overlaps//2), :]

    def rmps(self) -> int:
        from pkufiber.dsp.nonlinear_compensation.rmps import rmps_edc, rmps_fft
        
        FFT_size = self.strides + self.overlaps
        return (self.pbc.rmps()*FFT_size + rmps_fft(FFT_size)*2)/self.strides


class EqStftSnsePBC(nn.Module):
    '''
    PBC step.
    M: int, default=41, the length of the filter.
    rho: float, default=1, the parameter of the filter.
    index_type: str, default="reduce-1", ['full', 'reduce-1', 'reduce-2']
    pol_share: bool, default=False, whether the two polarizations share the same filter.

    '''
    def __init__(self, M: int = 41, n_fwm:int=0, overlaps: int = 40, strides: int = -1):
        assert overlaps % 2 == 0 and M % 2 == 1, "overlaps should be even and M should be odd."
        super(EqStftSnsePBC, self).__init__()
        self.M = M
        self.n_fwm = n_fwm
        self.pbc = SnseFilter(Nmodes=2, step=1, ntaps=self.M, share=True, n_fwm=n_fwm)
        self.overlaps = overlaps
        self.strides = strides
        self.n_fft = self.overlaps + strides
        self.hop_length = strides if strides > 0 else overlaps // 2  # 设置hop_length
        
    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, L, Nmodes]
        task_info: [batch, 4]
        --> [batch, L - overlaps, Nmodes]
        """
        batch, L, Nmodes = x.shape
        P = get_power(task_info, Nmodes, x.device)
        x0 = x

        # STFT: 对x应用STFT
        x_stft,_,_,_ = stft_on_dimension(x, self.n_fft, self.hop_length, self.n_fft, dim=1)  # [batch, Nmodes, n_fft, steps]
        steps, n_fft = x_stft.shape[-1], x_stft.shape[-2]
        x_stft = x_stft.permute([0, 3, 2, 1])   # [batch, steps, n_fft, Nmodes]
        x_stft = x_stft.reshape(-1, x_stft.shape[-2], x_stft.shape[-1]) # [batch*steps, n_fft, Nmodes]
        x_stft = self.pbc(periodic_padding(x_stft, self.M//2, dim=1), task_info.repeat(steps, 1), 0) # [batch*steps, n_fft, Nmodes]
        x_stft = x_stft.reshape(batch, steps, n_fft, Nmodes) # [batch, steps, n_fft, Nmodes]
        x_stft = x_stft.permute([0, 3, 2, 1])   # [batch, Nmodes, n_fft, steps]

        x_istft = istft_on_dimension(x_stft, x0.shape[1], self.n_fft, self.hop_length, self.n_fft, x0.shape, dim=1)  # [batch, L, Nmodes]
        # 截取中心部分以匹配原始信号长度
        return x_istft[:, (self.overlaps//2):L-(self.overlaps//2), :]

    def rmps(self) -> int:
        from pkufiber.dsp.nonlinear_compensation.rmps import rmps_edc, rmps_fft
        
        FFT_size = self.strides + self.overlaps
        return (self.pbc.rmps()*FFT_size + rmps_fft(FFT_size)*2)/self.strides




# class EqStftAMPBC(nn.Module):
#     '''
#     PBC step.
#     M: int, default=41, the length of the filter.
#     rho: float, default=1, the parameter of the filter.
#     index_type: str, default="reduce-1", ['full', 'reduce-1', 'reduce-2']
#     pol_share: bool, default=False, whether the two polarizations share the same filter.

#     '''
#     def __init__(self, M: int = 41, rho: float = -1, overlaps: int = 40, strides: int = -1):
#         assert overlaps % 2 == 0 and M % 2 == 1, "overlaps should be even and M should be odd."
#         super(EqStftAMPBC, self).__init__()
#         self.M = M
#         self.rho = rho if rho > 0 else M/2
#         self.index = self.get_index()
#         self.fc = ComplexLinear(len(self.index), 1)
#         self.overlaps = overlaps
#         self.strides = strides
#         self.n_fft = self.overlaps + strides
#         self.hop_length = strides if strides > 0 else overlaps // 2  # 设置hop_length
        

#     def get_index(self):
#         S = []
#         for n1 in range(-(self.M//2), self.M//2):
#             for n2 in range(-(self.M//2), self.M//2):
#                 if abs(n1*n2) <= self.rho * (self.M//2):
#                     S.append((n1, n2))
#         return S

#     def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
#         """
#         x: [batch, L, Nmodes]
#         task_info: [batch, 4]
#         --> [batch, L - overlaps, Nmodes]
#         """
#         batch, L, Nmodes = x.shape
#         P = get_power(task_info, Nmodes, x.device)
#         x0 = x

#         # STFT: 对x应用STFT
#         x_stft,_,_,_ = stft_on_dimension(x, self.n_fft, self.hop_length, self.n_fft, dim=1)  # [batch, Nmodes, n_fft, steps]
#         E3 = []
#         E4 = []
#         for (n1, n2) in self.index:
#             A = torch.roll(x_stft, shifts=n1, dims=2) * torch.roll(x_stft, shifts=n1 + n2, dims=2).conj()
#             E3.append((A + A.roll(1, dims=1)) * torch.roll(x_stft, shifts=n2, dims=2))

#         E = torch.stack(E3, dim=-1)  # [batch, Nmodes, n_fft, steps, len(index)]
#         delta = self.fc(E).squeeze(-1)   # [batch, Nmodes, n_fft, steps]

#         # ISTFT: 对delta进行逆STFT
#         delta_istft = istft_on_dimension(delta, x0.shape[1], self.n_fft, self.hop_length, self.n_fft, x0.shape, dim=1)  # [batch, L, Nmodes]

#         # 组合原始信号和补偿后的信号
#         output = (x0 + delta_istft * P[:, None, None])

#         # 截取中心部分以匹配原始信号长度
#         return output[:, (self.overlaps//2):L-(self.overlaps//2), :]

#     def rmps(self) -> int:
#         from pkufiber.dsp.nonlinear_compensation.rmps import rmps_edc, rmps_fft
        
#         FFT_size = self.strides + self.overlaps
#         return (4*len(self.index)*3*FFT_size + rmps_fft(FFT_size)*2)/self.strides



if __name__ == '__main__':
    # net = EqStftAMPBC(strides=41)
    net = EqStftSnsePBC(strides=41)
    x = torch.rand(5,10000,2)+1j 
    z = torch.rand(5,4)

    y = net(x,z)
    print(y.shape)