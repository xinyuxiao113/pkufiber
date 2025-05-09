'''
DBP sps = 1:  after CDC + ADF.
SNSE-DBP均衡器实现，用于补偿光纤传输中的色散和非线性效应
'''

import torch.nn as nn, torch, numpy as np, torch
from typing import Union, List, Tuple, Optional

from pkufiber.core import TorchSignal, TorchTime
from pkufiber.dsp.layers import ComplexLinear, ComplexConv1d
from pkufiber.dsp.nonlinear_compensation.op import triplets, trip_op, get_power
from pkufiber.dsp.op import dispersion_kernel, dconv, nconv
from pkufiber.op import get_beta2, get_beta1
from pkufiber.dsp.nonlinear_compensation.pbc.features import TripletFeatures, IndexType
from pkufiber.dsp.nonlinear_compensation.pbc import EqPBCstep, EqAMPBCstep
from pkufiber.dsp.nonlinear_compensation.filter import DispersionFilter,NonlinearFilter, SNSEFilter, SnseFilter


'''
每个模型都需要 M, overlaps参数
M: 滤波器长度
overlaps: 重叠区域长度
'''

class EqSNSEDBP(nn.Module):
    """
    SNSE-DBP均衡器类
    参数:
        Nmodes: 模式数
        step: DBP步数
        dtaps: 色散滤波器抽头数
        ntaps: 非线性滤波器抽头数
        d_share: 是否共享色散滤波器
        n_share: 是否共享非线性滤波器
        Fs: 采样率
        D: 色散系数
        Fc: 载波频率
        Fi: 输入信号频率
        L: 传输距离
        gamma: 非线性系数
        test_version: 是否使用测试版本
        no_icixpm: 是否忽略ICI和XPM效应
        n_fwm: FWM阶数
    """

    def __init__(self, Nmodes, step, dtaps=2001, ntaps=401, d_share=False, n_share=False, Fs=80e9, D=16.5, Fc: float = 299792458 / 1550e-9, Fi: float = 299792458 / 1550e-9, L=2000e3, gamma=0.0016567, test_version=True, no_icixpm=False, n_fwm=0):
        super(EqSNSEDBP, self).__init__()
        # 初始化基本参数
        self.Nmodes = Nmodes  # 模式数
        self.step = step      # DBP步数
        self.Fs = Fs          # 采样率
        self.D = D           # 色散系数
        self.Fc = Fc         # 载波频率
        self.Fi = Fi         # 输入信号频率
        self.L = L           # 传输距离
        self.gamma = gamma   # 非线性系数
        self.dtaps = dtaps   # 色散滤波器抽头数
        self.ntaps = ntaps   # 非线性滤波器抽头数
        self.n_share = n_share  # 是否共享非线性滤波器
        self.overlaps = ((dtaps - 1)*2 + ntaps - 1) * step  # 计算重叠区域长度
        self.beta2 = get_beta2(self.D, self.Fc) / 1e3      # 二阶色散系数 [s^2/m]
        self.beta1 = get_beta1(self.D, self.Fc, self.Fi) / 1e3  # 一阶色散系数 [s/m]

        # 初始化滤波器
        n_num = 1 if n_share else self.step
        self.linear = DispersionFilter(step, dtaps, L/step, d_train=True, d_share=d_share, Fs=Fs, D=D, Fc=Fc, Fi=Fi)

        # 选择非线性滤波器版本
        if test_version:
            self.nonlinear = SnseFilter(Nmodes, step, ntaps, share=n_share, no_icixpm=no_icixpm, n_fwm=n_fwm)
        else:
            raise ValueError("We don't recommend to use this model in the production environment.")
            self.nonlinear = SNSEFilter(Nmodes, step, ntaps, share=n_share)

    def disp_freq(self, x, dz, dtaps):
        """
        频域色散补偿
        参数:
            x: 输入信号
            dz: 传输距离
            dtaps: 滤波器抽头数
        返回:
            补偿后的信号
        """
        kernel = dispersion_kernel(dz, x.shape[1], self.Fs, self.beta2, self.beta1, domain="freq").to(torch.complex64) 
        kernel = kernel.to(x.device)
        x = torch.fft.ifft(torch.fft.fft(x, dim=1) * kernel[...,None], dim=1)
        return x[:, (dtaps//2):-(dtaps//2),:]
    
    def disp_time(self, x, dz, dtaps):
        """
        时域色散补偿
        参数:
            x: 输入信号
            dz: 传输距离
            dtaps: 滤波器抽头数
        返回:
            补偿后的信号
        """
        kernel = dispersion_kernel(dz, dtaps, self.Fs, self.beta2, self.beta1, domain="time").to(torch.complex64) 
        kernel = kernel.to(x.device)
        return dconv(x, kernel.expand(x.shape[0], dtaps), stride=1)

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        输入:
            x: [batch, M, Nmodes] 输入信号
            task_info: [B, 4] 任务信息 [P,Fi,Fs,Nch]
        输出:
            [batch, Nmodes] 补偿后的信号
        """
        Nmodes = x.shape[-1]
        if Nmodes == 1 and self.pol_num == 2:
            raise ValueError(
                "Nmodes=1 and pol_num=2 is not a good choise, please set pol_share=True."
            )
        
        # 步骤1: 添加色散补偿
        x = self.disp_freq(x, self.L, (self.dtaps-1)*self.step+1)

        # 步骤2: 执行DBP补偿
        n_num = 1 if self.n_share else self.step
        for i in range(self.step):
            x = self.linear(x, i)  # 线性补偿
            x = self.nonlinear(x, task_info, i)  # 非线性补偿

        return x

    def rmps(self) -> int:
        """
        计算每个样本的实数乘法次数
        返回:
            实数乘法次数
        """
        from pkufiber.dsp.nonlinear_compensation.rmps import rmps_edc
        sps = 1
        return rmps_edc((self.dtaps-1)*self.step+1) + self.step * sps * (rmps_edc(self.dtaps) + self.nonlinear.rmps())

if __name__ == "__main__":
    # 测试代码
    x = torch.randn(5, 40000, 2) + 1j
    task_info = torch.randn(5, 4)
    model = EqSNSEDBP(Nmodes=2, step=5, dtaps=1001, ntaps=41, n_fwm=1)
    y = model(x, task_info)
    print(y.shape)
    print(model.overlaps)
    print(model)
    print(model.__class__.__name__)