'''
DBP sps = 1:  after CDC + ADF.
'''

import torch.nn as nn, torch, numpy as np, torch
from torch.fft import fft, ifft, fftshift, ifftshift
import torch.nn.functional as F
from typing import Union, List, Tuple, Optional

from pkufiber.core import TorchSignal, TorchTime
from pkufiber.dsp.layers import ComplexLinear, ComplexConv1d
from pkufiber.dsp.nonlinear_compensation.op import triplets, trip_op, get_power
from pkufiber.dsp.op import dispersion_kernel, dconv, nconv
from pkufiber.op import get_beta2, get_beta1


def dispersion_loss(h_list, dz, Fs, Nfft):
    '''
        h_list: [N, dtaps]
        dz: total distance.
        Fs: float
        Nfft: int
    '''

    dtaps = h_list.shape[-1]
    m = (Nfft - dtaps) // 2
    h = F.pad(h_list, (m, m))
    H = dispersion_kernel(-dz, h.shape[-1], Fs, domain='freq').to(h_list.device)
    H_hat = fft(ifftshift(h, dim=-1), dim=-1)
    H_hat = torch.prod(H_hat, dim=0)
    return torch.mean(torch.abs(H - H_hat)**2)


class DispersionFilter(nn.Module):
    def __init__(self, step, dtaps, dz, d_train, d_share, Fs, D = 16.5, Fc = 299792458 / 1550e-9, Fi = 299792458 / 1550e-9):
        '''
        step: int
        dtaps: int
        dz: float  [m]
        d_train: bool
        d_share: bool
        '''
        super(DispersionFilter, self).__init__()
        self.step = step
        self.dz = dz
        self.d_train = d_train
        self.d_share = d_share
        self.Fs = Fs
        self.D = D 
        self.Fc = Fc
        self.Fi = Fi
        self.beta2 = get_beta2(self.D, self.Fc) / 1e3            # Second-order dispersion coefficient [s^2/m]
        self.beta1 = get_beta1(self.D, self.Fc, self.Fi) / 1e3   # First-order dispersion coefficient [s/m]
        self.dtaps = dtaps
        d_num = 1 if d_share else self.step

        Dkernel_init = dispersion_kernel(
            -self.dz, self.dtaps, Fs, self.beta2, self.beta1, domain="time"
        ).to(torch.complex64)   # [1, dtaps]

        D_init = torch.concatenate([Dkernel_init]*d_num)  # [d_num, dtaps]

        self.Dkernel_real = nn.Parameter(D_init.real, requires_grad=d_train)
        self.Dkernel_imag = nn.Parameter(D_init.imag, requires_grad=d_train)

        # Train D-filter
        # self.train()

    def forward(self, x, i):
        '''
        [B, M, Nmodes] -> [B, M - dtaps + 1, Nmodes]
        '''
        assert x.dtype == torch.complex64 or x.dtype == torch.complex128
        Dk = self.Dkernel_real[min(i, len(self.Dkernel_real) - 1)].expand(
                x.shape[0], self.dtaps
            ) + 1j * self.Dkernel_imag[min(i, len(self.Dkernel_imag) - 1)].expand(
                x.shape[0], self.dtaps
            )
        x = dconv(x, Dk, stride=1)
        return x
    
    def dispersion_loss(self, weight=None):
        '''
        weight: [d_num]
        '''
        Dkernel = self.Dkernel_real + 1j * self.Dkernel_imag  # [d_num, dtaps]
        
        if Dkernel.shape[0] == 1:
            Dkernel = Dkernel.expand(self.step, -1)
        
        if weight == None:
            weight = torch.ones(Dkernel.shape[0])

        loss = [weight[i] * dispersion_loss(Dkernel[0:i+1], self.dz*(i+1), self.Fs, (self.dtaps - 1)*self.step + 1) for i in range(Dkernel.shape[0])]

        return torch.stack(loss).mean()

    def train_filter(self, weight=None, lr=1e-3, epoch=6000, show_interval=200):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch//3, gamma=0.1)
        # Training loop
        Loss = []
        num_epochs = epoch

        print('Training Dispersion Filter...')
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.dispersion_loss(weight=weight)
            loss.backward()
            optimizer.step()
            scheduler.step()
            Loss.append(loss.item())
            if (epoch) % show_interval == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        print('Training Done!')
        return Loss

class NonlinearFilter(nn.Module):
    
    def __init__(self, Nmodes, step, ntaps, share, gamma=0.0016567, L=2000e3):  
        super(NonlinearFilter, self).__init__()
        self.Nmodes = Nmodes
        self.L = L
        self.step = step
        self.ntaps = ntaps
        self.gamma = gamma

        n_num = 1 if share else self.step
        self.Nkernel = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        self.Nmodes, self.Nmodes, self.ntaps, dtype=torch.float32
                    )
                )
                for _ in range(n_num)
            ]
        )  # Nonlinear kernel [Nmodes, Nmodes, ntaps]

    def forward(self, x, task_info, i):
        '''
        [B, M, Nmodes] -> [B, M - ntaps + 1, Nmodes]
        '''
        P = 1e-3 * 10 ** (task_info[:, 0] / 10) / self.Nmodes  # Power [W]
        phi = nconv(torch.abs(x) ** 2,
                self.Nkernel[min(i, len(self.Nkernel) - 1)].expand(
                    x.shape[0], self.Nmodes, self.Nmodes, self.ntaps
                ),
                1,)
        x = x[:, self.ntaps//2:x.shape[1] - (self.ntaps//2)] * torch.exp(1j*phi*self.gamma*P[:,None,None]*self.L / self.step)
        return x
        
