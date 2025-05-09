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
from pkufiber.op import get_beta2, get_beta1, periodic_padding


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
        

class SharedKernelConv1d(nn.Module):
    def __init__(self, kernel_size):
        super(SharedKernelConv1d, self).__init__()
        # 定义1D卷积核，输入输出通道都为1，groups=1表示共享一个卷积核
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False)
        torch.nn.init.zeros_(self.conv.weight)
    
    def forward_axis(self, x):
        # 输入形状为 [batch, L, Ci]
        batch_size, L, Ci = x.shape
        
        # 调整输入形状为 [batch*Ci, 1, L]，然后应用1D卷积
        x = x.permute(0, 2, 1).contiguous().view(batch_size * Ci, 1, L)
        
        # 应用共享卷积核
        x = self.conv(x)
        
        # 调整输出形状回到 [batch, L, Ci]
        L_out = x.shape[-1]
        x = x.view(batch_size, Ci, L_out).permute(0, 2, 1)
        
        return x
    
    def forward(self, x):
        # 对每个通道应用共享卷积核
        if x.dtype == torch.complex64 or x.dtype == torch.complex128:
            y = self.forward_axis(x.real) + 1j * self.forward_axis(x.imag)
        else:
            y = self.forward_axis(x)
        return y




class ZcvFilter(nn.Module):
    """
        zeros center vmap filter.
        x: real [B, L, Nmodes] -> real [B, L -  xpm_size + 1, Nmodes]
        """
    def __init__(self, kernel_size):
        super(ZcvFilter, self).__init__()
        self.kernel_size = kernel_size
        self.overlaps = kernel_size - 1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        """
        zeros center vmap filter.
        x: real [B, L, Nmodes] -> real [B, L -  xpm_size + 1, Nmodes] 
        """
        B = x.shape[0]
        Nmodes = x.shape[-1]
        x = x.transpose(1, 2)  # x [B, Nmodes, L]
        x = x.reshape(-1, 1, x.shape[-1])  # x [B*Nmodes, 1, L]
        c0 = self.conv.weight[0, 0, self.kernel_size // 2]


        x = (
            self.conv(x) - c0 * x[:, :, (self.overlaps // 2) : -(self.overlaps // 2)]
        )  # x [B*Nmodes, 1, L - xpm_size + 1]  
        x = x.reshape(B, Nmodes, x.shape[-1])  # x [B, Nmodes, L - xpm_size + 1]  
        x = x.transpose(1, 2)  # x [B, L - xpm_size + 1, Nmodes] 
        return x


class FwmFilter(nn.Module):
    """
        zeros center vmap filter.
        x: real [B, L, Nmodes] -> real [B, L -  xpm_size + 1, Nmodes] 
        """
    def __init__(self, kernel_size, channels):
        super(FwmFilter, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels   
        self.overlaps = kernel_size - 1
        self.conv = ComplexConv1d(channels, channels, kernel_size=kernel_size, bias=False, groups=channels, init='zeros')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        zeros center vmap filter.
        x: real [2n+1, B, L, Nmodes] -> real [2n+1, B, L -  xpm_size + 1, Nmodes]
        """
        num = x.shape[0]
        B = x.shape[1]
        Nmodes = x.shape[-1]
        x = x.permute(1, 3, 0, 2)  # x [B, Nmodes, 2n+1, L]
        x = x.reshape(-1, x.shape[-2], x.shape[-1])  # x [B*Nmodes, 2n+1, L]
        x = self.conv(x)                             # x [B*Nmodes, 2n+1, L - xpm_size + 1]
        x = x.reshape(B, Nmodes, num, x.shape[-1])   # x [B, Nmodes, 2n+1, L - xpm_size + 1]
        x = x.permute(2, 0, 3, 1)                    # x [2n+1, B, L - xpm_size + 1, Nmodes]
        return x



class SNSEFilter(nn.Module):
    
    def __init__(self, Nmodes, step, ntaps, share, gamma=0.0016567, L=2000e3):  
        super(SNSEFilter, self).__init__()
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

        self.conv1d = nn.ModuleList([SharedKernelConv1d(kernel_size=self.ntaps) for i in range(n_num)])

    def forward(self, x, task_info, i):
        '''
        [B, M, Nmodes] -> [B, M - ntaps + 1, Nmodes]
        '''
        P = get_power(task_info, x.shape[-1], x.device)
        phi = nconv(torch.abs(x) ** 2,
                self.Nkernel[min(i, len(self.Nkernel) - 1)].expand(
                    x.shape[0], self.Nmodes, self.Nmodes, self.ntaps
                ),
                1,)
        
        add_term = 1j*self.conv1d[min(i, len(self.conv1d) - 1)](x * torch.roll(x.conj(), 1, dims=-1)) * torch.roll(x, 1, dims=-1)[:, self.ntaps//2:x.shape[1] - (self.ntaps//2)]
        x = x[:, self.ntaps//2:x.shape[1] - (self.ntaps//2)] * torch.exp(1j*phi*P[:,None,None])  + add_term*P[:,None,None]
        return x

    def rmps(self):
        from pkufiber.dsp.nonlinear_compensation.rmps import rmps_edc
        return rmps_edc(self.ntaps)/4 + 6 + rmps_edc(self.ntaps)/2 + 2


class SnseFilter(nn.Module):
    
    def __init__(self, Nmodes, step, ntaps, share, gamma=0.0016567, L=2000e3, no_icixpm=False, n_fwm=0):  
        super(SnseFilter, self).__init__()
        self.Nmodes = Nmodes
        self.L = L
        self.step = step
        self.ntaps = ntaps
        self.gamma = gamma
        self.n_fwm = n_fwm

        self.no_icixpm = no_icixpm  
        n_num = 1 if share else self.step

        self.C00 = nn.Parameter(torch.zeros(()), requires_grad=True)  # SPM coeff
        self.ixpm_conv1d = nn.ModuleList([ZcvFilter(kernel_size=self.ntaps) for i in range(n_num)])
        self.icixpm_conv1d = nn.ModuleList([ZcvFilter(kernel_size=self.ntaps) for i in range(n_num)])

        if n_fwm > 0:
            self.fwm_filter = FwmFilter(kernel_size=ntaps, channels=2*n_fwm)
    
    def truncate(self, x):
        '''
        [B,L,Nmodes] -> [B,L-ntaps+1,Nmodes]
        '''
        return x[:, self.ntaps//2:x.shape[1] - (self.ntaps//2)]


    def icixpm(self, E, i):
        '''
        [B,L,Nmodes] -> [B,L-ntaps+1,Nmodes]
        '''
        conv = self.icixpm_conv1d[min(i, len(self.icixpm_conv1d) - 1)]
        x = E * torch.roll(E.conj(), 1, dims=-1)  # x [B, M Nmodes]
        x = conv(x.real) + (1j)*(conv(x.imag))
        x = self.truncate(E).roll(1, dims=-1) * x  # [B, M - ntaps + 1, Nmodes]
        return (1j)*x
    
    def fwm(self, x, i):
        # x: [B, M, 2]
        xt = self.truncate(x)

        ax = x 
        ax_s = torch.stack([torch.roll(ax, shifts=i, dims=1) for i in range(-self.n_fwm,self.n_fwm+1) if i != 0]) # [2n+1, B, M, 2]
        ay = torch.roll(x, shifts=1, dims=-1)                            # [2n, B, M, 2]
        ay_s = torch.roll(ax_s, shifts=1, dims=-1)                       # [2n, B, M, 2]
        fwm_A = 2*ax[None,...]*ax_s.conj() +  ay[None,...] * ay_s.conj() # [2n, B, M, 2]
        fwm_B = ax[None,...] * ay_s.conj()                               # [2n, B, M, 2]

        return torch.sum(ax_s[:,:,(self.ntaps//2):-(self.ntaps//2)]*self.fwm_filter(fwm_A),dim=0) + torch.sum(ay_s[:,:,(self.ntaps//2):-(self.ntaps//2)]*self.fwm_filter(fwm_B), dim=0)
        
    def forward(self, x, task_info, i):
        '''
        [B, M, Nmodes] -> [B, M - ntaps + 1, Nmodes]
        '''
        # P 的 scale不对的话，训练出来的结果不好！！！
        P = get_power(task_info, x.shape[-1], x.device)
        x = x * torch.sqrt(P[:, None, None])
        power = torch.abs(x)**2
        ps = 2*power + torch.roll(power, 1, dims=-1)  # [B, M, Nmodes]

        # SPM + iXPM
        spm = self.C00 * self.truncate(power).sum(dim=-1, keepdim=True)  # [B, M - ntaps + 1, 1]
        ixpm = 2*self.ixpm_conv1d[min(i, len(self.ixpm_conv1d) - 1)](ps.real) 
        phi = spm + ixpm  # [B, M - ntaps + 1, Nmodes]

        # fwm
        if self.n_fwm > 0:
            fwm = self.fwm(x, i)

        # iciXPM
        if self.no_icixpm:
            x = self.truncate(x) * torch.exp(1j*phi)
        else:
            x = self.truncate(x) * torch.exp(1j*phi)  + self.icixpm(x, i) # self.gamma*self.L / self.step
        
        if self.n_fwm > 0:
            x = x + fwm

        x = x / torch.sqrt(P[:, None, None])
        return x

    def rmps(self):
        from pkufiber.dsp.nonlinear_compensation.rmps import rmps_edc
        return rmps_edc(self.ntaps)/4 + 6 + rmps_edc(self.ntaps)/2 + 2 + self.n_fwm*(2*rmps_edc(self.ntaps) + 12)




# class SnseFilterPlus(nn.Module):
    
#     def __init__(self, Nmodes, step, ntaps, share, n_max=0, gamma=0.0016567, L=2000e3):  
#         super(SnseFilterPlus, self).__init__()
#         self.Nmodes = Nmodes
#         self.L = L
#         self.step = step
#         self.ntaps = ntaps
#         self.gamma = gamma

#         self.n_max = n_max
#         n_num = 1 if share else self.step

#         self.fwm_filter_a = FwmFilter(kernel_size=ntaps, channels=2*n_max+1)
#         self.fwm_filter_b = FwmFilter(kernel_size=ntaps, channels=2*n_max+1)

#     def features(self,x):
#         # x: [B, M, 2]
#         ax = x 
#         ax_s = torch.stack([torch.roll(ax, shifts=i, dims=1) for i in range(-self.n_max,self.n_max+1)]) # [2n+1, B, M, 2]
#         ay = torch.roll(x, shifts=1, dims=-1)
#         ay_s = torch.roll(ax_s, shifts=1, dims=-1)
#         A = 2*ax[None,...]*ax_s.conj() +  ay[None,...] * ay_s.conj() # [2n+1, B, M, 2]
#         B = ax[None,...] * ay_s.conj() # [2n+1, B, M, 2]
#         return A, B
        

#     def truncate(self, x):
#         '''
#         [B,L,Nmodes] -> [B,L-ntaps+1,Nmodes]
#         '''
#         return x[:, self.ntaps//2:x.shape[1] - (self.ntaps//2)]


#     def forward(self, x, task_info, i):
#         '''
#         [B, M, Nmodes] -> [B, M - ntaps + 1, Nmodes]
#         '''
#         # P 的 scale不对的话，训练出来的结果不好！！！
#         P = get_power(task_info, x.shape[-1], x.device)
#         x = x * torch.sqrt(P[:, None, None])
#         fwm_A, fwm_B = self.features(x)
#         x = x + self.fwm_filter_a(fwm_A)+ self.fwm_filter_b(fwm_B)

#         x = x / torch.sqrt(P[:, None, None])
#         return x

#     def rmps(self):
#         from pkufiber.dsp.nonlinear_compensation.rmps import rmps_edc
#         return rmps_edc(self.ntaps)/4 + 6 + rmps_edc(self.ntaps)/2 + 2


if __name__ == "__main__":
    # test FwmFilter
    x = torch.randn(4, 5, 40000, 2) + 1j
    task_info = torch.randn(5, 4)
    fwm = FwmFilter(kernel_size=5, channels=4)
    y = fwm(x)
    print(y.shape)

    # test SnseFilter
    x = torch.randn(5, 40000, 2) + 1j
    task_info = torch.randn(5, 4)
    model = SnseFilter(Nmodes=2, step=5, ntaps=1001, share=True, n_fwm=1)
    y = model(x, task_info, 0)
    print(y.shape)

