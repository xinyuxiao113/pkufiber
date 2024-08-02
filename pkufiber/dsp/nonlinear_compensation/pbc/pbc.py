"""
    NN equalizer.
"""

import torch.nn as nn, torch, numpy as np, torch
from typing import Union, List, Tuple, Optional

from pkufiber.core import TorchSignal, TorchTime
from pkufiber.dsp.layers import ComplexLinear, ComplexConv1d
from pkufiber.dsp.nonlinear_compensation.op import triplets, trip_op, get_power

from pkufiber.dsp.nonlinear_compensation.pbc.features import TripletFeatures, IndexType

'''
每个模型都需要 M， overlaps参数
'''


class EqPBC(nn.Module):
    """
    PBC equalizer.
    M: int, default=41, the length of the filter.
    rho: float, default=1, the parameter of the filter.
    index_type: str, default="reduce-1", ['full', 'reduce-1', 'reduce-2']
    pol_share: bool, default=False, whether the two polarizations share the same filter.

    """

    def __init__(self, M: int = 41, rho: float=1., index_type: Union[IndexType, str]="reduce-1", pol_share: bool=False, decision=False):
        super(EqPBC, self).__init__()
        assert M % 2 == 1, "M must be odd."
        self.M = M
        self.overlaps = M - 1
        self.features = TripletFeatures(M, rho, index_type, decision)
        self.pol_num = 1 if pol_share else 2
        self.nn = nn.ModuleList(
            [
                ComplexLinear(self.features.hdim, 1, bias=False)
                for _ in range(self.pol_num)
            ]
        )

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        """
        Input:
            signal:  [batch, M, Nmodes]
            task_info: torch.Tensor or None. [B, 4] ot None.    [P,Fi,Fs,Nch]
        Output:
            Tensor. [batch, Nmodes]
            Nmodes = 1:
                O_{b,k,i} = gamma P0^{3/2} * sum_{m,n} E_{b, k+n, i} E_{b, k+m+n, i}^* E_{b, k+m, i} C_{m,n}
            Nmodes = 2:
                O_{b,k,i} = gamma P0^{3/2} * sum_{m,n} (E_{b, k+n, i} E_{b, k+m+n, i}^* +  E_{b, k+n, -i} E_{b, k+m+n, -i}^*) E_{b, k+m, i} C_{m,n}
        """
        Nmodes = x.shape[-1]
        if Nmodes == 1 and self.pol_num == 2:
            raise ValueError(
                "Nmodes=1 and pol_num=2 is not a good choise, please set pol_share=True."
            )
        P = get_power(task_info, Nmodes, x.device)
        features = self.features.nonlinear_features(x)  # [batch, Nmodes, len(S)]
        E = [
            self.nn[min(i, self.pol_num - 1)](features[..., i, :])
            for i in range(Nmodes)
        ]  # [batch, 1]
        E = torch.cat(E, dim=-1)  # [batch, Nmodes]
        return x[:, self.features.M // 2, :] + E * P[:, None]  # [batch, Nmodes]


class EqAMPBC(nn.Module):
    """
    Latest version of AmFoPBC. Nmodes=2.
    M: int, default=41, the length of the filter.
    rho: float, default=1, the parameter of the filter.
    fwm_share: bool, default=False, whether the two FWM filters share the same filter.
    """

    def __init__(self, M: int = 41, rho: float=1, fwm_share: bool=False, decision=False):
        super(EqAMPBC, self).__init__()
        self.M, self.L, self.rho = M, M - 1, rho
        self.overlaps = M - 1
        self.xpm_size, self.overlaps = M, M - 1
        self.fwm_modes = 1 if fwm_share else 2
        self.features = TripletFeatures(M, rho, index_type="FWM", decision=decision)

        self.C00 = nn.Parameter(torch.zeros(()), requires_grad=True)  # SPM coeff
        self.fwm_nn = nn.ModuleList(
            [
                ComplexLinear(self.features.hdim, 1, bias=False)
                for _ in range(self.fwm_modes)
            ]
        )  # FWM coeff
        self.xpm_conv1 = nn.Conv1d(1, 1, M, bias=False)
        self.xpm_conv2 = nn.Conv1d(1, 1, M, bias=False)
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.fwm_nn:
            nn.init.zeros_(layer.real.weight)
            nn.init.zeros_(layer.imag.weight)
        for conv in [self.xpm_conv1, self.xpm_conv2]:
            nn.init.zeros_(conv.weight)

    def zcv_filter(self, conv, x):
        """
        zeros center vmap filter.
        x: real [B, L, Nmodes] -> real [B, L -  xpm_size + 1, Nmodes]
        """
        B = x.shape[0]
        Nmodes = x.shape[-1]
        x = x.transpose(1, 2)  # x [B, Nmodes, L]
        x = x.reshape(-1, 1, x.shape[-1])  # x [B*Nmodes, 1, L]
        c0 = conv.weight[0, 0, self.xpm_size // 2]
        x = (
            conv(x) - c0 * x[:, :, (self.overlaps // 2) : -(self.overlaps // 2)]
        )  # x [B*Nmodes, 1, L - xpm_size + 1]
        x = x.reshape(B, Nmodes, x.shape[-1])  # x [B, Nmodes, L - xpm_size + 1]
        x = x.transpose(1, 2)  # x [B, L - xpm_size + 1, Nmodes]
        return x

    def IXIXPM(self, E):
        """
        E: [batch, M, Nmodes]
        """
        x = E * torch.roll(E.conj(), 1, dims=-1)  # x [B, M Nmodes]
        x = self.zcv_filter(self.xpm_conv2, x.real) + (1j) * self.zcv_filter(
            self.xpm_conv2, x.imag
        )  # x [B, M - xpm_size + 1, Nmodes]
        x = (
            E[..., (self.overlaps // 2) : -(self.overlaps // 2), :].roll(1, dims=-1) * x
        )  # x [B, M - xpm_size + 1, Nmodes]
        return x[:, 0, :] * (1j)  #   [B, Nmodes]

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        """
        Input:
            signal:  [batch, M, Nmodes] or [M, Nmodes]
            task_info: torch.Tensor or None. [B, 4] ot None.    [P,Fi,Fs,Nch]
        Output:
            TorchSignal.
            Nmodes = 1:
                O_{b,k,i} = gamma P0^{3/2} * sum_{m,n} E_{b, k+n, i} E_{b, k+m+n, i}^* E_{b, k+m, i} C_{m,n}
            Nmodes = 2:
                O_{b,k,i} = gamma P0^{3/2} * sum_{m,n} (E_{b, k+n, i} E_{b, k+m+n, i}^* +  E_{b, k+n, -i} E_{b, k+m+n, -i}^*) E_{b, k+m, i} C_{m,n}
        """

        P = get_power(task_info, x.shape[-1], x.device)
        x = x * torch.sqrt(P[:, None, None])

        # IFWM term
        features = self.features.nonlinear_features(x)  # [batch, Nmodes, hdim]
        E = [
            self.fwm_nn[min(self.fwm_modes - 1, i)](features[..., i, :])
            for i in range(x.shape[-1])
        ]  # [batch, 1] x Nmodes
        E = torch.cat(E, dim=-1)  # [batch, Nmodes]

        # SPM + IXPM
        power = torch.abs(x) ** 2  # [B, M, Nmodes]
        ps = 2 * power + torch.roll(power, 1, dims=-1)  # [B, M, Nmodes]
        phi = (
            self.C00 * power[:, self.M // 2, :].sum(dim=-1, keepdim=True)
            + 2 * self.zcv_filter(self.xpm_conv1, ps)[:, 0, :]
        )  # [B, Nmodes]

        E = E + self.IXIXPM(x)  # [batch, Nmodes]
        E = E + x[:, self.M // 2, :] * torch.exp(1j * phi)  # [batch, Nmodes]
        E = E / torch.sqrt(P[:, None])  # [batch, Nmodes]
        return E


class EqPBCstep(nn.Module):
    '''
    PBC step.
    M: int, default=41, the length of the filter.
    rho: float, default=1, the parameter of the filter.
    index_type: str, default="reduce-1", ['full', 'reduce-1', 'reduce-2']
    pol_share: bool, default=False, whether the two polarizations share the same filter.

    '''
    def __init__(self, M: int = 41, rho=1.0, index_type: Union[IndexType, str]="reduce-1", pol_share: bool=False, decision=False):
        super(EqPBCstep, self).__init__()
        self.M, self.rho, self.index_type = M, rho, IndexType(index_type)
        self.PBC = EqPBC(M, rho, index_type, pol_share, decision)
        self.overlaps = self.M - 1

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, L, Nmodes]
        task_info: [batch, 4]
        --> [batch, L - M + 1, Nmodes]
        """
        batch, L, Nmodes = x.shape
        x = x.unfold(1, self.M, 1)  # [batch, L - M + 1, Nmodes, M]
        x = x.permute(0, 1, 3, 2)  # [batch,L - M + 1, M, Nmodes]
        x = x.reshape(-1, x.shape[-2], x.shape[-1])  # [batch*(L - M + 1), M, Nmodes]
        x = self.PBC(
            x,
            task_info.view(batch, 1, -1)
            .expand(batch, L - self.M + 1, -1)
            .reshape((batch * (L - self.M + 1), -1)),
        )  # [batch*(L - M + 1), Nmodes]
        x = x.reshape(batch, -1, x.shape[-1])  # [batch, L - M + 1, Nmodes]
        return x


class EqAMPBCstep(nn.Module):
    '''
    AMPBC step.
    M: int, default=41, the length of the filter.
    rho: float, default=1, the parameter of the filter.
    fwm_share: bool, default=False, whether the two FWM filters share the same filter.

    '''

    def __init__(self, M: int = 41, rho: float=1.0, fwm_share: bool=False, decision=False):
        super(EqAMPBCstep, self).__init__()
        self.M, self.rho = M, rho
        self.PBC = EqAMPBC(M, rho, fwm_share, decision)
        self.overlaps = self.M - 1

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, L, Nmodes]
        task_info: [batch, 4]
        --> [batch, L - M + 1, Nmodes]
        """
        batch, L, Nmodes = x.shape
        x = x.unfold(1, self.M, 1)  # [batch, L - M + 1, Nmodes, M]
        x = x.permute(0, 1, 3, 2)  # [batch,L - M + 1, M, Nmodes]
        x = x.reshape(-1, x.shape[-2], x.shape[-1])  # [batch*(L - M + 1), M, Nmodes]
        x = self.PBC(
            x,
            task_info.view(batch, 1, -1)
            .expand(batch, L - self.M + 1, -1)
            .reshape((batch * (L - self.M + 1), -1)),
        )  # [batch*(L - M + 1), Nmodes]
        x = x.reshape(batch, -1, x.shape[-1])  # [batch, L - M + 1, Nmodes]
        return x


class MultiStepPBC(nn.Module):
    '''
    Multi-step PBC.
    steps: int, the number of steps.
    M: int, default=41, the length of the filter.
    rho: float, default=1, the parameter of the filter.
    index_type: str, default="reduce-2", ['full', 'reduce-1', 'reduce-2']
    pol_share: bool, default=False, whether the two polarizations share the same filter.
    '''
    def __init__(
        self, steps: int, M: int = 41, rho=1.0, index_type: Union[IndexType, str]="reduce-2", pol_share=False, decision=False
    ):
        super(MultiStepPBC, self).__init__()
        self.steps = steps
        self.M = M
        self.PBC_list = nn.ModuleList(
            [EqPBCstep(M, rho, index_type, pol_share, decision) for i in range(steps)]
        )
        self.overlaps = sum([PBC.overlaps for PBC in self.PBC_list])

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, L, Nmodes]
        task_info: [batch, 4]
        --> [batch, L - (M-1)*steps, Nmodes]
        """
        for i in range(self.steps):
            x = self.PBC_list[i](x, task_info)
        return x


class MultiStepAMPBC(nn.Module):
    '''
    Multi-step AMPBC.
    steps: int, the number of steps.
    M: int, default=41, the length of the filter.
    rho: float, default=1, the parameter of the filter.
    fwm_share: bool, default=False, whether the two FWM filters share the same
    '''

    def __init__(self, steps: int, M: int = 41, rho: float=1, fwm_share:bool=False, decision:bool=False):
        super(MultiStepAMPBC, self).__init__()
        self.steps = steps
        self.M = M
        self.PBC_list = nn.ModuleList(
            [EqAMPBCstep(M, rho, fwm_share, decision) for i in range(steps)]
        )
        self.overlaps = sum([PBC.overlaps for PBC in self.PBC_list])

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, L, Nmodes]
        task_info: [batch, 4]
        --> [batch, L - (M-1)*steps, Nmodes]
        """
        for i in range(self.steps):
            x = self.PBC_list[i](x, task_info)
        return x


if __name__ == "__main__":
    M = 41
    Nmodes = 2
    L = 100
    # x = torch.randn(10, M, Nmodes) + 1j*torch.randn(10, M, Nmodes)
    # task_info = torch.randn(10, 4)
    # eq = eqPBC_step(M, 1)
    # y1 = eq.PBC(x, task_info)
    # y = eq(x, task_info)
    # print(y1.shape, y.shape)
    # print(y1 - y[:,0])

    x = torch.randn(10, L, Nmodes) + 1j * torch.randn(10, L, Nmodes)
    task_info = torch.randn(10, 4)
    eq = MultiStepPBC(2, M, 1)
    # eq = eqPBC_step(M, 1)
    y = eq(x, task_info)
    print(y.shape)
