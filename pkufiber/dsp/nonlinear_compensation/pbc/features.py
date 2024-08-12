"""
    NN equalizer.
"""

import torch.nn as nn, torch, numpy as np, torch, matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional
from enum import Enum

from pkufiber.core import TorchSignal, TorchTime
from pkufiber.dsp.layers import ComplexLinear, ComplexConv1d
from pkufiber.dsp.nonlinear_compensation.op import triplets, trip_op, get_power
from pkufiber.simulation.receiver import nearst_symb




class IndexType(Enum):
    full = "full"
    reduce_1 = "reduce-1"
    reduce_2 = "reduce-2"
    FWM = "FWM"


class TripletFeatures(nn.Module):
    '''
    Triplet features for nonlinear compensation.

    Args:
        M: int, default 41, the number of triplets.
        rho: int, default 1, the ratio of the number of triplets.
        index_type: str, default "reduce-1", the type of index.
        decision: bool, default False, if True, the input is decision symbols.

    Attributes:
        M: int, the number of triplets.
        L: int, the number of samples.
        rho: int, the ratio of the number of triplets.
        index_type: str, the type of index.  ['full','reduce-1', 'reduce-2','FWM']
        index: list, the valid index.
        hdim: int, the number of triplets.

    Methods:
        valid_index: check if the index is valid.
        get_index: get valid index.
        nonlinear_features: get the nonlinear features.
    '''

    def __init__(self, M: int = 41, rho: float=1, index_type: Union[IndexType, str]=IndexType.reduce_1, decision=False):
        super(TripletFeatures, self).__init__()
        self.M, self.L, self.rho, self.index_type = M, M - 1, rho, IndexType(index_type)
        self.index = self.get_index()
        self.hdim = len(self.index)
        self.decision = decision

    def valid_index(self, m, n):
        index_checks = {
            IndexType.full: lambda: abs(m * n) <= self.rho * self.L // 2 and abs(m) + abs(n) <= self.L // 2,
            IndexType.reduce_1: lambda: abs(m * n) <= self.rho * self.L // 2 and n >= m and abs(m) + abs(n) <= self.L // 2,
            IndexType.reduce_2: lambda: abs(m * n) <= self.rho * self.L // 2 and n >= abs(m) and abs(m) + abs(n) <= self.L // 2,
            IndexType.FWM: lambda: abs(m * n) <= self.rho * self.L // 2 and n >= abs(m) and abs(m) + abs(n) <= self.L // 2 and m * n != 0
        }
        if self.index_type not in index_checks:
            raise ValueError("Invalid index type. please choose from ['full','reduce-1', 'reduce-2','FWM']")
        return index_checks[self.index_type]()

    def sort_index(self, index):
        if self.index_type == IndexType.full:
            return index
        elif self.index_type == IndexType.reduce_1:
            idx1 = [(m, n) for m, n in index if m == n]
            idx2 = [(m, n) for m, n in index if m != n]
            return idx1 + idx2
        elif self.index_type == IndexType.reduce_2:
            idx1 = [(m, n) for m, n in index if m == 0 and n == 0]
            idx2 = [(m, n) for m, n in index if m == n and n != 0]
            idx3 = [(m, n) for m, n in index if m != n and m+n == 0]
            idx4 = [(m, n) for m, n in index if m != n and m+n != 0]
            return idx1 + idx2 + idx3 + idx4
        elif self.index_type == IndexType.FWM:
            idx1 = [(m, n) for m, n in index if m ==n]
            idx2 = [(m, n) for m, n in index if m != n and m+n == 0]
            idx3 = [(m, n) for m, n in index if m != n and m+n != 0]
            return idx1 + idx2 + idx3
        else:
            raise ValueError("Invalid index type. please choose from ['full','reduce-1', 'reduce-2','FWM']")
            

    def get_index(self):
        """
        Get valid index.
        """
        S = [(m, n) for m in range(-self.L // 2, self.L // 2 + 1) for n in range(-self.L // 2, self.L // 2 + 1) if self.valid_index(m, n)]
        return self.sort_index(S)
    
    def get_m_n(self, valid, device):
        idx = [(m, n) for m, n in self.index if valid(m, n)]
        if len(idx) > 0:
            m,n = zip(*idx)
        else:
            m,n = [], []
        m,n = torch.tensor(m, device=device), torch.tensor(n, device=device)
        return m,n

    def nonlinear_features(self, E):
        
        if self.decision: E = nearst_symb(E)
        if self.index_type == IndexType.full:
            m,n = self.get_m_n(lambda m,n: True, E.device)
            return triplets(E, m, n).transpose(1,2)  # [batch,  Nmodes, len(S)]
        
        elif self.index_type == IndexType.reduce_1:
            m0, n0 = self.get_m_n(lambda m,n: m == n, E.device)
            m1, n1 = self.get_m_n(lambda m,n: m != n, E.device)
            E0 = triplets(E, m0, n0).transpose(1,2)                                       # [batch, Nmodes, len(m0)]
            E1 = triplets(E, m1, n1).transpose(1,2) + triplets(E, n1, m1).transpose(1,2)  # [batch, Nmodes, len(m1)]
            return torch.cat([E0, E1], dim=-1)  # [batch, Nmodes, len(S)]
        elif self.index_type == IndexType.reduce_2:
            m0,n0 = self.get_m_n(lambda m,n: m == 0 and n == 0, E.device)
            m1, n1 = self.get_m_n(lambda m,n: m == n and n != 0, E.device)
            m2, n2 = self.get_m_n(lambda m,n: m != n and m+n == 0, E.device)
            m3, n3 = self.get_m_n(lambda m,n: m != n and m+n != 0, E.device)

            E0 = triplets(E, m0, n0).transpose(1,2)  # [batch, Nmodes, len(m0)]
            E1 = triplets(E, m1, n1).transpose(1,2) + triplets(E, -m1, -n1).transpose(1,2)  # [batch, Nmodes, len(m1)]
            E2 = triplets(E, m2, n2).transpose(1,2) + triplets(E, n2, m2).transpose(1,2)    # [batch, Nmodes, len(m2)]
            E3 = triplets(E, m3, n3).transpose(1,2) + triplets(E, n3, m3).transpose(1,2)  +  triplets(E, -m3, -n3).transpose(1,2) + triplets(E, -n3, -m3).transpose(1,2)    # [batch, Nmodes, len(m3)]
            return torch.cat([E0, E1, E2, E3], dim=-1)  # [batch, Nmodes, len(S)]
        elif self.index_type == IndexType.FWM:
            m1, n1 = self.get_m_n(lambda m,n: m ==n, E.device)
            m2, n2 = self.get_m_n(lambda m,n: m != n and m+n == 0, E.device)
            m3, n3 = self.get_m_n(lambda m,n: m != n and m+n != 0, E.device)
            

            E1 = triplets(E, m1, n1).transpose(1,2) + triplets(E, -m1, -n1).transpose(1,2)  # [batch, Nmodes, len(m1)]
            E2 = triplets(E, m2, n2).transpose(1,2) + triplets(E, n2, m2).transpose(1,2)    # [batch, Nmodes, len(m2)]
            E3 = triplets(E, m3, n3).transpose(1,2) + triplets(E, n3, m3).transpose(1,2)  +  triplets(E, -m3, -n3).transpose(1,2) + triplets(E, -n3, -m3).transpose(1,2)    # [batch, Nmodes, len(m3)]
            return torch.cat([E1, E2, E3], dim=-1)  # [batch, Nmodes, len(S)]
        else:
            raise ValueError("Invalid index type. please choose from ['full','reduce-1', 'reduce-2','FWM']")
        
    def rmps(self) -> int:
        '''
        real multiplications per sample. 
            4x: 2 for real and 2 for imag.
            2x: 2 multiply operator for each triplet.
        '''
        if self.index_type == IndexType.full:
            return 4*2*len(self.index)
        elif self.index_type == IndexType.reduce_1:
            idx1 = [(m, n) for m, n in self.index if m == n]
            idx2 = [(m, n) for m, n in self.index if m != n]
            return 4*2*(len(idx1)*1+ len(idx2)*2)
        elif self.index_type == IndexType.reduce_2:
            idx1 = [(m, n) for m, n in self.index if m == 0 and n == 0]
            idx2 = [(m, n) for m, n in self.index if m == n and n != 0]
            idx3 = [(m, n) for m, n in self.index if m != n and m+n == 0]
            idx4 = [(m, n) for m, n in self.index if m != n and m+n != 0]
            return 4*2*(len(idx1)*1+ len(idx2)*2 + len(idx3)*2 + len(idx4)*4)
        elif self.index_type == IndexType.FWM:
            idx1 = [(m, n) for m, n in self.index if m ==n]
            idx2 = [(m, n) for m, n in self.index if m != n and m+n == 0]
            idx3 = [(m, n) for m, n in self.index if m != n and m+n != 0]
            return 4*2*(len(idx1)*1+ len(idx2)*2 + len(idx3)*4)
        else:
            raise ValueError("Invalid index type. please choose from ['full','reduce-1', 'reduce-2','FWM']")
    



class SoFeatures(nn.Module):
    '''
    Seccond order features for nonlinear compensation.

    Args:
        M: int, default 41, the number of triplets.
        rho: int, default 1, the ratio of the number of triplets.
        decision: bool, default False, if True, the input is decision symbols.

    Attributes:
        M: int, the number of triplets.
        L: int, the number of samples.
        rho: int, the ratio of the number of triplets.
        index_type: str, the type of index.  ['full','reduce-1', 'reduce-2','FWM']
        index: list, the valid index.
        hdim: int, the number of triplets.

    Methods:
        valid_index: check if the index is valid.
        get_index: get valid index.
        nonlinear_features: get the nonlinear features.
    '''

    def __init__(self, M: int = 41, Nmodes: int=2, rho: float=1, decision=False):
        super(SoFeatures, self).__init__()
        self.M, self.L, self.rho  = M, M - 1, rho
        self.Nmodes = Nmodes

        self.index = self.get_index()
        self.hdim = len(self.index)*2
        self.decision = decision

    def valid_index(self, m, n, k):
        if abs(m) <= self.M // 2 and abs(n) <= self.M // 2 and abs(k) <= self.M // 2 and abs(m+n) <= self.M//2:
            return True

    def get_index(self):
        """
        Get valif index.
        """
        S = [
            (m, n, k)
            for m in range(-self.L // 2, self.L // 2 + 1)
            for n in range(-self.L // 2, self.L // 2 + 1)
            for k in range(-self.L //2,  self.L//2 + 1)
            if self.valid_index(m, n, k)
        ]
        return S
    
    def forward(self, E):
        '''
            E: [batch, M, Nmodes]
        '''
        assert E.shape[-1] == 2, "Nmodes must be 2, single mode not implemented."
        if self.decision:
            E = nearst_symb(E)
        Es = []
        p = E.shape[1] // 2
        for (m,n,k) in self.index:
            # SO term 1  https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9457130 Second-Order Perturbation Theory-Based Digital Predistortion for Fiber Nonlinearity Compensation
            E1 = triplets(E, m, n)
            term1 = torch.sum(torch.abs(E[:,p+k,:])**2, dim=-1) # [batch]
            Es.append(E1 * term1[:,None])
            # SO term 2 https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9457130 
            term2 = torch.sum(E[:, p+k]*E[:, p-k], dim=-1) # [batch]
            Es.append(E1.conj() * term2[:,None]) 
        return torch.stack(Es, dim=-1)
    
    def rmps(self) -> int:
        '''
        real multiplications per sample.
        4x: 2 for real and 2 for imag.
        4x: 4 multiply operator for each feature.
        '''
        return 4*len(self.index)*(2 + 2 + 2)
            



def show_pbc(C: torch.Tensor, index: list, index_type: IndexType=IndexType.reduce_1, figsize: tuple=(5,5),dpi: int=100, vmax=None, vmin=None, s=3):
    '''
    Show Pbc coeff heatmap.
    C: list, the PBC coefficients.
    index: list, the index of PBC coeffs.
    index_type: IndexType, the type of index.
    figsize: tuple, the size of figure.
    dpi: int, thse dpi of figure.
    vmax: float, the max value of colorbar.s
    vmin: float, the min value of colorbar.
    s: int, the size of points.
    '''
    x,y = zip(*index)
    values = np.log10(np.abs(C) + 1e-8)
    vmax = torch.log10(torch.max(torch.abs(C)) + 1e-8).item() if vmax == None else vmax
    vmin = torch.log10(torch.min(torch.abs(C)) + 1e-8).item() if vmin == None else vmin
    
    plt.figure(figsize=figsize, dpi=dpi)
    x = np.array(x)
    y = np.array(y)
    cmap = 'coolwarm'
    if index_type == IndexType.full:
        plt.scatter(x, y, c=values, cmap=cmap, s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
    elif index_type == IndexType.reduce_1:
        plt.scatter(x, y, c=values, cmap=cmap, s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
        plt.scatter(y, x, c=values, cmap=cmap, s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小s
    elif index_type == IndexType.reduce_2:
        plt.scatter(x, y, c=values, cmap=cmap, s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
        plt.scatter(y, x, c=values, cmap=cmap, s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
        plt.scatter(-x, -y, c=values, cmap=cmap, s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
        plt.scatter(-y, -x, c=values, cmap=cmap, s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
    else:
        raise ValueError("Invalid index type. please choose from ['full','reduce-1', 'reduce-2','FWM']")
    plt.colorbar(label='Value')
    plt.xlabel('m Coordinate')
    plt.ylabel('n Coordinate')
    plt.title(f'Heatmap of C_m,n (log10 scale)')


if __name__ == "__main__":
    M = 41
    Nmodes = 2
    L = 100

    x = torch.randn(10, L, Nmodes) + 1j * torch.randn(10, L, Nmodes)
