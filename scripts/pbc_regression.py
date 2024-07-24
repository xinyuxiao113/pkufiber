
import pkufiber as pf 
import pkufiber.dsp.nonlinear_compensation as nl
import pkufiber.dsp.nonlinear_compensation.pbc as pbc
import pkufiber.dsp.nonlinear_compensation.nneq as eq
import torch 
from torch.utils.data import DataLoader

train_path = '/home/xiaoxinyu/TorchFiber/dataset_A800/train.h5'
test_path = '/home/xiaoxinyu/TorchFiber/dataset_A800/test.h5'
# train_path = '/home/xiaoxinyu/dataset/train_80km.h5'
# test_path = '/home/xiaoxinyu/dataset/test_80km.h5'
Rs = 40
Nch = 1
Nmodes = 2
Pch = 2


def get_loader(path: str, window_size: int=41, num_symb: int=100000):
    '''
    Input:
        path, window_size, num_symb
    Output:
        Rx_window, Tx, P, info
    '''
    train_data = pf.data.PbcDataset(path=path, Nmodes=Nmodes, Nch=Nch, Rs=Rs, Pch=Pch, window_size=window_size, num_symb=num_symb)
    train_loader = DataLoader(train_data, batch_size=num_symb-window_size*2, shuffle=False)
    Rx_window, Tx, info = next(iter(train_loader))
    P = pf.get_power(info, Nmodes=Nmodes, device='cpu')
    return Rx_window, Tx, P, info

def feature_extraction(Rx_window: torch.Tensor, P: torch.Tensor, rho:float=-1, decision=False):
    '''
    Input:
        Rx_window: [B, M, Nmodes]
        P: [B]
        rho: float, default -1
    Output:
        features: [B, Nmodes, M+p] or [B, Nmodes, p]
        index: list of valid index.
    '''

    if rho < 0: rho = Rx_window.shape[1]
    net = pbc.TripletFeatures(M=Rx_window.shape[1], rho=rho, decision=decision)
    symbol_features = Rx_window.transpose(1,2)                                              # [N, Nmodes, M]
    pbc_features = net.nonlinear_features(Rx_window) * P[:,None,None]   # [N, Nmodes, p]
    
    features = torch.cat([pbc_features], dim=2)                                           # [N, Nmodes, M+p]
    # features = torch.cat([symbol_features, pbc_features], dim=2)                            # [N, Nmodes, *]
    return features, net.index


class RegressionPBC:

    def __init__(self, window_size=41, rho=1):
        self.window_size = window_size
        self.rho = rho 
        self.coeffs = torch.zeros(1)
        self.indexs = []

    def train(self,num_symb=100000, p:float=2.0, gamma: float=1.0, lamb_l2: float=0.1):
        # data
        Rx_window, Tx, P, info = get_loader(train_path, self.window_size, num_symb)
        Rx = Rx_window[:,self.window_size//2]
        bias = Tx - Rx

        # fit data
        features, indexs = feature_extraction(Rx_window, P, rho = self.rho)
        weight = pbc.kernel(features, bias, None, p=p, gamma=gamma, k_type='p-gamma')
        coeffs = pbc.fit(features, bias, weight=weight, pol_sep=True, lamb_l2=lamb_l2)
        self.coeffs = coeffs 
        self.indexs = indexs
        print('Training done!')
    
    def test(self, num_symb=100000):
        '''
        
        '''
        import numpy as np
        Rx_window, Tx, P, info = get_loader(test_path, self.window_size, num_symb)
        Rx = Rx_window[:,self.window_size//2]
        Rx_hat = self.process(Rx_window, P)

        qsq = pf.qfactor(np.mean(pf.ber(Rx_hat, Tx)['BER']))
        print('CDC Q factor:', pf.qfactor(np.mean(pf.ber(Rx, Tx)['BER'])))
        print('PBC Q factor:', qsq)
        print('Test done!')
        return Rx, Rx_hat, Tx, qsq

    
    def process(self, Rx_window, P):
        Rx = Rx_window[:,self.window_size//2]
        features, indexs = feature_extraction(Rx_window, P, rho=self.rho)
        bias_hat = pbc.predict(self.coeffs, features)
        return Rx + bias_hat
        
    
    def show(self):
        pbc.show_pbc(self.coeffs[0], self.indexs, figsize=(6,5))
        pbc.show_pbc(self.coeffs[1], self.indexs, figsize=(6,5))




    







