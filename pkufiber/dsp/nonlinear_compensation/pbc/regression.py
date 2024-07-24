import pickle , matplotlib.pyplot as plt, torch, numpy as np, argparse, time, os, scipy
from .features import TripletFeatures
from pkufiber.simulation.receiver import ber


def kernel(X, Y, C, p=2.0, gamma=1.0, k_type='w=0'):
    '''
    Input:
        X:  [N, Nmodes, p]
        Y:  [N, Nmodes]
        C:  [Nmodes, p]
        p, gamma: hyperparameters.
        k_type: 'w=0', 'w!=0', 'p-gamma'
    Output:
            [N, Nmodes]
    '''
    if k_type=='w=0':
        return torch.abs(Y)**(p - 2)
    elif k_type=='w!=0':
        return torch.abs(Y - predict(C, X))**(p - 2)
    elif k_type=='p-gamma':
        return torch.minimum(torch.ones(()), gamma / torch.abs(Y))**2 * torch.abs(Y)**(p - 2)
    else:
        raise ValueError('k_type should be w=0 or w!=0')


def predict(C, X):
    '''
    Input:
        C: [Nmodes, p].       PBC coefficients.   
        X: [N, Nmodes, p]. Nonlinear features.
    Output:
        [N, Nmodes]
    '''
    return torch.einsum('nmp, mp->nm', X, C)  # [B, N, Nmodes]



def fit(X,  Y, weight, lamb_l2:float=0, pol_sep=True):
    '''
    fit the coeff for PBC, if pol_sep==True, then the PBC is applied to each polarization separately
    Input:
        X: features with shape        [N, Nmodes, p].
        Y: target with shape          [N, Nmodes].
        weight: weight for each data. [N, Nmodes]
    output:
        C: features coeff.            [Nmodes, p]
        C = argmin_{C}  sum weight*|X1 @ C - Y|^2 + lamb_l2/2 * |C|^2
    '''

    Nmodes = Y.shape[-1]
    weight = weight.to(torch.complex64)

    if pol_sep:
        A = torch.einsum('nm, nmp, nmq -> mpq', weight, X.conj(), X) / X.shape[0]   # [m, p, p]
        b = torch.einsum('nm, nmp, nm  -> mp', weight, X.conj(), Y) / X.shape[0]    # [m, p]
        C = [torch.linalg.solve(A[i] + lamb_l2 * torch.eye(A.shape[-1]), b[i]) for i in range(A.shape[0])]
        return torch.stack(C, dim=0)
    else:
        A = torch.einsum('nm, nmp, nmq -> pq', weight, X.conj(), X) / X.shape[0]    # [p, p]
        b = torch.einsum('nm, nmp, nm  -> p', weight, X.conj(), Y) / X.shape[0]     # [p]
        C = torch.linalg.solve(A + lamb_l2 * torch.eye(A.shape[-1]), b)
        return torch.stack([C]*Nmodes, dim=0)
