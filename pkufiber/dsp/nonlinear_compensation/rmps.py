import numpy as np, matplotlib.pyplot as plt

Nd_EDC = 1501 * 25

def cmps_conv(N, Nd):
    '''
    Complex Multiply per symbl (CMPS) for convolution.
    Input:
        N: FFT size.
        Nd: convolution kernel size. 
    Output:
        Number of complex multiply per sample.
    '''
    return N*(np.log2(N) + 1)/(N - Nd + 1)

def rmps_fft(Nfft):
    '''
        Real Multiply per Sample (RMPS) for FFT.
    '''
    return 4 * Nfft/2*np.log2(Nfft)   # only multiplication is considered.

def rmps_edc(Nd):
    ''''
        Real Multiply per Sample (RMPS) for EDC.
    
    rmps_edc(Nd) <= 8 (log2(Nd) + 2)
    '''
    k = int(np.log2(Nd))
    return 4*np.min([cmps_conv(int(xi*Nd), Nd) for xi in range(k//2, k)])


def rmps_fdbp(Nd, Nf, step, sps=2):
    '''
        Real Multiply per Sample (RMPS) for FDBP.
    Input:
        Nd: dispersion kernel size for each step. 
        Nf: nonlinearity kernel size for each step.
        step: number of steps.
        sps: samples per symbol.
    '''
    return sps*step*(rmps_edc(Nd) + rmps_edc(Nf)/4 * 2 + 6) 


def rmps_dbp(Nstps, Nspan, Nd_EDC):
    '''
        Real Multiply per Sample (RMPS) for DBP.
    Input:
        Nstps: number of steps.
        Nspan: number of spans.
        Nd_EDC: dispersion kernel size for whole EDC.
    '''
    Nd = Nd_EDC / Nspan / Nstps
    return Nspan*Nstps * rmps_edc(Nd)




