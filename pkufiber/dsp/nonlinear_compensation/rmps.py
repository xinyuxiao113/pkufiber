import numpy as np, matplotlib.pyplot as plt

Nd_EDC = 1501 * 25

def rmps_conv(N, Nd):
    '''
    RMPS for convolution.
    '''
    return 2*N*(np.log2(N) + 1)/(N - Nd + 1)

def rmps_fft(Nfft):
    return 4 * Nfft/2*np.log2(Nfft)

def rmps_edc(Nd=Nd_EDC):
    k = int(np.log2(Nd))
    return 4*np.min([rmps_conv(int(xi*Nd), Nd) for xi in range(k//2, k)])


def rmps_fdbp(Nd, Nf, step):
    return step*(rmps_edc(Nd) + rmps_edc(Nf)) 


def rmps_dbp(Nstps, Nspan=25):
    Nd = Nd_EDC / Nspan / Nstps
    return Nspan*Nstps * rmps_edc(Nd)




