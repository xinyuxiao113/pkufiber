import numpy as np, matplotlib.pyplot as plt

Nd_EDC = 1501 * 25

def cmps_conv(N, Nd):
    '''
    Complex Multiply per symbl (CMPS) for convolution.
    '''
    return N*(np.log2(N) + 1)/(N - Nd + 1)

def rmps_fft(Nfft):
    return 4 * Nfft/2*np.log2(Nfft)   # only multiplication is considered.

def rmps_edc(Nd=Nd_EDC):
    k = int(np.log2(Nd))
    return 4*np.min([cmps_conv(int(xi*Nd), Nd) for xi in range(k//2, k)])


def rmps_fdbp(Nd, Nf, step, sps=2):
    return sps*step*(rmps_edc(Nd) + rmps_edc(Nf)) 


def rmps_dbp(Nstps, Nspan=25):
    Nd = Nd_EDC / Nspan / Nstps
    return Nspan*Nstps * rmps_edc(Nd)




