import scipy.special as sp
import numpy as np
from scipy.integrate import quad

def pbc_coeff(m, n, beta2, T, tau, L):
    if m==0 and n == 0:
        def integrand(z, tau, beta_2):
            return 1 / np.sqrt((tau**4) / (3 * beta_2**2) + z**2)
        result, error = quad(integrand, 0, L, args=(tau, beta2))
        return result
    elif m*n == 0 and m != n:
        return 0.5 * sp.exp1((m-n)**2*T**2*tau**2/(3*beta2**2*L**2))
    else:
        return sp.exp1(-1j*m*n*T**2/(beta2*L))


def vstf_coeff(m, n, alpha, beta2, L, Ls, fftsize, samplerate):
    omega_m = samplerate * 1/fftsize * m 
    omega_n = samplerate * 1/fftsize * n 
    
    F = np.exp(-1j * beta2 * omega_m * omega_n / 2 * (L - Ls)) * np.sin(beta2*omega_m * omega_n*L/2)/np.sin(beta2*omega_m * omega_n*Ls/2)

    return (1 - np.exp(alpha*Ls - 1j*beta2*omega_m * omega_n*Ls))/(-alpha + 1j*beta2*omega_m * omega_n) * F

