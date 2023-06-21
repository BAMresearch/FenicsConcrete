import numpy as np
def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    
    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf = acf / (len(x)*np.ones(len(x)) - np.arange(len(x)))
    #acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr_new(y, c=5.0):
    total_chain_steps = y.shape[0]
    number_of_chains = y.shape[1]
    f = np.zeros(total_chain_steps)
    for i in range(number_of_chains): # for each chain
        f += autocorr_func_1d(y[:,i]) # array with ACFs for different gaps.
    f /= number_of_chains             # average autocorrelation function over all chains.
    taus = 2.0 * np.cumsum(f) + 1.0   # ACT calculated calculated upto different gaps.
    #return taus[-1]
    window = auto_window(taus, c)    # ACT selected for a particular gap.
    return taus[window]
