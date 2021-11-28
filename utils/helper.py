import numpy as np


def DSt(t, times, DS):
    '''
    function to return time series value at any time using linear interpolation
    t: time you want to evaluate \Delta S at
    times: timestamps associated with the actual measurements of DS
    DS: time series (for one spatial location only) of \Delta S values
    '''
    return np.interp(t, times, DS)


def forward(theta, DSt, t, tb):
    '''
    theta: (r, t_a, S_c, A\sigma)
    Computes the foward map
    DSt: the \Delta S values at that location up to and including the time index t
    '''
    r, ta, Sc, Asig = theta
    if t < tb:
        return 0
    num = np.exp((DSt[t]-Sc)/Asig)
    t_primes = [int(k) for k in np.linspace(tb, t, num=np.int(t-tb+1))]
    den = 1/ta*np.trapz([np.exp((DSt[tp]-Sc)/Asig)
                         for tp in t_primes], t_primes) + 1
    return r*num/den


def int_R(theta, DS, t):
    '''
    Function that computes the spatial integral over R in equation (4) for given parameter values theta at a specific time index t
    '''
    # TODO: rewrite integral
    r, ta, Sc, Asig = theta
    total = 0
    for i in range(DX):
        for j in range(DY):
            # print(len(DS[0,0,:]))
            Dsij = DS[i, j, :t+1]
            tb = find_nearest(Dsij, Sc)
            R = forward(theta, Dsij, t, tb)
            total += R*(0.25**2)

    return total


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
