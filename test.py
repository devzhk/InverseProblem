import math
import scipy
from tqdm import tqdm
import random

# Array and Dataformating
import numpy as np
import h5py
import pandas as pd


f = h5py.File('data/Groningen_Data_Example.hdf5', 'r')
# keys() shows the contents of the file

# extract all the data.
# note that these hdf4 datasets should behave exactly as np arrays

# the maximum coulomb stress smoothed with a 6km Gaussian2D kernel calculated 10m below the reservoir.
smoothed_coulomb_stress = f['smoothed_coulomb_stress']
cat = pd.read_csv('data/catalog2.txt', sep='\t',
                  names=['Date', 'Time', 'Mag', 'Depth', 'RDX', 'RDY'])
# creating the dates and times
cat['DateTime'] = pd.to_datetime(
    cat.Date + ' ' + cat.Time, format='%d %m %Y %H %M %S')
temp_index = pd.DatetimeIndex(cat.DateTime)
cat['decDate'] = temp_index.year + temp_index.month/12 + (temp_index.day + (
    temp_index.hour + (temp_index.minute + (temp_index.second/60)/60)/24))/365.25
# filtering the catalog to magnitudes > mc
mc = 1.5
cat = cat[cat.Mag > mc]
cat = cat.reset_index()
# The simulation given in this example is from 1956 to 2019 for a total of 756 months
y0, y1, dy = 1956, 2019, 12

dates = np.linspace(y0, y1+1, (y1-y0)*dy+1)

DS = smoothed_coulomb_stress
dates_R = np.array(cat.decDate)
print(np.linspace(1992, 2019, (2019-1992)+1))
R0, years = np.histogram(dates_R, np.linspace(1992, 2019, (2019-1992)+1))

DX, DY, DT = smoothed_coulomb_stress.shape


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


def int_R_ref(theta, DS, t):
    '''
    Function that computes the spatial integral over R in equation (4) for given parameter values theta at a specific time index t
    '''
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


def int_R(theta, DS, t):
    '''
    Function that computes the spatial integral over R in equation (4) for given parameter values theta at a specific time index t
    '''
    r, ta, Sc, Asig = theta
    data = DS[:, :, :t+1]
    tbs = np.argmin(np.abs(data - Sc), axis=2)  # shape: DX x DY
    num = np.exp((data[:, :, t] - Sc) / Asig)   # shape: DX x DY

    DX, DY, DT = data.shape
    for i in range(DX):
        for j in range(DY):
            data[i, j, 0: tbs[i, j]] = 0.0
    t_prime = np.linspace(0, t, num=int(t+1)).reshape(1, 1, -1)
    tps = t_prime.repeat(DX, axis=0).repeat(DY, axis=1)
    den = 1.0 + np.trapz(np.exp((data - Sc) / Asig), tps, axis=2) / ta
    result = np.sum(r * num / den) / 4
    return result


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def pi(R0, theta):
    '''
    distribution we want to sample from 
    Ri0 is the vector of observed rate density (in months)
    theta is vector of parameter values

    Compute RHS of eqn (4)
    '''
    total = 0
    for i in range(2017-1992+1):
        year = 1992 + i
        # Need to find index t that corresponds to the year in the dates for the \Delta S data
        # taking the value halfway through the year: potentially we should average?
        t = find_nearest(dates, year + 0.5)
        res = int_R(theta, DS, t)
        diff = R0[i] - res
        total += diff**2

    return -0.5*total


def q(u_theta, v_theta):
    '''
    proposal distribution q, assume Gaussian with mean vector (u_theta,v_theta) and covariance sigma
    '''
    # TODO


def sample_q(u_theta):
    '''
    returns a sample from q(u_theta,*)
    Assuming Gaussian with mean at u_theta
    '''
    sigma = 0.1  # constant SD
    samp = u_theta + np.random.randn(4)*sigma
    return samp


def acc(u_theta, v_theta, R0):
    '''
    Acceptance probability of being at sample u_theta and transitioning to sample v_theta
    '''
    p1 = pi(R0, v_theta)
    p2 = pi(R0, u_theta)
    return min(1, np.exp(p1 - p2))


def sample_pi0(M):
    '''
    Initial distribution of parameters; paper assumes uniform
    Returns: 
        samples: M samples from this dist in 4xM np array
    '''

    Asig = np.random.rand(M)*0.999+0.001
    r = np.random.rand(M)*(10**(-2.6)-10**(-6.2))+10**(-6.2)
    ta = np.random.rand(M)*(10000-0.5)+0.5
    DSc = np.random.rand(M)*(0.3-0.05)+0.05
    samples = np.stack([r, ta, DSc, Asig], axis=0)
    return samples


def EnMCMC(M, N):
    '''
    Implements ensemble MCMC sampling: draws M samples but doing a burn-in of N steps each
    '''
    thetas_all = np.zeros((4, M))
    thetas0 = sample_pi0(M)
    for m in tqdm(range(M)):
        theta0 = thetas0[:, m]
        theta_n = theta0
        for n in tqdm(range(N)):
            theta_v = sample_q(theta_n)
            acc_p = acc(theta_n, theta_v, R0)
            p = random.random()
            if p < acc_p:
                theta_n = theta_v

        thetas_all[:, m] = theta_n

    return thetas_all


EnMCMC(3, 10)
