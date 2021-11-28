import numpy as np
from tqdm import tqdm
import random

from .helper import find_nearest, int_R


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
        diff = R0[i] - int_R(theta, DS, t)
        total += diff**2

    return np.exp(-0.5*total)


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
    return min(1, pi(R0, v_theta)/(pi(R0, u_theta)))


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
            p = random.rand()
            if p < acc_p:
                theta_n = theta_v

        thetas_all[:, m] = theta_n

    return thetas_all


if __name__ == '__main__':
    EnMCMC(3, 3)
