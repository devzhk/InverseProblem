import numpy as np
import math
from numpy import random




# Establishing variables (could probably be in separate main file)

# Data (need to import, but for now making it up)
times = np.linspace(0,399,400)

DS = random.rand(400)


# Note: DS also varies with space, but I haven't implemented that yet- one task it to add this feature and use the actual data 

def DSt(t):
    '''
    function to return time series value at any time using linear interpolation
    '''
    return np.interp(t,times,DS)

def forward(theta, t, tb):
    r, ta, Sc, Asig = theta
    if t < tb: return 0
    num = np.exp((DSt(t)-Sc)/Asig)
    t_primes = np.linspace(tb,t,num = int((t-tb)/10))
    den = 1/ta*np.trapz([np.exp((DSt(tp)-Sc)/Asig) for tp in t_primes],t_primes) + 1
    return r*num/den

test_theta = np.array([1,30,2,0.5])
print(forward(test_theta, 50,10))

    
def int_R(theta):
    '''
    Function that computes the spatial integral over R in equation (4) for given parameter values theta
    '''
    r, ta, Sc, Asig = theta
    #TODO 
    

# Implementing MCMC
def pi(Ri0, theta):
    '''
    Log of the distribution we want to sample from 
    Ri0 is the vector of observed rate density (in months)
    theta is vector of parameter values

    Compute RHS of eqn (4)
    '''
    pass
    #TODO 

def q(u_theta,v_theta):
    '''
    proposal distribution q, assume Gaussian with mean vector (u_theta,v_theta) and covariance sigma
    '''
    #TODO

def sample_q(u_theta):
    '''
    returns a sample from q(u_theta,*)
    '''
    #TODO 
    pass

def acc(u_theta,v_theta,Ri0):
    '''
    Acceptance probability of being at sample u_theta and transitioning to sample v_theta
    '''
    return min(1,pi(Ri0,v_theta)*q(v_theta,u_theta)/(pi(Ri0,u_theta)*q(u_theta,v_theta)))

def sample_pi0(M):
    '''
    Initial distribution of parameters; paper assumes uniform
    returns M samples from this dist in 4xM np array
    '''
    #TODO
    pass

def EnMCMC(M,N):
    '''
    Implements ensemble MCMC sampling: draws M samples but doing a burn-in of N steps each
    '''
    thetas_all = np.zeros((4,M))
    thetas0 = sample_pi0(M)
    for m in range(M):
        theta0 = thetas0[:,m]
        theta_n = theta0
        for n in range(N):
            theta_v = sample_q(theta_n)
            acc_p = acc(theta_n,theta_v,Ri0)
            p = random.rand()
            if p < acc_p:
                theta_n = theta_v
        
        thetas_all[:,m] = theta_n
    
    return thetas_all

    

