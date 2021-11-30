# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 14:07:57 2021

@author: Hojjat
"""
import numpy as np
import matplotlib.pylab as plt
# Identified values in the paper :
    # r=5e-6
    # Asig=0.006e6
    # t_a=8700
    # Ds_c=0.17e6

def FindR(Ds_t,u,time):
    # u s is parameters
    r=u[0]
    t_a=u[1]
    Asigma=u[2]
    Ds_c=u[3]
    R=np.zeros(Ds_t.shape)
    y=np.exp((Ds_t-Ds_c)/Asigma)
    for i in range(0,Ds_t.size):
        Num=r*y[i]
        Den=(np.trapz(y[0:i+1],time[0:i+1])/t_a)+1
        R[i]=Num/Den
    return R

def GenLinDs_t(time,u,m):
    Ds_c=u[3]
    Ds_t=m*time+Ds_c
    return Ds_t
    
def GenbiLinDs_t(time,u,m):
    t_max= time.size
    t_half=np.int(t_max/2)
    print(t_half)
    Ds_c=u[3]
    Ds_t= np.zeros(time.shape)
    Ds_t[0:t_half]=m*time[0:t_half]+Ds_c
    Ds_t[t_half:] = Ds_t[t_half-1]
    return Ds_t

def StatePerturb(u,u_min,u_max,u_sigma):
    # u_range=u_max-u_min 
    # u_sigma= u_range/200 # the sigma of the purterbation in Particle filter
    # We can change the value and specify for each u[i]

    Noise=np.zeros(u.shape)
    u_new=np.zeros(u.shape)
    for i in range(0,u.size):
        Noise[i]=np.random.normal(0,u_sigma[i])
        u_new[i]=u[i]+Noise[i]
        # while u_new[i]<u_min[i] or u_new[i]>u_max[i]: # To ensure to be inside the range
        #     Noise[i]=np.random.normal(0,u_sigma[i])
        #     u_new[i]=u[i]+Noise[i]

    return u_new

def FindLikelihood (u,R0,Ds_t,time):
    R = FindR(Ds_t,u,time)
    w= np.exp(-0.5*(np.linalg.norm(R-R0))**2) #1351*0.25*
    return w

def FindWeights (U,R0,Ds_t,time):
    N = U.shape[0]
    weights = np.zeros(N)
    for i in range(N):
        weights[i] = FindLikelihood (U[i,:],R0,Ds_t,time)
        
    norm_weights = weights/np.sum(weights)
    
    return norm_weights
    

def ResampleParticles (U,weights):
    N = U.shape[0]

    indices = np.random.choice(N, N, replace=True, p=weights)
    
    U_new = U[indices]
    
    return U_new

def ParticleFilterStep (U,R0,Ds_t,time,u_min,u_max,u_sigma):
    N = U.shape[0]
    U1 = np.zeros(U.shape)
    for i in range(N):
        U1[i,:] = StatePerturb(U[i,:],u_min,u_max,u_sigma)
    
    w = FindWeights (U1,R0,Ds_t,time)
    
    print(w)
    
    U2 = ResampleParticles (U1,w)
    
    return U2

def ParticleFilter (u_min,u_max,U0,R0,Ds_t,time,iters):
    N,M = U0.shape
    # U_history = np.zeros((iters+1,N,M))
    # U_history[0,:,:] = U0
    # for i in range (iters):
    #     print(i)
    #     U_history[i+1,:,:] = ParticleFilterStep (U_history[i,:,:],R0,Ds_t,time,u_min,u_max)
    
    # return U_history

    U = U0
    for i in range (iters):
        print(i)
        u_sigma = (u_max-u_min)/(40*np.sqrt(i+1))
        U = ParticleFilterStep (U,R0,Ds_t,time,u_min,u_max,u_sigma)
    
        if i%50==0:
            for m in range(4):
                plt.show()
                plt.hist(U[:,m],bins=100, range=(u_min[m], u_max[m]))
                plt.title( "Variable: %d Iteration: %d" %(m,i))
                plt.show()
    return U

def GetU0_Normal (N,u,u_min,u_max):
    
    u_range=u_max-u_min 
    u_sigma=u_range/5
    
    cov = np.diag(u_sigma)
    
    U0 = np.random.multivariate_normal(u,cov,N)
    
    return U0

def GetU0_Uniform (N,M,u_min,u_max):
    
    return np.random.uniform(u_min,u_max,[N,M])



def MonteCarloStep (u,R0,Ds_t,time,u_min,u_max,u_sigma):
    
    M = u.size
    cov = np.diag(u_sigma)
    v = np.random.multivariate_normal(u,cov,1)
    # print(n.shape)
    # v = u + n
    v = v.reshape(M,)
    
    like_ratio = FindLikelihood(v,R0,Ds_t,time)/FindLikelihood(u,R0,Ds_t,time)

    acc_prob = np.min([like_ratio,1])
    # print(acc_prob)
    if (np.random.uniform() <= acc_prob):
        u_new = v
    else:
        u_new=  u
        
    # print(FindLikelihood (u_new,R0,Ds_t,time))
    # print(u_new-u)
            
    return u_new

def MonteCarloSingleChain (u_min,u_max,u0,R0,Ds_t,time,iters):
    
    M = u0.size
    u_history = np.zeros((iters+1,M))
    
    u=u0
    u_history[0,:] = u
    
    u_sigma = (u_max-u_min)/75
    
    for i in range (iters):
        # print(i)
        
        u = MonteCarloStep (u,R0,Ds_t,time,u_min,u_max,u_sigma)
        u_history[i+1,:] = u
        
        if i%1000==0 and i!=0:
            for m in range(4):
                plt.show()
                plt.hist(u_history[i-200:i+1,m],bins=100) #, range=(u_min[m], u_max[m])
                plt.title( "Variable: %d Iteration: %d" %(m,i))
                plt.show() 
                
            U_retain = u_history[i-1000:i+1,:]
            wts = FindWeights (U_retain,R0,Ds_t,time)
            u_pred = np.dot(wts,U_retain)
            
            R_pred= FindR(Ds_t,u_pred,time)
            
            plt.plot(time,R0, label='Actual')
            plt.plot(time,R_pred, label='Predicted')
            plt.xlabel("Time (Years)")
            plt.ylabel("R")
            plt.title("")
            plt.legend()
            plt.show()
        
    return u_history

def M_coarse_DS(DS,fac,dates):
    '''
    coarsifies the DS data by a factor of fac:
    # DS is smoothed coloub stress
    Ex: if fac = 4, then at each time the DS data is averaged over blocks of size 4x4
    Also bins by years

    Returns: data spatially coarsified by a factor of fac and time-coarsified into years
    '''
    DX,DY,a=np.shape(DS)
    Dx_new = DX//fac
    Dy_new = DY//fac
    data_counts, years = np.histogram(dates, np.linspace(1956,2019,(2019-1956)+1))
    coarse_data = np.zeros((Dx_new,Dy_new,len(years)-1))
    stop_i = 0
    for ti,t in enumerate(data_counts):
      stop_i+= t
      for i in range(Dx_new):
        for j in range(Dy_new):
          coarse_data[i,j,ti] = sum(DS[i*fac:(i+1)*fac,j*fac:(j+1)*fac,stop_i - t:stop_i].flatten())/(fac**2*t)
   
    return coarse_data