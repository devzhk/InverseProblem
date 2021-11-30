#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:06:14 2021

@author: pranavkulkarni
"""

import numpy as np
import Functions
import matplotlib.pylab as plt


r = 5e-6
t_a = 8700
Asig = 0.006e6
Ds_c = 0.17e6

# Defining ranges of priors
r_min = 4.7e-6  # 6.2e-7
r_max = 5.3e-6  # 2.5e-5

t_a_min = 6000  # .5
t_a_max = 1e4

Asig_min = 0.003e6  # 1e3
Asig_max = 0.01e6  # 1e6

Ds_c_min = 0.1e6  # 0
Ds_c_max = 0.3e6  # .5e6

r_range = r_max-r_min
t_a_range = t_a_max-t_a_min
Asig_range = Asig_max-Asig_min
Ds_c_range = Ds_c_max-Ds_c_min


u = np.array([r, t_a, Asig, Ds_c])
u_min = np.array([r_min, t_a_min, Asig_min, Ds_c_min])
u_max = np.array([r_max, t_a_max, Asig_max, Ds_c_max])

m = Asig
time = np.linspace(0, 20, 100)

Ds_t = Functions.GenLinDs_t(time, u, m)
R0 = Functions.FindR(Ds_t, u, time)
plt.plot(time, R0)
plt.show()
plt.plot(time, Ds_t)


# %% Ensemble monte carlo

n_chains = 1000
n_retain = 100
n_iters = 500

U_final = np.zeros((n_retain*n_chains, 4))

for chain_n in range(n_chains):

    print(chain_n)

    u0 = Functions.GetU0_Uniform(1, u_min.size, u_min, u_max).reshape(4,)
    # u0 = 1.05*u

    U = Functions.MonteCarloSingleChain(
        u_min, u_max, u0, R0, Ds_t, time, n_iters)

    U_final[chain_n*n_retain:(chain_n+1)*n_retain,
            :] = U[n_iters-n_retain+1:n_iters+1, :]

# u_pred = np.mean(U[500:1000,:],0)

# R_pred=Functions.FindR(Ds_t,u_pred,time)

# plt.plot(time,R0, label='Actual')
# plt.plot(time,R_pred, label='Predicted')
# plt.xlabel("Time (Years)")
# plt.ylabel("R")
# plt.title("")
# plt.legend()
# plt.show()
# [5e-6,8700,0.17,0.006]
# U_retain = U[iters//2:iters+1,:]
wts = Functions.FindWeights(U_final, R0, Ds_t, time)
u_pred = np.dot(wts, U_final)

R_pred = Functions.FindR(Ds_t, u_pred, time)

plt.plot(time, R0, label='Actual')
plt.plot(time, R_pred, label='Predicted')
plt.xlabel("Time (Years)")
plt.ylabel("R")
plt.title("")
plt.legend()
plt.show()
