
'''
Forward operator that maps rho_0 to rho_T
'''
import numpy as np
from scipy.linalg import expm
h_bar = 1 # actual value is 1.054571817×10-34 J-s

def forward_map(H, rho_0, T):
    '''
    Args: 
        H: Hamiltonian matrix (shape: 2^n x 2^n)
        rho_0: Initital state vector (shape: 2^n x 1)
        T: Evolution time (size: 1x1)
    Returns:
    	rho_T: State vector after T evolutions (shape: 2^n x 1)
    '''

    rho_T = expm(-1j*H*T/h_bar)*rho_0

    return rho_T