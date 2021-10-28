'''
This file will implement classic shadow that takes input rho, output rho tilde
'''

import numpy as np


def classical_shadow_generation(rho):
    """
    Summary:
        Implementation of classical shadow approach follow the approach of 

         Hsin-Yuan Huang, Richard Kueng, and John Preskill.
         Predicting many properties of a quantumsystem from very few measurements.
         Nature, 16, 2020.

    Inputs:
        rho         - Type: Complex ndarray (Shape: k x 2^n x 2^n)
                    - k copes of Quantum state (Hermitian matrix)
    Outputs:
        sigma       - Type: Complex ndarray (Shape: 2^n x 2^n)
                    - Classical shadow of the input state rho
                    - NOTE: May be efficiently represented with O(Kn) elements
    """

    #Initialize \sigma_k
    sigma = np.zeros_like((rho.shape[1],rho.shape[2]))

    #TODO: Can we do this in a vector format. Start with looped version
    for i in np.arange(rho.shape[0]):

        #Alias our current sample
        rho_k = np.squeeze(rho[i,:,:])

        #Draw U_k ~ \mu 
        U_k = ...

        #Apply \rho \mapsto U \rho U^\dagger
        #TODO Uncertain if .H notation works for ndarrays. could do .conjugate().T
        rho_k_trans = U_k @ rho_k @ U_k.H

        #Perform computational basis measurement of U \rho U^\dagger
        #Basis vector b_k \in {0,1}^n
        b_k = ...


        #Store an efficient representation of g_k
        #g_k = TODO decode bra-kets
        g_k = ...


        #Apply "inverse channel" \mathcal{M}^{-1} to g_k
        # \hat{\rho}_k = \mathcal{M}^{-1} (TODO)
        rho_hat_k = ...



        sigma += rho_hat_k
        pass

    #Average sigma
    sigma = sigma/rho.shape[0]



    return sigma



if __name__ == '__main__':
    #Testing of funciton here
    pass
