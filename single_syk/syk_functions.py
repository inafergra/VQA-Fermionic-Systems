import numpy as np
import matplotlib.pyplot as plt
import itertools as itertools
from scipy.optimize import minimize, differential_evolution
#np.set_printoptions(precision=2)
import pdb
from itertools import combinations
from math import floor
from matrix_elements import *
import pdb


def init_syk_tensor(J_dict, N):
    '''
    Builds the initial matrix of coefficients with elements draw from a normal distribution with mean=mean 
    variance=6*J**2/N**3.
    '''
    #variance = 1
    J = np.zeros([2*N,2*N,2*N,2*N])
    for ind in list(combinations(range(0, 2*N), 4)):
        a = ind[0]
        b = ind[1]
        c = ind[2]
        d = ind[3]
        #print(ind)
        J[a,b,c,d] = J_dict[ind]
        
        J[b,c,d,a] = -J[a,b,c,d]
        J[c,d,a,b] = J[a,b,c,d]
        J[d,a,b,c] = -J[a,b,c,d]

        J[b,a,c,d] = -J[a,b,c,d]
        J[a,c,d,b] = J[a,b,c,d]
        J[c,d,b,a] = -J[a,b,c,d]
        J[d,b,a,c] = J[a,b,c,d]

        J[c,b,a,d] = -J[a,b,c,d]
        J[b,a,d,c] = J[a,b,c,d]
        J[a,d,c,b] = -J[a,b,c,d]
        J[d,c,b,a] = J[a,b,c,d]

        J[d,b,c,a] = -J[a,b,c,d]
        J[b,c,a,d] = J[a,b,c,d]
        J[c,a,d,b] = -J[a,b,c,d]
        J[a,d,b,c] = J[a,b,c,d]

        J[a,c,b,d] = -J[a,b,c,d]
        J[c,b,d,a] = J[a,b,c,d]
        J[b,d,a,c] = -J[a,b,c,d]
        J[d,a,c,b] = J[a,b,c,d]

        J[a,d,c,b] = -J[a,b,c,d]
        J[d,c,b,a] = J[a,b,c,d]
        J[c,b,a,d] = -J[a,b,c,d]
        J[b,a,d,c] = J[a,b,c,d]
        
        J[a,b,d,c] = -J[a,b,c,d] 
        J[b,d,c,a] = J[a,b,c,d]
        J[d,c,a,b] = -J[a,b,c,d]
        J[c,a,b,d] = J[a,b,c,d]
    return J


def init_TFD_model(N, J):
    '''
    '''
    syk_L_indices = list(combinations(range(0, 2*N), 4))

    # dictionary
    var = 6*J**2/N**3
    couplings = np.random.normal(scale=np.sqrt(var), size=len(syk_L_indices))
    J_dict = {ind:couplings[i] for ind,i in zip(syk_L_indices,range(len(couplings)))}

    # tensor
    J = init_syk_tensor(J_dict,N)

    return J, J_dict

def syk_energy(J):
    '''
    Returns the energy of the independent SYK models w.r.t. the vacuum
    '''
    
    N = int(np.size(J,axis=0)/2)
    energy = 0
    for a in range(N):
        for b in range(N):
            for c in range(N):
                for d in range(N):
                    for alpha_a in range(2):
                        for alpha_b in range(2):
                            for alpha_c in range(2):
                                for alpha_d in range(2):
                                    #for alpha_i in range(2):
                                    #    for alpha_j in range(2):
                                    #print(alpha_a,alpha_b,alpha_c,alpha_d)
                                    #i = -1
                                    #j = -1
                                    #energy += J[2*a+alpha_a, 2*b+alpha_b, 2*c+alpha_c, 2*d+alpha_d] * mat_element_4maj_exc(i,j,a, alpha_a, b, alpha_b, c, alpha_c, d, alpha_d)
                                    energy += J[2*a+alpha_a, 2*b+alpha_b, 2*c+alpha_c, 2*d+alpha_d] * mat_elem_4maj(a, alpha_a, b, alpha_b, c, alpha_c, d, alpha_d)
    #print(energy)
    return energy
"""

def syk_energy(J):
    '''
    Returns the energy of the independent SYK models w.r.t. the |phi> state
    '''
    N = int(np.size(J,axis=0)/2)
    energy = 0
    for i in range(N):
        for j in range(N):
            #for alpha_i in range(2):
            #    for alpha_j in range(2):
            for a in range(N):
                for b in range(N):
                    for c in range(N):
                        for d in range(N):
                            for alpha_a in range(2):
                                for alpha_b in range(2):
                                    for alpha_c in range(2):
                                        for alpha_d in range(2):
                                            #energy +=  J[2*a+alpha_a, 2*b+alpha_b, 2*c+alpha_c, 2*d+alpha_d] * \
                                            #    ( mat_element_6maj(i,alpha_i,a,alpha_a,b,alpha_b,c,alpha_c,d,alpha_d,j,alpha_j) \
                                            #    + mat_elem_2maj(i,alpha_i,j,alpha_j) * mat_elem_4maj(a,alpha_a,b,alpha_b,c,alpha_c,d,alpha_d) )
                                            #print(alpha_a,alpha_b,alpha_c,alpha_d)
                                            
                                            energy += J[2*a+alpha_a, 2*b+alpha_b, 2*c+alpha_c, 2*d+alpha_d] * \
                                            mat_element_4maj_exc(i,j,a,alpha_a,b,alpha_b,c,alpha_c,d,alpha_d)
                                            #( mat_element_4maj_exc(i,j,a,alpha_a,b,alpha_b,c,alpha_c,d,alpha_d) \
                                            #+ kdf(i,j) * mat_elem_4maj(a,alpha_a,b,alpha_b,c,alpha_c,d,alpha_d) )
                                            
                                            #energy += J[2*a+alpha_a, 2*b+alpha_b, 2*c+alpha_c, 2*d+alpha_d] * \
                                            #( (1j)**(alpha_j+alpha_i)*(-1)**(alpha_j)* \
                                            #mat_element_4maj_exc(i,j,a,alpha_a,b,alpha_b,c,alpha_c,d,alpha_d) \
                                            #+ mat_elem_2maj(i, alpha_i,j, alpha_j) * \
                                            #mat_elem_4maj(a,alpha_a,b,alpha_b,c,alpha_c,d,alpha_d) )
    #print(energy)
    return energy/(N)
"""

def tfd_energy(J):
    
    #print('Energy', syk_energy(J))
    
    # Have to cast to real because there is a small imaginary part not cancelling from numerical imprecisions
    return np.real(syk_energy(J))

def apply_unitary(J, t, indices):
    '''
    Calculates the new matrix of coefficients of the interaction term after applying a left h.
    '''
    
    N = int(np.size(J,axis=0)/2)
    i = indices[0] ; alpha = indices[1]; j = indices[2]; beta = indices[3]

    #Compute the A matrix
    A = np.zeros([2*N,2*N])
    A[2*i+alpha,2*j+beta] = 2
    A[2*j+beta,2*i+alpha] = -2

    #Compute exp(A)
    I = np.identity(2*N)
    A2 = np.matmul(A,A)
    exp_A = I + np.sin(2*t)*A/2 - (np.cos(2*t)-1)*A2/4
    #exp_A_T = I - np.sin(2*t)*A/2 - (np.cos(2*t)-1)*A2/4

    #Compute new matrix 
    J = np.einsum('ijkl, mi, nj, ok, pl->mnop', J, exp_A, exp_A, exp_A, exp_A)

    return J

def new_tfd_energy(t, indices, J):
    '''
    Function to input in the minimization loop. Computes the energy of the coupling matrices after 
    going through the transformation h[indices] for a time t.
    '''
    J = apply_unitary(J, t, indices)
    new_energy = tfd_energy(J)
    return new_energy