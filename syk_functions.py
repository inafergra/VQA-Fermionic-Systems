import numpy as np
import matplotlib.pyplot as plt
import itertools as itertools
from scipy.optimize import minimize, differential_evolution
#np.set_printoptions(precision=2)
import pdb
from itertools import combinations

from matrix_elements import *

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

def init_TFD_model(N, J, mu):
    '''
    '''
    L_indices = range(0, 2*N)
    R_indices = range(2*N, 4*N)
    syk_L_indices = list(combinations(L_indices, 4))
    syk_R_indices = list(combinations(R_indices, 4))
    interaction_indices = [(l, r) for l, r in zip(L_indices, R_indices)]

    # dictionaries with the couplings
    var = 6*J**2/N**3
    couplings = np.random.normal(scale=np.sqrt(var), size=len(syk_L_indices))
    J_L_dict = {ind:couplings[i] for ind,i in zip(syk_L_indices,range(len(couplings)))}
    J_R_dict = {ind:couplings[i] for ind,i in zip(syk_R_indices,range(len(couplings)))}
    H_int_dict = {ind: 1j * mu for ind in interaction_indices}

    # tensors
    J_L = init_syk_tensor(J_L_dict,N)
    J_R = J_L
    H_int = 1j* mu * np.identity(2*N)

    tensor_list = [J_L, J_R, H_int]
    dict_list = [J_L_dict,J_R_dict,H_int_dict]

    return tensor_list, dict_list


def syk_energy(J):
    '''
    Returns the energy of the independent SYK models w.r.t. the |phi> state
    '''
    N = int(np.size(J,axis=0)/2)
    energy = 0
    for i in range(N):
        for j in range(N):
            for a in range(N):
                for b in range(N):
                    for c in range(N):
                        for d in range(N):
                            for alpha_i in range(2):
                                for alpha_j in range(2):
                                    for alpha_a in range(2):
                                        for alpha_b in range(2):
                                            for alpha_c in range(2):
                                                for alpha_d in range(2):
                                                    energy += J[2*a+alpha_a, 2*b+alpha_b, 2*c+alpha_c, 2*d+alpha_d] * \
                                                    ( (1j)**(alpha_j+alpha_i)*(-1)**(alpha_j)* \
                                                    mat_element_4maj_exc(i,j,a,alpha_a,b,alpha_b,c,alpha_c,d,alpha_d) \
                                                    + mat_elem_2maj(i, alpha_i,j, alpha_j) * \
                                                    mat_elem_4maj(a,alpha_a,b,alpha_b,c,alpha_c,d,alpha_d) )
    return (1/(4*N)**2)*energy

def interaction_energy(H_int):
    '''
    Calculates the interaction energy of the TFD Hamiltonian. The energy is taken w.r.t to 
    the |phi> state. 
    '''
    N = int(np.size(H_int,axis=0)/2)
    energy = 0
    for a in range(N):
            for b in range(N):
                for i in range(N):
                        for j in range(N): 
                                for alpha_a in range(2):
                                    for alpha_b in range(2):
                                        for alpha_i in range(2):
                                            for alpha_j in range(2):
                                                energy += H_int[2*a+alpha_a,2*b+alpha_b] * 1j**(1+alpha_a + alpha_b + alpha_i + alpha_j) * ((-1)**(alpha_b + alpha_j) + (-1)**(alpha_a + alpha_j))
                                                #energy += H_int[2*a+alpha_a,2*b+alpha_b] * (mat_elem_2maj(a,alpha_a,j,alpha_j)*mat_elem_2maj(i,alpha_i,b,alpha_b) + mat_elem_2maj(i,alpha_i,a,alpha_a)*mat_elem_2maj(b,alpha_b,j,alpha_j))    
    return (1/(4*N)**2)*energy

def tfd_energy(TFD_model):
    J_L = TFD_model[0]
    J_R = TFD_model[1]
    H_int= TFD_model[2]

    #print(syk_energy(J_R))
    #print(syk_energy(J_L))
    #print(interaction_energy(H_int))

    # Have to cast to real because there is a small imaginary part not cancelling from numerical imprecisions
    return np.real(syk_energy(J_R) + syk_energy(J_L) + interaction_energy(H_int))

def apply_unitary(TFD_model, t, subsystem, indices):
    '''
    Calculates the new matrix of coefficients of the interaction term after applying a left h.
    '''
    J_L = TFD_model[0]
    J_R = TFD_model[1]
    H_int= TFD_model[2]

    N = int(np.size(J_L,axis=0)/2)
    i = indices[0] ; alpha = indices[1]; j = indices[2]; beta = indices[3]

    #Compute the A matrix
    A = np.zeros([2*N,2*N])
    A[2*i+alpha,2*j+beta] = 2
    A[2*j+beta,2*i+alpha] = -2

    #Compute exp(A)
    I = np.identity(2*N)
    A2 = np.matmul(A,A)
    exp_A = I + np.sin(2*t)*A/2 - (np.cos(2*t)-1)*A2/4
    exp_A_T = I - np.sin(2*t)*A/2 - (np.cos(2*t)-1)*A2/4

    if subsystem == 'R':
        #Compute new interaction matrix of coefficients
        #H_int_ = np.matmul(H_int,exp_A)
        H_int = np.einsum('ij,jk->ik', H_int, exp_A)
        #print(H_int_==H_int_)

        #Compute new matrix for the R system
        #J_R = np.tensordot(np.tensordot(np.tensordot(np.tensordot(J_R, exp_A, axes=([0,1])), exp_A, axes=([0,1])), exp_A, axes=([0,1])), exp_A, axes=([0,1]))
        J_R = np.einsum('ijkl, mi, nj, ok, pl->mnop', J_R, exp_A, exp_A, exp_A, exp_A)

    elif subsystem == 'L':
        #Compute new interaction matrix of coefficients
        #H_int_ = exp_A_T @ H_int 
        H_int = np.einsum('ij,ik->jk', H_int, exp_A)
        #print(H_int_==H_int_)

        #Compute new matrix for the L system
        #J_L = np.tensordot(np.tensordot(np.tensordot(np.tensordot(J_L, exp_A, axes=([0,1])), exp_A, axes=([0,1])), exp_A, axes=([0,1])), exp_A, axes=([0,1]))
        J_L = np.einsum('ijkl, mi, nj, ok, pl->mnop', J_L, exp_A, exp_A, exp_A, exp_A)
    
    TFD_model = [J_L, J_R, H_int]
    return TFD_model

def new_tfd_energy(t, indices, subsystem, TFD_model):
    '''
    Function to input in the minimization loop. Computes the energy of the coupling matrices after 
    going through the transformation h[indices] for a time t.
    '''

    TFD_model = apply_unitary(TFD_model, t, subsystem, indices)
    new_energy = tfd_energy(TFD_model)
    return new_energy