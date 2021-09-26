import numpy as np
import matplotlib.pyplot as plt
import itertools as itertools
from scipy.optimize import minimize, differential_evolution
#np.set_printoptions(precision=2)
import pdb; 

def init_syk_tensor(N, J = 1, mean=0, ):
    '''
    Builds the initial matrix of coefficients with elements draw from a normal distribution with mean=mean 
    variance=6*J**2/N**3.
    '''
    variance = 6*J**2/N**3
    #variance = 1
    J = np.zeros([2*N,2*N,2*N,2*N])
    for i1 in range(2*N):
        for i2 in range(i1, 2*N): 
            for i3 in range(i2, 2*N):
                for i4 in range(i3, 2*N):
                    if (i1!=i2 and i3!=i4):
                        J[i1,i2,i3,i4] = np.random.normal(mean, variance)
                        J[i2,i1,i3,i4] = -J[i1,i2,i3,i4]
                        J[i1,i2,i4,i3] = -J[i1,i2,i3,i4]
                        J[i3,i4,i1,i2] = J[i1,i2,i3,i4] 
    return J

def init_TFD_model(N, J, mu):
    J_L, J_R = init_syk_tensor(N, J = J)
    J_int = 1j* mu * np.identity(N)
    return J_L, J_R, J_int

#J = init_syk_tensor(5)
#print(J[1,2,3,4], J[3,4,1,2])


def inter_energy(H_int):
    '''
    Calculates the interaction energy of the TFD Hamiltonian. The energy is taken w.r.t to 
    the |phi> state. 
    '''
    N = int(np.size(H,axis=0)/2)
    energy = 0
    for a in range(2*N):
            for b in range(2*N):
                for i in range(2*N):
                        for j in range(2*N): 
                            for l in range(2*N):
                                for k in range(2*N):
                                    alpha_i = i%2
                                    alpha_j = j%2
                                    alpha_k = k%2
                                    alpha_l = l%2
                                    alpha_a = a%2
                                    alpha_b = b%2
                                    energy -= H[2*a+alpha_a,2*b+alpha_b] * (1j**(alpha_j + alpha_k)*(-1)**(alpha_b + alpha_k) + 1j**(alpha_i + alpha_l)*(-1)**(alpha_a + alpha_l))
    return energy

def syk_energy(J):
    """
    Returns the energy of the independent SYK models w.r.t. the |phi> state
    """
    N = int(np.size(H,axis=0)/2)
    energy = 0
    for i in range(2*N):
        for j in range(2*N): 
            for l in range(2*N):
                for k in range(2*N):
                    alpha_i = i%2
                    alpha_j = j%2
                    alpha_k = k%2
                    alpha_l = l%2
                    energy += 0 #need to do the calculations
    return energy

def appy_right_unitary(J_L, J_R, J_int, t, indices):
    '''
    Calculates the new matrix of coefficients of the interaction term after applying a left h.
    '''
    N = int(np.size(H,axis=0)/2)
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

    #Compute new interaction matrix of coefficients
    
    J_int = J_int


    return J_L, J_R, J_int