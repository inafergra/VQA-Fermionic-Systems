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
    J_L = init_syk_tensor(N, J = J)
    J_R = J_L
    H_int = 1j* mu * np.identity(2*N)
    return J_L, J_R, H_int

#J = init_syk_tensor(5)
#print(J[1,2,3,4], J[3,4,1,2])


def energy_inter(H_int):
    '''
    Calculates the interaction energy of the TFD Hamiltonian. The energy is taken w.r.t to 
    the |phi> state. 
    '''
    N = int(np.size(H,axis=0)/2)
    energy = 0
    for a in range(N):
            for b in range(N):
                for i in range(N):
                        for j in range(N): 
                            for l in range(N):
                                for k in range(N):
                                    for alpha_i in range(2):
                                        for alpha_j in range(2):
                                            for alpha_k in range(2):
                                                    for alpha_l in range(2): 
                                                        for alpha_a in range(2):
                                                            for alpha_b in range(2):
                                                                energy += H[2*a+alpha_a,2*b+alpha_b] * (1j**(alpha_j + alpha_k)*(-1)**(alpha_b + alpha_k) + 1j**(alpha_i + alpha_l)*(-1)**(alpha_a + alpha_l))
    return energy


def mat_elem_syk(J, m1, m2):
    """
    Returns the energy of the independent SYK models w.r.t. the |phi> state
    """
    N = int(np.size(H,axis=0)/2)
    mat_elem = 0
    for i in range(N):
            for j in range(N):
                for l in range(N):
                    for k in range(N):
                        for alpha in range(0,2):
                            for beta in range(0,2):
                                for gamma in range(0,2):
                                    for delta in range(0,2):
                                        x = J[2*l+alpha, 2*k+beta, 2*i+delta, 2*j+gamma] * 1j**(alpha+beta+gamma+delta)
                                        if (l==k==m1==m2==i==j) or  (l==k==m1==j!=i==m2) or (k==m1!=l==i==j==m2) or (k==m1!=l==j!=i==m2):
                                            mat_elem += x*(-1)**(alpha*delta)
                                        elif (l==k==m1==m2!=i==j) or  (l==k==m1==i!=j==m2) or (k==m1!=l==m2!=i==j) or (k==m1!=l==i!=j==m2):
                                            mat_elem += x*(-1)**(alpha+gamma)
                                        elif (l==k!=m1==m2==i==j) or  (l==k!=m1==j!=i==m2) or (l==m1!=k==i==j==m2) or (l==m1!=k==j!=i==m2):
                                            mat_elem += x*(-1)**(beta+delta)
                                        elif (l==k!=m1==m2!=i==j) or (l==k!=m1==i!=j==m2) or (l==m1!=k==m2!=i==j) or (l==m1!=k==i!=j==m2):
                                            mat_elem += x*(-1)**(beta+gamma)
                                        elif (l==i!=k==j!=m1==m2) or (l==m2!=k==j!=m1==i) or (l==i!=k==m2!=m1==j):
                                            mat_elem += -x*(-1)**(alpha+beta)
                                        elif (l==j!=k==i!=m1==m2) or (l==m2!=k==i!=m1==j) or (l==j!=k==m2!=m1==i):
                                            mat_elem += x*(-1)**(alpha+beta)
    return mat_elem


def energy_syk(J):
    '''
    Computes the energy TO THE FIRST ORDER of the state given by doing an X rotation of theta[i] in qubit i. 
    args:
        theta: 1darray (len = number of fermions/qubits N)
    '''
    N = int(np.size(H,axis=0)/2)
    e = 0
    for m1 in range(N):
        for m2 in range(N):
            e += theta[m1]*theta[m2]*ham_matrix_element(H, m1, m2)
    #print(e)
    return energy(H) + e

def tfd_energy(J_L, J_R, H_int):
    return (syk_energy(J_R) + syk_energy(J_L) + inter_energy(H_int))

def apply_unitary(J_L, J_R, H_int, t, subsystem, indices):
    '''
    Calculates the new matrix of coefficients of the interaction term after applying a left h.
    '''
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

    print(len(A))
    print(len(exp_A))
    if subsystem == 'R':
        #Compute new interaction matrix of coefficients
        H_int = np.matmul(H_int,exp_A)

        #Compute new matrix for the R system
        #J_R = np.tensordot(np.tensordot(np.tensordot(np.tensordot(J_R, exp_A, axes=([0,1])), exp_A, axes=([0,1])), exp_A, axes=([0,1])), exp_A, axes=([0,1]))
        J_R = np.einsum('ijkl, ...i, ...j, ...k, ...l', J_R, exp_A, exp_A, exp_A, exp_A)


    elif subsystem == 'L':
        #Compute new interaction matrix of coefficients
        H_int = exp_A_T @ H_int 

        #Compute new matrix for the L system
        #J_L = np.tensordot(np.tensordot(np.tensordot(np.tensordot(J_L, exp_A, axes=([0,1])), exp_A, axes=([0,1])), exp_A, axes=([0,1])), exp_A, axes=([0,1]))
        J_L = np.einsum('ijkl, ...i, ...j, ...k, ...l', J_L, exp_A, exp_A, exp_A, exp_A)

    return J_L, J_R, H_int

