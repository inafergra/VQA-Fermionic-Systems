import numpy as np
import matplotlib.pyplot as plt
import itertools as itertools
from scipy.optimize import minimize, differential_evolution
#np.set_printoptions(precision=2)
import pdb; 

def energy(H):
    '''
    Calculates the energy of a given coupling matrix H. The energy is always taken w.r.t the
    fermionic vacuum state. 
    '''
    N = int(np.size(H,axis=0)/2)
    energy = 0
    for l in range(N):
        energy -= H[2*l+1,2*l]
    return energy

def energy_excited_state(H, k):
    '''
    Calculates the energy of a given coupling matrix H. The energy is taken w.r.t. the state defined by 
    the binary array k.
    '''
    N = int(np.size(H,axis=0)/2)
    excited_states = np.flatnonzero(k)
    #print(excited_states)
    m = 0
    for l in range(N):
        if l in excited_states:
            m -= H[2*l+1,2*l]
        else:
            m += H[2*l+1,2*l]
    energy = 0.5*(-2*m)
    return energy

def squared_hamiltonian_average(H):
    ''' Calculates the average of the squared hamiltonian with respect to the vacuum '''
    N = int(np.size(H,axis=0)/2)

    x=0
    for i in range(N):
        for j in range(N):
            for l in range(N):
                for k in range(N):
                    for alpha in range(0,2):
                        for beta in range(0,2):
                            for gamma in range(0,2):
                                for delta in range(0,2):
                                    if (i==j) and (l==k):
                                        x += H[2*l+alpha, 2*k+beta]*H[2*i+delta, 2*j+gamma] * 1j**(alpha+beta+gamma+delta) * (-1)**(gamma+beta)
                                    elif (l!=k) and (i==l) and (k==j):
                                        x += H[2*l+alpha, 2*k+beta]*H[2*i+delta, 2*j+gamma] * 1j**(alpha+beta+gamma+delta) * (-1)**(alpha+beta)
                                    elif (l!=k) and (j==l) and (k==i):
                                        x += H[2*l+alpha, 2*k+beta]*H[2*i+delta, 2*j+gamma] * 1j**(alpha+beta+gamma+delta) * (-1)**(alpha+beta)
    return -1/4*x

def squared_hamiltonian_average_excited_state(H, x):
    ''' Calculates the average of the squared hamiltonian with respect to the excited state x '''
    N = int(np.size(H,axis=0)/2)
    
    excited_states = (np.flatnonzero(x))
    sq_ham = 0
    for i in range(N):
        for j in range(N):
            for l in range(N):
                for k in range(N):
                    for alpha in range(0,2):
                        for beta in range(0,2):
                            for gamma in range(0,2):
                                for delta in range(0,2):
                                    x = H[2*l+alpha, 2*k+beta]*H[2*i+delta, 2*j+gamma] * 1j**(alpha+beta+gamma+delta)
                                    if (l not in excited_states) and (k not in excited_states) and (i not in excited_states) and (j not in excited_states):
                                    #if (l and k and i and j not in excited_states):
                                        if (l!=k) and (i==l) and (k==j):
                                            sq_ham += x*(-1)**(alpha+beta)
                                        elif (l!=k) and (j==l) and (k==i):
                                            sq_ham += x*(-1)**(alpha+beta)
                                        elif (i==j) and (l==k):
                                            sq_ham += x*(-1)**(beta+gamma)
                                    #elif (j and l in excited_states) and (k and i not in excited_states):
                                    elif (j in excited_states) and (l in excited_states) and (k not in excited_states) and (i not in excited_states):
                                        if (l!=k) and (j==l) and (k==i):
                                            sq_ham += x*(-1)**(beta+gamma)
                                    #elif (i and l in excited_states) and (k and j not in excited_states):
                                    elif (i in excited_states) and (l in excited_states) and (k not in excited_states) and (j not in excited_states):
                                        if (l!=k) and (i==l) and (k==j):
                                            sq_ham += x*(-1)**(beta+delta)
                                    #elif (i and j in excited_states) and (l and k not in excited_states):
                                    elif (i in excited_states) and (j in excited_states) and (l not in excited_states) and (k not in excited_states):
                                        if (i==j) and (l==k):
                                            sq_ham += x*(-1)**(beta+delta)
                                    #elif (k and i in excited_states) and (j and l not in excited_states):
                                    elif (k in excited_states) and (i in excited_states) and (j not in excited_states) and (l not in excited_states):
                                        if (l!=k) and (j==l) and (i==k):
                                            sq_ham += x*(-1)**(alpha+delta)
                                    #elif (l and k in excited_states) and (j and i not in excited_states):
                                    elif (l in excited_states) and (k in excited_states) and (j not in excited_states) and (i not in excited_states):
                                        if (l==k) and (i==j):
                                            sq_ham += x*(-1)**(alpha+gamma)
                                    #elif (j and k in excited_states) and (l and i not in excited_states):
                                    elif (j in excited_states) and (k in excited_states) and (l not in excited_states) and (i not in excited_states):
                                        if (l!=k) and (i==l) and (j==k):
                                            sq_ham += x*(-1)**(alpha+gamma)
                                    #elif (l and k and i and j in excited_states):
                                    elif (l in excited_states) and (k in excited_states) and (i in excited_states) and (j in excited_states):
                                        if (l!=k) and (i==l) and (k==j):
                                            sq_ham += x*(-1)**(delta+gamma)
                                        elif (l!=k) and (j==l) and (i==k):
                                            sq_ham += x*(-1)**(delta+gamma)
                                        elif (i==j) and (l==k):
                                            sq_ham += x*(-1)**(alpha+delta)
    return (-1/4)*sq_ham            

def energy_after_x_rotations(theta, H):
    '''
    Computes the energy of the state given by doing an X rotation of theta[i] in qubit i. 
    args:
        theta: 1darray (len = number of fermions/qubits)
    '''
    N = int(np.size(H,axis=0)/2)
    energy = 0
    for state in (itertools.product([0, 1], repeat=N)):
        k = np.array(state)
        coeff = 1
        for i in range(N): # coefficients
            if k[i]==0:
                coeff *= np.cos(theta[i]/2)
            else:
                coeff *= np.sin(theta[i]/2)
        energy += coeff*energy_excited_state(H, k)
    return energy

def squared_hamiltonian_after_x_rotations(theta, H):
    N = int(np.size(H,axis=0)/2)
    sq_ham = 0
    for state in list(itertools.product([0, 1], repeat=N)):
        k = np.array(state)
        coeff = 1
        for i in range(N): # coefficients
            if k[i]==0:
                coeff *= np.cos(theta[i]/2)
            else:
                coeff *= np.sin(theta[i]/2)
        sq_ham += coeff*squared_hamiltonian_average_excited_state(H, k)
    return sq_ham

def variance(theta, H):
    var = squared_hamiltonian_after_x_rotations(theta, H) - energy_after_x_rotations(theta, H)**2
    return np.abs(np.real(var))



def init_coeff_matrix(N, mean=0, variance=1):
    '''
    Builds the initial (antisymmetric) matrix of coefficients with elements draw from a normal distribution with mean=mean 
    variance=6*H_value**2/N**3. The elements of the matrix H are indexed by 4 indices (i,alpha,j,beta) as
    H_{2*i+alpha,2*j+beta}.
    '''
    #variance = 6*H_value**2/N**3
    #variance = 1
    H = np.zeros([2*N,2*N])
    for i in range(N):
        for j in range(i,N): #H[i,alpha,i,beta] is zero
            for alpha in range(0,2):
                for beta in range(0,2):
                    if (2*i+alpha!=2*j+beta): #diagonal is zero
                        H[2*i+alpha,2*j+beta] = np.random.normal(mean, variance)
                        H[2*j+beta,2*i+alpha] = -H[2*i+alpha,2*j+beta]
    return H

def appy_h_gate(t,H, indices):
    '''
    Gives the new coupling matrix after applying the h=e^{i*c_{i,alpha}*c_{j,beta}*t} transformation
    for a time t.
    '''

    N = int(np.size(H,axis=0)/2)
    i = indices[0] ; alpha = indices[1]; j = indices[2]; beta = indices[3]

    #Compute the A matrix
    A = np.zeros([2*N,2*N])
    A[2*i+alpha,2*j+beta] = 2
    A[2*j+beta,2*i+alpha] = -2
    #print('A is:')
    #print(A)

    #Compute exp(A)
    I = np.identity(2*N)
    A2 = np.matmul(A,A)
    exp_A = I + np.sin(2*t)*A/2 - (np.cos(2*t)-1)*A2/4
    exp_A_T = I - np.sin(2*t)*A/2 - (np.cos(2*t)-1)*A2/4
    #print('expA*expAT:')
    #print(np.matmul(exp_A,exp_A_T)) #Should be the identity!

    #Compute the new coupling matrix
    new_H = np.matmul(np.matmul(exp_A_T,H),exp_A)

    return new_H

def new_energy(optimal_time,H,indices):
    '''
    Function to input in the minimization loop. Computes the energy of the coupling matrix after 
    going through the transformation h[indices] for a time optimal_time.
    '''
    new_H = appy_h_gate(optimal_time,H,indices)
    new_energy = energy(new_H)
    return new_energy

def exact_energy_levels(H, k):
    '''
    Finds the first k exact energy levels through exact diagonalization
    '''
    energies = np.zeros(k)

    eig_vals, O = np.linalg.eigh(np.matmul(H,H))
    #print(eig_vals) #-(epsilon_k^2)
    #parity = np.linalg.det(O)
    #print(O)
    #print(f'Determinant of O: {parity}')
    epsilons = np.sqrt(abs(eig_vals))
    energies[0] = -np.sum(epsilons)/2

    a  = np.argmin(epsilons) 
    energies[1] = energies[0] + 2*abs(epsilons[a])
    #for i in range(1, k-1):
    #    a  = np.argmin(epsilons) 
    #    
    #    epsilons[a]+= -energies[0]
    #    print(epsilons[a])
    #    energies[i] = energies[i-1] + 2*epsilons[a]
    #    epsilons[a]+= -energies[0]

    return energies

#def williamson_decomposition(H):
#    ''' Returns the matrix O that takes H into W = O^T H O in block diagonal form'''
