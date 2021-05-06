import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
np.set_printoptions(precision=2)

def energy(J):
    '''
    Calculates the energy of a given coupling matrix J. The energy is always taken w.r.t the
    fermionic vacuum state. 
    '''
    N = int(np.size(J,axis=0)/2)
    m = 0
    for l in range(N):
        m += J[2*l+1,2*l]
    energy = 0.5*(-2*m) #0.5 due to the one half factor in front  of the hamiltonian
    return energy

def init_coeff_matrix(N, mean=0, J_value=1):
    '''
    Builds the initial (antisymmetric) matrix of coefficients with elements draw from a normal distribution with mean=mean 
    variance=6*J_value**2/N**3. The elements of the matrix J are indexed by 4 indices (i,alpha,j,beta) as
    J_{2*i+alpha,2*j+beta}.
    '''
    #variance = 6*J_value**2/N**3
    variance = 1
    J = np.zeros([2*N,2*N])
    for i in range(N):
        for j in range(i,N): #J[i,alpha,i,beta] is zero
            for alpha in range(0,2):
                for beta in range(0,2):
                    if (2*i+alpha!=2*j+beta): #diagonal is zero
                        J[2*i+alpha,2*j+beta] = np.random.normal(mean, variance)
                        J[2*j+beta,2*i+alpha] = -J[2*i+alpha,2*j+beta]
    return J

def appy_h_gate(t,J, indices):
    '''
    Gives the new coupling matrix after applying the h=e^{i*c_{i,alpha}*c_{j,beta}*t} transformation
    for a time t.
    '''

    N = int(np.size(J,axis=0)/2)
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
    new_J = np.matmul(np.matmul(exp_A_T,J),exp_A)

    return new_J

def new_energy(optimal_time,new_J,indices): 
    '''
    Function to input in the minimization loop. Computes the energy of the coupling matrix after 
    going through the transformation h[indices] for a time optimal_time.
    '''
    new_J = appy_h_gate(optimal_time,new_J,indices)
    new_energy = energy(new_J)
    return new_energy

def exact_energy_levels(J, k):
    '''
    Finds the first k exact energy levels through exact diagonalization
    '''
    energies = np.zeros(k)

    eig_vals, O = np.linalg.eigh(np.matmul(J,J))
    #print(eig_vals) #-(epsilon_k^2)
    parity = np.linalg.det(O)
    print(O)
    print(f'Parity: {parity}')
    epsilons = np.sqrt(abs(eig_vals))
    energies[0] = -np.sum(epsilons)/2
    k =3
    for i in range(1, k-1):
        #print('epsilons', epsilons)
        a  = np.argmin(epsilons) 
        epsilons[a]+= -energies[0]
        a  = np.argmin(epsilons)
        #print('index of the min', a)      
        energies[i] = energies[i-1] + 2*epsilons[a]
        epsilons[a]+= -energies[0]

    return energies

#def williamson_decomposition(J):
#    ''' Returns the matrix O that takes J into W = O^T J O in block diagonal form'''
