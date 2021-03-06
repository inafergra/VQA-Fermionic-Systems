import numpy as np
import matplotlib.pyplot as plt
import itertools as itertools
from scipy.optimize import minimize, differential_evolution
#np.set_printoptions(precision=2)
import pdb; 


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

def energy(H):
    '''
    Calculates the energy of a given coupling matrix H. The energy is always taken w.r.t the
    fermionic VACUUM state. 
    '''
    N = int(np.size(H,axis=0)/2)
    energy = 0
    for l in range(N):
        energy -= H[2*l+1,2*l]
    return energy

def squared_hamiltonian_average(H):
    ''' Calculates the average of the squared hamiltonian with respect to the vacuum '''
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
                                    y = H[2*l+alpha, 2*k+beta]*H[2*i+delta, 2*j+gamma] * 1j**(alpha+beta+gamma+delta) 
                                    if (i==j) and (l==k):
                                        mat_elem += y*(-1)**(gamma+beta)
                                    elif (i==l!=k==j) :
                                        mat_elem +=  -y*(-1)**(alpha+beta)
                                    elif (j==l!=k==i):
                                        mat_elem +=  y*(-1)**(alpha+beta)
    return -1/4*mat_elem

def ham_matrix_element(H, m1, m2):
    '''
    Calculates the matrix element <m1|H|m2>, where |m>=a_m^{dagger}|Omega>, i.e., |m> is a state where
     only the m-th mode is excited
    '''
    N = int(np.size(H,axis=0)/2)
    mat_elem = 0
    for l in range(N):
        for k in range(N):
            for alpha in range(0,2):
                for beta in range(0,2):
                    x = H[2*l+alpha,2*k+beta] * 1j**(alpha+beta)
                    if (l== k == m1 == m2) or  (l== m2 != m1 == k):
                        mat_elem += x*(-1)**alpha
                    elif (l == k != m1 == m2) or (l== m1 != m2 == k):
                        mat_elem += x*(-1)**beta
    return 0.5*1j*mat_elem

def sq_ham_matrix_element(H, m1, m2):
    '''
    Calculates the matrix element <m1|H??|m2>, where |m>=a_m^{dagger}|Omega>, i.e., |m> is a state where
     only the m-th mode is excited
    '''
    N = int(np.size(H,axis=0)/2)
    mat_elem = 0
    for i in range(N):
            for j in range(N):
                for l in range(N):
                    for k in range(N):
                        for alpha in range(0,2):
                            for beta in range(0,2):
                                for delta in range(0,2):
                                    for gamma in range(0,2):
                                        x =  H[2*l+alpha, 2*k+beta]*H[2*i+delta, 2*j+gamma] * 1j**(alpha+beta+gamma+delta)
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
    return -1/4*mat_elem



def coeffm1(N, m1, theta):
    s = 1
    for i in range(N):
        if i==m1:
            s *= np.sin(theta[i])
        else:
            s *= np.cos(theta[i])
    return s

def ham_average_rotated(theta, H):
    '''
    Computes the energy of the state given by doing an X rotation of theta[i] in qubit i. 
    args:
        theta: 1darray (len = number of fermions/qubits N)
    '''
    N = int(np.size(H,axis=0)/2)
    e = 0
    c = np.prod([np.cos(i)**2 for i in theta])
    for m1 in range(N):
        for m2 in range(N):
            e += coeffm1(N, m1, theta)*coeffm1(N, m2, theta)*ham_matrix_element(H, m1, m2)
    #print(e)
    return c*energy(H) + np.real(e)



def sq_ham_average_rotated(theta,H):
    """
    Computes the average of the squared hamiltonian TO THE FIRST ORDER of the state given by doing an X rotation of theta[i] in qubit i. 
    args:
        theta: 1darray (len = number of fermions/qubits N)
    """
    N = int(np.size(H,axis=0)/2)
    e = 0
    for m1 in range(N):
        for m2 in range(N):
            e += theta[m1]*theta[m2]*sq_ham_matrix_element(H, m1, m2)
    return squared_hamiltonian_average(H) +  (1/N**2)*np.real(e)

def variance(theta, H):
    return np.abs(sq_ham_average_rotated(theta, H) - ham_average_rotated(theta, H)**2)

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

def new_energy(t,H,indices):
    '''
    Function to input in the minimization loop. Computes the energy of the coupling matrix after 
    going through the transformation h[indices] for a time t.
    '''
    new_H = appy_h_gate(t,H,indices)
    new_energy = energy(new_H)
    return new_energy

def exact_energy_levels(H, k=2):
    '''
    Finds the first k exact energy levels through exact diagonalization
    '''
    energies = np.zeros(k)

    eig_vals, O = np.linalg.eigh(np.matmul(H,H)) #epsilons squared
    parity = np.linalg.det(O)
    #print(f'Determinant of O: {parity}')
    epsilons = np.sort(np.sqrt(abs(eig_vals)))

    # Ground state energie
    energies[0] = -np.sum(epsilons)/2

    # First excited state energy
    energies[1] = energies[0] + 2*abs(epsilons[0])

    return energies, parity

def randomized_cooling(H, gates, ground_energy=0, tolerance = 0.01):

    N = int(np.size(H,axis=0)/2)
    init_energy = energy(H)

    energy_list= []
    energy_list.append(init_energy)

    #while np.abs(ground_energy-final_energy)>tolerance:
    for k in range(gates):
        # -------------------------------Draw random i,alpha,j,beta---------------------
        i=0;j=0;alpha=0;beta=0
        while (i==j) and (alpha==beta):
            i = np.random.randint(0,N) ; j = np.random.randint(0,N)
            alpha = np.random.randint(0,2) ; beta = np.random.randint(0,1)
        indices = [i,alpha,j,beta]
        
        #----------------------------------------Optimizing the time---------------------
        t0=np.random.rand() #initial guess
        
        minimize_dictionary = minimize(new_energy, x0=t0,args=(H,indices), options={'disp': False}, method = 'Nelder-Mead')
        #minimize_dictionary = differential_evolution(new_energy,args=(new_H,indices), bounds=[(0,2)], disp = False)#, maxiter=10000)

        optimal_time_num = minimize_dictionary['x'][0]
        #print(f'Optimal time num: {optimal_time_num}')

        #--------------------------------------Computing energy after h---------------------
        H = appy_h_gate(optimal_time_num,H, indices)
        final_energy = energy(H)

        energy_list.append(final_energy)

        
    return energy_list