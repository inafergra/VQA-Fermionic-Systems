import numpy as np
import matplotlib.pyplot as plt
#np.set_printoptions(precision=2)

from syk_functions import *
from exact_diagonalization import *
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

def tfd_algor_cooling(N, num_gates, J, save = True):
    '''
    Runs algorithmic cooling for the TFD Hamiltonian with N fermions
    '''
    #np.random.seed(seed=seed)

    #print('Initializing tensor')
    J_tens, J_dict = init_TFD_model(N, J/np.math.factorial(4))

    init_energy = tfd_energy(J_tens)
    #print(f'Initial energy is {init_energy}')

    eig =tfd_exact(N,J_tens)
    #print("Eig")
    #print(eig)
    #pdb.set_trace()
    energy_list = [init_energy]
    np.random.seed(seed=2)

    for k in range(num_gates):

        #print('Block ', k)
        i=0;j=0;alpha=0;beta=0
        while (i==j):
            i = np.random.randint(0,N) ; j = np.random.randint(0,N)
            alpha = np.random.randint(0,2) ; beta = np.random.randint(0,2)
        indices = [i,alpha,j,beta]
        #print(indices)

        # Optimizing the time 
        t0=np.random.rand()
        minimize_dictionary = minimize(new_tfd_energy, x0=t0,args=(indices, J_tens), options={'disp': False}, method = 'SLSQP') #SLSQP  Nelder-Mead
        #minimize_dictionary = differential_evolution(new_tfd_energy,args=(indices, subsystem, TFD_model), bounds=[(0,2*np.pi)], disp = False)#, maxiter=10000)
        optimal_time = minimize_dictionary['x'][0]

        # Computing energy after unitary
        J_tens = apply_unitary(J_tens, optimal_time, indices)
        energy = tfd_energy(J_tens)
        #print('Energy after unitary', energy)
        energy_list.append(energy)

    return np.array(energy_list), np.array(eig)
