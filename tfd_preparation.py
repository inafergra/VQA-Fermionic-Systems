import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

from syk_functions import *
from exact_diagonalization_TFD import *

from scipy.optimize import minimize, differential_evolution

import cirq
from openfermion.ops import MajoranaOperator
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator

from itertools import combinations

def tfd_algor_cooling(N, num_gates, J, mu, seed = np.random.randint):
    '''
    Runs algorithmic cooling for the TFD Hamiltonian with N fermions
    '''
    np.random.seed(seed=seed)

    print('Initializing tensor')
    TFD_model, TFD_dict = init_TFD_model(N, J, mu)

    init_energy = tfd_energy(TFD_model)
    print(f'Initial energy is {init_energy}')

    print(tfd_exact(N, TFD_dict))

    energy_list = []

    for k in range(num_gates):

        # Draw random i,alpha,j,beta
        print('Gate ', k)
        i=0;j=0;alpha=0;beta=0
        while (i==j):
            i = np.random.randint(0,N) ; j = np.random.randint(0,N)
            alpha = np.random.randint(0,2) ; beta = np.random.randint(0,1)
        indices = [i,alpha,j,beta]
        
        # Optimizing the time (for one of the sides)
        t0=np.random.rand() #initial guess
        subsystem = 'R'
        minimize_dictionary = minimize(new_tfd_energy, x0=t0,args=(indices, subsystem, TFD_model), options={'disp': True}, method = 'Nelder-Mead')
        #minimize_dictionary = differential_evolution(new_tfd_energy,args=(indices, subsystem, J_L, J_R, H_int), bounds=[(0,2)], disp = False)#, maxiter=10000)
        optimal_time_R = minimize_dictionary['x'][0]
        print(f'Optimal time for R num: {optimal_time_R}')

        # Computing energy after right unitary
        TFD_model = apply_unitary(TFD_model, optimal_time_R, 'R', indices)
        left_energy = tfd_energy(TFD_model)
        print('Energy after right unitary', left_energy)
        energy_list.append(left_energy)

        # Optimizing time for the other side
        subsystem = 'L'
        minimize_dictionary = minimize(new_tfd_energy, x0=t0,args=(indices, subsystem, TFD_model), options={'disp': True}, method = 'Nelder-Mead')
        #minimize_dictionary = differential_evolution(new_tfd_energy,args=(indices, subsystem, J_L, J_R, H_int), bounds=[(0,2)], disp = False)#, maxiter=10000)
        optimal_time_L = minimize_dictionary['x'][0]
        print(f'Optimal time for L num: {optimal_time_L}')

        # Computing energy after left unitary
        TFD_model = apply_unitary(TFD_model, optimal_time_L, 'L', indices)
        right_energy = tfd_energy(TFD_model)
        print('Energy after left unitary', right_energy)
        energy_list.append(right_energy)
        #print(energy_list)

    print(f'Numerical energy is {right_energy}')

    plt.plot(energy_list, label = 'Algorithmic cooling')
    #plt.axhline(y=exact_energies[0], color =  'r', linestyle='--', label = f'Energy level 0')
    #plt.axhline(y=exact_energies[1], color =  'y', linestyle='--', label = f'Energy level 1')
    plt.xlabel('Number of unitaries')
    plt.ylabel('Energy')
    plt.legend()
    #plt.savefig(f'Plots_SYK/N={N}, number of gates={len(energy_list)}')
    plt.show()

