import numpy as np
import matplotlib.pyplot as plt
#np.set_printoptions(precision=2)

from syk_functions import *
from exact_diagonalization_TFD import *

from scipy.optimize import minimize, differential_evolution

import cirq
from openfermion.ops import MajoranaOperator
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator

from itertools import combinations

def tfd_algor_cooling(N, num_gates, J, mu, seed = np.random.randint, independent_optimization = False, save = True):
    '''
    Runs algorithmic cooling for the TFD Hamiltonian with N fermions
    '''
    np.random.seed(seed=seed)

    print('Initializing tensor')
    TFD_model, TFD_dict = init_TFD_model(N, J, mu)

    init_energy = tfd_energy(TFD_model)
    print(f'Initial energy is {init_energy}')

    eig =tfd_exact(N, TFD_model, TFD_dict[2])
    print(eig)
    energy_list = [100]
    #pdb.set_trace()
    for k in range(num_gates):

        # Draw random i,alpha,j,beta
        print('Block ', k)
        i=0;j=0;alpha=0;beta=0
        while (i==j):
            i = np.random.randint(0,N) ; j = np.random.randint(0,N)
            alpha = np.random.randint(0,2) ; beta = np.random.randint(0,2)
        indices = [i,alpha,j,beta]
        print(indices)

        # Optimizing the time for Right side
        t0=np.random.rand()
        subsystem = 'R'
        minimize_dictionary = minimize(new_tfd_energy, x0=t0,args=(indices, subsystem, TFD_model), options={'disp': False}, method = 'SLSQP') #SLSQP  Nelder-Mead
        #minimize_dictionary = differential_evolution(new_tfd_energy,args=(indices, subsystem, TFD_model), bounds=[(0,2*np.pi)], disp = False)#, maxiter=10000)
        optimal_time_R = minimize_dictionary['x'][0]
        #print(f'Optimal time for R num: {optimal_time_R}')

        flag = 0

        if independent_optimization:

            # Computing energy after right unitary
            TFD_model = apply_unitary(TFD_model, optimal_time_R, 'R', indices)
            right_energy = tfd_energy(TFD_model)
            print('Energy after right unitary', right_energy)
            energy_list.append(right_energy)

            #i=0;j=0;alpha=0;beta=0
            #while (i==j):
            #    i = np.random.randint(0,N) ; j = np.random.randint(0,N)
            #alpha = np.random.randint(0,2) ; beta = np.random.randint(0,2)
            #indices = [i,alpha,j,beta]
            #print(indices)

            # Optimizing time for L side
            subsystem = 'L'
            minimize_dictionary = minimize(new_tfd_energy, x0=t0,args=(indices, subsystem, TFD_model), options={'disp': False}, method = 'SLSQP') #SLSQP  Nelder-Mead
            #minimize_dictionary = differential_evolution(new_tfd_energy,args=(indices, subsystem, TFD_model), bounds=[(0,2*np.pi)], disp = False)#, maxiter=10000)
            optimal_time_L = minimize_dictionary['x'][0]
            #print(f'Optimal time for L num: {optimal_time_L}')

            # Computing energy after left unitary
            TFD_model = apply_unitary(TFD_model, optimal_time_L, 'L', indices)
            left_energy = tfd_energy(TFD_model)
            print('Energy after left unitary', left_energy)
            energy_list.append(left_energy)
            #print(energy_list)

        else:

            # Computing energy after right unitary
            TFD_model_temp = apply_unitary(TFD_model, optimal_time_R, 'R', indices)
            right_energy = tfd_energy(TFD_model_temp)
            #if right_energy < energy_list[-1]:
            print('Energy after right unitary', right_energy)
            TFD_model = TFD_model_temp
            energy_list.append(right_energy)
            #flag+=1

            # Computing energy after left unitary
            TFD_model_temp = apply_unitary(TFD_model, optimal_time_R, 'L', indices)
            left_energy = tfd_energy(TFD_model_temp)
            #if left_energy < energy_list[-1]:
            print('Energy after left unitary', left_energy)
            TFD_model = TFD_model_temp
            energy_list.append(left_energy)
            #flag+=1
        #eig = tfd_exact(N, TFD_model, TFD_dict[2])
        #print(eig)
        if save == True:
            np.save(f'Data/tfd_cooling_indep{independent_optimization}_N{N}_J{J}_mu{mu}_seed{seed}_numgates{num_gates}.npy',np.array(energy_list[1:]))
            np.save(f'Data/tfd_exact_indep{independent_optimization}_N{N}_J{J}_mu{mu}_seed{seed}_numgates{num_gates}.npy',np.array(eig))
        print('Number of times it went up: ', flag)
