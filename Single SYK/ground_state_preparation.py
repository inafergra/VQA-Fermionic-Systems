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

def tfd_algor_cooling(N, num_gates, J, seed = np.random.randint, save = True):
    '''
    Runs algorithmic cooling for the TFD Hamiltonian with N fermions
    '''
    np.random.seed(seed=seed)

    print('Initializing tensor')
    J_tens, J_dict = init_TFD_model(N, J)

    init_energy = tfd_energy(J_tens)
    print(f'Initial energy is {init_energy}')

    eig =tfd_exact(N,J_tens)
    print(eig)
    #pdb.set_trace()
    energy_list = [init_energy]
    for k in range(num_gates):

        # Draw random i,alpha,j,beta
        #print('Block ', k)
        #i=0;j=0;alpha=0;beta=0
        #while (i==j):
        #    i = np.random.randint(0,N) ; j = np.random.randint(0,N)
        #    alpha = np.random.randint(0,2) ; beta = np.random.randint(0,2)
        #indices = [i,alpha,j,beta]

        print('Block ', k)
        i=0;j=0;alpha=0;beta=0
        while (i==j):
            i = np.random.randint(0,N) ; j = np.random.randint(0,N)
            alpha = np.random.randint(0,2) ; beta = np.random.randint(0,2)
        indices = [i,alpha,j,beta]
        print(indices)

        # Optimizing the time 
        t0=np.random.rand()
        minimize_dictionary = minimize(new_tfd_energy, x0=t0,args=(indices, J_tens), options={'disp': False}, method = 'SLSQP') #SLSQP  Nelder-Mead
        #minimize_dictionary = differential_evolution(new_tfd_energy,args=(indices, subsystem, TFD_model), bounds=[(0,2*np.pi)], disp = False)#, maxiter=10000)
        optimal_time = minimize_dictionary['x'][0]



        # Computing energy after right unitary
        J_tens = apply_unitary(J_tens, optimal_time, indices)
        energy = tfd_energy(J_tens)
        print('Energy after right unitary', energy)
        energy_list.append(energy)

        #eig = tfd_exact(N, TFD_model, TFD_dict[2])
        #print(eig)
        if save == True:
            np.save(f'Data/SYK_N{N}_J{J}_seed{seed}_numgates{num_gates}.npy',np.array(energy_list))
            np.save(f'Data/SYKexact_N{N}_J{J}_seed{seed}_numgates{num_gates}.npy',np.array(eig))
