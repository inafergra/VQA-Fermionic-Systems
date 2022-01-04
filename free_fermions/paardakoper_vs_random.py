from algebra_functions import *
from functions import *
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
np.set_printoptions(precision=2)
#np.random.seed(seed=10)

N = 10
print(f'{N} fermions')

print()
error_random = []
error_paard = []
givens_paard_list = []
givens_random_list = []
sweeps_list = range(4)

H = init_coeff_matrix(N)
H_rand = np.copy(H)
H_paard = np.copy(H)

exact_energies = exact_energy_levels(H,2)
exact_gs_energy = exact_energies[0][1]

for sweeps in sweeps_list:
    energy_error, number_of_givens = greedy_paard(H_paard, exact_gs_energy, sweeps=sweeps)
    error_paard.append(energy_error)
    givens_paard_list.append(number_of_givens)
    print(energy_error)

 
#--------------------------------- Randomized cooling
new_H_num = H_rand
init_energy = energy(new_H_num)

energy_list_num = []
energy_list_num.append(init_energy)

for k in range(int(number_of_givens)):
    # ------Draw random i,alpha,j,beta-----
    i=0;j=0;alpha=0;beta=0
    while (i==j) and (alpha==beta):
        i = np.random.randint(0,N) ; j = np.random.randint(0,N)
        alpha = np.random.randint(0,2) ; beta = np.random.randint(0,1)
    indices = [i,alpha,j,beta]
    
    #--Optimizing the time---
    t0=np.random.rand() #initial guess
    
    minimize_dictionary = minimize(new_energy, x0=t0,args=(new_H_num,indices), options={'disp': False}, method = 'Nelder-Mead')
    #minimize_dictionary = differential_evolution(new_energy,args=(new_H,indices), bounds=[(0,2)], disp = False)#, maxiter=10000)

    optimal_time_num = minimize_dictionary['x'][0]
    new_H_num = appy_h_gate(optimal_time_num,new_H_num, indices)
    final_energy_num = energy(new_H_num)

    #energy_list_num.append(final_energy_num)
    energy_error = abs((final_energy_num - exact_gs_energy)*100/exact_gs_energy)
    if k%5==0:
        error_random.append(energy_error)
        givens_random_list.append(k)
        print(energy_error)




start=0
plt.plot(givens_random_list[start:], error_random[start:], ".-", label = "Randomized cooling")
plt.plot(givens_paard_list[:], error_paard[:], ".-", label = "Paardakoper")
plt.legend()
plt.xlabel("Number of Givens rotations")
plt.ylabel("Ground state energy error (%)")
plt.show()
