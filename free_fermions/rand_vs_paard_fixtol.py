from algebra_functions import *
from functions import *
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
np.set_printoptions(precision=2)
#np.random.seed(seed=10)

#N = 10
error_random = []
error_paard = []
givens_paard_list = []
givens_random_list = []
energy_tol = 10
N_list = range(6,20)
avrg_number = 25

for N in N_list:
    print(f'{N} fermions')

    avr_list_paard = []
    avr_list_rand = []

    for l in range(avrg_number): #take the average

        H = init_coeff_matrix(N)
        H_rand = np.copy(H)
        H_paard = np.copy(H)

        exact_energies, parity = exact_energy_levels(H,2)
        if parity>0:
            #print("hehe")
            exact_gs_energy = exact_energies[0]

        else:
            exact_gs_energy = exact_energies[1]

        energy_error_paard = 100
        sweeps = 1
        while energy_error_paard>energy_tol:
            energy_error_paard, givens_paard = greedy_paard(H_paard, exact_gs_energy, sweeps=sweeps)
            
            avr_list_paard.append(givens_paard)
            print(energy_error_paard)
            if sweeps>7:
                break
            sweeps += 1
        #print()
        #print(energy_error_paard)

        #--------------------------------- Randomized cooling
        new_H_num = H_rand
        init_energy = energy(new_H_num)
        #print(init_energy)

        energy_list_num = []
        energy_list_num.append(init_energy)
        energy_error_rand = 100
        k=0
        while energy_error_rand>energy_error_paard:
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
            energy_error_rand = abs((final_energy_num - exact_gs_energy)*100/exact_gs_energy)
            k += 1

        avr_list_rand.append(k)

        
    avr_givens_paard = sum(avr_list_paard)/len(avr_list_paard)
    givens_paard_list.append(avr_givens_paard)

    avr_givens_rand = sum(avr_list_rand)/len(avr_list_rand)
    givens_random_list.append(avr_givens_rand)


np.save(f"free_fermions/data/givens_random_list1_avrg{avrg_number}_listuntil{N_list[-1]}",np.array(givens_random_list))
np.save(f"free_fermions/data/givens_paard_list1_avrg{avrg_number}_listuntil{N_list[-1]}",np.array(givens_paard_list))
