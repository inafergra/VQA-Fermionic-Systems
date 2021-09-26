from syk_functions import *
from algebra_functions import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
np.set_printoptions(precision=2)
np.random.seed(seed=3)

#------------------Number of fermions
N = 5

#------------------Initial state
J_L = init_syk_tensor(N)
J_R = init_syk_tensor(N)
#print('Initial H:')
#print(H)
init_energy = energy(J_L)
print(f'Initial energy is {init_energy}')

new_H_eq_mix = H
new_H_num = H

energy_list_mix = []
energy_list_num = []
energy_list_num.append(init_energy)
#energy_list_eq.append(init_energy)
num_gates = 3000

for k in range(num_gates):
    # -------------------------------Draw random i,alpha,j,beta---------------------
    i=0;j=0;alpha=0;beta=0
    while (i==j) and (alpha==beta):
        i = np.random.randint(0,N) ; j = np.random.randint(0,N)
        alpha = np.random.randint(0,2) ; beta = np.random.randint(0,1)
    indices = [i,alpha,j,beta]
    
    #----------------------------------------Optimizing the time---------------------
    t0=np.random.rand() #initial guess
    
    minimize_dictionary = minimize(new_energy, x0=t0,args=(new_H_num,indices), options={'disp': False}, method = 'Nelder-Mead')
    #minimize_dictionary = differential_evolution(new_energy,args=(new_H,indices), bounds=[(0,2)], disp = False)#, maxiter=10000)

    optimal_time_num = minimize_dictionary['x'][0]
    #print(f'Optimal time num: {optimal_time_num}')

    #-------------------------------------Computing exact optimal t-------------------
    
    #minimize_dictionary = minimize(f, x0=t0,args=(indices,new_H), options={'disp': False}, method = 'Nelder-Mead')
    #optimal_time = minimize_dictionary['x'][0]
    #optimal_time_eq = optimal_t(indices,new_H_eq)
    #optimal_time_eq_shifted = optimal_time_eq - np.pi
    #print(f'Optimal time eq: {optimal_time_eq}')
    #--------------------------------------Computing energy after h---------------------
    new_H_num = appy_h_gate(optimal_time_num,new_H_num, indices)
    final_energy_num = energy(new_H_num)

    #new_H_eq = appy_h_gate(optimal_time_eq,new_H_eq, indices)
    #final_energy_eq= energy(new_H_eq)

    #new_H_eq_shifted = appy_h_gate(optimal_time_eq_shifted,new_H_eq, indices)
    #final_energy_eq_shifted = energy(new_H_eq_shifted)

    #if final_energy_eq_shifted < final_energy_eq:
    #    final_energy_eq = final_energy_eq_shifted
    #    new_H_eq = new_H_eq_shifted
    #    print('Heyyyyy')

    #print(f'Exact energy :{final_energy_eq}')
    #print(f'Numeric energy :{final_energy_num}')

    #energy_list_eq.append(final_energy_eq)
    energy_list_num.append(final_energy_num)
    #print(energy_list)

#num_gates = len(energy_list)
print(new_H_num)
exact_energies = exact_energy_levels(H,2)
theta = np.zeros(N)
print(f'Exact ground energy: {exact_energies[0]}')
print(f'Numerical energy is {final_energy_num}')
#print(f'Energy function rotate {energy_after_x_rotations(theta, new_H_num)}')
#print(f'Variance is {squared_hamiltonian_average(new_H_num) - energy(new_H_num)**2}')#{variance(np.zeros(N), new_H_num)}')
#print(f'Variance rotating is {variance(theta, new_H_num)}')
#plt.plot(range(round(num_gates/2), num_gates), energy_list_num[-round(num_gates/2):], label = 'Algorithmic cooling numeric')
#plt.plot(range(round(num_gates/2), num_gates), energy_list_exact[-round(num_gates/2):], label = 'Algorithmic cooling exact')

plt.plot(energy_list_num, label = 'Algorithmic cooling numeric')
#plt.plot(energy_list_eq, label = 'Algorithmic cooling exact')

#--------------------plotting the exact energies
#for i in range(len(exact_energies)): 
#plt.axhline(y=exact_energies[i], linestyle='-', label = f'Energy level {i}')
plt.axhline(y=exact_energies[0], color =  'r', linestyle='--', label = f'Energy level 0')
plt.axhline(y=exact_energies[1], color =  'y', linestyle='--', label = f'Energy level 1')

plt.xlabel('Number of unitaries')
plt.ylabel('Energy')
plt.legend()
#plt.savefig(f'Plots/N={N}, p={len(energy_list)}')
plt.show()