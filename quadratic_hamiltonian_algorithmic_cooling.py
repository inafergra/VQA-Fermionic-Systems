from functions import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
np.set_printoptions(precision=2)
#np.random.seed(seed=1)

#Number of fermions
N = 5

#Initial state
J = init_coeff_matrix(N, mean=0, J_value=1)
#print('Initial J:')
#print(J)
init_energy = energy(J)
print(f'Initial energy is {init_energy}')

#How to know we are in the ground state??
new_J_num = J
new_J_exact = J
new_J_exact_shifted = J
energy_list_num= []
energy_list_exact= []
energy_list_num.append(init_energy)
energy_list_exact.append(init_energy)
num_gates = 2000

for k in range(num_gates):
    # -------------------------------Draw random i,alpha,j,beta---------------------
    i=0;j=0;alpha=0;beta=0
    while (i==j) and (alpha==beta):
        i = np.random.randint(0,N) ; j = np.random.randint(0,N)
        alpha = np.random.randint(0,2) ; beta = np.random.randint(0,1)
    indices = [i,alpha,j,beta]
    
    #----------------------------------------Optimizing the time---------------------
    t0=np.random.rand() #initial guess
    
    minimize_dictionary = minimize(new_energy, x0=t0,args=(new_J_num,indices), options={'disp': False}, method = 'Nelder-Mead')
    #minimize_dictionary = differential_evolution(new_energy,args=(new_J,indices), bounds=[(0,2)], disp = False)#, maxiter=10000)

    optimal_time_num = minimize_dictionary['x'][0]
    
    #-------------------------------------Computing exact optimal t-------------------
    
    #minimize_dictionary = minimize(f, x0=t0,args=(indices,new_J), options={'disp': False}, method = 'Nelder-Mead')
    #optimal_time = minimize_dictionary['x'][0]
    optimal_time_exact = optimal_t(indices,new_J_exact)
    optimal_time_exact_shifted = optimal_time_exact - np.pi
    #--------------------------------------Computing energy after h---------------------
    new_J_num = appy_h_gate(optimal_time_num,new_J_num, indices)
    final_energy_num = energy(new_J_num)

    new_J_exact = appy_h_gate(optimal_time_exact,new_J_exact, indices)
    final_energy_exact= energy(new_J_exact)

    new_J_exact_shifted = appy_h_gate(optimal_time_exact_shifted,new_J_exact_shifted, indices)
    final_energy_exact_shifted = energy(new_J_exact_shifted)

    if final_energy_exact_shifted < final_energy_exact:
        final_energy_exact = final_energy_exact_shifted

    print(final_energy_exact)
    print(final_energy_num)

    energy_list_exact.append(final_energy_exact)
    energy_list_num.append(final_energy_num)
    #print(energy_list)

#num_gates = len(energy_list)
#print(f'Final energy is {final_energy_num}')

exact_energies = exact_energy_levels(J,2)
print(f'Exact ground energy: {exact_energies[0]}')

#plt.plot(range(round(num_gates/2), num_gates), energy_list_num[-round(num_gates/2):], label = 'Algorithmic cooling numeric')
#plt.plot(range(round(num_gates/2), num_gates), energy_list_exact[-round(num_gates/2):], label = 'Algorithmic cooling exact')

plt.plot(energy_list_num, label = 'Algorithmic cooling numeric')
plt.plot(energy_list_exact, label = 'Algorithmic cooling exact')

#plotting the exact energies
#for i in range(len(exact_energies)): 
#plt.axhline(y=exact_energies[i], linestyle='-', label = f'Energy level {i}')
plt.axhline(y=exact_energies[0], color =  'r', linestyle='--', label = f'Energy level 0')
plt.axhline(y=exact_energies[1], color =  'y', linestyle='--', label = f'Energy level 1')

plt.xlabel('Number of unitaries')
plt.ylabel('Energy')
plt.legend()
#plt.savefig(f'Plots/N={N}, p={len(energy_list)}')
plt.show()