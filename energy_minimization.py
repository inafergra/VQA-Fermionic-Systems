from functions import *
from algebra_functions import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
np.set_printoptions(precision=2)
#np.random.seed(seed=1)

#------------------Number of fermions
N = 2

#------------------Initial state
J = init_coeff_matrix(N, mean=0, J_value=1)
#print('Initial J:')
#print(J)
init_energy = energy(J)
print(f'Initial energy is {init_energy}')

new_J_num = J
new_J_eq = J
new_J_eq_shifted = J
energy_list_num= []
energy_list_eq= []
energy_list_num.append(init_energy)
energy_list_eq.append(init_energy)
num_gates = 500

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
    print(f'Optimal time num: {optimal_time_num}')

    #-------------------------------------Computing exact optimal t-------------------
    
    #minimize_dictionary = minimize(f, x0=t0,args=(indices,new_J), options={'disp': False}, method = 'Nelder-Mead')
    #optimal_time = minimize_dictionary['x'][0]
    optimal_time_eq = optimal_t(indices,new_J_eq)
    optimal_time_eq_shifted = optimal_time_eq - np.pi
    print(f'Optimal time eq: {optimal_time_eq}')
    #--------------------------------------Computing energy after h---------------------
    new_J_num = appy_h_gate(optimal_time_num,new_J_num, indices)
    final_energy_num = energy(new_J_num)

    new_J_eq = appy_h_gate(optimal_time_eq,new_J_eq, indices)
    final_energy_eq= energy(new_J_eq)

    new_J_eq_shifted = appy_h_gate(optimal_time_eq_shifted,new_J_eq, indices)
    final_energy_eq_shifted = energy(new_J_eq_shifted)

    if final_energy_eq_shifted < final_energy_eq:
        final_energy_eq = final_energy_eq_shifted
        new_J_eq = new_J_eq_shifted
        print('Heyyyyy')

    print(f'Exact energy :{final_energy_eq}')
    print(f'Numeric energy :{final_energy_num}')

    energy_list_eq.append(final_energy_eq)
    energy_list_num.append(final_energy_num)
    #print(energy_list)

#num_gates = len(energy_list)
#print(f'Final energy is {final_energy_num}')

exact_energies = exact_energy_levels(J,2)
print(f'Exact ground energy: {exact_energies[0]}')

#plt.plot(range(round(num_gates/2), num_gates), energy_list_num[-round(num_gates/2):], label = 'Algorithmic cooling numeric')
#plt.plot(range(round(num_gates/2), num_gates), energy_list_exact[-round(num_gates/2):], label = 'Algorithmic cooling exact')

plt.plot(energy_list_num, label = 'Algorithmic cooling numeric')
plt.plot(energy_list_eq, label = 'Algorithmic cooling exact')

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