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
print('Initializing tensor')
J_L, J_R, H_int = init_TFD_model(N, 1, 0.1)

init_energy = tfd_energy(J_L, J_R, H_int)
print(f'Initial energy is {init_energy}')

J_L, J_R, H_int = apply_unitary(J_L, J_R, H_int, 1.2, 'L', (1,1,0,0))

print(f'Next energy is {init_energy}')

energy_list = []
num_gates = 10

for k in range(num_gates):
    # -------------------------------Draw random i,alpha,j,beta---------------------
    i=0;j=0;alpha=0;beta=0
    while (i==j) and (alpha==beta):
        i = np.random.randint(0,N) ; j = np.random.randint(0,N)
        alpha = np.random.randint(0,2) ; beta = np.random.randint(0,1)
    indices = [i,alpha,j,beta]
    
    #----------------------------------------Optimizing the time---------------------
    t0=np.random.rand() #initial guess
    subsystem = 'R'
    minimize_dictionary = minimize(new_tfd_energy, x0=t0,args=(indices, subsystem, J_L, J_R, H_int), options={'disp': False}, method = 'Nelder-Mead')
    #minimize_dictionary = differential_evolution(new_energy,args=(new_H,indices), bounds=[(0,2)], disp = False)#, maxiter=10000)

    optimal_time = minimize_dictionary['x'][0]
    #print(f'Optimal time num: {optimal_time}')

    #--------------------------------------Computing energy after h---------------------
    J_L, J_R, H_int = apply_unitary(J_L, J_R, H_int, optimal_time, 'R', indices)
    final_energy = tfd_energy(J_L, J_R, H_int)
    energy_list.append(final_energy)
    #print(energy_list)


print(f'Numerical energy is {final_energy}')

plt.plot(energy_list_num, label = 'Algorithmic cooling numeric')

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