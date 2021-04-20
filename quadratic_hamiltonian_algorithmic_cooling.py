from functions import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
np.set_printoptions(precision=2)

#Number of fermions
N = 10

#Initial state
J = init_coeff_matrix(N, mean=0, J_value=1)
#print('Initial J:')
#print(J)
init_energy = energy(J)
print(f'Initial energy is {init_energy}')

#How to know we are in the ground state??
new_J = J
energy_list= []
num_gates = 300

for k in range(num_gates):
    #draw random i,alpha,j,beta 
    i=0;j=0;alpha=0;beta=0
    while (i==j) and (alpha==beta):
        i = np.random.randint(0,N) ; j = np.random.randint(0,N)
        alpha = np.random.randint(0,2) ; beta = np.random.randint(0,1)
    indices = [i,alpha,j,beta]
    #print(indices)

    #Optimizing the time
    t0=np.random.rand() #initial guess
    minimize_dictionary = minimize(new_energy, x0=t0,args=(new_J,indices), options={'disp': False}, method = 'Nelder-Mead')
    #minimize_dictionary = differential_evolution(new_energy,args=(new_J,indices), bounds=[(0,2)], disp = False)#, maxiter=10000)

    optimal_time = minimize_dictionary['x'][0]
    #print(f'Optimal time is: {optimal_time}')
    new_J = appy_h_gate(optimal_time,new_J, indices)
    #print(new_J)
    final_energy = energy(new_J)
    energy_list.append(final_energy)
    #print(f'Energy is {final_energy}')

print(f'Final energy is {final_energy}')

print(f'Exact ground energy: {energy_levels(J,2)[0]}')

plt.plot(energy_list)
plt.xlabel('Number of unitaries')
plt.ylabel('Energy')
#plt.savefig(f'Plots/N={N}, p={len(energy_list)}')
#plt.show()