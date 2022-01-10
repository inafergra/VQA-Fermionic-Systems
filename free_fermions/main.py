from functions import *
import numpy as np
import matplotlib.pyplot as plt

N=10
gates = 3000

H = init_coeff_matrix(N)
energy_list = randomized_cooling(H, ground_energy=0, gates = gates)
exact_energies = exact_energy_levels(H,2)[0]

plt.plot(energy_list[500:], label = 'Algorithmic cooling numeric')
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

np.save(f"energy_list_algocooling_parity-1", np.array(energy_list))
np.save(f"exact_energies_parity-1", np.array(exact_energies))
