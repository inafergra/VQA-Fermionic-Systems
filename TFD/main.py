from tfd_preparation import * 
from syk_functions import * 

N = 3 # N fermions, 2N Majoranas
num_gates = 500 #number of left-right blocks
J = 1
mu = 0
seed = 99
independent_optimization = False
save = True

#tfd_algor_cooling(N, num_gates, J, mu, seed = seed, independent_optimization=independent_optimization, save=save)
      
energy_list = np.load(f'Data/tfd_cooling_indep{independent_optimization}_N{N}_J{J}_mu{mu}_seed{seed}_numgates{num_gates}.npy')
exact_energies = np.load(f'Data/tfd_exact_indep{independent_optimization}_N{N}_J{J}_mu{mu}_seed{seed}_numgates{num_gates}.npy')

print(exact_energies)
plt.plot(energy_list, label = 'Algorithmic cooling')
for i in range(7):
    plt.axhline(y=exact_energies[i], linestyle='--', label = f'Energy level {i}')
#plt.axhline(y=exact_energies[1], color =  'y', linestyle='--', label = f'Energy level 1')
#plt.axhline(y=exact_energies[2], color =  'y', linestyle='--', label = f'Energy level 2')
plt.title('Algorithmic cooling of the TFD Hamiltonian')
plt.xlabel('Number of givens rotations')
plt.ylabel('Energy')
plt.legend()
#plt.savefig(f'Plots_SYK/Data/tfd_cooling_indep{independent_optimization}_N{N}_J{J}_mu{mu}_seed{seed}_numgates{num_gates}')
plt.show()