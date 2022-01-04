from math import factorial
from tfd_preparation import * 
from syk_functions import * 

N = 4 # N fermions, 2N Majoranas
num_gates = 500 #number of left-right blocks
J = 1 # divide by comb(2N,4)
mu = 1
seed = 19
independent_optimization = False
save = True

tfd_algor_cooling(N, num_gates, J, mu, seed = seed, independent_optimization=independent_optimization, save=save)
      
energy_list = np.load(f'TFD/Data/tfd_cooling_indep{independent_optimization}_N{N}_J{J}_mu{mu}_seed{seed}_numgates{num_gates}.npy')
exact_energies = np.load(f'TFD/Data/tfd_exact_indep{independent_optimization}_N{N}_J{J}_mu{mu}_seed{seed}_numgates{num_gates}.npy')

print(exact_energies)
plt.plot(energy_list[:200]/24, '.', label = 'Algorithmic cooling')
#for i in range(2):
#    plt.axhline(y=exact_energies[i]/24, linestyle='--', label = f'Energy level {i}')
plt.axhline(y=exact_energies[0]/24, color =  'r', linestyle='--', label = f'Ground energy')
plt.axhline(y=exact_energies[4]/24, color =  'g', linestyle='--', label = f'First excited energy')
plt.title('Algorithmic cooling of the TFD Hamiltonian')
plt.xlabel('Number of givens rotations')
plt.ylabel('Energy')
plt.legend()
#plt.savefig(f'Plots_SYK/Data/tfd_cooling_indep{independent_optimization}_N{N}_J{J}_mu{mu}_seed{seed}_numgates{num_gates}')
plt.show()