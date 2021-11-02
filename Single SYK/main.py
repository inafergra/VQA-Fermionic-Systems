from ground_state_preparation import * 
from syk_functions import * 

N = 3 # N fermions, 2N Majoranas
num_gates = 500 #number of left-right blocks
J = 1
seed = 3

tfd_algor_cooling(N, num_gates, J, seed = seed)

energy_list = np.load(f'Data/SYK_N{N}_J{J}_seed{seed}_numgates{num_gates}.npy')
exact_energies = np.load(f'Data/SYKexact_N{N}_J{J}_seed{seed}_numgates{num_gates}.npy')

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

cmap = get_cmap(len(exact_energies))
print(exact_energies)

plt.plot(energy_list[:40], '.' , label = 'Algorithmic cooling')
#for i in range(4):
#    plt.axhline(y=exact_energies[i], linestyle='--',c=cmap(i), label = f'Energy level {i}')
plt.axhline(y=exact_energies[0], color =  'r', label = f'Ground energy')
plt.axhline(y=exact_energies[2], color =  'g', label = f'First excited energy') #, linestyle='--'
plt.title('Algorithmic cooling of the SYK model')
plt.xlabel('Number of givens rotations')
plt.ylabel('Energy')
plt.legend()
#plt.savefig(f'Plots_SYK/Data/tfd_cooling_indep{independent_optimization}_N{N}_J{J}_mu{mu}_seed{seed}_numgates{num_gates}')
plt.show()