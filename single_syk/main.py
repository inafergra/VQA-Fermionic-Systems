from math import factorial
from ground_state_preparation import * 
from syk_functions import * 

Nlist = range(3,6)
number_iters = 10
num_gates = [100,130, 250] #number of left-right blocks
J = 1
seed = 1

approx_ratios = []
t = 0
for N in Nlist:
    N = int(N)
    print(f"{2*N} Majoranas")
    app_ratio_iter_list = np.zeros(number_iters)
    for iter in range(number_iters):
        print(f"Iteration {iter}")
        energy_list, exact_energies =  tfd_algor_cooling(N, num_gates[t], J)

        #np.save(f'data_average/SYK_N{N}_J{J}_numgates{num_gates}.npy',energy_list)
        #np.save(f'data_average/SYKexact_N{N}_J{J}_numgates{num_gates}.npy',exact_energies)

        #energy_list = np.load(f'data_average/SYK_N{N}_J{J}_numgates{num_gates}.npy')
        #exact_energies = np.load(f'data_average/SYKexact_N{N}_J{J}_numgates{num_gates}.npy')

        app_ratio_iter_list[iter] = energy_list[-1]/exact_energies[0]
        print(app_ratio_iter_list[iter])
    
    approx_ratios.append(np.average(app_ratio_iter_list))
    print(approx_ratios)
    t+=1
plt.plot(approx_ratios)
plt.show()

"""

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

cmap = get_cmap(len(exact_energies))
print(exact_energies)

en_list = []
givens = []
for k in range(len(energy_list)):
    if k%2==0:
        givens.append(k)
        en_list.append(energy_list[k])

plt.plot(givens, en_list, '.' , markersize = 6 , label = 'Algorithmic cooling')
#for i in range(2):
#    plt.axhline(y=exact_energies[i], linestyle='--',c=cmap(i), label = f'Energy level {i}')
plt.axhline(y=exact_energies[0], color =  'r', label = f'Ground energy')
plt.axhline(y=exact_energies[2], color =  'g', label = f'First excited energy') #, linestyle='--'
plt.title(f'Algorithmic cooling of a n={N} SYK model')
plt.xlabel('Number of givens rotations')
plt.ylabel('Energy')
plt.legend()
#plt.savefig(f'Plots_SYK/Data/tfd_cooling_indep{independent_optimization}_N{N}_J{J}_mu{mu}_seed{seed}_numgates{num_gates}')
plt.show()




en_list_p = []
en_list_m = []
givens_p = []
givens_m = []
for k in range(450, len(energy_list_minus)):
    if k%5==0:
        givens_p.append(k)
        givens_m.append(k)
        en_list_p.append(energy_list_plus[k])
        en_list_m.append(energy_list_minus[k])

plt.rcParams.update({'font.size': 12})
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Randomized algorithmic cooling for n = 10')
ax1.plot(givens_m, en_list_m, '.' , markersize = 3, label = "Algorithmic cooling" )
ax2.plot(givens_p, en_list_p, '.' , markersize = 3 )

ax1.axhline(y=exact_energies_minus[0], color =  'r', linestyle='--')
ax1.axhline(y=exact_energies_minus[1], color =  'y', linestyle='--')
ax2.axhline(y=exact_energies_plus[0], color =  'r', linestyle='--', label = f'Ground energy')
ax2.axhline(y=exact_energies_plus[1], color =  'y', linestyle='--', label = f'First excited energy')


ax1.set_title('Ground state parity p=-1')
ax2.set_title('Ground state parity p=+1')
ax1.set_xlabel("Number of Givens rotations")
ax2.set_xlabel("Number of Givens rotations")
ax1.set_ylabel("Energy")
#fig.set_title("Randomized algorithmic cooling")

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc = (0.65, 0.70))

plt.show()

"""