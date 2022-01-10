from math import factorial
from ground_state_preparation import * 
from syk_functions import * 

N = 6
num_gates = 600
J = 1
seed = 10
print(f"{2*N} Majoranas")

#tfd_algor_cooling(N, num_gates, J)

energy_list_N3 = np.load(f'Data/SYK_N3_J1_seed12_numgates500.npy')
exact_energies_N3 = np.load(f'Data/SYKexact_N3_J1_seed12_numgates500.npy')

energy_list_N4 = np.load(f'Data/SYK_N4_J1_seed10_numgates500.npy')
exact_energies_N4 = np.load(f'Data/SYKexact_N4_J1_seed10_numgates500.npy')

energy_list_N5 = np.load(f'Data/SYK_N5_J1_seed10_numgates50000.npy')
exact_energies_N5 = np.load(f'Data/SYKexact_N5_J1_seed10_numgates50000.npy')

energy_list_N6 = np.load(f'Data/SYK_N6_J1_numgates600.npy')
exact_energies_N6 = np.load(f'Data/SYKexact_N6_J1_numgates600.npy')

#energy_list = np.load(f'Data/SYK_N{N}_J{J}_seed{seed}_numgates{num_gates}.npy')
#exact_energies = np.load(f'Data/SYKexact_N{N}_J{J}_seed{seed}_numgates{num_gates}.npy')

app_ratio3 = energy_list_N3[-1]/exact_energies_N3[0]
app_ratio4 = energy_list_N4[-1]/exact_energies_N4[0]
app_ratio5 = energy_list_N5[-1]/exact_energies_N5[0]
app_ratio6 = energy_list_N6[-1]/exact_energies_N6[0]

en_list_3 = []
en_list_4 = []
en_list_5 = []
en_list_6 = []

givens_3 = []
givens_4 = []
givens_5 = []
givens_6 = []

for k in range(len(energy_list_N3)):
    if k%1==0:
        givens_3.append(k)
        en_list_3.append(energy_list_N3[k])

for k in range(0, len(energy_list_N4)):
    if k%1==0:
        givens_4.append(k)
        en_list_4.append(energy_list_N4[k])

for k in range(0,120): #len(energy_list_N5)):
    if k%1==0:
        givens_5.append(k)
        en_list_5.append(energy_list_N5[k])

for k in range(25, len(energy_list_N6)-20):
    if k%5==0:
        givens_6.append(k)
        en_list_6.append(energy_list_N6[k])

plt.rcParams.update({'font.size': 14})
#plt.rcParams["figure.autolayout"] = True
fig, axes = plt.subplots(2, 2)
ax3 = axes[0,0]
ax4 = axes[0,1]
ax5 = axes[1,0]
ax6 = axes[1,1]

fig.suptitle('Algorithmic cooling of the SYK model')

ax3.plot(givens_3, en_list_3, '.' , markersize = 4, label = "Algorithmic cooling" )
ax4.plot(givens_4, en_list_4, '.' , markersize = 3 )
ax5.plot(givens_5, en_list_5, '.' , markersize = 3 )
ax6.plot(givens_6, en_list_6, '.' , markersize = 3 )

ax3.axhline(y=exact_energies_N3[0], color =  'r', linestyle='--', label = f'Ground energy')
ax3.axhline(y=exact_energies_N3[2], color =  'y', linestyle='--', label = f'First excited energy')
ax4.axhline(y=exact_energies_N4[0], color =  'r', linestyle='--')
ax4.axhline(y=exact_energies_N4[2], color =  'y', linestyle='--')
ax5.axhline(y=exact_energies_N5[0], color =  'r', linestyle='--')
ax5.axhline(y=exact_energies_N5[2], color =  'y', linestyle='--')
ax6.axhline(y=exact_energies_N6[0], color =  'r', linestyle='--')
ax6.axhline(y=exact_energies_N6[2], color =  'y', linestyle='--')


ax3.set_title(r"n=3, $r_{average}=$" + f"{round(app_ratio3, 3)}")
ax4.set_title(r"n=4, $r_{average}=$" + f"{round(app_ratio4, 3)}")
ax5.set_title(r"n=5, $r_{average}=$" + f"{round(app_ratio5, 3)}")
ax6.set_title(r"n=6, $r_{average}=$" + f"{round(app_ratio6, 3)}")
ax3.set_ylabel("Energy")
ax5.set_ylabel("Energy")
ax5.set_xlabel("Number of Givens rotations")
ax6.set_xlabel("Number of Givens rotations")


#fig.set_title("Randomized algorithmic cooling")

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc = (0.25, 0.76))

plt.show()

"""
def plot_one(energy_list, e)
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

#cmap = get_cmap(len(exact_energies))
#print(exact_energies)

en_list = []
givens = []
for k in range(len(energy_list)):
    if k%1==0:
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

"""
