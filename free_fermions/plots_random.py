import matplotlib.pyplot as plt
import numpy as np


energy_list_minus = np.load(f"free_fermions/data/energy_list_algocooling_parity-1.npy")
exact_energies_minus = np.load(f"free_fermions/data/exact_energies_parity-1.npy")
energy_list_plus = np.load(f"free_fermions/data/energy_list_algocooling_parity+.npy")
exact_energies_plus = np.load(f"free_fermions/data/exact_energies_parity+.npy")

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
fig.suptitle('Algorithmic cooling for n = 10')
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

