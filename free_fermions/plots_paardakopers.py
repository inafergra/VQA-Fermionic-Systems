import matplotlib.pyplot as plt
import numpy as np
N_list = range(6,20)
avrg_number = 25
givens_random_list = np.load(f"free_fermions/data/givens_random_list_avrg{avrg_number}_listuntil{N_list[-1]}.npy")
givens_paard_list = np.load(f"free_fermions/data/givens_paard_list_avrg{avrg_number}_listuntil{N_list[-1]}.npy")

print(givens_paard_list)

start=0
plt.plot(N_list, givens_random_list, "ro", label = "Algorithmic cooling")
plt.plot(N_list, givens_paard_list, "gv", label = "Paardakoper-based algorithm")
plt.legend()
plt.xticks(N_list)
plt.xlabel("Number of fermions N")
plt.ylabel("Number of Givens rotations")
plt.title("Number of Givens rotations required to achieve \n an energy error smaller than 1% ")
plt.show()
