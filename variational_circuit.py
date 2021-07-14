from algebra_functions import *
from functions import *
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
np.set_printoptions(precision=2)
np.random.seed(seed=1)

N = 4
print(f'{N} fermions')

# Vanilla implementation (No variational circuit)

print('Initializing matrix, initial state vacuum ...')
H0 = init_coeff_matrix(N)
print(f'Initial energy {energy(H0)}')
print(f'Initial off-diagonal norm {off_diag_frobenius_norm(H0)}') 
print(f"Variance {variance(np.zeros(N), H0)}")
print()

print('Applying Paardakopers (Uj)...')
H, norm, sweeps, angles = paardekoper_algorithm(H0, tolerance = 1e-15)
print(f'Number of sweeps = {sweeps}')
print(f'Off-diagonal norm {norm}') 
print(f'Energy {energy(H)}')
print(f"Variance {variance(np.zeros(N), H)}")
print()

print('Applying greedy algorithm...')
H = greedy_algorithm(H)
print(f'Off-diagonal norm {off_diag_frobenius_norm(H)}') 
print(f'Energy {energy(H)}')
print(f"Variance {variance(np.zeros(N), H)}")
print()

exact_energies = exact_energy_levels(H0,2)
print(f'Exact ground energy: {exact_energies[0]}')
print(f'Exact second excited energy: {exact_energies[1]}')

print()
pdb.set_trace()


#H, norm, sweeps, angles = paardekoper_algorithm(H0, tolerance = 1e-15)
#print(f'Energy {energy(H)}')


#Variational circuit


H0 = init_coeff_matrix(N)
init_norm = off_diag_frobenius_norm(H0)
print(f'Initial off-diagonal norm {init_norm}') 
print(f'Initial variance {variance(np.zeros(N), H0)}')
print(f'Initial variance {squared_hamiltonian_average(H0) - energy(H0)**2}')
print(f'Initial energy {energy(H0)}')


# Get Uj
print("Calculating the angles of Uj")
H, norm, sweeps, angles = paardekoper_algorithm(H0, tolerance = 1e-10)
print(f'Off-diagonal norm applying Uj: {norm}')
#print(f'Number of sweeps = {sweeps}')

# Apply Uj
H, norm, sweeps = paardekoper_algorithm(H0, tolerance = 1e-2, saved_angles =  angles)
print(f'Final off-norm without variational circuit {norm}')
#print(f'Number of sweeps = {sweeps}')

print(f'Energy not rotating {energy_after_x_rotations(np.zeros(N), H)}')
print(f'Variance not rotating {variance(np.zeros(N), H)}')
var = partial(variance, H = H)
#en = partial(energy_after_x_rotations, H=H)

theta0 = np.random.random(N)
minimize_dictionary = minimize(var, x0=theta0,
                               options={'disp': True, 'maxiter': 20},
                               method = 'Nelder-Mead')
optimal_theta = minimize_dictionary['x']

print(optimal_theta)
print(f'Variance with rotation {variance(optimal_theta, H)}')
print(f'Energy with rotation {energy_after_x_rotations(optimal_theta, H)}')
#print(f'Final norm {off_diag_frobenius_norm(H)} with variational circuit')

exact_energies = exact_energy_levels(H,2)
print(f'Exact ground energy: {exact_energies[0]}')
print(f'Exact second excited energy: {exact_energies[1]}')