from algebra_functions import *
from functions import *
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)
np.random.seed(seed=1)

N = 5
print(f'{N} fermions')

# We start out with the vacuum |0>^(xn) 
# and then apply X rotation to every qubit
# so every qubit is in a superposition e**(itX/2)|0> = cos(t/2)|0> + i sin(t/2) |0>



# Get Uj
H0 = init_coeff_matrix(N, mean=0, H_value=1)
init_norm = off_diag_frobenius_norm(H0)
print(f'Initial norm {init_norm}') 

H, norm, sweeps, angles = paardekoper_algorithm(H0, tolerance = 1e-10)

print(f'Final norm {norm}')
print(f'Number of sweeps = {sweeps}')
print(len(angles))
# Apply Uj
init_norm = off_diag_frobenius_norm(H0)
print(f'Initial norm {init_norm}') 

H, norm, sweeps = paardekoper_algorithm(H0, tolerance = 1e-10, saved_angles =  angles)

print(f'Final norm {norm}')
print(f'Number of sweeps = {sweeps}')



energy = energy_after_x_rotations(H,theta)
