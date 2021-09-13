from algebra_functions import *
from functions import *
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
np.set_printoptions(precision=2)
np.random.seed(seed=10)

N = 3
variational_layer = True
tolerance = 1e2
#print(f'{N} fermions')

# Vanilla implementation (No variational layer)
if not variational_layer:
    print('Initializing matrix, initial state vacuum ...')
    H0 = init_coeff_matrix(N)
    print(f'Initial energy {energy(H0)}')
    print(f'Initial off-diagonal norm {off_diag_frobenius_norm(H0)}') 
    print(f"Variance {variance(np.zeros(N), H0)}")

    print()

    print('Applying Paardakopers (Uj)...')
    H, norm, sweeps, angles = paardekoper_algorithm(H0, tolerance = 1e2)
    print(f'Number of sweeps = {sweeps}')
    print(f'Off-diagonal norm {norm}') 
    print(f'Energy {energy(H)}')
    print(f"Variance {variance(np.zeros(N), H)}")
    print()

    print('Applying greedy algorithm...')
    H = greedy_algorithm(H)
    print(f'Off-diagonal norm {off_diag_frobenius_norm(H)}') 
    print(f'Energy {energy(H)}')
    #print(f"Second energy {energy_1(H)}")
    print(f"Variance {variance(np.zeros(N), H)}")
    print()

    exact_energies = exact_energy_levels(H0,2)
    print(f'Exact ground energy: {exact_energies[0]}')
    print(f'Exact second excited energy: {exact_energies[1]}')

    print()
    pdb.set_trace()


# Variational layer
else:

    fermion_list = range(3,7)
    ground_energy_error_novar = []
    ground_energy_error_var = []
    variance_novar = []
    variance_var = []

    for N in fermion_list:
        tolerance = 1e-1
        print(f"{N} fermions")
        print('Initializing matrix, initial state vacuum ...')
        H0 = init_coeff_matrix(N)
        init_norm = off_diag_frobenius_norm(H0)
        exact_energies, parity = exact_energy_levels(H0,2)
        print(f'Exact ground energy: {exact_energies[0]}')
        print(f'Exact second excited energy: {exact_energies[1]}')
        print() 

        print(f'Initial off-diagonal norm {init_norm}') 
        print(f'Initial variance {variance(np.zeros(N), H0)}')
        #print(f'Initial variance {squared_hamiltonian_average(H0) - energy(H0)**2}')
        print(f'Initial energy {energy(H0)}')
        print()

        print("Calculating the angles of Uj")
        H, norm, sweeps, angles = paardekoper_algorithm(H0, tolerance = tolerance)
        #print(f'Off-diagonal norm applying Uj: {norm}')
        print(f'Number of sweeps = {sweeps}')
        print()

        print("Applying Paardekopers (Uj)")
        H, norm, sweeps = paardekoper_algorithm(H0, tolerance = tolerance, saved_angles =  angles)
        print(f'Off-diagonal norm {norm}')
        #print(f'Number of sweeps = {sweeps}')
        print(f'Energy {energy(H)}')
        print(f'Variance {variance(np.zeros(N), H)}')
        print()

        print('Applying greedy algorithm...')
        H = greedy_algorithm(H)
        print(f'Off-diagonal norm {off_diag_frobenius_norm(H)}') 
        en = energy(H)

        if np.abs(en-exact_energies[0]) < np.abs(en-exact_energies[1]):
            ground_energy = exact_energies[0]
        else:
            ground_energy = exact_energies[1]
        print(f"Ground energy {ground_energy} up to parity")
        print(f"Energy {en}")
        err = 100*np.abs((en-ground_energy)/ground_energy)
        print(f'Ground energy error (%) without variational layer {err}')
        var = variance(np.zeros(N), H)
        print(f"Variance {var}")
        ground_energy_error_novar.append(err)
        variance_novar.append(var)
        print()

        print("Optimizing variational layer")
        vari = partial(variance, H = H)
        theta0 = np.zeros(N)
        minimize_dictionary = minimize(vari, x0=theta0,
                                    options={'disp': True},#, 'maxiter': 30},
                                    method = 'Nelder-Mead')
        optimal_theta = minimize_dictionary['x']

        print(optimal_theta)
        print(f'Energy with variational layer {ham_average_rotated(optimal_theta, H)}')
        var = variance(optimal_theta, H)
        print(f'Variance with variational layer {var}')
        err = -100*np.abs(ham_average_rotated(optimal_theta, H)-ground_energy)/ground_energy
        print(f'Ground energy error (%) with variational layer {err}')
        ground_energy_error_var.append(err)
        variance_var.append(var)
    
        #print(f'Final norm {off_diag_frobenius_norm(H)} with variational circuit')

        print(ground_energy_error_novar) 
        print(ground_energy_error_var)
        print(variance_novar)
        print(variance_var)
        print()


plt.plot(fermion_list, ground_energy_error_novar, ".-", label = "Without variational layer")
plt.plot(fermion_list, ground_energy_error_var, ".-", label = "With variational layer")
plt.legend()
plt.xlabel("Number of fermions")
plt.ylabel("Ground energy error (%)")
plt.show()

plt.plot(fermion_list, variance_novar, ".-", label = "Without variational layer")
plt.plot(fermion_list, variance_var, ".-", label = "With variational layer")
plt.legend()
plt.xlabel("Number of fermions")
plt.ylabel("Variance")
plt.show()

"""the more fermions, the smaller the tolerance has to be in the first place in order to 
distinguish the ground/first excited energy (could happen that optimizing the variance the
system is drived to another unwanted eigenstate)
"""