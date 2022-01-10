from algebra_functions import *
from functions import *
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
np.set_printoptions(precision=2)
#np.random.seed(seed=10)

N = 10
variational_layer = True
tolerance = 1e2

fermion_list = range(3,7)
sweeps_list = range(1,5)
iterations = 15


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

    ground_energy_error_novar = []
    ground_energy_error_var = []
        
    for N in fermion_list:
        avr_list_err_novar = np.zeros(iterations)
        avr_list_err_var = np.zeros(iterations)

        for iteration in range(iterations):

            sweeps = 1

            print(f"{N} fermions")
            print(f"{sweeps} sweeps")

            H0 = init_coeff_matrix(N)
            init_norm = off_diag_frobenius_norm(H0)
            exact_energies, parity = exact_energy_levels(H0,2)
            print(f'Exact ground energy: {exact_energies[0]}')
            print(f'Exact first excited energy: {exact_energies[1]}')

            H, norm, sweeps, angles = paardekoper_algorithm(H0, sweeps = sweeps)
            H, number_of_givens = greedy_algorithm(H)
            en = energy(H)

            H0 = np.copy(H)
            cooled_energy = randomized_cooling(H0)[-1]
            print(cooled_energy)

            if np.abs(cooled_energy-exact_energies[0])<np.abs(cooled_energy-exact_energies[1]):
                ground_energy = exact_energies[0]
            else:
                ground_energy = exact_energies[1]

            print(np.abs(cooled_energy-exact_energies[0]))
            print(np.abs(cooled_energy-exact_energies[1]))
            print(f"Ground energy {ground_energy} up to parity")

            print(f"Energy without variational layer {en}")
            err_novar = 100*np.abs((en-ground_energy)/ground_energy)
            avr_list_err_novar[iteration] = err_novar

            print(f'Ground energy error (%) without variational layer {err_novar}')
            vari = partial(ham_average_rotated, H=H)
            theta0 = np.zeros(N)
            #print(vari(theta0))
            bounds = [(0, 2*np.pi)]*N

            minimize_dictionary = minimize(vari, x0=theta0,
                                        options={'disp': True,'maxiter': 1000},
                                        method = 'Nelder-Mead')
            #minimize_dictionary = differential_evolution(vari, maxiter=10000,# 'maxiter': 30},
            #                            bounds = bounds)
            optimal_theta = minimize_dictionary['x']

            print(optimal_theta)
            print(f'Energy with variational layer {ham_average_rotated(optimal_theta, H)}')
            #var = variance(optimal_theta, H)
            #print(f'Variance with variational layer {var}')
            err_var = -100*np.abs(ham_average_rotated(optimal_theta, H)-exact_energies[0])/exact_energies[0]
            avr_list_err_var[iteration] = err_var
            print(f'Ground energy error (%) with variational layer {err_var}')
            #variance_var.append(var)
            #print(f'Final norm {off_diag_frobenius_norm(H)} with variational circuit')

            print()


        ground_energy_error_var.append(np.average(avr_list_err_var))
        ground_energy_error_novar.append(np.average(avr_list_err_novar))




plt.plot(fermion_list, ground_energy_error_novar, ".-", label = "Without variational layer")
plt.plot(fermion_list, ground_energy_error_var, ".-", label = "With variational layer")
plt.legend()
plt.xlabel("Number of fermions")
plt.ylabel("Ground energy error (%)")
plt.show()



"""the more fermions, the smaller the tolerance has to be in the first place in order to 
distinguish the ground/first excited energy (could happen that optimizing the variance the
system is drived to another unwanted eigenstate)
"""