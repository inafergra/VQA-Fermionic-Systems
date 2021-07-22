from algebra_functions import *
from functions import *
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)
np.random.seed(seed=1)


sweeps_list = []
for N in range(2,100,10):
    H = init_coeff_matrix(N, mean=0, H_value=1)
    init_norm = off_diag_frobenius_norm(H)
    H, norm, sweeps, angles = paardekoper_algorithm(H, tolerance = 1e-10)

    sweeps_list.append(sweeps)

    print(f'{N} fermions')
    print(f'Initial norm {init_norm}') 
    print(f'Final norm {norm}')
    print(f'Number of sweeps = {sweeps}')

    print(sweeps_list)

#sweeps_list = [1.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0]
plt.plot(range(2,100,10), sweeps_list)
plt.xlabel('Number of fermions')
plt.ylabel('Number of sweeps')
plt.title('Tolerance 1e-10')
plt.show()


'''
# Exact decomposition from the Williamson transformation O
for j in range(2*N-1,0,-1):
    for i in range(0,j):
        print(f'Apply G{i+1,i+2}')
        print(f'theta = 2t is : {2*t_list[n]}')
        G = givens_rotation_matrix(H,i,i+1,t_list[n])
        H = G @ H @ np.transpose(G)
        n += 1
        print(H)
'''
'''
H = np.array([[ 0,  3,  2,  -7],
       [ -3,  0, -5,  -12],
       [ -2, 5, 0, -8],
       [ 7, 12, 8, 0]])

# Paardekooper
n_iter = 0
norm = off_diag_frobenius_norm(H)
print('Initial H')
#print(H)
print(f'Initial norm {norm}')
while norm > 1e-10:
    for j in range(2*N-2,1,-2):
        for i in range(0,j,2):
            #print(i,j)

            phi = 0.5 * np.arctan(2 * (H[i,i+1]*H[i+1,j] - H[i,j+1]*H[j,j+1]) / ( H[i,j+1]**2  + H[i,i+1]**2 - H[i+1,j]**2 - H[j,j+1]**2))
            G = givens_rotation_matrix(H,i,j,0.5*phi)
            H = np.transpose(G) @ H @ G

            #print(H)

            phi =  np.arctan(-H[i, j+1]/H[i,i+1])
            G = givens_rotation_matrix(H,i+1,j+1,0.5 *phi)
            H = np.transpose(G) @ H @ G

            #print(H)

            phi = 0.5 * np.arctan(2 * (H[i,j]*H[j,j+1] + H[i,i+1]*H[i+1,j+1]) / ( H[i,i+1]**2  + H[i,j]**2 - H[i+1,j+1]**2 - H[j,j+1]**2))
            G = givens_rotation_matrix(H,i,j+1,0.5 *phi)
            H = np.transpose(G) @ H @ G

            #print(H)

            phi = np.arctan(-H[i, j]/H[i,i+1])
            G = givens_rotation_matrix(H,i+1,j,0.5 *phi)
            H = np.transpose(G) @ H @ G

            norm = off_diag_frobenius_norm(H)
            n_iter += 1
#print(H)
print(f'Number of iterations = {n_iter}')
print(f'Number of sweeps = {2*n_iter/(N*(N-1))}')
print(f'Norm {norm}')


#Tridiagonal matrix
for j in range(2*N-1,1,-1):
    for i in range(0,j-1):
        t = 0.5 * np.arctan(new_H[i,j]/new_H[i+1,j])
        G = givens_rotation_matrix(new_H,i,i+1,t)
        new_H = np.transpose(G) @ new_H @ G

        print(G)
        print(new_H)
        energy_list.append(energy(new_H))

#now we have to zero the elements (i,i+1) in the tridiagonal that do not belong to the diagonal blocks.
#we do this through G(i+1,2*N) rotations that induce a non zero element on the corner of the matrix
for i in range(1,2*N-1,2):
    print(i)
    t = -0.5 * np.arctan(new_H[i,i+1]/new_H[i+1,2*N-1])
    G = givens_rotation_matrix(new_H,i,2*N-1,t)
    new_H = np.transpose(G) @ new_H @ G
    print(G)
    print(new_H)
    energy_list.append(energy(new_H))


for j in range(N,1,-1):
    for b in range(1,-1,-1):
        for i in range(0,j-1):
            for a in range(0,2): 
                print(i+a,j+b)
                if b==1:
                    t = 0.5 * np.arctan(new_H[i+a,j+b]/new_H[i+a+1,j+b])
                elif b==0:
                    t = 0.5 * np.arctan(new_H[i+a,j+b]/new_H[i+a+1,j+b])
                G = givens_rotation_matrix(new_H,i+a,i+a+1,t)
                new_H = np.transpose(G) @ new_H @ G

                print(G)
                print(new_H)
                energy_list.append(energy(new_H))

exact_energies = exact_energy_levels(H,2)
print(f'Exact ground energy: {exact_energies[0]}')

#plt.plot(range(round(num_gates/2), num_gates), energy_list_num[-round(num_gates/2):], label = 'Algorithmic cooling numeric')
#plt.plot(range(round(num_gates/2), num_gates), energy_list_exact[-round(num_gates/2):], label = 'Algorithmic cooling exact')

plt.plot(energy_list, label = 'Givens decomposition')

#plotting the exact energies
#for i in range(len(exact_energies)): 
#plt.axhline(y=exact_energies[i], linestyle='-', label = f'Energy level {i}')
plt.axhline(y=exact_energies[0], color =  'r', linestyle='--', label = f'Energy level 0')
plt.axhline(y=exact_energies[1], color =  'y', linestyle='--', label = f'Energy level 1')

plt.xlabel('Number of unitaries')
plt.ylabel('Energy')
plt.legend()
#plt.savefig(f'Plots/N={N}, p={len(energy_list)}')
plt.show()
'''
