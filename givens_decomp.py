from algebra_functions import *

from functions import *
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)
#np.random.seed(seed=1)

#Number of fermions
N = 2

#Initial state
J = init_coeff_matrix(N, mean=0, J_value=1)
print('Initial J:')
print(J)
init_energy = energy(J)
print(f'Initial energy is {init_energy}')

new_J = J
energy_list=[]
energy_list.append(init_energy)

# Find the O
eig_vals, O = np.linalg.eigh(np.matmul(J,J))
#print(np.linalg.matrix_rank(O))
#print(O)
print('Williamson matrix W:')
print(np.matmul(np.matmul(np.transpose(O),J),O))
print(f'Determinant of O is {np.linalg.det(O)}')

t_list = []
indices = []

for j in range(2*N-1,0,-1):
    for i in range(0,j):
        print(i,j)
        
        t = 0.5 * np.arctan(O[i,j]/O[i+1,j])
        t_list.append(t)
        G = givens_rotation_matrix(O,i,i+1,t)
        O = G @ O 

        print(O)

print('Now the Js:')
print('----------------------')
n = 0
for j in range(2*N-1,0,-1):
    for i in range(0,j):
        print(i,j)
        G = givens_rotation_matrix(J,i,i+1,t_list[n])
        J = np.transpose(G) @ J @ G
        n += 1
        print(J)



'''
#Tridiagonal matrix
for j in range(2*N-1,1,-1):
    for i in range(0,j-1):
        t = 0.5 * np.arctan(new_J[i,j]/new_J[i+1,j])
        G = givens_rotation_matrix(new_J,i,i+1,t)
        new_J = np.transpose(G) @ new_J @ G

        print(G)
        print(new_J)
        energy_list.append(energy(new_J))

#now we have to zero the elements (i,i+1) in the tridiagonal that do not belong to the diagonal blocks.
#we do this through G(i+1,2*N) rotations that induce a non zero element on the corner of the matrix
for i in range(1,2*N-1,2):
    print(i)
    t = -0.5 * np.arctan(new_J[i,i+1]/new_J[i+1,2*N-1])
    G = givens_rotation_matrix(new_J,i,2*N-1,t)
    new_J = np.transpose(G) @ new_J @ G
    print(G)
    print(new_J)
    energy_list.append(energy(new_J))


for j in range(N,1,-1):
    for b in range(1,-1,-1):
        for i in range(0,j-1):
            for a in range(0,2): 
                print(i+a,j+b)
                if b==1:
                    t = 0.5 * np.arctan(new_J[i+a,j+b]/new_J[i+a+1,j+b])
                elif b==0:
                    t = 0.5 * np.arctan(new_J[i+a,j+b]/new_J[i+a+1,j+b])
                G = givens_rotation_matrix(new_J,i+a,i+a+1,t)
                new_J = np.transpose(G) @ new_J @ G

                print(G)
                print(new_J)
                energy_list.append(energy(new_J))

exact_energies = exact_energy_levels(J,2)
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
