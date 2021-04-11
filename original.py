import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

#Comezar coa matriz J de coeficientes (distribución gaussiana?)
#O estado inicial é o vacío: calcular a súa enerxía. 
#Pra saber como actúa e^{iht} sobre J: calcular a matriz A e a súa exponencial coa expresión analítica
#e multiplicar as matrices pra conseguir J'
#Calcular outra vez a enerxía. Optimizar pra que a enerxía sempre baixe en cada paso

def energy(J):
    '''
    Calculates the energy of a given coupling matrix J. The energy is always taken w.r.t the
    fermionic vacuum state. 
    '''
    N = int(np.size(J,axis=0)/2)
    m = 0
    for l in range(N):
        m += J[l+1,l]
    energy = -2*m
    return energy

def init_coeff_matrix(N, mean=0, variance=1):
    return np.random.normal(mean, variance, (2*N, 2*N))

def h_gate(t,J, indices):
    '''
    Gives the new coupling matrix after applying the h tranformation
    '''

    N = int(np.size(J,axis=0)/2)
    i = indices[0] ; alpha = indices[1]; j = indices[2]; beta = indices[3]

    A = np.zeros([2*N,2*N])
    I = np.identity(2*N)
    A[i+alpha,j+beta] = 2
    A[j+beta,i+alpha] = -2

    A2 = A*A
    exp_A = I + (np.sin(2*t)/2)*A + ((np.cos(2*t)-1)/4)*A2
    exp_A_dagger = I - (np.sin(2*t)/2)*A + ((np.cos(2*t)-1)/4)*A2

    new_J = exp_A * J * exp_A_dagger

    return new_J


N = 5

J = init_coeff_matrix(N)
init_energy = energy(J)

print(f'Initial energy is {init_energy}')

#draw random i,alpha,j,beta 
indices = [1,0,0,0]
t=.3
new_J = h_gate(t,J, indices)

final_energy = energy(new_J)
print(f'Final energy is {final_energy}')