import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

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
    J = np.zeros([2*N,2*N])
    for i in range(N):
        for j in range(i+1,N): #J[i,alpha,j,alpha] is zero
            for alpha in range(0,2):
                for beta in range(alpha,2):
                    J[i+alpha,j+beta] = np.random.normal(mean, variance)
                    J[j+beta,i+alpha] = -J[i+alpha,j+beta]
    return J

def appy_h_gate(t,J, indices):
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
    exp_A = I + (np.sin(2*t)/2)*A - ((np.cos(2*t)-1)/2)*A2
    exp_A_dagger = I - (np.sin(2*t)/2)*A - ((np.cos(2*t)-1)/2)*A2

    new_J = exp_A * J * exp_A_dagger

    return new_J


def new_energy(t,J,indices):
    new_J = appy_h_gate(t,J,indices)
    new_energy = energy(new_J)
    return new_energy
N = 2

#Initial state
J = init_coeff_matrix(N)
#print(J)
init_energy = energy(J)
print(f'Initial energy is {init_energy}')

'''
i = np.random.randint(0,N) ; j = np.random.randint(0,N)
alpha = np.random.randint(0,2) ; beta = np.random.randint(0,2)
indices = [i,alpha,j,beta]
t = 0.4
new_J = appy_h_gate(t,J, indices)
print(new_J)
final_energy = energy(new_J)

print(f'Final energy is {final_energy}')
'''

#How to know we are in the ground state??
new_J = J
i=0;j=0
for num_gates in range(5):
    #draw random i,alpha,j,beta 
    while (i==j):
        i = np.random.randint(0,N) ; j = np.random.randint(0,N)
        alpha = np.random.randint(0,2) ; beta = np.random.randint(0,2)
    indices = [i,alpha,j,beta]
    t0=np.random.rand() #initial guess
    print(indices)

    #minimize_dictionary = minimize(new_energy, x0=t0,args=(new_J,indices), options={'disp': False}, method = 'Nelder-Mead')
    minimize_dictionary = differential_evolution(new_energy,args=(new_J,indices), bounds=[(-2,2)], disp = False)#, maxiter=10000)

    optimal_time = minimize_dictionary['x']
    print(f'Optimal time is: {optimal_time}')
    new_J = appy_h_gate(optimal_time,new_J, indices)
    np.set_printoptions(precision=2)
    #print(new_J)
    final_energy = energy(new_J)
    print(f'Energy is {final_energy}')

#Habera que normalizar as newJ ou algo??