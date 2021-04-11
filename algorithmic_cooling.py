import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
np.set_printoptions(precision=2)

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
        m += J[2*l+1,2*l]
    energy = -2*m
    return energy

def init_coeff_matrix(N, mean=0, J_value=1):
    variance = 6*J_value/N**3
    J = np.zeros([2*N,2*N])
    for i in range(N):
        for j in range(i,N): #J[i,alpha,i,beta] is zero
            for alpha in range(0,2):
                for beta in range(0,2):
                    if (2*i+alpha!=2*j+beta): #diagonal is zero
                        J[2*i+alpha,2*j+beta] = np.random.normal(mean, variance)
                        J[2*j+beta,2*i+alpha] = -J[2*i+alpha,2*j+beta]
    return J

def appy_h_gate(t,J, indices):
    '''
    Gives the new coupling matrix after applying the h tranformation
    '''

    N = int(np.size(J,axis=0)/2)
    i = indices[0] ; alpha = indices[1]; j = indices[2]; beta = indices[3]

    A = np.zeros([2*N,2*N])
    A[2*i+alpha,2*j+beta] = 2
    A[2*j+beta,2*i+alpha] = -2
    #print('A is:')
    #print(A)

    I = np.identity(2*N)
    A2 = np.matmul(A,A)
    exp_A = I + np.sin(2*t)*A/2 - (np.cos(2*t)-1)*A2/4
    exp_A_dagger = I - np.sin(2*t)*A/2 - (np.cos(2*t)-1)*A2/4
    #print('expA*expAT:')
    #print(np.matmul(exp_A,exp_A_dagger)) #wrong! not the identity
    new_J = np.matmul(np.matmul(exp_A,J),exp_A_dagger)

    return new_J


def new_energy(optimal_time,new_J,indices):
    new_J = appy_h_gate(optimal_time,new_J,indices)
    new_energy = energy(new_J)
    return new_energy


N = 100

#Initial state
J = init_coeff_matrix(N)
print('Initial J:')
print(J)
init_energy = energy(J)
print(f'Initial energy is {init_energy}')

'''
i = np.random.randint(0,N) ; j = np.random.randint(0,N)
alpha = np.random.randint(0,2) ; beta = np.random.randint(0,2)
indices = [i,alpha,j,beta]
print(indices)
t = 1
new_J = appy_h_gate(t,J, indices)
print('New J:')
print(new_J)
final_energy = energy(new_J)

print(f'Final energy is {final_energy}')
'''

#How to know we are in the ground state??
new_J = J

for num_gates in range(5):
    #draw random i,alpha,j,beta 
    i=0;j=0
    while (i==j):
        i = np.random.randint(0,N) ; j = np.random.randint(0,N)
        alpha = np.random.randint(0,2) ; beta = np.random.randint(0,1)
    indices = [i,alpha,j,beta]
    print(indices)

    t0=np.random.rand() #initial guess
    #minimize_dictionary = minimize(new_energy, x0=t0,args=(new_J,indices), options={'disp': False}, method = 'Nelder-Mead')
    minimize_dictionary = differential_evolution(new_energy,args=(new_J,indices), bounds=[(0,2)], disp = False)#, maxiter=10000)

    optimal_time = minimize_dictionary['x'][0]
    print(f'Optimal time is: {optimal_time}')
    new_J = appy_h_gate(optimal_time,new_J, indices)
    #print(new_J)
    final_energy = energy(new_J)
    print(f'Energy is {final_energy}')

#Habera que normalizar as newJ ou algo?? energy is always zero if alpha!=beta

