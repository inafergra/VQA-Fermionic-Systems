import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
np.set_printoptions(precision=2)

def energy(J):
    '''
    Calculates the energy of a given coupling matrix J. The energy is always taken w.r.t the
    fermionic vacuum state. 
    '''
    N = int(np.size(J,axis=0)/2)
    m = 0
    for l in range(N):
        m += J[2*l+1,2*l]
    energy = 0.5*(-2*m) #0.5 due to the one half factor in front  of the hamiltonian
    return energy

def init_coeff_matrix(N, mean=0, J_value=1):
    '''
    Builds the initial (antisymmetric) matrix of coefficients with elements draw from a normal distribution with mean=mean 
    variance=6*J_value**2/N**3. The elements of the matrix J are indexed by 4 indices (i,alpha,j,beta) as
    J_{2*i+alpha,2*j+beta}.
    '''
    #variance = 6*J_value**2/N**3
    variance = 1
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
    Gives the new coupling matrix after applying the h=e^{i*c_{i,alpha}*c_{j,beta}*t} transformation
    for a time t.
    '''

    N = int(np.size(J,axis=0)/2)
    i = indices[0] ; alpha = indices[1]; j = indices[2]; beta = indices[3]

    #Compute the A matrix
    A = np.zeros([2*N,2*N])
    A[2*i+alpha,2*j+beta] = 2
    A[2*j+beta,2*i+alpha] = -2
    #print('A is:')
    #print(A)

    #Compute exp(A)
    I = np.identity(2*N)
    A2 = np.matmul(A,A)
    exp_A = I + np.sin(2*t)*A/2 - (np.cos(2*t)-1)*A2/4
    exp_A_T = I - np.sin(2*t)*A/2 - (np.cos(2*t)-1)*A2/4
    #print('expA*expAT:')
    #print(np.matmul(exp_A,exp_A_T)) #Should be the identity!

    #Compute the new coupling matrix
    new_J = np.matmul(np.matmul(exp_A_T,J),exp_A)

    return new_J

def new_energy(optimal_time,new_J,indices): 
    '''
    Function to input in the minimization loop. Computes the energy of the coupling matrix after 
    going through the transformation h[indices] for a time optimal_time.
    '''
    new_J = appy_h_gate(optimal_time,new_J,indices)
    new_energy = energy(new_J)
    return new_energy

def exact_energy_levels(J, k):
    '''
    Finds the first k exact energy levels through exact diagonalization
    '''
    energies = np.zeros(k)

    eig_vals, O = np.linalg.eig(np.matmul(J,J))
    #print(eig_vals) #-(epsilon_k^2)
    parity = np.linalg.det(O)
    print(parity)
    epsilons = np.sqrt(abs(eig_vals))
    energies[0] = -np.sum(epsilons)/2
    k =3
    for i in range(1, k-1):
        #print('epsilons', epsilons)
        a  = np.argmin(epsilons) 
        epsilons[a]+= -energies[0]
        a  = np.argmin(epsilons)
        #print('index of the min', a)      
        energies[i] = energies[i-1] + 2*epsilons[a]
        epsilons[a]+= -energies[0]

    return energies


from scipy.optimize import fsolve
import sympy as sy
import math

def equations(x, indices, J):
    x, y = x[0], x[1]

    #Compute the A matrix
    N = int(np.size(J,axis=0)/2)
    i = indices[0] ; alpha = indices[1]; j = indices[2]; beta = indices[3]
    A = np.zeros([2*N,2*N])
    A[2*i+alpha,2*j+beta] = 2
    A[2*j+beta,2*i+alpha] = -2

    JA = np.matmul(J,A)
    AJ = np.matmul(A,J)
    AA = np.matmul(A,A)
    comm_JA = JA - AJ
    a = energy(comm_JA)
    b = energy(J@AA + AA@J)
    g = energy(np.matmul(np.matmul(A,comm_JA), A))
    d = energy(A@J@A)
    e = energy(AA@J@AA)

    return (a*y+b*x/2+g*(y*y-y-x*x)/4-d*x*y+e*x*(1-y)/4, x**2+y**2-1)

def equations_sympy(indices, J):
    #Compute the A matrix
    N = int(np.size(J,axis=0)/2)
    i = indices[0] ; alpha = indices[1]; j = indices[2]; beta = indices[3]
    A = np.zeros([2*N,2*N])
    A[2*i+alpha,2*j+beta] = 2
    A[2*j+beta,2*i+alpha] = -2
    JA = np.matmul(J,A)
    AJ = np.matmul(A,J)
    AA = np.matmul(A,A)
    comm_JA = JA - AJ
    a = energy(comm_JA)
    b = energy(J@AA + AA@J)
    g = energy(A@comm_JA@A)
    #g = energy(np.matmul(np.matmul(A,comm_JA), A))
    d = energy(A@J@A)
    e = energy(AA@J@AA)

    x, y = sy.symbols('x y')
    eq1 =a*y+b*x/2+g*(y*y-y-x*x)/4-d*x*y+e*x*(1-y)/4
    eq2 = x**2+y**2-1
    sol = sy.solve([eq1, eq2], [x, y])
    soln = [tuple(v.evalf() for v in s) for s in sol]
    print(soln)
    return soln
#Groener basis--------------------------

def groebner(x, indices, J):

    x, y = x[0], x[1]

    #Compute the A matrix
    N = int(np.size(J,axis=0)/2)
    i = indices[0] ; alpha = indices[1]; j = indices[2]; beta = indices[3]
    A = np.zeros([2*N,2*N])
    A[2*i+alpha,2*j+beta] = 2
    A[2*j+beta,2*i+alpha] = -2

    JA = np.matmul(J,A)
    AJ = np.matmul(A,J)
    AA = np.matmul(A,A)
    comm_JA = JA - AJ
    a = energy(comm_JA)
    b = energy(J@AA + AA@J)
    g = energy(A@comm_JA@A)
    d = energy(A@J@A)
    e = energy(AA@J@AA)

    a1 = (16 * d ** 2 + 8 * d * e + e ** 2 + 4 * g ** 2)
    a2 = (16 * a * g - 16 * b * d - 4 * b * e - 8 * d * e - 2 * e ** 2)
    a3 = (16 * a ** 2 + 4 * b ** 2 + 4 * b * e - 16 * d ** 2 - 8 * d * e - 8 * g ** 2)
    a4 = (-16 * a * g + 16 * b * d + 4 * b * e + 8 * d * e + 2 * e ** 2)
    a5 = - 4 * b ** 2 - 4 * b * e - e ** 2 + 4 * g ** 2

    b1 = (32 * a * b * d + 8 * a * b * e + 16 * a * d * e + 4 * a * e ** 2 + 8 * b ** 2 * g + 8 * b * e * g - 32 * d ** 2 * g - 16 * d * e * g)
    b2 = (64 * d ** 3 + 48 * d ** 2 * e + 12 * d * e ** 2 + 16 * d * g ** 2 + e ** 3 + 4 * e * g ** 2)
    b3 = (64 * a * d * g + 16 * a * e * g - 32 * b * d ** 2 - 16 * b * d * e - 2 * b * e ** 2 + 8 * b * g ** 2 - 16 * d ** 2 * e - 8 * d * e ** 2 - e ** 3 + 4 * e * g ** 2)
    b4 = (64 * a ** 2 * d + 16 * a ** 2 * e + 16 * a * b * g + 8 * a * e * g - 64 * d ** 3 - 48 * d ** 2 * e - 12 * d * e ** 2 - 16 * d * g ** 2 - e ** 3 - 4 * e * g ** 2)
    b5 = - 32 * a * d * g - 8 * a * e * g + 32 * b * d ** 2 + 16 * b * d * e + 2 * b * e ** 2 - 8 * b * g ** 2 + 16 * d ** 2 * e + 8 * d * e ** 2 + e ** 3 - 4 * e * g ** 2    
    
    return (a1 * y**4 + a2 * y**3 + a3 * y**2 + a4 * y + a5, b1 * x + b2 * y**3 + b3 * y**2 + b4 * y +b5)

def groebner_sympy(indices, J):

    #Compute the A matrix
    N = int(np.size(J,axis=0)/2)
    i = indices[0] ; alpha = indices[1]; j = indices[2]; beta = indices[3]
    A = np.zeros([2*N,2*N])
    A[2*i+alpha,2*j+beta] = 2
    A[2*j+beta,2*i+alpha] = -2
    print(A)
    JA = np.matmul(J,A)
    AJ = np.matmul(A,J)
    AA = np.matmul(A,A)
    comm_JA = JA - AJ
    a = energy(comm_JA)
    b = energy(J@AA + AA@J)
    #g = energy(A@comm_JA@A)
    g = energy(np.matmul(np.matmul(A,comm_JA), A))
    d = energy(A@J@A)
    e = energy(AA@J@AA)
    print(a,b,g,d,e)
    a1 = (16 * d ** 2 + 8 * d * e + e ** 2 + 4 * g ** 2)
    a2 = (16 * a * g - 16 * b * d - 4 * b * e - 8 * d * e - 2 * e ** 2)
    a3 = (16 * a ** 2 + 4 * b ** 2 + 4 * b * e - 16 * d ** 2 - 8 * d * e - 8 * g ** 2)
    a4 = (-16 * a * g + 16 * b * d + 4 * b * e + 8 * d * e + 2 * e ** 2)
    a5 = - 4 * b ** 2 - 4 * b * e - e ** 2 + 4 * g ** 2
    print(a1,a2,a3,a4,a5)
    b1 = (32 * a * b * d + 8 * a * b * e + 16 * a * d * e + 4 * a * e ** 2 + 8 * b ** 2 * g + 8 * b * e * g - 32 * d ** 2 * g - 16 * d * e * g)
    b2 = (64 * d ** 3 + 48 * d ** 2 * e + 12 * d * e ** 2 + 16 * d * g ** 2 + e ** 3 + 4 * e * g ** 2)
    b3 = (64 * a * d * g + 16 * a * e * g - 32 * b * d ** 2 - 16 * b * d * e - 2 * b * e ** 2 + 8 * b * g ** 2 - 16 * d ** 2 * e - 8 * d * e ** 2 - e ** 3 + 4 * e * g ** 2)
    b4 = (64 * a ** 2 * d + 16 * a ** 2 * e + 16 * a * b * g + 8 * a * e * g - 64 * d ** 3 - 48 * d ** 2 * e - 12 * d * e ** 2 - 16 * d * g ** 2 - e ** 3 - 4 * e * g ** 2)
    b5 = - 32 * a * d * g - 8 * a * e * g + 32 * b * d ** 2 + 16 * b * d * e + 2 * b * e ** 2 - 8 * b * g ** 2 + 16 * d ** 2 * e + 8 * d * e ** 2 + e ** 3 - 4 * e * g ** 2    
    print(b1,b2,b3,b4,b5)


    x, y = sy.symbols('x y')
    eq1 = a1 * y**4 + a2 * y**3 + a3 * y**2 + a4 * y + a5
    eq2 = b1 * x + b2 * y**3 + b3 * y**2 + b4 * y + b5
    print(eq2)
    sol = sy.solveset([eq1, eq2], [x, y])
    print(sol)
    soln = [tuple(v.evalf() for v in s) for s in sol]
    print(soln)
    return soln


def arc_things(x,y):
    '''
    Returns the angle t corresponding to x = sin(2t), y = cos(2t).
    '''
    t = 0
    if x>0: #upper half of the circle --> arcos works since returns between (0,pi)
        t = np.arccos(y)/2 #(0,pi)
    elif x<0 and y>0: #right bottom half --> arcsin works since returns between (-pi/2, pi/2)
        t = np.arcsin(x)/2 #(-pi/2, pi/2)
    elif x<0 and y<0:
        t = - np.arccos(y)/2
    return t


from scipy.optimize import root    
def optimal_t(indices, J):
    #x, y =  fsolve(equations, (1,1), args=(indices, J))
    #x, y =  fsolve(groebner, (.5,.5), args=(indices, J))
    #x,y = groebner_sympy(indices, J)
    x = float(equations_sympy(indices, J)[0][0])
    y = float(equations_sympy(indices, J)[0][1])

    #x, y =  root(groebner_sympy, args=(indices, J)).x
    if abs(x)>1 or abs(y)>1:
        t=0
    else:
        t= arc_things(x,y)
    return t

def f(t, indices, J):
    #Compute the A matrix
    N = int(np.size(J,axis=0)/2)
    i = indices[0] ; alpha = indices[1]; j = indices[2]; beta = indices[3]
    A = np.zeros([2*N,2*N])
    A[2*i+alpha,2*j+beta] = 2
    A[2*j+beta,2*i+alpha] = -2

    c = (np.cos(2*t)-1)/4
    s = np.sin(2*t)/2
    JA = np.matmul(J,A)
    AJ = np.matmul(A,J)
    AA = np.matmul(A,A)
    comm_JA = JA - AJ
    a = energy(comm_JA)
    b = energy(J@AA + AA@J)
    g = energy(A@comm_JA@A)
    d = energy(A@J@A)
    e = energy(AA@J@AA)

    return a*s-b*c+g*s*c-d*(s*s)+e*(c*c)

