from functions import *
import numpy as np
from scipy.optimize import fsolve
import sympy as sy
import math
from sympy import *


def givens_rotation_symbolic(H,i,j,t):

    G = eye(len(H))
    G[i, i] = cos(2*t)
    G[j, j] = cos(2*t)
    G[i, j] = sin(2*t)
    G[j, i] = -sin(2*t)
    return G


def equations(x, indices, H):
    x, y = x[0], x[1]

    #Compute the A matrix
    N = int(np.size(H,axis=0)/2)
    i = indices[0] ; alpha = indices[1]; j = indices[2]; beta = indices[3]
    A = np.zeros([2*N,2*N])
    A[2*i+alpha,2*j+beta] = 2
    A[2*j+beta,2*i+alpha] = -2

    HA = np.matmul(H,A)
    AH = np.matmul(A,H)
    AA = np.matmul(A,A)
    comm_HA = HA - AH
    a = energy(comm_HA)
    b = energy(H@AA + AA@H)
    g = energy(np.matmul(np.matmul(A,comm_HA), A))
    d = energy(A@H@A)
    e = energy(AA@H@AA)

    return (a*y+b*x/2+g*(y*y-y-x*x)/4-d*x*y+e*x*(1-y)/4, x**2+y**2-1)

def equations_sympy(indices, H):
    #Compute the A matrix
    N = int(np.size(H,axis=0)/2)
    i = indices[0] ; alpha = indices[1]; j = indices[2]; beta = indices[3]
    A = np.zeros([2*N,2*N])
    A[2*i+alpha,2*j+beta] = 2
    A[2*j+beta,2*i+alpha] = -2
    HA = np.matmul(H,A)
    AH = np.matmul(A,H)
    AA = np.matmul(A,A)
    comm_HA = HA - AH
    a = energy(comm_HA)
    b = energy(H@AA + AA@H)
    g = energy(A@comm_HA@A)
    #g = energy(np.matmul(np.matmul(A,comm_HA), A))
    d = energy(A@H@A)
    e = energy(AA@H@AA)

    x, y = sy.symbols('x y')
    eq1 =a*y+b*x/2+g*(y*y-y-x*x)/4-d*x*y+e*x*(1-y)/4
    eq2 = x**2+y**2-1
    sol = sy.solve([eq1, eq2], [x, y])
    soln = [tuple(v.evalf() for v in s) for s in sol]
    print(soln)
    return soln

#Groener basis--------------------------

def groebner(x, indices, H):

    x, y = x[0], x[1]

    #Compute the A matrix
    N = int(np.size(H,axis=0)/2)
    i = indices[0] ; alpha = indices[1]; j = indices[2]; beta = indices[3]
    A = np.zeros([2*N,2*N])
    A[2*i+alpha,2*j+beta] = 2
    A[2*j+beta,2*i+alpha] = -2

    HA = np.matmul(H,A)
    AH = np.matmul(A,H)
    AA = np.matmul(A,A)
    comm_HA = HA - AH
    a = energy(comm_HA)
    b = energy(H@AA + AA@H)
    g = energy(A@comm_HA@A)
    d = energy(A@H@A)
    e = energy(AA@H@AA)

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

def groebner_sympy(x, indices, H):

    x, y = x[0], x[1]
    #Compute the A matrix
    N = int(np.size(H,axis=0)/2)
    i = indices[0] ; alpha = indices[1]; j = indices[2]; beta = indices[3]
    A = np.zeros([2*N,2*N])
    A[2*i+alpha,2*j+beta] = 2
    A[2*j+beta,2*i+alpha] = -2

    HA = np.matmul(H,A)
    AH = np.matmul(A,H)
    AA = np.matmul(A,A)
    comm_HA = HA - AH
    a = energy(comm_HA)
    b = energy(H@AA + AA@H)
    g = energy(A@comm_HA@A)
    #g = energy(np.matmul(np.matmul(A,comm_HA), A))
    d = energy(A@H@A)
    e = energy(AA@H@AA)
    #print(a,b,g,d,e)
    a1 = (16 * d ** 2 + 8 * d * e + e ** 2 + 4 * g ** 2)
    a2 = (16 * a * g - 16 * b * d - 4 * b * e - 8 * d * e - 2 * e ** 2)
    a3 = (16 * a ** 2 + 4 * b ** 2 + 4 * b * e - 16 * d ** 2 - 8 * d * e - 8 * g ** 2)
    a4 = (-16 * a * g + 16 * b * d + 4 * b * e + 8 * d * e + 2 * e ** 2)
    a5 = - 4 * b ** 2 - 4 * b * e - e ** 2 + 4 * g ** 2
    #print(a1,a2,a3,a4,a5)
    b1 = (32 * a * b * d + 8 * a * b * e + 16 * a * d * e + 4 * a * e ** 2 + 8 * b ** 2 * g + 8 * b * e * g - 32 * d ** 2 * g - 16 * d * e * g)
    b2 = (64 * d ** 3 + 48 * d ** 2 * e + 12 * d * e ** 2 + 16 * d * g ** 2 + e ** 3 + 4 * e * g ** 2)
    b3 = (64 * a * d * g + 16 * a * e * g - 32 * b * d ** 2 - 16 * b * d * e - 2 * b * e ** 2 + 8 * b * g ** 2 - 16 * d ** 2 * e - 8 * d * e ** 2 - e ** 3 + 4 * e * g ** 2)
    b4 = (64 * a ** 2 * d + 16 * a ** 2 * e + 16 * a * b * g + 8 * a * e * g - 64 * d ** 3 - 48 * d ** 2 * e - 12 * d * e ** 2 - 16 * d * g ** 2 - e ** 3 - 4 * e * g ** 2)
    b5 = - 32 * a * d * g - 8 * a * e * g + 32 * b * d ** 2 + 16 * b * d * e + 2 * b * e ** 2 - 8 * b * g ** 2 + 16 * d ** 2 * e + 8 * d * e ** 2 + e ** 3 - 4 * e * g ** 2    
    #print(b1,b2,b3,b4,b5)

    x, y = sy.symbols('x y')
    eq1 = a1 * y**4 + a2 * y**3 + a3 * (y**2) + a4 * y + a5
    eq2 = b1 * x + b2 * y**3 + b3 * (y**2) + b4 * y + b5
    
    #print(eq2)
    sol = sy.solve([eq1, eq2], [x, y],simplify=False)
    print(sol)
    soln = [tuple(v.evalf() for v in s) for s in sol]
    print(soln)
    return soln


def arc_things(x,y):
    '''
    Returns the angle t corresponding to x = sin(2t), y = cos(2t).
    '''
    if x>0:
        t =  0.5*np.arctan(x/y)
    else:
        t = 0.5*np.arctan(x/y) + np.pi
    return t
    '''
    t = 0
    if x>0: #upper half of the circle --> arcos works since returns between (0,pi)
        t = np.arccos(y)/2 #(0,pi)
    elif x<0 and y>0: #right bottom half --> arcsin works since returns between (-pi/2, pi/2)
        t = np.arcsin(x)/2 #(-pi/2, pi/2)
    elif x<0 and y<0:
        t = - np.arccos(y)/2
    return t
    '''

from scipy.optimize import root    
from sympy.solvers import solve

def optimal_t(indices, H):

    #x, y =  fsolve(equations, (1,1), args=(indices, H))
    #x, y =  fsolve(groebner, (.5,.5), args=(indices, H))
    #x,y = groebner_sympy(indices, H)
    #print(equations_sympy(indices, H))
    x = float(equations_sympy(indices, H)[0][0])
    y = float(equations_sympy(indices, H)[0][1])

    #x, y =  root(groebner_sympy, (.5,.5), args=(indices, H)).x
    t = arc_things(x,y)
    print(f't={t}')  #x = sin(2t), y = cos(2t)
    return t
    '''
    if abs(x)>1 or abs(y)>1:
        t=0
    else:
        t= arc_things(x,y)
    return t
    '''

def f(t, indices, H):
    #Compute the A matrix
    N = int(np.size(H,axis=0)/2)
    i = indices[0] ; alpha = indices[1]; j = indices[2]; beta = indices[3]
    A = np.zeros([2*N,2*N])
    A[2*i+alpha,2*j+beta] = 2
    A[2*j+beta,2*i+alpha] = -2

    c = (np.cos(2*t)-1)/4
    s = np.sin(2*t)/2
    HA = np.matmul(H,A)
    AH = np.matmul(A,H)
    AA = np.matmul(A,A)
    comm_HA = HA - AH
    a = energy(comm_HA)
    b = energy(H@AA + AA@H)
    g = energy(A@comm_HA@A)
    d = energy(A@H@A)
    e = energy(AA@H@AA)

    return a*s-b*c+g*s*c-d*(s*s)+e*(c*c)

