from functions import *
import numpy as np
from scipy.optimize import fsolve
import sympy as sy
import math
from sympy import *

def greedy_algorithm(H):
    ''' 
    Given that H is already in Williamson form, applies a series of givens rotations
    that brings H to the lower energy state
    '''
    N = int(np.size(H,axis=0)/2)
    E = energy(H)
    x = 0
    for i in range(2*N-1):
        for l in range(1,2*N-i):
            G = givens_rotation_matrix(H, i, i+l, np.pi/2)
            H_1 = np.transpose(G) @ H @ G
            E_1 = energy(H_1)
            if  E_1 < E:
                H = H_1
                E = E_1
                x+=1
    return H, x

def givens_rotation_matrix(H, i, j, t):
    """Compute matrix for Givens rotation G(i,j;t)"""

    G = np.identity(len(H))

    G[i, i] = np.cos(2*t)
    G[j, j] = np.cos(2*t)
    G[i, j] = np.sin(2*t)
    G[j, i] = -np.sin(2*t)
    return G

def off_diag_frobenius_norm(H):
    """Computes the off-diagonal norm of the matrix H"""
    n = int(len(H)/2)
    x=0
    for i in range(n):
        for j in range(i+1,n):
            x += np.sqrt(H[2*i,2*j]**2 + H[2*i+1,2*j]**2 + H[2*i,2*j+1]**2 + H[2*i+1,2*j+1]**2)
    return 2*x

def paardekoper_algorithm(H, tolerance = 1e-5, saved_angles = [], sweeps = 1):
    '''
    Naive strategy
    '''
    N = N = int(np.size(H,axis=0)/2)
    save_angles = bool(len(saved_angles) == 0)

    n_iter = 0
    k = 0
    norm = off_diag_frobenius_norm(H)
    angles = []

    #while norm > tolerance:
    for i in range(int(sweeps)):
        for j in range(2*N-2,1,-2):
            for i in range(0,j,2):

                if save_angles:
                    phi = 0.5 * np.arctan(2 * (H[i,i+1]*H[i+1,j] - H[i,j+1]*H[j,j+1]) / ( H[i,j+1]**2  + H[i,i+1]**2 - H[i+1,j]**2 - H[j,j+1]**2))
                    angles.append(phi)#angles[i,j] = phi
                else:
                    #phi = saved_angles[i,j]
                    phi = saved_angles[k]
                G = givens_rotation_matrix(H,i,j,0.5*phi)
                H = np.transpose(G) @ H @ G

                if save_angles:
                    phi =  np.arctan(-H[i, j+1]/H[i,i+1])
                    #angles[i+1,j+1] = phi
                    angles.append(phi)
                else:
                    #phi = saved_angles[i+1,j+1]
                    phi = saved_angles[k+1]
                G = givens_rotation_matrix(H,i+1,j+1,0.5 *phi)
                H = np.transpose(G) @ H @ G

                if save_angles:
                    phi = 0.5 * np.arctan(2 * (H[i,j]*H[j,j+1] + H[i,i+1]*H[i+1,j+1]) / ( H[i,i+1]**2  + H[i,j]**2 - H[i+1,j+1]**2 - H[j,j+1]**2))
                    #angles[i,j+1] = phi
                    angles.append(phi)
                else:
                    #phi = saved_angles[i,j+1] 
                    phi = saved_angles[k+2]
                G = givens_rotation_matrix(H,i,j+1,0.5 *phi)
                H = np.transpose(G) @ H @ G

                if save_angles:
                    phi = np.arctan(-H[i, j]/H[i,i+1])
                    #angles[i+1,j] = phi
                    angles.append(phi)
                else:
                    #phi = saved_angles[i+1,j] 
                    phi = saved_angles[k+3]
                G = givens_rotation_matrix(H,i+1,j,0.5 *phi)
                H = np.transpose(G) @ H @ G

                norm = off_diag_frobenius_norm(H)
                n_iter += 1
                k += 4

        if not(save_angles) and k > len(saved_angles):
            break

    sweeps = 2*n_iter/(N*(N-1))

    if len(saved_angles) == 0:
        return H, norm, sweeps, angles
    else:
        return H, norm, sweeps


def optimal_givens_sequence(H):

    N = int(np.size(H,axis=0)/2)

    # Find the O
    eig_vals, O = np.linalg.eigh(np.matmul(H,H))
    #print(f'Determinant of O is {np.linalg.det(O)}')

    t_list = []

    #Decompose O in Givens rotations
    for j in range(2*N-1,0,-1):
        for i in range(0,j):            
            t = 0.5 * np.arctan(-O[i,j]/O[i+1,j])
            t_list.append(t)
            G = givens_rotation_matrix(O,i,i+1,t)
            O = G @ O 

    # Exact decomposition from the Williamson transformation O
    n=0
    for j in range(2*N-1,0,-1):
        for i in range(0,j):
            print(f'Apply G{i+1,i+2}')
            print(f'theta = 2t is : {2*t_list[n]}')
            G = givens_rotation_matrix(H,i,i+1,t_list[n])
            H = G @ H @ np.transpose(G)
            n += 1

def greedy_paard(H_0, exact_gs_energy, sweeps):
    N = int(np.size(H_0,axis=0)/2)

    #print('Applying Paardakopers (Uj)...')
    H, offnorm, sweeps, angles = paardekoper_algorithm(H_0, sweeps = sweeps)

    #print('Applying greedy algorithm...')
    H, givens_used_in_greedy = greedy_algorithm(H)

    final_energy = energy(H)
    energy_error = abs((final_energy - exact_gs_energy)*100/exact_gs_energy)

    number_of_givens = 4*(N*(N-1)/2)*sweeps + givens_used_in_greedy

    return energy_error, number_of_givens