import numpy as np

import cirq
from openfermion.ops import MajoranaOperator
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator

from itertools import combinations, permutations

def convert_H_majorana_to_qubit_L(inds, J):
    '''
    Convert SYK hamiltonian (dictionary) from majorana terms to Pauli terms
    '''
    #print(inds)
    #rint()
    ham_terms = [MajoranaOperator(ind, J[ind]) for ind in inds]
    #print(ham_terms)
    #print()
    ham_sum = sum_ops(ham_terms)
    #print(ham_sum)
    #print()
    return jordan_wigner(ham_sum)

def convert_H_majorana_to_qubit_R(inds, J, N):
    '''
    Convert SYK hamiltonian (dictionary) from majorana terms to Pauli terms
    '''
    #print(inds)
    #print()
    ham_terms = [MajoranaOperator(ind, J[tuple(np.array(ind)-2*N)]) for ind in inds]
    ham_sum = sum_ops(ham_terms)
    #print(ham_terms)
    #print()
    #print(ham_sum)
    #print()
    return jordan_wigner(ham_sum)

def sum_ops(operators):
    '''
    Wrapper for summing a list of majorana operators
    '''
    return sum(operators, MajoranaOperator((), 0))

def tfd_exact(N, TFD_model, int_dict):
    '''
    Solves the TFD model exactly
    '''
    J_L_tens = TFD_model[0]
    J_R_tens = TFD_model[1]

    L_indices = range(0, 2*N)
    R_indices = range(2*N, 4*N)
    SYK_L_indices = list(permutations(L_indices,4))
    SYK_R_indices = list(permutations(R_indices,4))
    int_indices = [(l, r) for l, r in zip(L_indices, R_indices)]

    #print(SYK_L_indices)
    #print()
    #print(SYK_R_indices)
    H_L = convert_H_majorana_to_qubit_L(SYK_L_indices, J_L_tens)
    H_R = convert_H_majorana_to_qubit_R(SYK_R_indices, J_R_tens, N)
    H_int = convert_H_majorana_to_qubit_L(int_indices, int_dict)

    total_ham = H_L + H_R + H_int
    matrix_ham = get_sparse_operator(total_ham) # todense() allows for ED

    # Diagonalize qubit hamiltonian to compare the spectrum of variational energy
    e, v = np.linalg.eigh(matrix_ham.todense())    

    return e[:10]



"""    


def convert_H_majorana_to_qubit(J, N):
    '''
    Convert SYK hamiltonian (dictionary) from majorana terms to Pauli terms
    '''

    ham_terms = [MajoranaOperator(ind, J[ind]) for ind in permutations(range(1,2*N), 4)]
    ham_sum = sum_ops(ham_terms)
    return jordan_wigner(ham_sum)

def sum_ops(operators):
    '''
    Wrapper for summing a list of majorana operators
    '''
    return sum(operators, MajoranaOperator((), 0))

def tfd_exact(N, TFD_model):
    '''
    Solves the TFD model exactly
    '''
    J_L = TFD_model[0]
    J_R = TFD_model[1]
    H_int = TFD_model[2]

    L_indices = range(0, 2*N)
    R_indices = range(2*N, 4*N)
    SYK_L_indices = list(combinations(L_indices,4))
    SYK_R_indices = list(combinations(R_indices,4))
    int_indices = [(l, r) for l, r in zip(L_indices, R_indices)]
    mu =1

    H_L = jordan_wigner(sum_ops([MajoranaOperator(ind, J_L[ind]) for ind in SYK_L_indices]))
    H_R = jordan_wigner(sum_ops([MajoranaOperator(tuple(np.array(ind)+4), J_R[ind]) for ind in SYK_L_indices]))
    H_int = jordan_wigner(sum_ops([MajoranaOperator(ind, H_int[0,0]) for ind in int_indices]))

    total_ham = H_L + H_R + H_int
    matrix_ham = get_sparse_operator(total_ham) # todense() allows for ED

    # Diagonalize qubit hamiltonian to compare the spectrum of variational energy
    e, v = np.linalg.eigh(matrix_ham.todense())    

    return e[:10]

#a = [print(i) for i in permutations(range(1,4), 4)]
#print(a)
"""    


#########

"""
def convert_H_majorana_to_qubit(inds, J_dict, N):
    '''
    Convert SYK hamiltonian (dictionary) from majorana terms to Pauli terms
    '''
    ham_terms = [MajoranaOperator(ind, J_dict[ind]) for ind in inds]
    ham_sum = sum_ops(ham_terms)
    #print(ham_sum)
    return jordan_wigner(ham_sum)

def sum_ops(operators):
    '''
    Wrapper for summing a list of majorana operators
    '''
    return sum(operators, MajoranaOperator((), 0))

def tfd_exact(N, TFD_dict):
    '''
    Solves the TFD model exactly
    '''
    J_L_dict = TFD_dict[0]
    J_R_dict = TFD_dict[1]
    int_dict = TFD_dict[2]

    L_indices = range(0, 2*N)
    R_indices = range(2*N, 4*N)
    SYK_L_indices = list(combinations(L_indices,4))
    SYK_R_indices = list(combinations(R_indices,4))
    int_indices = [(l, r) for l, r in zip(L_indices, R_indices)]

    H_L = convert_H_majorana_to_qubit(SYK_L_indices, J_L_dict, 2*N)
    H_R = convert_H_majorana_to_qubit(SYK_R_indices, J_R_dict, 2*N)
    H_int = convert_H_majorana_to_qubit(int_indices, int_dict, 2*N)

    total_ham = H_L + H_R + H_int
    matrix_ham = get_sparse_operator(total_ham) # todense() allows for ED

    # Diagonalize qubit hamiltonian to compare the spectrum of variational energy
    e, v = np.linalg.eigh(matrix_ham.todense())    

    return e[:10]


"""