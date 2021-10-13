import numpy as np

import cirq
from openfermion.ops import MajoranaOperator
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator

from itertools import combinations

def convert_H_majorana_to_qubit(inds, J_dict, N):
    """Convert SYK hamiltonian (dictionary) from majorana terms to Pauli terms"""
    ham_terms = [MajoranaOperator(ind, J_dict[ind]) for ind in inds]
    ham_sum = sum_ops(ham_terms)
    return jordan_wigner(ham_sum)

def sum_ops(operators):
    """Wrapper for summing a list of majorana operators"""
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

    return e[:4]