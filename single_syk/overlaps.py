import numpy as np

import cirq
from openfermion.ops import MajoranaOperator
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator

from itertools import combinations, permutations

def convert_H_majorana_to_qubit(inds, J):
    '''
    Convert SYK hamiltonian (dictionary) from majorana terms to Pauli terms
    '''
    
    ham_terms = [MajoranaOperator(ind, J[ind]) for ind in inds]
    ham_sum = sum_ops(ham_terms)
    return jordan_wigner(ham_sum)

def sum_ops(operators):
    '''
    Wrapper for summing a list of majorana operators
    '''
    return sum(operators, MajoranaOperator((), 0))

def tfd_exact(N, J):
    '''
    Solves the TFD model exactly
    '''

    indices = list(permutations(range(0, 2*N),4))
    ham = convert_H_majorana_to_qubit(indices, J)
    matrix_ham = get_sparse_operator(ham) # todense() allows for ED

    # Diagonalize qubit hamiltonian 
    e, v = np.linalg.eigh(matrix_ham.todense())    

    return e[:10]

#def givens_rotation(indeces, t):
#    i = indeces[0]
#    j = indeces[1]
#    return jordan_wigner(MajoranaOperator(i, np.sin(t)))


N = 4
qubit = cirq.LineQubit(N)
circuit = cirq.Circuit()

giv = jordan_wigner(MajoranaOperator( (1,2),1.0))
print(giv)
circuit.append(cirq.H(qubit[0]))

sim = cirq.Simulator()
result = sim.simulate(circuit)
