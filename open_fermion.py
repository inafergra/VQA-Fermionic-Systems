from openfermion import *
import numpy as np


def Majorana(i,alpha):
    '''
    arg:
        index: list containing two integers that define the Majorana fermion c_{i,alpha}
    '''
    if (alpha == 0 or alpha == 1):
        return (1j)**(alpha)*(FermionOperator(f'{i}') + (-1)**(alpha) * FermionOperator(f'{i}^'))
    else:
        print('Alpha should be either 0 or 1')

print(jordan_wigner(Majorana(4,1)))

print(np.e**(1j*Majorana(4,1)))