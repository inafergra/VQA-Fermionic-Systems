
# Mathematica calculations of the matrix elements:

def kdf(a,b):
    if a==b:
        return 1
    else:
        return 0

def mat_elem_2maj(i,alpha,j,beta):
    '''
    Matrix element <Omega|c_{i,alpha}c_{j,alpha}|Omega>
    '''
    if i==j:
        return 1j**(alpha+beta)*(-1)**(beta)
    else:
        return 0

def mat_elem_4maj(i, delta, j, gamma, l, alpha, k, beta):
    '''
    Matrix element <Omega|c_{i,delta}c_{j,gamma}c_{l,alpha}c_{k.beta}|Omega>
    '''
    mat_elem = 1j**(alpha+beta+gamma+delta) * ( (-1)**(alpha+beta) * (-kdf(i,l)*kdf(j,k) + kdf(i,k)*kdf(j,l)) + \
    (-1)**(gamma+beta) * kdf(i,j)*kdf(k,l))
    return mat_elem

def mat_element_4maj_exc(m1, m2, i, delta, j, gamma, l, alpha, k, beta):
    '''
    Matrix element <1_{m1}|c_{i,delta} c_{j,gamma} c_{l,alpha} c_{k,beta}|1_{m2}>
    '''
    mat_elem = 1j**(alpha+beta+gamma+delta)* \
        ((-1)**(delta+alpha) * kdf(i,m1)*kdf(j,l)*kdf(k,m2) + \
        (-1)**(delta+beta) * kdf(i,m1) * (kdf(j,m2)*kdf(k,l) - kdf(j,k)*kdf(l,m2) ) + \
        (-1)**(alpha+beta) * kdf(l,m1) * (kdf(i,m2)*kdf(j,k) - kdf(i,k)*kdf(j,m2) ) - \
        (-1)**(alpha+beta) * kdf(j,l) * (kdf(i,m2)*kdf(k,m1) - kdf(i,k)*kdf(m1,m2) ) + \
        (-1)**(alpha+beta) * kdf(i,l) * (kdf(j,m2)*kdf(k,m1) - kdf(j,k)*kdf(m1,m2) ) + \
        (-1)**(gamma+alpha) * (- kdf(i,l)*(kdf(j,m1)*kdf(k,m2) + kdf(i,j)*kdf(k,m2)*kdf(l,m1)) ) + \
        (-1)**(gamma+beta) * (- kdf(j,m1)*(kdf(i,m2)*kdf(k,l) - kdf(i,k)*kdf(l,m2) ) ) + \
        (-1)**(gamma+beta) * ( kdf(i,j) * (-kdf(k,m1)*kdf(l,m2) + kdf(k,l)*kdf(m1,m2) ) ) )
    return mat_elem