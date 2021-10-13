def dmat_elem_4maj(i, delta, j, gamma, l, alpha, k, beta):
    '''
    Matrix element <Omega|c_{i,delta}c_{j,gamma}c_{l,alpha}c_{k.beta}|Omega>
    '''
    mat_elem = 1j**(alpha+beta+gamma+delta)
    if (i==j) and (l==k):
        mat_elem *= (-1)**(gamma+beta)
    elif (i==l!=k==j) :
        mat_elem *= -(-1)**(alpha+beta)
    elif (j==l!=k==i):
        mat_elem *= (-1)**(alpha+beta)
    return mat_elem

def dmat_element_4maj_exc(m1, m2, i, delta, j, gamma, l, alpha, k, beta):
    '''
    Matrix element <1_{m1}|c_{i,delta} c_{j,gamma} c_{l,alpha} c_{k,beta}|1_{m2}>
    '''
    mat_elem = 1j**(alpha+beta+gamma+delta)
    if (l==k==m1==m2==i==j) or  (l==k==m1==j!=i==m2) or (k==m1!=l==i==j==m2) or (k==m1!=l==j!=i==m2):
        mat_elem *= (-1)**(alpha*delta)
    elif (l==k==m1==m2!=i==j) or  (l==k==m1==i!=j==m2) or (k==m1!=l==m2!=i==j) or (k==m1!=l==i!=j==m2):
        mat_elem *= (-1)**(alpha+gamma)
    elif (l==k!=m1==m2==i==j) or  (l==k!=m1==j!=i==m2) or (l==m1!=k==i==j==m2) or (l==m1!=k==j!=i==m2):
        mat_elem *= (-1)**(beta+delta)
    elif (l==k!=m1==m2!=i==j) or (l==k!=m1==i!=j==m2) or (l==m1!=k==m2!=i==j) or (l==m1!=k==i!=j==m2):
        mat_elem *= (-1)**(beta+gamma)
    elif (l==i!=k==j!=m1==m2) or (l==m2!=k==j!=m1==i) or (l==i!=k==m2!=m1==j):
        mat_elem *= (-1)**(alpha+beta)
    elif (l==j!=k==i!=m1==m2) or (l==m2!=k==i!=m1==j) or (l==j!=k==m2!=m1==i):
        mat_elem *= (-1)**(alpha+beta)
    return mat_elem



def init_syk_tensor_1(N, J, mean=0):
    '''
    Builds the initial matrix of coefficients with elements draw from a normal distribution with mean=mean 
    variance=6*J**2/N**3.
    '''
    variance = 6*J**2/N**3
    #variance = 1
    J = np.zeros([2*N,2*N,2*N,2*N])
    for a in range(2*N):
        for b in range(2*N): 
            for c in range(2*N):
                for d in range(2*N):
                    if (a!=b and a!=c and a!=d and b!=c and b!=d and c!=d):
                        # Conditions over J to make the Hamiltonian hermitian
                        J[a,b,c,d] = np.random.normal(mean, variance)
                        J[b,c,d,a] = -J[a,b,c,d]
                        J[c,d,a,b] = J[a,b,c,d]
                        J[d,a,b,c] = -J[a,b,c,d]

                        J[b,a,c,d] = -J[a,b,c,d]
                        J[a,c,d,b] = J[a,b,c,d]
                        J[c,d,b,a] = -J[a,b,c,d]
                        J[d,b,a,c] = J[a,b,c,d]

                        J[c,b,a,d] = -J[a,b,c,d]
                        J[b,a,d,c] = J[a,b,c,d]
                        J[a,d,c,b] = -J[a,b,c,d]
                        J[d,c,b,a] = J[a,b,c,d]

                        J[d,b,c,a] = -J[a,b,c,d]
                        J[b,c,a,d] = J[a,b,c,d]
                        J[c,a,d,b] = -J[a,b,c,d]
                        J[a,d,b,c] = J[a,b,c,d]

                        J[a,c,b,d] = -J[a,b,c,d]
                        J[c,b,d,a] = J[a,b,c,d]
                        J[b,d,a,c] = -J[a,b,c,d]
                        J[d,a,c,b] = J[a,b,c,d]

                        J[a,d,c,b] = -J[a,b,c,d]
                        J[d,c,b,a] = J[a,b,c,d]
                        J[c,b,a,d] = -J[a,b,c,d]
                        J[b,a,d,c] = J[a,b,c,d]
                        
                        J[a,b,d,c] = -J[a,b,c,d] 
                        J[b,d,c,a] = J[a,b,c,d]
                        J[d,c,a,b] = -J[a,b,c,d]
                        J[c,a,b,d] = J[a,b,c,d]
    return J


def tfd_exact(N, J_L, mu):
    '''
    Solves the TFD model exactly
    '''
    L_indices = range(0, 2*N)
    R_indices = range(2*N, 4*N)
    SYK_L_indices = list(combinations(L_indices,4))
    SYK_R_indices = list(combinations(R_indices,4))
    interaction_indices = [(l, r) for l, r in zip(L_indices, R_indices)]

    # Generate dictionaries
    J_L_dict = {ind:J_L[ind] for ind in SYK_L_indices}
    J_R_dict = {ind:J_L[tuple(np.array(ind)-2*N)] for ind in SYK_R_indices}
    int_dict = {ind: 1j * mu for ind in interaction_indices}

    H_L = convert_H_majorana_to_qubit(SYK_L_indices, J_L_dict, 2*N)
    H_R = convert_H_majorana_to_qubit(SYK_R_indices, J_R_dict, 2*N)
    H_int = convert_H_majorana_to_qubit(interaction_indices, int_dict, 2*N)

    total_ham = H_L + H_R + H_int
    matrix_ham = get_sparse_operator(total_ham) # todense() allows for ED

    # Diagonalize qubit hamiltonian to compare the spectrum of variational energy
    e, v = np.linalg.eigh(matrix_ham.todense())    

    return e[:4]

