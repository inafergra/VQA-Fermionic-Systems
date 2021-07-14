def energy_excited_state(H, k):
    '''
    Calculates the energy of a given coupling matrix H. The energy is taken w.r.t. the state defined by 
    the binary array k.
    '''
    N = int(np.size(H,axis=0)/2)
    excited_states = np.flatnonzero(k)
    #print(excited_states)
    m = 0
    for l in range(N):
        if l in excited_states:
            m -= H[2*l+1,2*l]
        else:
            m += H[2*l+1,2*l]
    energy = 0.5*(-2*m)
    return energy

def squared_hamiltonian_average(H):
    ''' Calculates the average of the squared hamiltonian with respect to the vacuum '''
    N = int(np.size(H,axis=0)/2)

    x=0
    for i in range(N):
        for j in range(N):
            for l in range(N):
                for k in range(N):
                    for alpha in range(0,2):
                        for beta in range(0,2):
                            for gamma in range(0,2):
                                for delta in range(0,2):
                                    if (i==j) and (l==k):
                                        x += H[2*l+alpha, 2*k+beta]*H[2*i+delta, 2*j+gamma] * 1j**(alpha+beta+gamma+delta) * (-1)**(gamma+beta)
                                    elif (l!=k) and (i==l) and (k==j):
                                        x += H[2*l+alpha, 2*k+beta]*H[2*i+delta, 2*j+gamma] * 1j**(alpha+beta+gamma+delta) * (-1)**(alpha+beta)
                                    elif (l!=k) and (j==l) and (k==i):
                                        x += H[2*l+alpha, 2*k+beta]*H[2*i+delta, 2*j+gamma] * 1j**(alpha+beta+gamma+delta) * (-1)**(alpha+beta)
    return -1/4*x

def energy_after_x_rotations(theta, H):
    '''
    Computes the energy of the state given by doing an X rotation of theta[i] in qubit i. 
    args:
        theta: 1darray (len = number of fermions/qubits N)
    '''
    N = int(np.size(H,axis=0)/2)
    energy = 0
    for state in (itertools.product([0, 1], repeat=N)):
        k = np.array(state)
        coeff = 1
        for i in range(N): # coefficients
            if k[i]==0:
                coeff *= np.cos(theta[i]/2)
            else:
                coeff *= np.sin(theta[i]/2)
        energy += coeff*energy_excited_state(H, k)
    return energy

def squared_hamiltonian_average_excited_state(H, x):
    ''' Calculates the average of the squared hamiltonian with respect to the excited state x '''
    N = int(np.size(H,axis=0)/2)
    
    excited_states = (np.flatnonzero(x))
    sq_ham = 0
    for i in range(N):
        for j in range(N):
            for l in range(N):
                for k in range(N):
                    for alpha in range(0,2):
                        for beta in range(0,2):
                            for gamma in range(0,2):
                                for delta in range(0,2):
                                    x = H[2*l+alpha, 2*k+beta]*H[2*i+delta, 2*j+gamma] * 1j**(alpha+beta+gamma+delta)
                                    if (l not in excited_states) and (k not in excited_states) and (i not in excited_states) and (j not in excited_states):
                                    #if (l and k and i and j not in excited_states):
                                        if (l!=k) and (i==l) and (k==j):
                                            sq_ham += x*(-1)**(alpha+beta)
                                        elif (l!=k) and (j==l) and (k==i):
                                            sq_ham += x*(-1)**(alpha+beta)
                                        elif (i==j) and (l==k):
                                            sq_ham += x*(-1)**(beta+gamma)
                                    #elif (j and l in excited_states) and (k and i not in excited_states):
                                    elif (j in excited_states) and (l in excited_states) and (k not in excited_states) and (i not in excited_states):
                                        if (l!=k) and (j==l) and (k==i):
                                            sq_ham += x*(-1)**(beta+gamma)
                                    #elif (i and l in excited_states) and (k and j not in excited_states):
                                    elif (i in excited_states) and (l in excited_states) and (k not in excited_states) and (j not in excited_states):
                                        if (l!=k) and (i==l) and (k==j):
                                            sq_ham += x*(-1)**(beta+delta)
                                    #elif (i and j in excited_states) and (l and k not in excited_states):
                                    elif (i in excited_states) and (j in excited_states) and (l not in excited_states) and (k not in excited_states):
                                        if (i==j) and (l==k):
                                            sq_ham += x*(-1)**(beta+delta)
                                    #elif (k and i in excited_states) and (j and l not in excited_states):
                                    elif (k in excited_states) and (i in excited_states) and (j not in excited_states) and (l not in excited_states):
                                        if (l!=k) and (j==l) and (i==k):
                                            sq_ham += x*(-1)**(alpha+delta)
                                    #elif (l and k in excited_states) and (j and i not in excited_states):
                                    elif (l in excited_states) and (k in excited_states) and (j not in excited_states) and (i not in excited_states):
                                        if (l==k) and (i==j):
                                            sq_ham += x*(-1)**(alpha+gamma)
                                    #elif (j and k in excited_states) and (l and i not in excited_states):
                                    elif (j in excited_states) and (k in excited_states) and (l not in excited_states) and (i not in excited_states):
                                        if (l!=k) and (i==l) and (j==k):
                                            sq_ham += x*(-1)**(alpha+gamma)
                                    #elif (l and k and i and j in excited_states):
                                    elif (l in excited_states) and (k in excited_states) and (i in excited_states) and (j in excited_states):
                                        if (l!=k) and (i==l) and (k==j):
                                            sq_ham += x*(-1)**(delta+gamma)
                                        elif (l!=k) and (j==l) and (i==k):
                                            sq_ham += x*(-1)**(delta+gamma)
                                        elif (i==j) and (l==k):
                                            sq_ham += x*(-1)**(alpha+delta)
    return (-1/4)*sq_ham            

def squared_hamiltonian_after_x_rotations(theta, H):
    N = int(np.size(H,axis=0)/2)
    sq_ham = 0
    for state in list(itertools.product([0, 1], repeat=N)):
        k = np.array(state)
        coeff = 1
        for i in range(N): # coefficients
            if k[i]==0:
                coeff *= np.cos(theta[i]/2)
            else:
                coeff *= np.sin(theta[i]/2)
        sq_ham += coeff*squared_hamiltonian_average_excited_state(H, k)
    return sq_ham


def variance(theta, H):
    var = squared_hamiltonian_after_x_rotations(theta, H) - energy_after_x_rotations(theta, H)**2
    return np.abs(np.real(var))

