import sys
sys.path.append("./")

from classes.bacon import bacon
import numpy as np
np.set_printoptions(precision=3)
'''
This is the hamiltonian for the 5 qubit system, with rescaling and mapping from the network to the ising hamiltonian
'''
import numpy as np
#parameters
N = 9
na = int((N-1)/2)
nb = int((N+1)/2)
W = 1
dW = 0.01
Jzz = 5.33 # possibly 5.33?
#Jxx = 0
Escale = 15
# Get hzs and Jzz

# - mapping from network to ising hamiltonian
k = (na + nb) / (4 * (na * nb * Jzz - W))
Jzz = k * Jzz
# Jzz in the local field penalises coupled adjacent spins
hzs = [(nb * Jzz - 2 * k * (W + dW) / na), (na * Jzz - 2 * k * W / nb)]

spin_coeff = [
    *[hzs[0] for i in range(na)],
    *[hzs[1] for i in range(nb)],   
]


#specifically upper triangular adjacency matrix
#0.5*n*(n-1) elements



def generate_coupling_coeff(na=na,nb=nb, Jzz=Jzz):
    '''
    creates coupling coefficients for general bipartite graph in a way that is
    interpretable in the construction of bacon. computes the upper triangular matrix
    and then flattens it. this gives 0.5*N*N-1 elements where N = na+nb
    na - number of qubits in A
    nb - number of qubits in B
    Jxx - coupling strength
    '''
    base_matrix = np.zeros((na+nb,na+nb))
    base_matrix[:na, -nb:] = np.full((na, nb), Jzz)
    return base_matrix[np.triu_indices_from(base_matrix,1)]

coupling_coeff = generate_coupling_coeff()

H_input = bacon(N, spin_coeff, coupling_coeff)
Hd = H_input.driver()
Hp = H_input.problem() * Escale