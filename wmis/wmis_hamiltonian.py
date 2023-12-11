import sys
sys.path.append("./")

from classes.bacon import bacon
import numpy as np
np.set_printoptions(precision=3)
'''
This is the hamiltonian for the 5 qubit system, with rescaling and mapping from the network to the ising hamiltonian
'''
import numpy as np

def compute_k_factor(na,nb, W, Jzz):
    return (na + nb) / (4 * (na * nb * Jzz - W))

def generate_spin_coeff(N, W, dW, Jzz):
    '''
    generates spin coefficients for general bipartite graph
    IMPORTANT: Jzz is RESCALED in this function due to penalty term
    the RESCALED version must be passed to the generate_coupling_coeff function
    '''
    na = int((N-1)/2)
    nb = int((N+1)/2)
    # - mapping from network to ising hamiltonian
    k = (na + nb) / (4 * (na * nb * Jzz - W))
    Jzz = k * Jzz
    # Jzz in the local field penalises coupled adjacent spins
    hzs = [(nb * Jzz - 2 * k * (W + dW) / na), (na * Jzz - 2 * k * W / nb)]
    spin_coeff = [
        *[hzs[0] for i in range(na)],
        *[hzs[1] for i in range(nb)],   
    ]
    return spin_coeff, Jzz


def generate_coupling_coeff(na,nb, Jzz):
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

