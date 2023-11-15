import sys
sys.path.append("./")

from classes.bacon import bacon

'''
This is the hamiltonian for the 5 qubit system, with rescaling and mapping from the network to the ising hamiltonian
'''

#parameters
na = 2
nb = 3
W = 1
dW = 0.01
Jzz = 5.77
Jxx = 0
Escale = 15
# Get hzs and Jzz

# - mapping from network to ising hamiltonian
k = (na + nb) / (4 * (na * nb * Jzz - W))
Jzz = k * Jzz
# Jzz in the local field penalises coupled adjacent spins
hzs = [(nb * Jzz - 2 * k * (W + dW) / na), (na * Jzz - 2 * k * W / nb)]

spin_coeff = [
    hzs[0],
    hzs[0],
    hzs[1],
    hzs[1],
    hzs[1],
]


coupling_coeff = [0,Jzz,Jzz,Jzz,
    # 12 - 15
    Jzz,Jzz,Jzz,
    # 23-25
    0,0,0,
]



H_input = bacon(5, spin_coeff, coupling_coeff)
Hd = H_input.driver()
Hp = H_input.problem() * Escale