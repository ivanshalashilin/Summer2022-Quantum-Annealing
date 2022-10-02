# creates general input Hamiltonian
import numpy as np
import qutip as qt
from numpy.linalg import eigh
import matplotlib.pyplot as plt
import math

class bacon:
    '''
    n - number of qubits
    spin_coeff - vertex coefficients, in order of the spin strength
    '''
    def __init__(self, n, spin_coeff = None, coupling_coeff = None):
        self._n = int(n)
        self._spin_coeff = spin_coeff
        self._coupling_coeff = coupling_coeff

        pass

    def __create_sigma_i(self, i, sigma = qt.sigmax()):
        '''
        creates a list to be inputted into tp
        for a particular matrix
        sigma - type of pauli matrix
        '''
        base = []
        # create list of I2
        for j in range(self._n):
                base.append(qt.qeye(2))
        # change ith component to sigmax
        base[i] = sigma
        sigma_i = qt.tensor(base)
        return sigma_i

    def __create_sigmaz_ij(self,i,j):
        '''
        creates coupling hamiltonians for 
        problem hamiltonian
        '''
        base = []
        k = self._n
        # create list of I2
        for l in range(self._n):
                base.append(qt.qeye(2))
        base[i] = qt.sigmaz()
        base[j] = qt.sigmaz()
        sigmaz_ij = qt.tensor(base)
        return sigmaz_ij



    def driver(self):
        '''
        creates drive Hamiltonian, which is the sum of -sigmax_i operators 
        
        '''
        k = self._n
        Hd = np.zeros((2**k, 2**k))
        Hd = qt.Qobj(Hd, dims=[[2 for i in range(k)],[2 for i in range(k)]])
        for i in range(self._n):
            sigmax_i = self.__create_sigma_i(i)
            Hd -= sigmax_i
        return Hd

    def problem(self):
        '''
        creates problem Hamiltonian

        '''
        # initialise Hp array
        p = self._n # to save on writing
        Hp = np.zeros((2**p, 2**p))
        Hp = qt.Qobj(Hp, dims=[[2 for i in range(p)],[2 for i in range(p)]])

        #vertex coefficients
        if self._spin_coeff is None:
            pass
        else:
            for i in range(self._n):
                sigmaz_i = self.__create_sigma_i(i, sigma = qt.sigmaz())
                Hp += self._spin_coeff[i] * sigmaz_i
        #coupling coefficients
        if self._coupling_coeff is None:
            pass
        else:
            k = -1
            for i in range(p):
                for j in range(i+1,p):
                    k += 1
                    print(str(i), str(j))
                    sigma_ij = self.__create_sigmaz_ij(i,j)
                    Hp += self._coupling_coeff[k] * sigma_ij
        return Hp


