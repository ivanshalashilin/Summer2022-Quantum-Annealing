import sys
#append to classes directory
sys.path.append('dir here')
from hamv2 import wmis
from hamv2 import ham
from bacon import bacon
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=5)

'''
demo of custom hamiltonians with bacon class
'''

na = 2
nb = 3
W = 1
dW = 0.01
Jzz = 5.77
Jxx = 0
Escale = 15
# Get hzs and Jzz

# - mapping from network to ising hamiltonian
k = (na+nb)/(4*(na*nb*Jzz-W))
Jzz = k*Jzz
# Jzz in the local field penalises coupled adjacent spins
hzs = [(nb*Jzz - 2*k*(W+dW)/na) , (na*Jzz - 2*k*W/nb)]

spin_coeff = [
    hzs[0],
    hzs[0],
    hzs[1],
    hzs[1],
    hzs[1],
]


coupling_coeff = [
    0,
    Jzz,
    Jzz,
    Jzz,
    #12 - 15
    Jzz,
    Jzz,
    Jzz,
    #23-25
    0,
    0,
    0,
]

H_input = bacon(
    5,
    spin_coeff,
    coupling_coeff
)

Hd = H_input.driver()
Hp = H_input.problem() * Escale
Hc = 1.83 * qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(2), qt.sigmax(), qt.sigmax())


def init_superpsn(n):
    '''
    n - number of qubits
    '''
    state = np.ones((2**n), dtype = int)/np.sqrt(2**n)
    state = qt.Qobj(state)
    return state

initial_state = init_superpsn(5)

grain = 5000
anneal_time = 50

#putting state into qutip
# initial coefficients
def d_coeff(t,params): 
    return (1-t/params["T"])
def p_coeff(t,params):
    return (t/params["T"])
def c_coeff(t, params):
    return t/params["T"] * (1-t/params["T"])

Hs = [[Hd, d_coeff], [Hp, p_coeff], [Hc, c_coeff]]
# "writes down" hamiltonian at time t 
H_dyn = qt.QobjEvo(Hs,args={"T":anneal_time}) 
# solves SE for a particular hamiltonian that we wrote down
s = np.linspace(0, anneal_time, grain)
sln = qt.mesolve(H_dyn, initial_state, s) # 100 states between 0 and 20 
states0 = sln.states

Hs = ham(Hd,Hp, anneal_time,grain)

m = 7

evals, evecs = Hs.sta_get_data()

#Hs.sta_plot_spectrum(evals, 8)

#Hs.sta_plot_spectrum_floor(evals,2, plot = True )

#Hs.sta_plot_evec(evals, evecs, 0,1)

#Hs.dyn_plot_comp(states0,1)

#Hs.dyn_plot_cont(evecs, states0, 2)

#H = Hs.reorder(Hd)

#Hs = ham(Hd,Hp, anneal_time,grain)#, None, None, Hc)
#evals, evecs = Hs.sta_get_data()
##Hs.sta_plot_evec(evals, evecs, 0,8)
#Hs.dyn_plot_comp(states0, 2)


#hamv2 returns data outside of class
overlaps, overlaps_squared = Hs.dyn_plot_comp(states0, n=2)
overlap_sta = Hs.sta_plot_evec(evals, evecs, 0, 2)
tsteps = s * anneal_time


fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(s/anneal_time, overlap_sta[1])
ax2.plot(tsteps, overlaps[1])
ax3.plot(tsteps, overlaps_squared[1])
plt.show()



