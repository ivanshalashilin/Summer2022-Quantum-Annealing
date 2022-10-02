import sys
#append to classes directory
sys.path.append('dir here')
from matplotlib import pyplot as plt
from cmath import sqrt
from hamv2 import wmis
import numpy as np
import qutip as qt
from numpy.linalg import eigh



def get_spins_lists(na,nb,fully_coupled=False):

    # get spin values for each sub-graph
    if not fully_coupled:
        s_values = [na/2,(nb-2)/2,1.0]
    elif fully_coupled:
        s_values = [na/2,nb/2]
    rN = int(np.prod([(2*s + 1) for s in s_values]))

    # get list of m values for each sub-graph
    m_lists = [np.zeros(int(2*value) + 1) for value in s_values]
    for i,l in enumerate(m_lists):
        m = -s_values[i]
        j = 0
        while m <= s_values[i]:
            l[j] = m
            m = m + 1
            j = j + 1

    return s_values,m_lists

def build_operators(s_values,m_lists,fully_coupled=False):
   
    # Initialise Sx and S
    matrix_size = 1
    for s in s_values:
        matrix_size = matrix_size*(int(2*s) + 1)
    if not fully_coupled:
        matrices = [np.zeros([matrix_size,matrix_size]) for i in range(6)]
    elif fully_coupled:
        matrices = [np.zeros([matrix_size,matrix_size]) for i in range(4)]

    # Create Sx and Sz matrices
    if not fully_coupled:
        for i,a in enumerate(m_lists[0]):
            for j,b in enumerate(m_lists[1]):
                for k,c in enumerate(m_lists[2]):
                    for l,d in enumerate(m_lists[0]):
                        for m,e in enumerate(m_lists[1]):
                            for n,f in enumerate(m_lists[2]):
                                e1 = 3*i*(nb-1) + 3*j + k
                                e2 = 3*l*(nb-1) + 3*m + n
                                if a == d and b == e and c == f:
                                    matrices[0][e1][e2] = a
                                    matrices[1][e1][e2] = b
                                    matrices[2][e1][e2] = c
                                if a == d-1 and b == e and c == f:
                                    matrices[3][e1][e2] = 0.5*sqrt(s_values[0]*(s_values[0]+1)-a*(a+1)).real
                                    matrices[3][e2][e1] = 0.5*sqrt(s_values[0]*(s_values[0]+1)-a*(a+1)).real
                                if a == d and b == e-1 and c == f:
                                    matrices[4][e1][e2] = 0.5*sqrt(s_values[1]*(s_values[1]+1)-b*(b+1)).real
                                    matrices[4][e2][e1] = 0.5*sqrt(s_values[1]*(s_values[1]+1)-b*(b+1)).real
                                if a == d and b == e and c == f-1:
                                    matrices[5][e1][e2] = 0.5*sqrt(s_values[2]*(s_values[2]+1)-c*(c+1)).real
                                    matrices[5][e2][e1] = 0.5*sqrt(s_values[2]*(s_values[2]+1)-c*(c+1)).real
    elif fully_coupled:
        for i,a in enumerate(m_lists[0]):
            for j,b in enumerate(m_lists[1]):
                    for l,d in enumerate(m_lists[0]):
                        for m,e in enumerate(m_lists[1]):
                                e1 = i*(nb+1) + j
                                e2 = l*(nb+1) + m
                                if a == d and b == e:
                                    matrices[0][e1][e2] = a
                                    matrices[1][e1][e2] = b
                                if a == d-1 and b == e:
                                    matrices[2][e1][e2] = 0.5*sqrt(s_values[0]*(s_values[0]+1)-a*(a+1)).real
                                    matrices[2][e2][e1] = 0.5*sqrt(s_values[0]*(s_values[0]+1)-a*(a+1)).real
                                if a == d and b == e-1:
                                    matrices[3][e1][e2] = 0.5*sqrt(s_values[1]*(s_values[1]+1)-b*(b+1)).real
                                    matrices[3][e2][e1] = 0.5*sqrt(s_values[1]*(s_values[1]+1)-b*(b+1)).real

    return matrices

def build_driver(x_operators):

    return -2*sum(x_operators)

def build_catalyst(cat_x_operator,Jxx):

    return 2*Jxx*np.matmul(cat_x_operator,cat_x_operator)

def build_problem(z_operators,na,nb,W,dW,Jzz,Escale=15,fully_coupled=False):

    # Get hzs and Jzz
    k = (na+nb)/(4*(na*nb*Jzz-W))
    Jzz = k*Jzz
    hzs = [nb*Jzz - 2*k*(W+dW)/na , na*Jzz - 2*k*W/nb]

    if not fully_coupled:
        return 2*Escale*(hzs[0]*z_operators[0] + hzs[1]*(z_operators[1]+z_operators[2]) + 2*Jzz*np.matmul(z_operators[0],z_operators[1]+z_operators[2]))
    elif fully_coupled:
         return 2*Escale*(hzs[0]*z_operators[0] + hzs[1]*(z_operators[1]) + 2*Jzz*np.matmul(z_operators[0],z_operators[1]))

def problem_states(Hp,s_values,m_lists,N):

    Hp_energies, Hp_vecs = np.linalg.eigh(Hp)
    zipped = list(zip(Hp_energies, Hp_vecs.transpose()))
    sorted_states = sorted(zipped, key=lambda x: x[0])
    Hp_vecs = [ np.squeeze(np.asarray(state[1])) for state in sorted_states ]

    state_list = [ str(s_values[0]) + ',' + str(m) for m in m_lists[0] ]
    for v in range(1,len(s_values)):
        new_state_list = []
        for state in state_list:
            new_state_list.extend([ state + ";  " + str(s_values[v]) + ',' + str(m) for m in m_lists[v]])
        state_list = new_state_list

    state_dict = {}
    for state in range(N):
        vec = Hp_vecs[state]
        loc = 0 
        for pot_loc,c in enumerate(vec):
            if c == 1:
                loc = pot_loc
        state_dict[state] = state_list[loc]

    return state_dict

def build_initial_state(Hd):
    evals, evecs = eigh(Hd)
    evecs = evecs.transpose()
    # employ bubble sort to order them
    for i in range(len(evals)):
        for j in range(i+1, len(evals)):
                        if evals[i] > evals[j]:
                            evals[i],evals[j] = evals[j],evals[i]
                            evecs[[i,j]] = evecs[[j,i]]
    return evecs[0]

'''
demo of making plots with hamnv2 class
'''


na = 2
nb = 3
W = 1
dW = 0.01 #0.01 0.37
Jzz = 5.77  #5.77 37.3 dont change these comments

#catalyst
Jxx = 0
full = False

s_values, m_lists = get_spins_lists(na,nb)#, fully_coupled=True)
operators = build_operators(s_values,m_lists)#, fully_coupled = True)
Hd = build_driver([operators[3],operators[4],operators[5]])
Hc = build_catalyst(operators[5],Jxx)
Hp = build_problem([operators[0],operators[1],operators[2]],na,nb,W,dW,Jzz)
states = problem_states(Hp,s_values,m_lists,18)
# for state in states:
#     print(str(state) + ": " + states[state]) 
initial_state = qt.Qobj(build_initial_state(Hd))




#input to qutip

#initial coefficients
def d_coeff(t,params): 
    return (1-t/params["T"])
def p_coeff(t,params):
    return (t/params["T"])
def c_coeff(t, params):
    return t/params["T"] * (1-t/params["T"])

anneal_time = 100 # 100*17312472 #100/(min(Hs.sta_plot_spectrum_floor(evals, 2, plot = True)[1]))**2
grain = int(anneal_time/2)

Hd = qt.Qobj(Hd)
Hp = qt.Qobj(Hp)
Hc = qt.Qobj(Hc)

Hs = [[Hd, d_coeff], [Hp, p_coeff], [Hc, c_coeff]]

# "writes down" hamiltonian at time t 
H_dyn = qt.QobjEvo(Hs,args={"T":anneal_time}) 
# solves SE for a particular hamiltonian that we wrote down
s = np.linspace(0, anneal_time, grain)
sln = qt.mesolve(H_dyn, initial_state, s) # 100 states between 0 and 20 
states0 = sln.states



Hs = wmis(Hd, Hp, anneal_time, grain, s_values, m_lists, Hc)

m=2

evals, evecs = Hs.sta_get_data()



if Jzz == 5.77:
    filename = 'spectrum_strong_'+str(Jxx)
elif Jzz == 37.3:
    filename = 'spectrum_weak_'+str(Jxx)

#functions of wmis class

#Hs.sta_plot_spectrum(evals, m)

#Hs.sta_plot_spectrum_floor(evals,m, savefig=False, plot = False)

#Hs.sta_plot_evec(evals, evecs, 0, 2, plot = True)

#Hs.dyn_plot_cont(evecs, states0, 2, plot = True)

#Hs.dyn_plot_comp(states0,1)