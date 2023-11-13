
import numpy as np
import matplotlib.pyplot as plt
#import
import pickle
#redirect
import os
#reading data
from scipy.signal import find_peaks


#double check these ...

h=0.00020004000800160032
def FirstDiff(E, h=h):
    return (E[n+1]-E[n-1])/(2*h)

def SecondDiff(E, h=h):
    return (E[n+1] + E[n-1]- 2*E[n])/(4*h**2)

def ACoeff(E, EP, EPP, type = 'GS'):
    if type == 'GS':
        return EP + np.sqrt(0.5 * E * EPP)


def BCoeff(E,EP, EPP, type = 'GS'):
    if type == 'GS':
        return EP - np.sqrt(0.5 * E * EPP)
    

    return EP - np.sqrt(0.5 * E * EPP)

def LZFit(s, A,B,C, GSorFES = 'GS'):
    FirstTerm = 0.5 * (A+B)*s
    SecondTerm = 0.5 * np.sqrt((A-B)**2 * s**2 + 4 * C**2)
    if GSorFES == 'FES':
        return FirstTerm-SecondTerm
    else:
        return FirstTerm+SecondTerm 





catalyst_strengths = ['175','183','187','192','200']

exponents = np.zeros(5)
for i in range(5):
    #load data
    pickle_in = open('TwoLevelEvals'+catalyst_strengths[i]+'.pkl', 'rb')
    data = pickle.load(pickle_in)
    #extract energies and s
    GSEnergy = data[0]
    FESEnergy = data[1]
    s = data[2]
    #locate closing gap
    DeltaE = FESEnergy-GSEnergy
    h = s[1]-s[0]

    GapIndex = find_peaks(-DeltaE)[0][0]
    n = GapIndex #for no clutter

    MinGSE = GSEnergy[n]
    MinFESE = FESEnergy[n]
    CrossingEn = np.average([MinGSE,MinFESE])
    sloc = s[n]

    GSEnergy = GSEnergy-CrossingEn
    FESEnergy = FESEnergy-CrossingEn



    s = s-s[n]
    E_GS = GSEnergy[n]
    EP_GS = FirstDiff(GSEnergy)
    EPP_GS = SecondDiff(GSEnergy)

    E_FES = FESEnergy[n]
    EP_FES = FirstDiff(FESEnergy)
    EPP_FES = FirstDiff(FESEnergy)


    C = -E_GS
    A =  ACoeff(E_GS,EP_GS,EPP_GS)
    B = BCoeff(E_GS,EP_GS,EPP_GS)


    twoPiGamma = 2*((E_FES-E_GS)/2)**2/np.abs(B-A)

    exponents[i] = twoPiGamma

import matplotlib.pyplot as plt
#make figure
fig, ax = plt.subplots(1,1, figsize =(10,6))
ax.plot(s,GSEnergy, color = 'red', label = '')
ax.plot(s,FESEnergy, color = 'red', label = '')
ax.vlines(s[n], ymin= -3, ymax = 3)
ax.grid()
#plt.savefig('.png', dpi = 600)
plt.show()


E_GS = GSEnergy[n]
EP_GS = FirstDiff(GSEnergy)
EPP_GS = SecondDiff(GSEnergy)

E_FES = FESEnergy[n]
EP_FES = FirstDiff(FESEnergy)
EPP_FES = SecondDiff(FESEnergy)


C_GS = GSEnergy[n]
A_GS =  ACoeff(E_GS,EP_GS,EPP_GS)
B_GS = BCoeff(E_GS,EP_GS,EPP_GS)

C_FES = -FESEnergy[n]
A_FES =  ACoeff(E_FES,EP_FES,EPP_FES)
B_FES = BCoeff(E_FES,EP_FES,EPP_FES)


LZParamsGS =  [
    A_GS,
    B_GS,
    C_GS,
    ]

LZParamsFES = [
    A_FES,
    B_FES,
    C_FES,
    ]

GSLZFit = LZFit(s, *LZParamsGS, 'GS')
FESLZFit = LZFit(s, *LZParamsFES, 'FES')


#make figure
fig, ax = plt.subplots(1,1, figsize =(10,6))
ax.plot(s, GSEnergy-GSEnergy, label = 'ground state')
ax.plot(s, FESEnergy-GSEnergy, label = 'First Excited state')
ax.plot(s,GSLZFit-GSEnergy, '--', color = 'green', label = 'First Excited State LZ Fit')
ax.plot(s,FESLZFit-GSEnergy, '--', color = 'purple', label = 'Ground State LZ Fit')
ax.plot(s,0.5*(B_GS+B_FES)*s-GSEnergy, '', color = 'red')
ax.plot(s,0.5*(A_FES+A_GS)*s-GSEnergy, '', color = 'blue')
ax.legend()
ax.grid()
plt.show()

fig, ax = plt.subplots(1,1, figsize =(10,6))

ax.plot(s, GSEnergy, label = 'ground state', alpha = 0.3)
ax.plot(s, FESEnergy, label = 'First Excited state',alpha = 0.3)
ax.plot(s,GSLZFit, '--', color = 'green', label = 'Ground State LZ Fit', alpha = 0.8, lw = 4)
#ax.vlines(s[n], ymin= -10, ymax = 10)
#ax.hlines(0, xmin= -0.5, xmax = 0.5)
ax.plot(s,FESLZFit, '--', color = 'purple', label = 'Fist Excited State LZ Fit', alpha = 0.8, lw = 4)
ax.plot(s,0.5*(B_GS+B_FES)*s, color = 'red', alpha = 0.8)
ax.plot(s,0.5*(A_FES+A_GS)*s, color = 'blue', alpha = 0.8)

ax.legend()
ax.grid()
#plt.savefig('.png', dpi = 600)
plt.show()

#values that fit
CGSArr = np.array([
    0.0004945166124130682,#175
    4.520060792702318e-05,#183
    8.967536094371207e-07,#187
    0.00010922626571586632,#192
    0.0006530379342763278,#200
])

print(CGSArr/exponents)

#import
import pickle
#reading data
pickle_in = open('2d_s_2_3__t_1e-1_10000__j_175e-2_20e-1.pkl', 'rb')
data = pickle.load(pickle_in)

TAnneal = data[1]

import matplotlib.pyplot as plt
#make figure
fig, ax = plt.subplots(1,1, figsize =(10,6))
colsdark = ['darkred', 'darkorange', 'goldenrod', 'darkgreen', 'darkblue']
cols = ['red', 'orange', 'yellow', 'green', 'blue']

for i in range(5):
    ProbabilityMeas = np.array(data[2])[:,i]
    ProbabilityTheory = np.exp(-exponents[i] * TAnneal)
    ax.plot(TAnneal,ProbabilityMeas, color = colsdark[i] )
    ax.plot(TAnneal,ProbabilityTheory,'--', color = cols[i] , alpha = 0.8, lw =3)

#ax.plot(data[1],ProbabilityCat200 , color = 'red')#, label = '')
#plt.savefig('.png', dpi = 600)
plt.show()

print()



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
# overlaps, overlaps_squared = Hs.dyn_plot_comp(states0, n=2)
# overlap_sta = Hs.sta_plot_evec(evals, evecs, 0, 2)
# tsteps = s * anneal_time


# fig, (ax1, ax2, ax3) = plt.subplots(3)
# ax1.plot(s/anneal_time, overlap_sta[1])
# ax2.plot(tsteps, overlaps[1])
# ax3.plot(tsteps, overlaps_squared[1])
# plt.show()



