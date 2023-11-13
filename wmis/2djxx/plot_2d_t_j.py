import sys
sys.path.append('./')
from matplotlib import pyplot as plt
from cmath import sqrt
from classes.hamv2 import wmis
import numpy as np
import qutip as qt
from numpy.linalg import eigh
import pickle
from mpl_toolkits import mplot3d
from matplotlib import cm


#redirect


# pickle_in = open('TTS_hd_cat_3_4_20_100.pkl', 'rb')
# data1 = pickle.load(pickle_in)

#redirect
# import os
# os.chdir('/Users/ivanshalashilin/Desktop/UCL Placement /research placement/wmis/2djxx/assets')
# #reading data
# pickle_in = open('Jxx_2_3_1_100.pkl', 'rb')
# data = pickle.load(pickle_in)

# data = np.append(data0, data1)


# import os
# os.chdir('/Users/ivanshalashilin/Desktop/UCL Placement /research placement/')
#reading data
pickle_in = open('2d_s_2_3__t_1e-1_10000__j_175e-2_20e-1.pkl', 'rb')
data = pickle.load(pickle_in)
#data = [Jxx, evec_all, anneal_times, Jxx_best, TTS_all]
gs_prob = np.array(data[2])
#make figure
fig, ax = plt.subplots(1,1, figsize =(10,5))
ax.grid(linestyle='-', linewidth=1)
ax.set_ylabel('$|\\langle \Psi(T) | E_0(T) \\rangle|^{2}$', size = 20)
ax.set_xlabel('Anneal Time (ns)', size = 20)
ax.tick_params(labelsize = 20)
plt.subplots_adjust(bottom=0.15)
ax.plot(data[1], gs_prob[:,0], color = 'purple' , label = '$%.2f$' % data[0][0], lw = 2, zorder = 100)
ax.plot(data[1], gs_prob[:,1], color = 'green', label = '$%.2f$' % data[0][1], lw = 2)
ax.plot(data[1], gs_prob[:,2], color = 'blue', label = '$%.2f$' % data[0][2], lw = 2)
ax.plot(data[1], gs_prob[:,3], color = 'orange', label = '$%.2f$' % data[0][3], lw = 2)
ax.plot(data[1], gs_prob[:,4], color = 'red'   , label = '$%.2f$' % data[0][4], lw = 2, zorder = 101)
fig.legend(title = '$J_{xx}$', prop={'size': 8},bbox_to_anchor=(0.89,0.47))
#plt.savefig('Different Catalysts 10000.png', dpi = 600)#, bbox_inches='tight')
plt.show()

from scipy.optimize import curve_fit

def osc(t,k0,k1, k2):
    return np.cos(k0* np.abs(np.cos(k1 * t)) * np.exp(-k2 * t**2))
def OscExp(t, k0,k1, k2, k3, k4, k5):
    return k0*(np.cos(k1 * (t-k4)))**2*np.exp(-k5*(t-k4)) + k2*np.exp(-k3 * (t-k4))
GuessParams = [1 , 0.03, 1.1 ,0.0006, 20, 0.001]
freqs = np.zeros(len(data[0]))
x = np.linspace(data[1][13], data[1][-1], 10000)
for i in range(len(data[0])):
    Params = curve_fit(OscExp, data[1][17:], gs_prob[:,i][17:], p0 = GuessParams)
    y = OscExp(x, *Params[0])#, zorder = 1000)
    ax.plot(x,y, zorder =1000)
    print(str(data[0][i])+":"+str(Params[0][1]))
    freqs[i] = Params[0][1]
fig, ax  = plt.subplots()

#make figure
fig, ax = plt.subplots(1,1, figsize =(7,6))
ax.grid(linestyle='-', linewidth=1)
ax.set_ylabel('$\omega$ (kHz)', size = 20)
ax.set_xlabel('$J_{xx}$', size = 20)
ax.tick_params(labelsize = 20)
ax.plot(data[0], freqs*1000/(2*np.pi), 'o', ms = 10, color = 'red')
plt.subplots_adjust(bottom=0.15)
#ax.tick_params(axis='y', labelcolor='red')
#plt.savefig('omega100.png', dpi = 100)
plt.show()


#redirect
import os
os.chdir('/Users/ivanshalashilin/Desktop/Documents/UCL Placement /research placement/wmis/2djxx/assets')
#reading data
pickle_in = open('Jxx_2_3_1_100.pkl', 'rb')
data = pickle.load(pickle_in)

xline_all = np.array([])
yline_all = np.array([])
zline_all = np.array([])
for i in range(len(data[2])):
    xline = data[0]
    yline = np.full((len(data[0])), data[2][i])
    zline = data[1][i]
    xline_all = np.append(xline_all, xline)
    yline_all = np.append(yline_all, yline)
    zline_all = np.append(zline_all, zline)


xline_all = xline_all.astype('float')
yline_all = yline_all.astype('float')
zline_all = zline_all.astype('float')



#ax._axis3don = False
fig = plt.figure(figsize=plt.figaspect(0.32))
#plot 1
fig.tight_layout(pad = 1)
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.plot_trisurf(xline_all, yline_all, zline_all, cmap=cm.jet)
ax.set_xlabel('$J_{xx}$')
ax.view_init(45,50)
ax.set_ylabel('Anneal Time (ns)')
ax.set_zlabel('$|\langle \Psi | E_0 \\rangle|^2$')


#plot 2
ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.plot_trisurf(xline_all, yline_all, zline_all, cmap=cm.jet)
ax.set_xlabel('$J_{xx}$')
ax.set_ylabel('Anneal Time (ns)')
ax.view_init(65,90)
ax.set_zlabel('$|\langle \Psi | E_0 \\rangle|^2$')


#plot 3
ax = fig.add_subplot(1, 3, 3, projection='3d')#, frameone= False)
ax.plot_trisurf(xline_all, yline_all, zline_all, cmap=cm.jet)
ax.set_xlabel('$J_{xx}$')
ax.set_ylabel('Anneal Time (ns)')
ax.set_zlabel('$|\langle \Psi | E_0 \\rangle|^2$')

plt.savefig("2dmaster200.png", dpi = 200)
plt.show()




