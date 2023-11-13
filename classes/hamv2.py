'''
ham v2 - optimisations
'''
from array import ArrayType
from cProfile import label
import numpy as np
import qutip as qt
from numpy.linalg import eigh
import matplotlib.pyplot as plt
import math
from scipy.linalg import expm

class ParentHam:
    def __init__(self, Hd, Hp, anneal_time, grain, Hc = None):
        self._Hd = Hd
        self._Hp = Hp
        self._Hc = Hc
        self._s = np.linspace(0,1, grain)
        self._tsteps = anneal_time * self._s

    def Hd(self):
        return self._Hd

    def Hp(self):
        return self._Hp

    def s(self):
        return self._s

    def Ham(self, s):
        if self._Hc is None:
            return s * self._Hp + (1-s) * self._Hd
        return s * self._Hp + (1-s) * self._Hd + (s * (1-s)) * self._Hc

    def eval_spectrum(self, Hs):
        '''
        prints instantaneous energy spectrum for a particular 
        Hamiltonian by computing and reordering evals
        
        input - Qobj
        '''
        evals = self.reorder(Hs)[0]
        print('energy spectrum')
        for i in range(len(evals)): #wtf?
            print("%.1f" % evals[i])

    def eval_comp_spectrum(self):
        '''
        prints energy spectrum for computational spectrum

        '''
        evals = self.reorder(self._Hp)[0]
        labels = self._label_states(self._Hp)
        print("Computational Energy Spectrum \n  ------- ")
        for i in range(len(evals)): #wtf?
            print(labels[i]+ ": %.1f " % evals[i])

    def __reorder(self, Hi):
        '''
        reorders a particular Hamiltonian s.t. eigenvalues
        and corresponding eigenvectors 
        '''
        # get eigenvalues and eigenvectors
        evals, evecs = eigh(Hi)
        evecs = evecs.transpose()
        # employ bubble sort to order them
        for i in range(len(evals)):
            for j in range(i+1, len(evals)):
                            if evals[i] > evals[j]:
                                evals[i],evals[j] = evals[j],evals[i]
                                evecs[[i,j]] = evecs[[j,i]]
        reordered = [evals,evecs]
        return reordered

    def _label_states(self):
        raise NotImplementedError("Subclass needs to define this.")

    def __save_fig(name, dpi = 600):
            name = str(name) + '.png'
            plt.savefig(name, dpi = dpi)

    def _evolving_Ham(self):
        '''
        forms Hs for each time step in 1 array
        s should go from 0 to 1!
        '''
        Hs = []
        for i in range(len(self._s)):
            Hs.append(ham.Ham(self, self._s[i]))
        Hs = np.array(Hs)
        return Hs

    def find_flip_locations_1(self,coeffs,threshhold=0.05):
        prob_states = coeffs.shape[0]
        times = coeffs.shape[1]
        flip_locations = np.zeros(times)
        current_coeffs = coeffs[:,0]
        for t in range(times):
            sign_flips = np.zeros(prob_states)
            new_coeffs = coeffs[:,t]
            for i in range(prob_states):
                if np.sign(new_coeffs[i]) != np.sign(current_coeffs[i]):
                    sign_flips[i] = 1
            all_flipped = True
            for i in range(prob_states):
                flip = sign_flips[i]
                if flip == 0:
                    if abs(current_coeffs[i]) > threshhold and abs(new_coeffs[i]) > threshhold:
                        all_flipped = False
            if all_flipped:
                flip_locations[t] = 1
            current_coeffs = new_coeffs
        return flip_locations

    def __unflip(self, coeffs,locations):
        new_coeffs = np.zeros(coeffs.shape)
        sign = -1
        for t in range(len(new_coeffs)):
            if locations[t] == 1:
                sign = -sign
            new_coeffs[t] = sign*coeffs[t]
        return new_coeffs

    '''STATIC'''
    def sta_get_data(self):
        '''
        returns every value of array
        '''
        Hs = self._evolving_Ham()
        evals_all = []
        evecs_all = []
        for i in range(len(Hs)):
            evals, evecs = self.__reorder(Hs[i])
            evals_all.append(evals)
            evecs_all.append(evecs)
        return evals_all, evecs_all

    def sta_plot_spectrum(self, evals, n=1, plot = False, savefig = False, name = None):
        '''
        n - number of energy levels plotted, ascending from lowest
        '''
        fig, ax = plt.subplots(1,1, figsize=(10, 6))
        #fig.canvas.set_window_title('static eigenvalues')
        if plot == True:
            ax.set_title('Eigenvalues')
            ax.set_xlabel("Anneal Time")
            ax.set_ylabel("Eigenvalue")
        label = self._label_states()
        evals = np.array(evals)
        for j in range(0, n):
            #ax.legend(prop={'size': 8}) 
            ax.plot(self._s, evals[:,j], label = label)
        if savefig == True:
            self.__save_fig(name)
            plt.show()
        #no need for return, as we already have evals_all in get_sta_data
    
    def sta_plot_spectrum_floor(self, evals, n=1, plot = False, savefig = False, name = None):
        '''
        plots the level spectrum, but with the ground state subtracted
        n - number of energy levels plotted, ascending from lowest
        '''
        if plot == True:
            fig, ax = plt.subplots(1,1, figsize=(10, 6))
            #fig.canvas.set_window_title('Static Eigenvalues')
            ax.set_title('Energy Spectrum')
            ax.set_xlabel("$s$")
            ax.set_ylabel("$\Delta E$")
        label = self._label_states()
        evals = np.array(evals)
        DeltaE_all = []
        for j in range(0, n):
            state = label[j]
            DE = evals[:,j]-evals[:,0]
            DeltaE_all.append(DE)
            if plot == True:
                if j == 0 or j == 1:
                    ax.plot(self._s, DE, label = label[j], zorder = 100 , lw = 4)
                else: 
                #make higher energies gray
                    ax.plot(self._s, DE, color = 'gray', lw = 2)
        if savefig == True:
            self.__save_fig(name)
        if plot == True:
            #ax.legend(prop={'size': 8})
            plt.show()
        return DeltaE_all

    def sta_plot_evec(self,evals, evecs, n, m, plot = False, savefig = False, name = None):
        '''
        n - state in ascending energy (0 -> 2^n-1)
        m - number of components plotted (1 -> n)
        '''
        if plot == True:
            #bit_label = self._label_states()
            fig, ax = plt.subplots(1,1, figsize=(10, 6))
            fig.canvas.set_window_title('Static n=%s eigenvector components' % n)
            ax.set_title('Components for n = %s = %.1f Eigenvalue'% (str(n), evals[-1][n]))
            ax.set_xlabel('$s$')
            ax.set_ylabel('Component Value')
            ax.tick_params(labelsize = 20)
    
        evecs_in = np.array(evecs)[:,n]
        evecs_in = evecs_in.transpose()
        locs_overlap = self.find_flip_locations_1(evecs_in)
        overlaps = []

        for j in range(m):
            overlap = np.zeros(len(self._tsteps), dtype = complex)
            for i in range(len(self._s)):
                overlap[i] = np.inner(evecs[i][n], evecs[-1][j])
            overlap = self.__unflip(overlap, locs_overlap)
            overlaps.append(overlap)
            if plot == True:
                ax.legend(prop={'size': 8})
                if j == 0 or j == 1:
                    ax.plot(self._s, overlap, zorder = 100, lw = 4)
                else: 
                    #make higher energies gray
                    ax.plot(self._s, overlap, color = 'gray', lw = 2)
                plt.show()
        if savefig == True:
            self.__save_fig(name)
        return overlaps

    '''DYNAMIC - NEED STATES INPUT'''

    def dyn_plot_cont(self, evecs, states, n = 1, plot = False, savefig = False, name = None):
        '''
        overlap with E_i(s)
        states - array obtained from qt.mesolve
        n - number of energy levels plotted, in ascending energy (?)
        '''
        Hs = self._evolving_Ham()
        if plot == True:
            fig, ax = plt.subplots(1,1, figsize=(10, 6))
            fig.canvas.set_window_title('continuous overlap')
            ax.set_title('Continuous Overlap')
            ax.set_xlabel('Anneal Time (ns)', size = 15)
            ax.set_ylabel('Overlap Value', size = 15)
        overlaps =[]
        overlaps_squared = []
        for j in range(n):
            label = "$|\\langle\Psi(s)|E_%s(s)\\rangle|^2$" % str(j)
            overlap_squared = np.zeros(len(Hs), dtype = np.csingle)
            for i in range(len(Hs)):
                evec = qt.Qobj(evecs[i][j])
                overlap = states[i].overlap(evec)
                overlap_squared[i] = overlap * np.conj(overlap)
            overlaps.append(overlap)
            overlaps_squared.append(overlap_squared)
            if plot == True:
                if j == 0 or j == 1:
                    ax.plot(self._tsteps, overlap_squared, label = label, zorder = 100, lw = 4)
                else: 
                    #make higher energies gray
                    ax.plot(self._tsteps, overlap_squared, color = 'gray', lw = 2)
                ax.tick_params(labelsize = 20)
                ax.legend(title = 'Probability', prop={'size': 10})
                plt.show()
            if savefig == True: 
                self.__save_fig(name)
        return overlaps, overlaps_squared

    def dyn_plot_comp(self, states, n = 1, plot = False, savefig = False, name = None):
        '''
        plots overlap integral with the E_i basis 
        states - tuple solution obtained from qt.mesolve
        n - number of overlaps plotted
        '''
        # eigenvectors
        evecs = self.__reorder(self._Hp)[1]
        labels = self._label_states()
        if plot == True:
            fig, ax = plt.subplots(1,1, figsize=(10, 6))
            fig.canvas.set_window_title('Computational Overlap')
            ax.set_ylabel('Overlap', size = 20)
            ax.set_xlabel('Anneal Time (ns)', size = 20)
            ax.tick_params(labelsize = 20)
        overlaps=[]
        overlaps_squared = []
        for j in range(n):
            overlap_squared = np.zeros(len(self._s), dtype = np.csingle)
            for i in range(len(self._s)):
                evec = qt.Qobj(evecs[j])
                # states = np.array(states)
                # state = np.transpose(states[i])
                overlap = states[i].overlap(evec)
                overlap_squared[i] = overlap * np.conj(overlap)
            overlaps.append(overlap)
            overlaps_squared.append(overlap_squared)
            if plot == True:
                if j == 0 or j == 1:
                    ax.plot(self._tsteps, overlap_squared, label = label, zorder = 100, lw = 4 )
                else: 
                    #make higher energies gray
                    ax.plot(self._tsteps, overlap_squared, color = 'gray', lw = 2)
                ax.tick_params(labelsize = 20)
                ax.legend(title = 'spin state', prop={'size': 8})
                plt.show()
        if savefig == True:
            self.__save_fig(name)
        return overlaps, overlaps_squared

class ham(ParentHam):

    def __init__(self, Hd, Hp, anneal_time, grain, Hc=None):
        super().__init__(Hd, Hp, anneal_time, grain, Hc)
    def __generate_binary(self, n):
        ''' 
        Generate a list of all n bit strings
        '''    
        bin_arr = range(0, int(math.pow(2,n)))
        bin_arr = [bin(i)[2:] for i in bin_arr]
        max_len = len(max(bin_arr, key=len))
        bin_arr = [i.zfill(max_len) for i in bin_arr]
        strings = []
        for string in bin_arr:
            newString = ""
            for b in string:
                if b == "0":
                    newString = newString + "1"
                elif b == "1":
                    newString = newString + "0"
            strings.append(newString)
        return strings

    def _label_states(self):
        '''
        input - Qobj
        
        returns state labels according to their order in 
        increasing energy, for the problem Hamiltonian
        
        '''
        # get eigenvectors and create empty list
        evecs = eigh(self._Hp)[1]
        evecst = evecs.transpose()
        evecs = np.array(evecst)
        l = len(evecs)
        print(l)
        nbits = np.log2(l)
        print(nbits)
        temp = []
        binum = self.__generate_binary(nbits)
        #scans through each row, and orders them according

        for j in range(len(evecs[0])):
            for i in range(len(evecs[0])):
                if evecs[j][i] == 1:
                    temp.append(binum[i])
        return temp

class wmis(ParentHam):

    def __init__(self, Hd, Hp, anneal_time, grain, s_values, m_lists, Hc=None):
        super().__init__(Hd, Hp, anneal_time, grain, Hc)
        self._s_values = s_values
        self._m_lists = m_lists

    def _label_states(self):
        Hp_energies, Hp_vecs = np.linalg.eigh(self._Hp)
        zipped = list(zip(Hp_energies, Hp_vecs.transpose()))
        sorted_states = sorted(zipped, key=lambda x: x[0])
        Hp_vecs = [ np.squeeze(np.asarray(state[1])) for state in sorted_states ]
        state_list = [ str(self._s_values[0]) + ',' + str(m) for m in self._m_lists[0] ]
        for v in range(1,len(self._s_values)):
            new_state_list = []
            for state in state_list:
                new_state_list.extend([ state + ";  " + str(self._s_values[v]) + ',' + str(m) for m in self._m_lists[v]])
            state_list = new_state_list

        state_dict = {}
        for state in range(len(Hp_energies)):
            vec = Hp_vecs[state]
            loc = 0 
            for pot_loc,c in enumerate(vec):
                if c == 1:
                    loc = pot_loc
            state_dict[state] = state_list[loc]
        return state_dict

class wmisReport(wmis):

    '''
    class with slightly different plots especially for the report, redefining dyn_plot_cony
    '''
    def __init__(self, Hd, Hp, anneal_time, grain, s_values, m_lists, Hc=None):
        super().__init__(Hd, Hp, anneal_time, grain, s_values, m_lists, Hc)
    
    def dyn_plot_cont(self, evecs, states, n = 1, plot = False, savefig = False, name = None):
        '''
        overlap with E_i(s)
        states - array obtained from qt.mesolve
        n - number of energy levels plotted, in ascending energy (?)
        '''
        Hs = self._evolving_Ham()
        if plot == True:
            fig, ax = plt.subplots(1,1, figsize=(6, 8))
            #fig.canvas.set_window_title('continuous overlap')
            ax.set_xlabel('Anneal Time (ns)', size = 14)
            ax.set_ylabel('Overlap Value', size = 14)
            ax.grid(linestyle='-', linewidth=1)
        overlaps =[]
        for j in range(n):
            label = "$|\\langle\Psi(s)|E_%s(s)\\rangle|^2$" % str(j)
            overlap_squared = np.zeros(len(Hs), dtype = np.csingle)
            for i in range(len(Hs)):
                evec = qt.Qobj(evecs[i][j])
                overlap = states[i].overlap(evec)
                overlap_squared[i] = overlap * np.conj(overlap)
            overlaps.append(overlap)
            if plot == True:
                if j == 0 or j == 1:
                    ax.plot(self._tsteps, overlap_squared, label = label, zorder = 100, lw = 3)
                else: 
                    #make higher energies gray
                    ax.plot(self._tsteps, overlap_squared, color = 'gray', lw = 2)
                ax.tick_params(labelsize = 14)
                ax.legend(title = 'Probability', prop={'size': 10})
                plt.show()
            if savefig == True: 
                self.__save_fig(name)
        return overlaps

