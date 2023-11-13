import sys

sys.path.append("./")
from classes.hamv2 import wmis
from classes.hamv2 import ham
from classes.bacon import bacon
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=5)
from scipy.signal import find_peaks


from matplotlib import rcParams

colors = rcParams["axes.prop_cycle"].by_key()["color"]

"""
demo of custom hamiltonians with bacon class
"""

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


coupling_coeff = [
    0,
    Jzz,
    Jzz,
    Jzz,
    # 12 - 15
    Jzz,
    Jzz,
    Jzz,
    # 23-25
    0,
    0,
    0,
]

H_input = bacon(5, spin_coeff, coupling_coeff)
Hd = H_input.driver()
Hp = H_input.problem() * Escale


def H_catalyst_LZ(catalyst_strength):
    return catalyst_strength * qt.tensor(
        qt.qeye(2), qt.qeye(2), qt.qeye(2), qt.sigmax(), qt.sigmax()
    )


def init_superpsn(n):
    """
    n - number of qubits
    """
    state = np.ones((2**n), dtype=int) / np.sqrt(2**n)
    state = qt.Qobj(state)
    return state


initial_state = init_superpsn(5)

grain = 5000
anneal_time = 50
s = np.linspace(0, 1, grain)


# putting state into qutip
# initial coefficients
def d_coeff(t, params):
    return 1 - t / params["T"]


def p_coeff(t, params):
    return t / params["T"]


def c_coeff(t, params):
    return t / params["T"] * (1 - t / params["T"])


catalyst_strengths = ["1.75", "1.83", "1.87", "1.92", "2.00"]


def find_closing_gap_index(E_ground_state_shifted, E_first_excited_state_shifted):
    """
    Finds the index of the closing gap
    input: ground state energy (array), first excited state energy (array)
    output: index of closing gap (int)
    """
    # find the location of the closing gap
    Delta_E = E_first_excited_state_shifted - E_ground_state_shifted
    gap_index = find_peaks(-Delta_E)[0][0]
    return gap_index


Hc = H_catalyst_LZ(float(catalyst_strengths[0]))
H_LZ = ham(Hd, Hp, anneal_time, grain, Hc)


def energy_levels(Hs: ham):
    """
    Obtain ground state and and first excited pair of energies from a hamiltonian object
    input: Hamiltonian (ham)
    output: ground state energy, first excited state energy (array)
    """
    evals, _ = Hs.sta_get_data()
    evals = np.array(evals).T
    return evals[0], evals[1]


def centre_energies(E_ground_state, E_first_excited_state):
    """
    Centres the closing gap at the origin
    Performs a vertical shift on the energy levels from
    E -> E + shift.
    Locates the location of the closing gap, s0, Also performs a
    horizontal shift from s -> X = s-s0, where s0 is the location of the closing gap.

    input: E_ground_state (array), E_first_excited_state (array)
    returns: ground state energy shifted (array), first excited energy state shifted
    (array)
    """
    # find the location of the closing gap
    closing_gap_index = find_closing_gap_index(E_ground_state, E_first_excited_state)
    # find the corresponding s value
    min_gse = E_ground_state[closing_gap_index]
    min_fese = E_first_excited_state[closing_gap_index]
    # shift the energies
    energy_gap = np.average([min_gse, min_fese])
    return E_ground_state - energy_gap, E_first_excited_state - energy_gap


def energy_derivatives(energy_spectrum):
    """
    computes and returns first and second derivatives of energy spectrum
    input: energy_spectrum (array)
    output: first derivative of energy spectrum (array), second derivative of energy spectrum (array)
    """
    s = np.linspace(0, 1, len(energy_spectrum))
    h = s[1] - s[0]
    e_prime = np.diff(energy_spectrum) / h
    e_prime_prime = np.diff(e_prime) / h

    return e_prime, e_prime_prime


def landau_zener_fit(s, energies, energy_index=0):
    """
    finds the first order Landau Zener function for a given energy level, determind by derivatives

    input: ground state and first excited state (list), (array), first derivative of energy spectrum (array), second
    derivative of energy spectrum (array), index of closing gap (int), ground or first excited state (int)

    output: first order Landau Zener function (array)
    """
    closing_gap_index = find_closing_gap_index(energies[0], energies[1])

    e_prime, e_prime_prime = energy_derivatives(energies[energy_index])
    e = energies[energy_index][closing_gap_index]
    e_prime_cg = e_prime[closing_gap_index]
    e_prime_prime_cg = e_prime_prime[closing_gap_index]
    return e_prime_cg * s + (-1)** (energy_index+1) * np.sqrt(e**2 + 0.5 * e * e_prime_prime_cg * s**2)


E_ground_state, E_first_excited_state = energy_levels(H_LZ)

E_ground_state_shifted, E_first_excited_state_shifted = centre_energies(
    E_ground_state, E_first_excited_state
)

s_shifted = s - s[find_closing_gap_index(E_ground_state, E_first_excited_state)]

LZ_fit_gs = landau_zener_fit(
    s_shifted, [E_ground_state_shifted, E_first_excited_state_shifted], 0
)
LZ_fit_fes = landau_zener_fit(
    s_shifted, [E_ground_state_shifted, E_first_excited_state_shifted], 1
)


# plot GS and FES
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(s_shifted, E_ground_state_shifted, color=colors[0], label="gs")
ax.plot(s_shifted, E_first_excited_state_shifted, color=colors[1], label="fes")
# plot LZ fits
ax.plot(s_shifted, LZ_fit_gs, color=colors[2], linestyle="--", label="LZ fit gs")
ax.plot(s_shifted, LZ_fit_fes, color=colors[3], linestyle="--", label="LZ fit fes")
ax.legend()
# plt.savefig('.png', dpi = 600)
plt.show()
