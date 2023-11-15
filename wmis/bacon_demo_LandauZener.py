import sys

sys.path.append("./")
from classes.hamv2 import wmis
from classes.hamv2 import ham
from classes.bacon import bacon
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks
from wmis_hamiltonian import Hd, Hp

import mplhep as hep

hep.style.use("CMS")

np.set_printoptions(precision=5)


from matplotlib import rcParams

colors = rcParams["axes.prop_cycle"].by_key()["color"]

"""
demo of custom hamiltonians with bacon class
"""


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


def LandauZenerFormula(t, A, B, C):
    Gammaovert = C**2 / np.abs(B - A)
    return np.exp(-2 * np.pi * Gammaovert * t)


# putting state into qutip
# initial coefficients
def d_coeff(t, params):
    return 1 - t / params["T"]


def p_coeff(t, params):
    return t / params["T"]


def c_coeff(t, params):
    return t / params["T"] * (1 - t / params["T"])


# coefficients for landau zener fit (ONLY CORRECT ONCE SHIFTED)
def B_coeff(e, e_prime, e_prime_prime):
    return e_prime + np.sqrt(e * e_prime_prime)


def A_coeff(e, e_prime, e_prime_prime):
    return e_prime - np.sqrt(e * e_prime_prime)


def C_coeff(e):
    return e


def find_cgi(energies):
    """
    Finds the index of the closing gap
    input: ground state energy (array), first excited state energy (array)
    output: index of closing gap (int)
    """
    # find the location of the closing gap
    Delta_E = energies[1] - energies[0]
    gap_index = find_peaks(-Delta_E)[0][0]
    return gap_index


def energy_levels(Hs: ham):
    """
    Obtain ground state and and first excited pair of energies from a hamiltonian object
    input: Hamiltonian (ham)
    output: ground state energy, first excited state energy (array)
    """
    evals, _ = Hs.sta_get_data()
    evals = np.array(evals).T
    return [evals[0], evals[1]]


def centre_energies(energies):
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
    cgi = find_cgi(energies)
    # find the corresponding s value
    min_gse = energies[0][cgi]
    min_fese = energies[1][cgi]
    # shift the energies
    energy_avg = np.average([min_gse, min_fese])
    return [energies[0] - energy_avg, energies[1] - energy_avg]


def energy_derivatives(energy_spectrum):
    """
    computes and returns first and second derivatives of energy spectrum
    input: energy_spectrum (array)
    output: first derivative of energy spectrum (array), second derivative of energy spectrum (array)
    """
    s = np.linspace(0, 1, len(energy_spectrum))
    h = s[1] - s[0]
    e_prime = np.gradient(energy_spectrum, s)
    e_prime_prime = second_derivative(s, energy_spectrum)

    return e_prime, e_prime_prime


def second_derivative(s, e):
    h = s[1] - s[0]
    # inexact second derivative for boundary points (ok because not used)
    e_pp_mid = np.gradient(np.gradient(e, s), s)
    e_prime_prime = (np.roll(e, -1) + np.roll(e, 1) - 2 * e) / h**2
    # substitute boundary points
    e_prime_prime[0] = e_pp_mid[0]
    e_prime_prime[-1] = e_pp_mid[-1]
    return e_prime_prime


def error_firstdvtve(function):
    """
    Estimate the error on the numerical derivative when performing finite differences
    """
    s = np.linspace(0, 1, len(function))
    h = s[1] - s[0]
    # require 3d derivative
    e_p3 = np.gradient(np.gradient(np.gradient(function, s), s), s)

    return h**2 / 6 * np.max(np.abs(e_p3))


def error_seconddvtve(function):
    """
    Estimate the error on the numerical derivative when performing finite differences
    """
    s = np.linspace(0, 1, len(function))
    h = s[1] - s[0]
    # require 3d derivative
    e_p4 = (
        np.gradient(np.gradient(np.gradient(np.gradient(function, s), s), s), s) / 100
    )
    return h**2 / 12 * np.max(np.abs(e_p4))


def error_a_coeffs(
    a_coeff,
    err_e_prime,
    err_e_prime_prime,
    e,
    e_prime_prime,  # required for term with error on e_prime_prime
):
    return np.abs(
        a_coeff
        * np.sqrt(err_e_prime**2 + (e / 4 * e_prime_prime) * err_e_prime_prime**2)
    )


def landau_zener_fit(s, energies, energy_index=0):
    """
    finds the first order Landau Zener function for a given energy level, determined by derivatives

    input: ground state and first excited state (list), (array), first derivative of energy spectrum (array), second
    derivative of energy spectrum (array), index of closing gap (int), ground or first excited state (int)

    output: first order Landau Zener fit (array)
    """
    cgi = find_cgi(energies)

    e_prime, e_prime_prime = energy_derivatives(energies[energy_index])
    e = energies[energy_index][cgi]
    e_prime_cg = e_prime[cgi]
    e_prime_prime_cg = e_prime_prime[cgi]
    return e_prime_cg * s + (-1) ** (energy_index + 1) * np.sqrt(
        e**2 + e * e_prime_prime_cg * s**2
    )


def plot_spectrum_shifted(
    ax,
    plot_dict,
    labels_dict,
    shift=0,
    **kwargs,
):


    cols = [
        ["C0", "C1", "black", "gray"],
        ["C1", "C0", "black", "gray"],
        ]

    X = plot_dict["X"]
    for k in range(2):
        ax.plot(
            X,
            plot_dict["energies"][k] - shift,
            color=cols[k][0],
            label=labels_dict["energy"][k],
        )
        ax.plot(
            X,
            plot_dict["lz"][k] - shift,
            color=cols[k][1],
            linestyle="--",
            label=labels_dict["lz"][k],
        )
        ax.plot(
            X,
            plot_dict["abc_coeffs"][1][k] * X - shift,
            linestyle="dotted",
            color=cols[k][2],
            alpha=0.8,
            label=labels_dict["linear"][k],
        )
        ax.fill_between(
            X,
            plot_dict["abc_coeffs"][0][k] * X - shift,
            plot_dict["abc_coeffs"][1][k] * X - shift,
            color=cols[k][3],
            alpha=0.4,
        )
    ax.set(**kwargs)
    ax.legend()
    ax.grid()





"""
plot_spectrum = True
catalyst_num = 4
grain = 20000
anneal_time = 50
s = np.linspace(0, 1, grain)

catalyst_strengths = ["1.75", "1.83", "1.87", "1.92", "2.00"]
abc_coeffs_all_avg = np.zeros((5, 3))
abc_coeffs_upper = np.zeros((5, 3))
abc_coeffs_lower = np.zeros((5, 3))
for i in range(5):
    Hc = H_catalyst_LZ(float(catalyst_strengths[i]))
    H_LZ = ham(Hd, Hp, anneal_time, grain, Hc)

    energies = energy_levels(H_LZ)

    cgi = find_cgi(energies)

    plot_dict = s - s[cgi]

    plot_dict['energies_shifted'][k]centre_energies(energies)

    lz_fits = []
    # energy, derivative and 2nd derivative
    E_params = []
    ABC_coeffs = []
    ABC_errs = []
    for j in range(2):
        LZ_fit = landau_zener_fit(plot_dict, plot_dict['energies_shifted'][k])
        lz_fits.append(LZ_fit)
        E_prime, E_prime_prime = energy_derivatives(plot_dict['energies_shifted'][k])
        E_params.append([plot_dict['energies_shifted'][k][cgi], E_prime[cgi], E_prime_prime[cgi]])
        ABC_coeffs.append(
            [A_coeff(*E_params[j]), B_coeff(*E_params[j]), E_params[j][0]]
        )

    abc_coeffs_arr = np.array(ABC_coeffs)

    abc_coeffs_lower[i] = abc_coeffs_arr[0]
    abc_coeffs_upper[i] = abc_coeffs_arr[1]

    abc_coeffs_avg = np.average(abc_coeffs_arr, axis=0)
    abc_coeffs_avg[-1] = plot_dict['energies_shifted'][k][cgi]

    abc_coeffs_all_avg[i] = abc_coeffs_avg
    # subtract ground state energy

    # error on the coefficients
    err_e_prime = error_firstdvtve(plot_dict['energies_shifted'][k])
    err_e_prime_prime = error_seconddvtve(plot_dict['energies_shifted'][k])
    err_a = error_a_coeffs(
        abc_coeffs_avg[0],
        err_e_prime,
        err_e_prime_prime,
        plot_dict['energies_shifted'][k][cgi],
        E_params[0][2],
    )
    err_b = error_a_coeffs(
        abc_coeffs_avg[1],
        err_e_prime,
        err_e_prime_prime,
        plot_dict['energies_shifted'][k][cgi],
        E_params[0][2],
    )
    ABC_errs.append([err_a, err_b, None])

    if plot_spectrum and i == catalyst_num:
        
        shift = -(energies[0]+energies[1])/2
        
        fig, ax = plt.subplots(1, 1)
        labels_energies = ["Ground state", "First excited state"]
        labels_lz = ["LZ fit GS", "LZ fit FES"]
        labels_linear = [None, "$AX, BX$"]
        for k in range(2):
            ax.plot(
                plot_dict,
                plot_dict['energies_shifted'][k] - shift,
                color=colors[2 * k],
                label=labels_energies[k],
            )
            ax.plot(
                plot_dict,
                lz_fits[k] - shift,
                color=colors[2 * k + 1],
                linestyle="--",
                label=labels_lz[k],
            )
            ax.plot(
                plot_dict,
                abc_coeffs_all_avg[i][k] * plot_dict - shift,
                linestyle="dotted",
                color="black",
                alpha=0.8,
                label=labels_linear[k],
            )
            ax.fill_between(
                plot_dict,
                abc_coeffs_upper[i][k] * plot_dict - shift,
                abc_coeffs_lower[i][k] * plot_dict - shift,
                color="gray",
                alpha=0.4,
            )
        ax.set(xlabel="$X = s-s_0$", ylabel="$E-E_0$")
        ax.legend()
        ax.grid()
        plt.savefig(f"spectrum_{catalyst_strengths[i]}.pdf")
        plt.show()

print(np.abs((abc_coeffs_upper - abc_coeffs_lower) / abc_coeffs_all_avg) * 100)

# np.save("abc_coeffs_all.npy", abc_coeffs_all_avg)


# perform landau zener fit
pickle_in = open("2d_s_2_3__t_1e-1_10000__j_175e-2_20e-1.pkl", "rb")
data = pickle.load(pickle_in)

t_anneal = data[1]


colsdark = ["darkred", "darkorange", "darkgoldenrod", "darkgreen", "darkblue"]
cols = ["red", "orange", "goldenrod", "green", "blue"]
fig, ax = plt.subplots()

for i in range(5):
    probability_meas = np.array(data[2])[:, i]
    probability_theory = LandauZenerFormula(
        t_anneal,
        abc_coeffs_all_avg[i][0],
        abc_coeffs_all_avg[i][1],
        abc_coeffs_all_avg[i][2],
    )
    probability_theory_upper = LandauZenerFormula(
        t_anneal, abc_coeffs_upper[i][0], abc_coeffs_upper[i][1], abc_coeffs_upper[i][2]
    )
    probability_theory_lower = LandauZenerFormula(
        t_anneal, abc_coeffs_lower[i][0], abc_coeffs_lower[i][1], abc_coeffs_lower[i][2]
    )

    ax.fill_between(
        t_anneal,
        probability_theory_upper,
        probability_theory_lower,
        color=cols[i],
        alpha=0.4,
    )

    ax.set(title=f"grain = {grain}")
    ax.plot(t_anneal, probability_meas, color=colsdark[i], lw=1)
    ax.plot(t_anneal, probability_theory, "-", color=cols[i], alpha=0.8, lw=1)

plt.show()
"""
