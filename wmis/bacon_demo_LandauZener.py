import sys

sys.path.append("./")
from tqdm import tqdm
from classes.hamv2 import wmis
from classes.hamv2 import ham
from classes.bacon import bacon
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks
from wmis_hamiltonian import *
from matplotlib.ticker import ScalarFormatter, FuncFormatter, MultipleLocator


import mplhep as hep

hep.style.use("CMS")

np.set_printoptions(precision=5)


from matplotlib import rcParams

colors = rcParams["axes.prop_cycle"].by_key()["color"]

"""
demo of custom hamiltonians with bacon class
"""


def H_catalyst_LZ(N, catalyst_strength):
    return catalyst_strength * qt.tensor(
        *[qt.qeye(2) for i in range(N - 2)], qt.sigmax(), qt.sigmax()
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
    try:
        gap_index = find_peaks(-Delta_E)[0][0]
        return gap_index
    except IndexError:
        return 0


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


def energy_derivatives(s, energy_spectrum):
    """
    computes and returns first and second derivatives of energy spectrum
    input: energy_spectrum (array)
    output: first derivative of energy spectrum (array), second derivative of energy spectrum (array)
    """

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

    e_prime, e_prime_prime = energy_derivatives(s, energies[energy_index])
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
    if "linewidth" in kwargs.keys():
        linewidth = kwargs["linewidth"]
    else:
        linewidth = 3
    for k in range(2):
        ax.plot(
            X,
            plot_dict["energies"][k] - shift,
            color=cols[k][0],
            label=labels_dict["energy"][k],
            linewidth = linewidth
        )
        ax.plot(
            X,
            plot_dict["lz"][k] - shift,
            color=cols[k][1],
            linestyle="--",
            label=labels_dict["lz"][k],
            linewidth = linewidth
        )
        ax.plot(
            X,
            plot_dict["abc_coeffs"][2][k] * X - shift,
            linestyle="dotted",
            color=cols[k][2],
            alpha=0.8,
            label=labels_dict["linear"][k],
            linewidth = linewidth
        )
        ax.fill_between(
            X,
            plot_dict["abc_coeffs"][0][k] * X - shift,
            plot_dict["abc_coeffs"][1][k] * X - shift,
            color=cols[k][3],
            alpha=0.4,

        )
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    ax.set(**kwargs)

    if "xlabel" in kwargs.keys():
        ax.set_xlabel(kwargs["xlabel"], loc="center")
    if "ylabel" in kwargs.keys():
        ax.set_ylabel(kwargs["ylabel"], loc="center")

    ax.grid()


def split_array_at_max(s, Epp):
    '''
    splits array at maximum point
    '''

    max_index = np.argmax(np.abs(Epp))
    return [s[:max_index], s[max_index:]], [Epp[:max_index], Epp[max_index:]]