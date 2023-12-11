from bacon_demo_LandauZener import *
from wmis_hamiltonian import *

plot_flag = True

# physical parameters
N = 5
na = int((N - 1) / 2)
nb = int((N + 1) / 2)
W = 1
dW = 0.01
Jzz = 5.77  # possibly 5.33?
# Jxx = 0
Escale = 15
# Get hzs and Jzz

# simulation parameters
catalyst_num = 7
grain = 10000
anneal_time = 50
s = np.linspace(0, 1, grain)

catalyst_strengths = np.linspace(1.85, 1.89, 40)
ncats = len(catalyst_strengths)
Epp_all = np.zeros((2, ncats))


spin_coeff, Jzz_rescaled = generate_spin_coeff(N, W, dW, Jzz)
coupling_coeff = generate_coupling_coeff(na, nb, Jzz_rescaled)

H_input = bacon(N, spin_coeff, coupling_coeff)
Hd = H_input.driver()
Hp = H_input.problem() * Escale

if plot_flag:
    # catalyst_strengths = np.linspace(1.75,2.0, 100)
    from matplotlib.ticker import ScalarFormatter

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    Epp_all = np.load("Epp_all_5spin_highres.npy")
    fig, ax = plt.subplots(1, 2, figsize=(11, 6))
    colors = ["C0", "C1"]

    for i in range(2):
        split_cat, split_Epp = split_array_at_max(catalyst_strengths, Epp_all[i])
        ax[i].plot(split_cat[0], split_Epp[0], "x", lw=2, color=colors[i])
        ax[i].plot(split_cat[1], split_Epp[1], "x", lw=2, color=colors[i])
        ax[i].yaxis.set_major_formatter(formatter)

    ax[0].set_ylabel(r"$E^{\prime\prime}_0(J_{xx})$")
    ax[1].set_ylabel(r"$E^{\prime\prime}_1(J_{xx})$")
    ax[1].set_xlabel(r"$J_{xx}$ (5 spin)")
    fig.tight_layout()

    plt.show()

    from sys import exit as sys_exit

    sys_exit()


for i in tqdm(range(ncats)):
    Hc = H_catalyst_LZ(N, float(catalyst_strengths[i]))
    H_LZ = ham(Hd, Hp, anneal_time, grain, Hc)
    energies = energy_levels(H_LZ)

    cgi = find_cgi(energies)
    s_shifted = s - s[cgi]
    energies_shifted = centre_energies(energies)

    # store landau zener fits
    lz_fits = []
    # energy, derivative and 2nd derivative
    E_params = []

    for j in range(2):
        LZ_fit = landau_zener_fit(s_shifted, energies_shifted, j)
        lz_fits.append(LZ_fit)
        _, E_prime_prime = energy_derivatives(s_shifted, energies_shifted[j])
        E_params.append(E_prime_prime[cgi])

    # store the 2nd derivative at the CGI
    Epp_all[:, i] = E_params

np.save("Epp_all_5spin_highres.npy", Epp_all)
