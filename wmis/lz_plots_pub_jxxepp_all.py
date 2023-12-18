from bacon_demo_LandauZener import *
from wmis_hamiltonian import *

plot_flag = True

# physical parameters
N = np.array([5,7,9])
na = np.array(((N - 1) / 2), dtype=int)
nb = np.array(((N + 1) / 2), dtype=int)
W = 1
dW = 0.01
Jzz = 5.33  # possibly 5.33?
Escale = 15


# Jxx
Jxx_opt = np.array([1.928, 1.618, 1.492, 1.429])

# simulation parameters
catalyst_num = 7
grain = 1000
anneal_time = 50
s = np.linspace(0, 1, grain)
ncats = 50
filepath = "wmis/epp_data/"

if plot_flag:
    # catalyst_strengths = np.linspace(1.75,2.0, 100)
    from matplotlib.ticker import ScalarFormatter

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))


    for i in range(len(N)):
        catalyst_strengths = np.linspace(Jxx_opt[i] * 0.9, Jxx_opt[i] * 1.1, ncats)
        filename = filename = (
            "Epp_opt_"
            + str(N[i])
            + "spin_"
            + str(grain)
            + "grain_"
            + fl_to_str(Jzz)
            + "jzz.npy"
        )

        Epp_all = np.load(filepath + filename)
        fig, ax = plt.subplots(1, 2, figsize=(11, 6))
        colors = ["C0", "C1"]

        for j in range(2):
            split_cat, split_Epp = split_array_at_cat(catalyst_strengths, Epp_all[j], Jxx_opt[i])
            ax[j].plot(split_cat[0], split_Epp[0], lw=2, color=colors[j])
            ax[j].plot(split_cat[1], split_Epp[1], lw=2, color=colors[j])
            ax[j].vlines(
                Jxx_opt[i], 
                ymin=0,
                ymax=(-1) ** (j + 1) * np.max(Epp_all),
                color="k",
                linestyle="--",
            )
            ax[j].yaxis.set_major_formatter(formatter)

        ax[0].set_ylabel(r"$E^{\prime\prime}_0(J_{xx})$")
        ax[1].set_ylabel(r"$E^{\prime\prime}_1(J_{xx})$")
        ax[1].set_xlabel(r"$J_{xx}$ (" + str(N[i]) + " spin)")
        fig.tight_layout()

        plt.savefig("wmis/pub_plots/epp_jxx/"+filename[:-4]+".pdf")
        plt.show()

for i in range(len(N)):
    # catalyst strengths
    catalyst_strengths = np.linspace(Jxx_opt[i] * 0.9, Jxx_opt[i] * 1.1, ncats)
    Epp_all = np.zeros((2, ncats))

    spin_coeff, Jzz_rescaled = generate_spin_coeff(N[i], W, dW, Jzz)
    coupling_coeff = generate_coupling_coeff(na[i], nb[i], Jzz_rescaled)

    H_input = bacon(N[i], spin_coeff, coupling_coeff)
    Hd = H_input.driver()
    Hp = H_input.problem() * Escale
    filename = (
        "Epp_opt_"
        + str(N[i])
        + "spin_"
        + str(grain)
        + "grain_"
        + fl_to_str(Jzz)
        + "jzz.npy"
    )
    for j in tqdm(range(ncats)):
        Hc = H_catalyst_LZ(N[i], float(catalyst_strengths[j]))
        H_LZ = ham(Hd, Hp, anneal_time, grain, Hc)
        energies = energy_levels(H_LZ)

        cgi = find_cgi(energies)
        s_shifted = s - s[cgi]
        energies_shifted = centre_energies(energies)

        # store landau zener fits
        lz_fits = []
        # energy, derivative and 2nd derivative
        E_params = []

        for k in range(2):
            LZ_fit = landau_zener_fit(s_shifted, energies_shifted, k)
            lz_fits.append(LZ_fit)
            _, E_prime_prime = energy_derivatives(s_shifted, energies_shifted[k])
            E_params.append(E_prime_prime[cgi])

        # store the 2nd derivative at the CGI
        Epp_all[:, j] = E_params
    np.save(filepath + filename, Epp_all)
