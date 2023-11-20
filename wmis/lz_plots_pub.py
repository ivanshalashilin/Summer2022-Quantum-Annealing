from bacon_demo_LandauZener import *
from wmis_hamiltonian import *

plot_spectrum_avg = False
plot_spectrum_floor = False
plot_spectrum_noshift = False
plot_spectrum_ceiling = False
plot_lz_flag = True


catalyst_num = 7
grain = 1000
anneal_time = 50
s = np.linspace(0, 1, grain)

if N == 5:
    nine_spin = False
    catalyst_strengths = [1.75, 1.83, 1.87, 1.92, 2.00]
    ncats = len(catalyst_strengths)
if N == 9:
    nine_spin = True
    errors = [
        -0.075,
        -0.05,
        -0.025,
        0,
    ]  # 0.025, 0.05, 0.075]
    error_string = [str(error).replace(".", "p") for error in errors]
    catalyst_strengths = [1.4922 * (1 + error) for error in errors]
    ncats = len(catalyst_strengths)
    print(f"catalyst strengths: {catalyst_strengths}")


abc_coeffs_all = np.zeros((ncats, 3, 3))
for i in tqdm(range(ncats)):
    abc_coeffs = np.zeros((3, 3))

    if not nine_spin:
        Hc = H_catalyst_LZ(N, float(catalyst_strengths[i]))
        H_LZ = ham(Hd, Hp, anneal_time, grain, Hc)
        energies = energy_levels(H_LZ)

    if nine_spin:
        from ninespin_preprocessing import s_9spin, energies_9spin

        energies = energies_9spin[i]
        s = s_9spin

        # Hc = H_catalyst_LZ(N, float(catalyst_strengths[i]))
        # H_LZ = ham(Hd, Hp, anneal_time, grain, Hc)
        # energies = energy_levels(H_LZ)
        # s = np.linspace(0, 1, grain)

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
        E_prime, E_prime_prime = energy_derivatives(s_shifted, energies_shifted[j])
        E_params.append([energies_shifted[j][cgi], E_prime[cgi], E_prime_prime[cgi]])
        abc_coeffs[j] = np.array(
            [A_coeff(*E_params[j]), B_coeff(*E_params[j]), E_params[j][0]]
        )

    # append average
    abc_coeffs[2] = np.array(
        [*np.average(abc_coeffs[:-1], axis=0)[:-1], energies_shifted[j][cgi]]
    )

    # subtract ground state energy
    plot_dict = {
        "X": s_shifted,
        "energies": energies_shifted,
        "lz": lz_fits,
        "abc_coeffs": abc_coeffs,
    }
    labels_dict = {
        "energy": ["Ground state", "First excited state"],
        "lz": ["LZ fit GS", "LZ fit FES"],
        "linear": [None, "$AX, BX$"],
    }

    abc_coeffs_all[i] = abc_coeffs

    filepath = "wmis/pub_plots/nolegend/"
    extra = str(N) + "spin_5p33"
    figsize = (12, 7)
    if plot_spectrum_noshift:
        # plot average
        shift = 0
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_spectrum_shifted(
            ax,
            plot_dict,
            labels_dict,
            shift,
            # kwargs
            xlabel="$X = s-s_0$",
            ylabel="$E$",
            xlim=[-0.001, 0.001],
            ylim=[-0.04, 0.06],
        )

        plt.savefig(f"{filepath}spectrum_{error_string[i]}_noshift{extra}.pdf")
        plt.show()

    if plot_spectrum_avg:
        shift = 0.5 * (energies_shifted[0] + energies_shifted[1])
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_spectrum_shifted(
            ax,
            plot_dict,
            labels_dict,
            shift,
            # kwargs
            xlabel="$X = s-s_0$",
            ylabel="$E-\\bar{E}$",
            xlim=[-0.1, 0.1],
            ylim=[-0.04, 0.06],
        )

        plt.savefig(f"{filepath}spectrum_{error_string[i]}_avg{extra}.pdf")
        plt.show()

    if plot_spectrum_floor:
        shift = energies_shifted[0]
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_spectrum_shifted(
            ax,
            plot_dict,
            labels_dict,
            shift,
            xlabel="$X = s-s_0$",
            ylabel="$E-E_0$",
            xlim=[-0.1, 0.1],
            ylim=[-0.003, 0.033],
        )

        plt.savefig(f"{filepath}spectrum_{error_string[i]}_floor{extra}.pdf")
        plt.show()

    if plot_spectrum_ceiling:
        shift = energies_shifted[1]
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_spectrum_shifted(
            ax,
            plot_dict,
            labels_dict,
            shift,
            xlabel="$X = s-s_0$",
            ylabel="$E-E_1$",
            xlim=[-0.1, 0.1],
            ylim=[-0.033, 0.003],
        )

        plt.savefig(f"{filepath}spectrum_{error_string[i]}_ceiling{extra}.pdf")
        plt.show()


if plot_lz_flag:
    # perform landau zener fit
    if nine_spin:
        from ninespin_preprocessing import data_fidelty, T_9spin

        fidelity_measured = data_fidelty
        t_anneal = T_9spin
    else:
        pickle_in = open("2d_s_2_3__t_1e-1_10000__j_175e-2_20e-1.pkl", "rb")
        data = pickle.load(pickle_in)
        t_anneal = data[1]
        fidelity_measured = np.array(data[2]).T

    cols = ["C" + str(i) for i in range(ncats)]
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i in range(ncats):
        probability_theory = LandauZenerFormula(
            t_anneal,
            *abc_coeffs_all[i][2],
        )
        probability_theory_upper = LandauZenerFormula(
            t_anneal,
            *abc_coeffs_all[i][0],
        )
        probability_theory_lower = LandauZenerFormula(
            t_anneal,
            *abc_coeffs_all[i][1],
        )
        ax.plot(
            t_anneal,
            probability_theory,
            "--",
            color="gray",
            alpha=0.5,
            lw=4,
            label=f"{error_string[i]}",
        )

        ax.fill_between(
            t_anneal,
            probability_theory_upper,
            probability_theory_lower,
            color=cols[i],
            alpha=0.2,
        )

        ax.plot(t_anneal, fidelity_measured[i], color=cols[i], lw=1.4)
    ax.set(xlabel="Anneal time", ylabel="Ground state fidelity")
    # ax.grid()
    # ax.legend()
    # plt.savefig(f"{filepath}_lz_fidelity_{extra}.pdf")
    plt.show()
