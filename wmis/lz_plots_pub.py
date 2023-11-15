from bacon_demo_LandauZener import *


plot_spectrum_avg = False
plot_spectrum_floor = False
plot_spectrum_noshift = False
plot_spectrum_ceiling = False

plot_lz_flag = True


catalyst_num = 4
grain = 1000
anneal_time = 50
s = np.linspace(0, 1, grain)
ncat = 5
catalyst_strengths = [1.75, 1.83, 1.87, 1.92, 2.00]
abc_coeffs_all = np.zeros((5, 3, 3))


for i in range(5):
    abc_coeffs = np.zeros((3, 3))
    Hc = H_catalyst_LZ(float(catalyst_strengths[i]))
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
        E_prime, E_prime_prime = energy_derivatives(energies_shifted[j])
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

    filepath = "wmis/pub_plots/"

    if plot_spectrum_noshift:
        # plot average
        shift = 0
        fig, ax = plt.subplots(1, 1)
        plot_spectrum_shifted(
            ax,
            plot_dict,
            labels_dict,
            shift,
            # kwargs
            xlabel="$X = s-s_0$",
            ylabel="$E$",
            xlim= [-0.001, 0.001],
            ylim = [-0.05,0.05]
        )
        extra = "_mini"
        plt.savefig(f"{filepath}spectrum_{catalyst_strengths[i]}_noshift{extra}.pdf")
        plt.show()

    if plot_spectrum_avg:
        shift = 0.5 * (energies_shifted[0] + energies_shifted[1])
        fig, ax = plt.subplots(1, 1)
        plot_spectrum_shifted(
            ax,
            plot_dict,
            labels_dict,
            shift,
            # kwargs
            xlabel="$X = s-s_0$",
            ylabel="$E-\\bar{E}$",
            xlim =[-0.04, 0.04],
            ylim =[-0.04, 0.04]
        )
        
        extra = "_mini"

        plt.savefig(f"{filepath}spectrum_{catalyst_strengths[i]}_avg{extra}.pdf")
        plt.show()

    if plot_spectrum_floor:
        shift = energies_shifted[0]
        fig, ax = plt.subplots(1, 1)
        plot_spectrum_shifted(
            ax,
            plot_dict,
            labels_dict,
            shift,
            xlabel="$X = s-s_0$",
            ylabel="$E-\\bar{E}$",
            xlim = [-0.023,0.023],
            ylim = [-0.003, 0.055]
        )
        extra = "_mini"
        plt.savefig(f"{filepath}spectrum_{catalyst_strengths[i]}_floor{extra}.pdf")
        plt.show()

    if plot_spectrum_ceiling:
        shift = energies_shifted[1]
        fig, ax = plt.subplots(1, 1)
        plot_spectrum_shifted(
            ax, 
            plot_dict, 
            labels_dict, 
            shift, 
            xlabel="$X = s-s_0$", 
            ylabel="$E-E_1$",
            xlim = [-0.023,0.023],
            ylim = [-0.055, 0.003]
            )


        extra = "_mini"
        plt.savefig(f"{filepath}spectrum_{catalyst_strengths[i]}_ceiling{extra}.pdf")
        plt.show()


if plot_lz_flag:
    # perform landau zener fit
    pickle_in = open("2d_s_2_3__t_1e-1_10000__j_175e-2_20e-1.pkl", "rb")
    data = pickle.load(pickle_in)

    t_anneal = data[1]

    cols = ["C0", "C1", "C2", "C3", "C4"]
    fig, ax = plt.subplots()

    for i in range(5):
        probability_meas = np.array(data[2])[:, i]
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

        ax.fill_between(
            t_anneal,
            probability_theory_upper,
            probability_theory_lower,
            color=cols[i],
            alpha=0.4,
        )

        ax.plot(t_anneal, probability_meas, color=cols[i], lw=1)
        ax.plot(t_anneal, probability_theory, "-", color=cols[i], alpha=0.8, lw=1)
    ax.set(xlabel="Ground state fidelity", ylabel="Anneal time")
    ax.grid()
    plt.savefig(f"{filepath}5spin_lz_fidelity.pdf")
    plt.show()
