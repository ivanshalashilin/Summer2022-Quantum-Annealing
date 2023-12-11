from bacon_demo_LandauZener import *
from wmis_hamiltonian import *

plot_spectrum_avg = False
plot_spectrum_floor = False
plot_spectrum_noshift = False
plot_spectrum_ceiling = False
plot_lz_flag = True


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
grain = 1000
anneal_time = 50
s = np.linspace(0, 1, grain)


if N == 5:
    nine_spin = False
    catalyst_strengths = [1.75, 1.83, 1.87, 1.92, 2.00]
    ncats = len(catalyst_strengths)
if N == 9:
    nine_spin = True
    errors = [-0.075, -0.05, -0.025, 0, 0.025]  # 0.025, 0.05, 0.075]
    error_string = [str(error).replace(".", "p") for error in errors]
    catalyst_strengths = [1.4922 * (1 + error) for error in errors]
    ncats = len(catalyst_strengths)
    print(f"catalyst strengths: {catalyst_strengths}")


spin_coeff, Jzz_rescaled = generate_spin_coeff(N, W, dW, Jzz)
coupling_coeff = generate_coupling_coeff(na, nb, Jzz_rescaled)

H_input = bacon(N, spin_coeff, coupling_coeff)
Hd = H_input.driver()
Hp = H_input.problem() * Escale


Epp_all = np.zeros((2, 2, ncats))
catalyst_strengths_all = np.zeros((2, ncats))
for k in tqdm(range(2)):
    for i in tqdm(range(ncats)):
        if k == 0:
            catalyst_strengths_all[k] = catalyst_strengths
            Hc = H_catalyst_LZ(N, float(catalyst_strengths[i]))
            H_LZ = ham(Hd, Hp, anneal_time, grain, Hc)
            energies = energy_levels(H_LZ)

        if k == 1:
            from ninespin_preprocessing import s_9spin, energies_9spin

            errors = [-0.075, -0.05, -0.025, 0, 0.025]  # 0.025, 0.05, 0.075]
            catalyst_strengths = [1.4922 * (1 + error) for error in errors]
            catalyst_strengths_all[k] = catalyst_strengths
            energies = energies_9spin[i]
            s = s_9spin

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
        Epp_all[k, :, i] = E_params

fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))
for i in range(2):
    for j in range(2):
        ax[i].plot(catalyst_strengths, Epp_all[i][j], "x", lw=2)
plt.show()
