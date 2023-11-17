# import
import pickle
import matplotlib.pyplot as plt

# decay fidelity_9spin
pickle_in = open("wmis/9spin_data/9spin_fidelities.pickle", "rb")
fidelity_9spin = pickle.load(pickle_in)


T_9spin = fidelity_9spin["T"]
cats = fidelity_9spin["Jxx"]
data_fidelty = fidelity_9spin["final_overlaps"]
# fig, ax = plt.subplots()
# for i in range(len(data_fidelty)):
#     ax.plot(T_9spin, data_fidelty[i], label = fidelity_9spin["Jxx"][i])
# ax.legend()
# plt.show()


# pickle_in = open("wmis/9spin_data/9spin_gap-mins.pickle", "rb")
# gapmins_9spin = pickle.load(pickle_in)

# print()

error = ["-0p1", "-0p05", "-0p03", "0", "0p01", "0p05", "0p1"]
error = ["-0p075", "-0p05", "-0p025", "0", "0p025", "0p05", "0p075"]


catalyst_strengths_rounded = [round(c, 2) for c in cats]
error_9spin = [c - cats[4] for c in cats]
energies_9spin = []
# gap spectra
for err in error:
    pickle_in = open(f"wmis/9spin_data/spectra/9spin_Jxxerror_{err}.pickle", "rb")
    data_spectra = pickle.load(pickle_in)
    energies_9spin.append([data_spectra["energies"][0], data_spectra["energies"][1]])
    # fig, ax = plt.subplots()
    # ax.plot(data_spectra["energies"][1]-data_spectra["energies"][0])
    # ax.plot(data_spectra["energies"][0]-data_spectra["energies"][0])
    #ax.set(xlim =)
    #plt.show()
    print()
s_9spin = data_spectra["times"]
