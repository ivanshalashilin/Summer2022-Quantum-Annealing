using OpenQuantumTools, OrdinaryDiffEq, Plots, JLD2

#WMIS parameters
na = 1
nb = 2
W = 1
dW = 0.01
Jzz = 5.77
Jxx = 0
Escale = 15
# - mapping from network to ising hamiltonian
k = (na+nb)/(4*(na*nb*Jzz-W))
Jzz = k*Jzz
# Jzz in the local field penalises coupled adjacent spins
hzs = [(nb*Jzz - 2*k*(W+dW)/na) , (na*Jzz - 2*k*W/nb)]


H_p = Escale * (4.8*σz⊗σi⊗σi + 9.6*σi⊗σz⊗σi + 4.82*σi⊗σi⊗σz + 6.4*σz⊗σz⊗σi + 6.4*σi⊗σz⊗σz)


H_d = -standard_driver(3) 

#H_c = Jxx * σi⊗σi⊗σi⊗σx⊗σx

# initial state
u0 = PauliVec[1][1]⊗PauliVec[1][1]⊗PauliVec[1][1]
H = DenseHamiltonian([(s)->1-s, (s)->s], [H_d, H_p], unit = :ħ)

couplings = "ZZZ"
coupling = ConstantCouplings([couplings])#, unit=:ħ)
# bath
η = 1e-4
ωc = 8
T = 16

bath = Ohmic(η, ωc, T)
annealing = Annealing(H, u0,  coupling=coupling, bath=bath)

tf = 200
sol = solve_ame(annealing, tf; alg=Tsit5(), ω_hint=range(-6, 6, length=100), reltol=1e-4)

plot(sol, H, [:0, :1], 0:0.01:tf, linewidth=2, xlabel="t (ns)", ylabel="\$P(t)\$")

# s_list = 0:0.005:1
# y = []
# for s in s_list
#     #, _ makes w a hidden variable( ??)
#     w, _ = eigen_decomp(H, s; lvl=2)
#     w = w - [w[1], w[1]]
#     push!(y, w)
# end
# y = hcat(y...)'
# plot(s_list, y)#, ylims = (-0.001, 0.02), xlims = (0.75, 0.99))

mdat = [na, nb, tf, Jzz]
data = (sol, H, metadata)
filename = "open "
save_object("3s_open.jld2", data)
