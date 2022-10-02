# using OpenQuantumTools, OrdinaryDiffEq, Plots
# # define the Hamiltonian
# H = DenseHamiltonian([(s)->1.0], [σz], unit=:ħ)
# # define the initial state
# u0 = PauliVec[1][1]*PauliVec[1][1]'
# # define the Lindblad operator
# # the rate and Lindblad operator can also be time-dependent functions
# lind = Lindblad(0.1, σz)
# # combine them into an Annealing object
# annealing = Annealing(H, u0, interactions = InteractionSet(lind))

# σz


using OpenQuantumTools, OrdinaryDiffEq, Plots, Random



#WMIS parameters
na = 2
nb = 3
W = 1
dW = 0.01
Jzz = 5.77
Jxx = 1.87
Escale = 15
# - mapping from network to ising hamiltonian
k = (na+nb)/(4*(na*nb*Jzz-W))
Jzz = k*Jzz
# Jzz in the local field penalises coupled adjacent spins
hzs = [(nb*Jzz - 2*k*(W+dW)/na) , (na*Jzz - 2*k*W/nb)]


H_p = Escale * ((local_field_term([hzs[1], hzs[1], hzs[2], hzs[2], hzs[2]], [1,2,3,4,5], 5)+(two_local_term([Jzz, Jzz, Jzz, Jzz, Jzz, Jzz],[[1,3],[1,4],[1,5],[2,3],[2,4],[2,5]],5))))

H_d = -standard_driver(5)

H_c = Jxx * σi⊗σi⊗σi⊗σx⊗σx

# initial state
u0 = PauliVec[1][1]⊗PauliVec[1][1]⊗PauliVec[1][1]⊗PauliVec[1][1]⊗PauliVec[1][1]
H = DenseHamiltonian([(s)->1-s, (s)->s, (s)-> s * (1-s)], [H_d, H_p, H_c], unit = :ħ)

coupling = ConstantCouplings(["ZZZZZ"])#, unit=:ħ)
# bath
bath = Ohmic(1e-4, 8, 16)
annealing = Annealing(H, u0,  coupling=coupling, bath=bath)

tf = 200
sol = solve_ame(annealing, tf; alg=Tsit5(), ω_hint=range(-6, 6, length=100), reltol=1e-4)



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

plot(sol, H, [:0, :1], 0:0.01:tf, linewidth=2, xlabel="t (ns)", ylabel="\$P(t)\$")