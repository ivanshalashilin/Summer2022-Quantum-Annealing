using OpenQuantumTools, OrdinaryDiffEq, Plots, JLD2


# initial state
u0 = PauliVec[1][1]
H = DenseHamiltonian([(s)->1-s, (s)->s], [-σx, σz], unit = :ħ)

couplings = "Z"
# X and Z - system shares eigenstates with ohmic noise
coupling = ConstantCouplings([couplings])#, unit=:ħ)
# bath
η = 1e-6
ωc = 10
T = 30

bath = Ohmic(η, ωc, T)
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

mdat = [couplings, η, ωc, T, tf]
data = (sol, H, mdat)
id = randstring(12)
spin = string(1)
filename = "$(spin)spin_closed_$id.jld2"
#save_object(filename, data)

plot(sol, H, [:0, :1], 0:0.01:tf, linewidth=2, xlabel="t (ns)", ylabel="\$P(t)\$")
