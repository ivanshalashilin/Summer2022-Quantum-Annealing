using OpenQuantumTools, OrdinaryDiffEq, Plots

# initial state
u0 = PauliVec[1][2]#⊗PauliVec[1][1]⊗PauliVec[1][1]

#this is defined as an actual function

H = DenseHamiltonian([(s)->1-s, (s)->s], [-σx, σz], unit=:ħ)
annealing = Annealing(H, u0)#, coupling=coupling, bath=bath)
tf = 200
sol = solve_schrodinger(annealing, tf; alg=Tsit5(),  abstol=1e-6, reltol=1e-6)#, ω_hint=range(-6, 6, length=100), reltol=1e-4)


plot(sol, H, [:0, :1], 0:0.01:tf, linewidth=2, xlabel="Anneal Time", ylabel="\$P_G(t)\$")


