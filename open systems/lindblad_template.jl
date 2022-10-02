using OpenQuantumTools, OrdinaryDiffEq, Plots



H_p = (4.8*σz⊗σi⊗σi + 9.6*σi⊗σz⊗σi + 4.82*σi⊗σi⊗σz + 6.4*σz⊗σz⊗σi + 6.4*σi⊗σz⊗σz)


H_d = -standard_driver(3) 


H_c = σx⊗σx⊗σi

# H_p = local_field_term([4,82, 9.6], [1, 2], 2)
#  + two_local_term([6.4, 6.4], [[1,2]], 2)

# H_d = -standard_driver(2) 

# initial state
u0 = PauliVec[1][1]⊗PauliVec[1][1]⊗PauliVec[1][1]
# coupling
coupling = ConstantCouplings(["ZZZ"], unit=:ħ)
# bath
bath = Ohmic(1e-4, 8, 16)

H = DenseHamiltonian([(s)->1-s, (s)->s, (s)->s*(1-s)], [H_d, H_p, H_c], unit =:ħ)
#H = DenseHamiltonian([(s)->1-s, (s)->s], -[σx, σz]/2)

annealing = Annealing(H, u0, coupling=coupling, bath=bath)

tf = 1
sol = solve_ame(annealing, tf; alg=Tsit5(), ω_hint=range(-6, 6, length=100), reltol=1e-4)
plot(sol, H, [:0,:1,:4], 0:0.01:tf, linewidth=2, xlabel="t (ns)", ylabel="\$P_G(t)\$")