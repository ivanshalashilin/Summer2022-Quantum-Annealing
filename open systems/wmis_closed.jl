using OpenQuantumTools, OrdinaryDiffEq, Plots, JLD2


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

annealing = Annealing(H, u0)

coupling = ConstantCouplings(["ZZZ"], unit=:ħ)
# bath
#bath = Ohmic(1e-4, 8, 16)

tf = 800
sol = solve_schrodinger(annealing, tf; alg=Tsit5(), abstol=1e-6, reltol=1e-6)


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

#save data
mdat = [na_closed, nb, W, dW, Jzz, Jxx, tf]
data = (sol, H, mdat)
id = randstring(12)
spin = string(na+nb)
filename = "$(spin)spin_closed_$id.jld2"
save_object(filename, data)


plot(sol, H, [:0, :1], 0:0.01:tf, linewidth=2, xlabel="t (ns)", ylabel="\$P_G(t)\$")




