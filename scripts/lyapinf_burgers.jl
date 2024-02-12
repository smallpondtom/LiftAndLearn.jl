## Packages
using CairoMakie
using LinearAlgebra
import HSL_jll

## My modules
using LiftAndLearn
const LnL = LiftAndLearn
const LFI = LyapInf

## First order Burger's equation setup
burgers = LnL.burgers(
    [0.0, 1.0], [0.0, 1.0], [0.10, 0.10],
    2^(-7), 1e-4, 1, "periodic"
)

## Settings
rmin = 1  # min dimension
rmax = 10  # max dimension

opinf_options = LnL.LS_options(
    system=LnL.sys_struct(
        is_lin=true,
        is_quad=true,
    ),
    vars=LnL.vars(
        N=1,
    ),
    data=LnL.data(
        Δt=burgers.Δt,
        DS=100,
    ),
    optim=LnL.opt_settings(
        verbose=true,
        initial_guess=true,
        max_iter=1000,
        reproject=false,
        SIGE=false,
        linear_solver="ma86",
    ),
)

# Downsampling rate
DS = opinf_options.data.DS
Tdim_ds = size(1:DS:burgers.Tdim, 1)  # downsampled time dimension

## Initial condition
IC = Dict(
    :a => [3, 5, 7],
    :b => [-0.5, 0, 0.5],
    :c => [-0.25, 0, 0.25],
    :test => ([0.1, 10.0], [-1.0, 1.0], [-0.5, 0.5]),
)
IC[:n] = Int(length(IC[:a])*length(IC[:b])*length(IC[:c]))
IC[:f] = (a,b,c) -> exp.(-a * cos.(π .* burgers.x .+ b).^2) .- c

## Generate the training data
A, F = burgers.generateEPmatrix(burgers, burgers.μs[1])
op_fom = LnL.operators(A=A, F=F)
    
Xall = Vector{Matrix{Float64}}(undef, IC[:n])
Xdotall = Vector{Matrix{Float64}}(undef, IC[:n])

ct = 1
for a in IC[:a], b in IC[:b], c in IC[:c]
    states = burgers.semiImplicitEuler(A, F, burgers.t, IC[:f](a, b, c))
    tmp = states[:, 2:end]
    Xall[ct] = tmp[:, 1:DS:end]  # downsample data
    tmp = (states[:, 2:end] - states[:, 1:end-1]) / burgers.Δt
    Xdotall[ct] = tmp[:, 1:DS:end]  # downsample data
    @info "(Loop #$ct) Generated training data for a = $a, b = $b, c = $c"
    ct += 1
end
X = reduce(hcat, Xall)
R = reduce(hcat, Xdotall)

# Compute the POD basis from the training data
Vrmax = svd(X).U[:, 1:rmax]

## Compute the intrusive POD model
op_int = LnL.intrusiveMR(op_fom, Vrmax, opinf_options)

## Compute the OpInf model
op_inf = LnL.inferOp(X, zeros(Tdim_ds,1), zeros(Tdim_ds,1), Vrmax, Vrmax' * R, opinf_options)

## Create a batch of training data
# batchsize = 300
# X_batch = map(Iterators.partition(axes(X,2), batchsize)) do cols
#     X[:, cols]
# end

## Intrusive LyapInf for POD model
ds2= 5  # another downsampling for lyapinf
int_lyapinf_options = LFI.Int_LyapInf_options(
    extra_iter=3,
    optimizer="Ipopt",
    ipopt_linear_solver="ma86",
    verbose=true,
    optimize_PandQ="P",
    HSL_lib_path=HSL_jll.libhsl_path,
)
P_int, Q, cost, ∇cost = LFI.Int_LyapInf(op_int, Vrmax' * X[:,1:ds2:end], int_lyapinf_options)

## Intrusive LyapInf for OpInf model
P_inf, Q, cost, ∇cost = LFI.Int_LyapInf(op_inf, Vrmax' * X[:,1:ds2:end], int_lyapinf_options)

## Non-intrusive LyapInf
nonint_lyapinf_options = LFI.NonInt_LyapInf_options(
    extra_iter=3,
    optimizer="Ipopt",
    ipopt_linear_solver="ma86",
    verbose=true,
    optimize_PandQ="P",
    HSL_lib_path=HSL_jll.libhsl_path,
)
P_star, Q, cost, ∇cost = LFI.NonInt_LyapInf(Vrmax' * X[:,1:ds2:end], Vrmax' * R[:,1:ds2:end], nonint_lyapinf_options)

##
# function skp_stability_rad(Ahat::AbstractArray{T}, Hhat::AbstractArray{T}, Q::AbstractArray{T}; 
#         div_by_2::Bool=false) where T

#     if div_by_2
#         P = lyapc(Ahat', 0.5*Q)
#     else
#         P = lyapc(Ahat', Q)
#     end
#     L = cholesky(Q).L
#     σmin = minimum(svd(L).S)
#     ρhat = σmin / sqrt(norm(P,2)) / norm(Hhat,2) / 2
#     return ρhat
# end


## Sample the correct level surface
## POD
V = (x) -> x' * P_int * x
Vdot = (x) -> x' * P_int * op_int.A * x + x' * P_int * op_int.F * (x ⊘ x)
c_star1, c_all1, x_sample1 = LFI.doa_sampling(
    V,
    Vdot,
    1000000, rmax, Tuple(burgers.Omega);
    method="memory", history=true, uniform_state_space=true
)
ρmin1 = sqrt(c_star1/maximum(eigvals(P_int)))
ρskp1 = LFI.skp_stability_rad(op_int.A, op_int.H, Q)

##
fig1 = Figure(fontsize=20)
ax1 = Axis(fig1[1,1],
    title="Level Convergence",
    ylabel=L"c_*",
    xlabel="Sample Number",
    xticks=0:250000:length(c_all1),
)
lines!(ax1, 1:length(c_all1), c_all1)
fig1

## OpInf
V = (x) -> x' * P_inf * x
Vdot = (x) -> x' * P_inf * op_inf.A * x + x' * P_inf * op_inf.F * (x ⊘ x)
c_star2, c_all2, x_sample2 = LFI.doa_sampling(
    V,
    Vdot,
    1000000, rmax, Tuple(burgers.Omega);
    method="memory", history=true, uniform_state_space=true
)
ρmin2 = sqrt(c_star2/maximum(eigvals(P_inf)))
ρskp2 = LFI.skp_stability_rad(op_inf.A, op_inf.H, Q)

## Non-intrusive
V = (x) -> x' * P_star * x
Vdot = (x) -> x' * P_star * op_int.A * x + x' * P_star * op_int.F * (x ⊘ x)
c_star3, c_all3, x_sample3 = LFI.doa_sampling(
    V,
    Vdot,
    1000000, rmax, Tuple(burgers.Omega);
    method="memory", history=true, uniform_state_space=true
)
ρmin3 = sqrt(c_star3/maximum(eigvals(P_star)))
ρskp3 = LFI.skp_stability_rad(op_int.A, op_int.H, Q)