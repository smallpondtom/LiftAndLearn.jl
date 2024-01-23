## Packages
using LinearAlgebra
using ProgressMeter
using Random
import HSL_jll

# My modules
using LiftAndLearn
const LnL = LiftAndLearn
const LFI = LnL.LyapInf

## First order Burger's equation setup
burger = LnL.burgers(
    [0.0, 1.0], [0.0, 1.0], [0.10, 0.10],
    2^(-7), 1e-4, 1, "periodic"
)

## Setup
rmin = 1
rmax = 10

burger_system = LnL.sys_struct(
    is_lin=true,
    is_quad=true,
)
burger_vars = LnL.vars(
    N=1,
)
burger_data = LnL.data(
    Δt=1e-4,
    DS=500,
)
burger_optim = LnL.opt_settings(
    verbose=true,
    initial_guess=true,
    max_iter=1000,
)

options = LnL.LS_options(
    system=burger_system,
    vars=burger_vars,
    data=burger_data,
    optim=burger_optim,
)

# Downsampling rate
DS = options.data.DS
Tdim_ds = size(1:DS:burger.Tdim, 1)  # downsampled time dimension

## Initial condition
IC = Dict(
    :a => [2, 4, 6],
    :b => [-0.5, 0, 0.5],
    :c => [-0.25, 0, 0.25],
    :test => ([0.1, 8], [-0.8, 0.8], [-0.5, 0.5]),
)
IC[:num] = Int(length(IC[:a])*length(IC[:b])*length(IC[:c]))
IC[:func] = (a,b,c) -> exp.(-a * cos.(π .* burger.x .+ b).^2) .- c

## Generate training data
@info "Generate the FOM system matrices and training data."
μ = burger.μs
    
# Generate the FOM system matrices 
A, F = burger.generateEPmatrix(burger, μ)
op_fom = LnL.operators(A=A, F=F)
states_fom = burger.semiImplicitEuler(A, F, burger.t, burger.IC)
    
# training data for OpInf
Xall = Vector{Matrix{Float64}}(undef, IC[:num])
Xdotall = Vector{Matrix{Float64}}(undef, IC[:num])

ct = 1
for a in IC[:a], b in IC[:b], c in IC[:c]
    states = burger.semiImplicitEuler(A, F, burger.t, IC[:func](a, b, c))
    tmp = states[:, 2:end]
    Xall[ct] = tmp[:, 1:DS:end]  # downsample data
    tmp = (states[:, 2:end] - states[:, 1:end-1]) / burger.Δt
    Xdotall[ct] = tmp[:, 1:DS:end]  # downsample data
    
    @info "(Loop $ct of $(IC[:num])) Generated training data for a = $a, b = $b, c = $c"
    ct += 1
end
X = reduce(hcat, Xall)
R = reduce(hcat, Xdotall)

# Compute the POD basis from the training data
tmp = svd(X)
Vrmax = tmp.U[:, 1:rmax]

## Compute the intrusive model
@info "Compute the intrusive model"
op_int = LnL.intrusiveMR(op_fom, Vrmax, options)

## Compute the OpInf model
@info "Compute the standard OpInf solution."
op_std = LnL.inferOp(X, zeros(Tdim_ds,1), zeros(Tdim_ds,1), Vrmax, Vrmax' * R, options)

## Define some storage variables
rsize = rmax - rmin + 1
P_res = Dict(
    "int"=> Vector{Matrix{Float64}}(undef, rsize),
    "std"=> Vector{Matrix{Float64}}(undef, rsize)
)
Q_res = Dict(
    "int"=> Vector{Matrix{Float64}}(undef, rsize),
    "std"=> Vector{Matrix{Float64}}(undef, rsize)
)
Jzubov_res = Dict(
    "int"=> Vector{Float64}(undef, rsize),
    "std"=> Vector{Float64}(undef, rsize)
)
∇Jzubov_res = Dict(
    "int"=> Vector{Float64}(undef, rsize),
    "std"=> Vector{Float64}(undef, rsize)
)
ρ_max_res = Dict(
    "int"=> Vector{Float64}(undef, rsize),
    "std"=> Vector{Float64}(undef, rsize)
)
ρ_min_res = Dict(
    "int"=> Vector{Float64}(undef, rsize),
    "std"=> Vector{Float64}(undef, rsize)
)
ρ_est_res = Dict(
    "int"=> Vector{Float64}(undef, rsize),
    "std"=> Vector{Float64}(undef, rsize)
)

## Options for LyapInf
lyapinf_options = LFI.Int_LyapInf_options(
    extra_iter=3,
    optimizer="Ipopt",
    ipopt_linear_solver="ma86",
    verbose=true,
    optimize_PandQ="P",
    HSL_lib_path=HSL_jll.libhsl_path,
)


## Infer the Lyapunov function for the intrusive model
@info "Infer the Lyapunov function for the intrusive model"
# for r in rmin:rmax
r = 10
i = r - rmin + 1  # index

A = op_int.A[1:r,1:r]
F = LnL.extractF(op_int.F, r)
H = LnL.extractH(op_int.H, r)
op_tmp = LnL.operators(A=A, F=F, H=H)

## Solve the non-intrusive Lyapunov Function Inference Problem
Qinit = 1.0I(r)
foo = round.(rand(r,r), digits=4)
Pinit = foo * foo'

P, Q, Jzubov_res["int"][i], ∇Jzubov_res["int"][i] = LFI.Int_LyapInf(op_tmp, Vrmax[:,1:r]' * X, lyapinf_options; Qi=Qinit)
P_res["int"][i] = P
Q_res["int"][i] = Q

## Compute the DoA 
ρ_min_res["int"][i], ρ_max_res["int"][i] = LFI.DoA(P)

## Compute the estimated domain of attraction from Boris's method
ρ_est_res["int"][i] = LFI.est_stability_rad(A, H, P)
##
@info "Done with r = $r"
# end