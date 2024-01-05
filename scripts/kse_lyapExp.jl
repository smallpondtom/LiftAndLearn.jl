## Setup
using FileIO
using JLD2
using LaTeXStrings
using LinearAlgebra
using Plots

using LiftAndLearn
const LnL = LiftAndLearn
const CG = LiftAndLearn.ChaosGizmo

# Settings for the KS equation
KSE = LnL.ks(
    [0.0, 22.0], [0.0, 300.0], [1.0, 1.0],
    512, 0.001, 1, "ep"
)

# Create file name to save data
datafile = "../examples/data/kse_data_L22.jld2"
opfile = "../examples/data/kse_operators_L22.jld2"
resultfile = "../examples/data/kse_results_L22.jld2"

# Settings for Operator Inference
KSE_system = LnL.sys_struct(
    is_lin=true,
    is_quad=true,
)
KSE_vars = LnL.vars(
    N=1,
)
KSE_data = LnL.data(
    Δt=KSE.Δt,
    DS=100,
)
KSE_optim = LnL.opt_settings(
    verbose=true,
    initial_guess=false,
    max_iter=1000,
    reproject=false,
    SIGE=false,
    with_bnds=true,
    HSL_lib_path=HSL_jll.libhsl_path,
    linear_solver="ma86",
)

options = LnL.LS_options(
    system=KSE_system,
    vars=KSE_vars,
    data=KSE_data,
    optim=KSE_optim,
)

# Downsampling rate
DS = KSE_data.DS

# Down-sampled dimension of the time data
Tdim_ds = size(1:DS:KSE.Tdim, 1)  # downsampled time dimension

# Number of random test inputs
num_test_ic = 50

# Prune data to get only the chaotic region
prune_data = false
prune_idx = KSE.Tdim ÷ 2
t_prune = KSE.t[prune_idx-1:end]

# Parameters of the initial condition
ic_a = [0.8, 1.0, 1.2]
ic_b = [0.2, 0.4, 0.6]

num_ic_params = Int(length(ic_a) * length(ic_b))
L = KSE.Omega[2] - KSE.Omega[1]  # length of the domain

# Parameterized function for the initial condition
u0 = (a,b) -> a * cos.((2*π*KSE.x)/L) .+ b * cos.((4*π*KSE.x)/L)  # initial condition

## Load data
DATA = load(datafile);
OPS = load(opfile);

# Function to compute the Lyapunov exponent and Kaplan-Yorke dimension for one initial condition
function analyze_chaos_oneIC(model, op, IC, Vr, ro, integrator, jacobian, options)
    # Lypuanov Exponents
    LE = Array{Array{Float64}}(undef, length(ro), model.Pdim)
    LE_all = Array{Array{Float64}}(undef, length(ro), model.Pdim)
    Dky = Array{Float64}(undef, length(ro), )

    for i in eachindex(model.μs)
        for (j,r) in enumerate(ro)
            if options.history
                LE[j,i], LE_all[j,i]  = lyapunovExponentJacobian(op[i], integrator, IC, Vr[i][:,1:r], jacobian, options)
            else
                LE[j,i] = lyapunovExponentJacobian(op[i], integrator, IC, Vr[i][:,1:r], jacobian, options)
            end
            @info "Reduced order of $(r) completed..."
        end
        @info "Loop $(i) out of $(model.Pdim) completed..."
    end
    if options.history
        return LE, LE_all
    else
        return LE
    end
end



