using FileIO
using JLD2
using LinearAlgebra
using ProgressMeter

include("../src/model/KS.jl")
include("../src/LiftAndLearn.jl")
const LnL = LiftAndLearn

# Settings for the KS equation
KSE = KS(
    [0.0, 100.0], [0.0, 200.0], [1.0, 1.0],
    512, 0.001, 1, "ep"
)


# Create file name to save data
datafile = "examples/data/kse_data.jld2"

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
    verbose=false,
    initial_guess=false,
    max_iter=1000,
    reproject=false,
    SIGE=false,
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

# Parameters of the initial condition
ic_a = [0.5, 1.0, 1.5]
ic_b = [0.1, 0.2, 0.3]

ic_a_out = [-1.0, 3.0]
ic_b_out = [-1.0, 1.0]
num_ic_params = Int(length(ic_a) * length(ic_b))

# Parameterized function for the initial condition
u0 = (a,b) -> a * cos.((2*π*KSE.x)/KSE.Xdim) .+ b * cos.((4*π*KSE.x)/KSE.Xdim)  # initial condition

DATA = load(datafile)
Xtr = DATA["Xtr"]
Rtr = DATA["Rtr"]
Vr = DATA["Vr"]

@info "Compute the EPHEC model"

options = LnL.EPHEC_options(
    system=KSE_system,
    vars=KSE_vars,
    data=KSE_data,
    optim=KSE_optim,
)
op_ephec =  Array{LnL.operators}(undef, KSE.Pdim)

for i in eachindex(KSE.μs)
    op_ephec[i] = LnL.inferOp(Xtr[i], zeros(Tdim_ds,1), zeros(Tdim_ds,1), Vr[i], Vr[i]' * Rtr[i], options)
    @info "Loop $(i) out of $(KSE.Pdim) completed..."
end