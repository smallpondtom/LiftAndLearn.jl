using FileIO
using JLD2
using LinearAlgebra
using ProgressMeter
using Random

include("../src/model/KS.jl")
include("../src/LiftAndLearn.jl")
const LnL = LiftAndLearn

# Settings for the KS equation
KSE = KS(
    [0.0, 100.0], [0.0, 200.0], [1.0, 1.0],
    512, 0.001, 1, "ep"
)

# WARNING:DO YOU WANT TO SAVE DATA?
save_data = true

# Create file name to save data
datafile = "examples/data/kse_data.jld2"
opfile = "examples/data/kse_operators.jld2"
resultfile = "examples/data/kse_results.jld2"

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
)

# Downsampling rate
DS = KSE_data.DS

# Down-sampled dimension of the time data
Tdim_ds = size(1:DS:KSE.Tdim, 1)  # downsampled time dimension

# Number of random test inputs
num_test_ic = 50


DATA = load(datafile)


Xtr = DATA["Xtr"]
Rtr = DATA["Rtr"]
Vr = DATA["Vr"]
ro = DATA["ro"]


DATA = nothing

GC.gc()

@info "Compute the EPHEC model"

options = LnL.EPHEC_options(
    system=KSE_system,
    vars=KSE_vars,
    data=KSE_data,
    optim=KSE_optim,
    A_bnds=(-1000.0, 1000.0),
    ForH_bnds=(-100.0, 100.0),
)
op_ephec =  Array{LnL.operators}(undef, KSE.Pdim)

@views function makechunks(X::AbstractVector{Int64}, n::Integer)
    c = length(X) ÷ n
    return [X[1+c*k:(k == n-1 ? end : c*k+c)] for k = 0:n-1]
end

@views function randomchunks(N::Integer,k::Integer)
    n,r = divrem(N,k)
    b = collect(1:n:N+1)
    for i in eachindex(b)
        b[i] += i > r ? r : i-1  
    end
    p = randperm(N)
    return [p[r] for r in [b[i]:b[i+1]-1 for i=1:k]]
end

data_size = size(Xtr[1], 2)
num_of_batches = 100
ordered_batch = makechunks(collect(1:data_size), num_of_batches)
rand_batch = randomchunks(data_size, num_of_batches)


for i in eachindex(KSE.μs)
    op_ephec[i] = LnL.inferOp(Xtr[i][:, ordered_batch[1]], zeros(Tdim_ds,1), zeros(Tdim_ds,1),
        Vr[i][:, 1:ro[end]], Vr[i][:, 1:ro[end]]' * Rtr[i][:, ordered_batch[1]], options)
    @info "Loop $(i) out of $(KSE.Pdim) completed..."
end