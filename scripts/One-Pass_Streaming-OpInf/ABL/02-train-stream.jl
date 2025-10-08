"""
ABL: Compute basis
"""

#=================#
## Load Packages ##
#=================#
using FileIO
using JLD2
using IncrementalSVD
using LinearAlgebra
using BlockDiagonals
using ProgressMeter
using SparseArrays
import LiftAndLearn as LnL

#=================================#
## Configure filepath for saving ##
#=================================#
DATAPATH = "../../../../../DATA/NREL/ABL"
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"One-Pass_Streaming-OpInf/ABL") : 
           joinpath(pwd(), "scripts/One-Pass_Streaming-OpInf/ABL")
fn = "ABL_0_10000.h5"
datafile = joinpath(DATAPATH, fn)

#===========================================#
## Load struct to read data in HDF5 format ##
#===========================================#
include(joinpath(FILEPATH, "datasource.jl"))

#========================#
## Additional functions ##
#========================#
include(joinpath(FILEPATH, "preprocess.jl"))

#=============================#
## Load the training dataset ##
#=============================#
ds = ChannelDataSource(datafile, ["z", "y", "x", "fields", "times"])
Nz, Ny, Nx, n_fields, n = ds.dims
nxyz = Nz * Ny * Nx
n_test = 2000
n_train = n - n_test

#==============================#
## Load the mean and scalings ##
#==============================#
means  = load(joinpath(FILEPATH, "data/mean.jld2"))["xbar"]
shifts = load(joinpath(FILEPATH, "data/minmax.jld2"))["minmax"]["shifts"]
scales = load(joinpath(FILEPATH, "data/minmax.jld2"))["minmax"]["scales"]

#=======================================#
## Some options for operator inference ##
#=======================================#
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=[1,2],
        constant=1,
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=sum(diff(ds["times"][1:n_train])) / (length(ds["times"][1:n_train])-1),
        deriv_type="FBCT4"
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
    use_backslash=false,
    use_svd_truncation=true,
)
rmax = 200

#====================================#
## Compute One-Pass Streaming-OpInf ##
#====================================#
LOAD_STREAM = true

if LOAD_STREAM
    stream = load(joinpath(FILEPATH, "data/results/onepass_stream.jld2"), "stream")
else
    stream = LnL.OnePassStreamingOpInf(
        preprocess!(ds[1], means, shifts, scales); 
        options=options, 
        n=Int(nxyz * n_fields), 
        rank=rmax, 
        finite_diff=true
    )
    @showprogress for i in 2:n_train
        LnL.stream!(stream, preprocess!(ds[i], means, shifts, scales), tol=1e-10)
    end
    save(joinpath(FILEPATH, "data/results/onepass_stream.jld2"), "stream", stream)
end

## Construct the reduced data matrices
E, Δidx = LnL.finite_diff_matrix(
    options.data.deriv_type, n_train, options.data.Δt
)
r = rmax
Xhat = Diagonal(stream.Σ[1:r]) * stream.W[:, 1:r]'
Xhatdot = Xhat * E
Xhat = Xhat[:, Δidx]

## Solve the OpInf problem 
options.with_reg = true
options.λ = LnL.TikhonovParameter(A=1e13, K=1e13, A2=1e13)
op_stream = LnL.opinf(Xhat, options; Xhatdot=Xhatdot)

## Save operators
save(joinpath(FILEPATH, "data/results/op_stream_r$(r).jld2"), "op_stream", op_stream)

##
include(joinpath(FILEPATH, "integrate.jl"))
include(joinpath(FILEPATH, "preprocess.jl"))
tspan = ds["times"][1:n_train] .- ds["times"][1]
x0 = stream.V[:,1:r]' * preprocess!(ds[1], means, shifts, scales)
states, _ = rk4_integrate(x0, tspan, op_stream.A, op_stream.A2u, op_stream.K)