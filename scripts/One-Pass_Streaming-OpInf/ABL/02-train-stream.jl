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
)
rmax = 400

#====================================#
## Compute One-Pass Streaming-OpInf ##
#====================================#
options.with_reg = true
options.λ = LnL.TikhonovParameter(A=1e12, K=1e12, A2=1e12)
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
E, Δidx = LnL.finite_diff_matrix(
    options.data.deriv_type, n_train, options.data.Δt
)

# r = 400
op_stream_r400 = LnL.compute_stream_operators(
    stream, E, (Δidx[1], Δidx[end])
)

# r = 350
op_stream_r350 = LnL.compute_stream_operators(
    stream, E, (Δidx[1], Δidx[end]); rank=350
)

# r = 300
op_stream_r300 = LnL.compute_stream_operators(
    stream, E, (Δidx[1], Δidx[end]); rank=300
)

# r = 250
op_stream_r250 = LnL.compute_stream_operators(
    stream, E, (Δidx[1], Δidx[end]); rank=250
)

# r = 200
op_stream_r200 = LnL.compute_stream_operators(
    stream, E, (Δidx[1], Δidx[end]); rank=200
)

# Save the stream object and operators
save(joinpath(FILEPATH, "data/results/onepass_stream.jld2"), "stream", stream)
save(joinpath(FILEPATH, "data/results/op_stream_r400.jld2"), "op_stream_r400", op_stream_r400)
save(joinpath(FILEPATH, "data/results/op_stream_r350.jld2"), "op_stream_r350", op_stream_r350)
save(joinpath(FILEPATH, "data/results/op_stream_r300.jld2"), "op_stream_r300", op_stream_r300)
save(joinpath(FILEPATH, "data/results/op_stream_r250.jld2"), "op_stream_r250", op_stream_r250)
save(joinpath(FILEPATH, "data/results/op_stream_r200.jld2"), "op_stream_r200", op_stream_r200)