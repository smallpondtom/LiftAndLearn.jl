"""
MHD256: Compute basis
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
DATAPATH = "../../../../../scratch1/1/tkoike3"
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"One-Pass_Streaming-OpInf/mhd256") : 
           joinpath(pwd(), "scripts/One-Pass_Streaming-OpInf/mhd256")
fn = "MHD_Ma_0.7_Ms_0.5.hdf5"
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
ds = DataSource(fn)
nx, ny, nz, n_fields, n_time, n_traj = ds.dims
n = n_time * n_traj
nxyz = nx * ny * nz

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
        state=[1,2,3],
        constant=1,
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=sum(diff(ds.grid["time"])) / (length(ds.grid["time"])-1),
        deriv_type="FBCT4"
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)
rmax = 100

#====================================#
## Compute One-Pass Streaming-OpInf ##
#====================================#
options.with_reg = true
options.λ = LnL.TikhonovParameter(A=1e14, K=1e14, A2=1e14)
stream = LnL.OnePassStreamingOpInf(
    preprocess!(ds[1], means, shifts, scales); 
    options=options, 
    n=Int(nxyz * n_fields), 
    rank=rmax, 
    finite_diff=true
)
@showprogress for i in 2:n
    LnL.stream!(stream, preprocess!(ds[i], means, shifts, scales), tol=1e-8)
end

GC.gc()  # clean up memory

E, Δidx = LnL.finite_diff_matrix(
    options.data.deriv_type, n_time, options.data.Δt
)
E = BlockDiagonal([E, E, E, E, E])

# r = 100
op_stream_r100 = LnL.compute_stream_operators(
    stream, E, (Δidx[1], Δidx[end])
)

# r = 75
op_stream_r75 = LnL.compute_stream_operators(
    stream, E, (Δidx[1], Δidx[end]); rank=350
)

# r = 50
op_stream_r50 = LnL.compute_stream_operators(
    stream, E, (Δidx[1], Δidx[end]); rank=300
)

# Save the stream object and operators
save(joinpath(FILEPATH, "data/results/onepass_stream.jld2"), "stream", stream)
save(joinpath(FILEPATH, "data/results/op_stream_r100.jld2"), "op_stream_r100", op_stream_r100)
save(joinpath(FILEPATH, "data/results/op_stream_r75.jld2"), "op_stream_r75", op_stream_r75)
save(joinpath(FILEPATH, "data/results/op_stream_r50.jld2"), "op_stream_r50", op_stream_r50)
