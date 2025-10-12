"""
3D Channel flow: Compute basis
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
DATAPATH = "../../../../../DATA/NREL/3D_CHANNEL"
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"One-Pass_Streaming-OpInf/3d_channel") : 
           joinpath(pwd(), "scripts/One-Pass_Streaming-OpInf/3d_channel")
fn = "channel_5200_data_0_10000.h5"
datafile = joinpath(DATAPATH, fn)
FILEPATH2 = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"Two-Pass_Streaming-OpInf/3d_channel") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/3d_channel")


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
    use_backslash=true,
)
rmax = 500

#====================================#
## Compute One-Pass Streaming-OpInf ##
#====================================#
LOAD_STREAM = false
GRID_SEARCH = false

if LOAD_STREAM
    # stream_res = load(joinpath(FILEPATH, "data/results/onepass_stream.jld2"), "stream")
else
    # # Baker
    # stream_res = LnL.OnePassStreamingOpInf(
    #     preprocess!(ds[1], means, shifts, scales); 
    #     options=options, 
    #     n_state=Int(nxyz * n_fields), 
    #     rank=rmax, 
    #     finite_diff=true
    # )
    # @showprogress for i in 2:n_train
    #     LnL.stream!(stream_res, preprocess!(ds[i], means, shifts, scales), tol=1e-10)
    # end

    # Sketchy
    stream_res = LnL.OnePassStreamingOpInf(
        Float64[0.0];
        options=options, 
        isvd_method=:sketchy,
        n_state=Int(nxyz * n_fields), 
        n_snapshots=n_train,
        rank=rmax, 
    )
    @showprogress for i in 1:n_train  # make sure to include first snapshot
        LnL.stream!(stream_res, preprocess!(ds[i], means, shifts, scales))
    end
    LnL.compute_svd_sketchy!(stream_res)
    # Free up memory
    stream_res.Xrange = [0.0]
    stream_res.Xcorange = [0.0]
    stream_res.Xcore = [0.0]
    stream_res.H = [0.0]
    stream_res.Ξ = [0.0]
    stream_res.Ω = [0.0]
    stream_res.Φ = [0.0]
    stream_res.Ψ = [0.0]

    save(joinpath(FILEPATH, "data/results/onepass_stream.jld2"), "stream", stream_res)
end

# ## Construct the reduced data matrices
# E, Δidx = LnL.finite_diff_matrix(
#     options.data.deriv_type, n_train, options.data.Δt
# )
# r = 300
# Xhat = Diagonal(stream_res.Σ[1:r]) * stream_res.W[:, 1:r]'
# Xhatdot = Xhat * E
# Xhat = Xhat[:, Δidx]

# ## Run grid Search
# if GRID_SEARCH
#     @info "Running grid search for regularization parameters"
#     include(joinpath(FILEPATH, "grid_search.jl"))
#     B1 = 10.0 .^ range(10, 12, length=12)  # best_beta1 = 1.0e12
#     B2 = 10.0 .^ range(12, 13, length=8)  # best_beta2 = 5.179474679231202e12
#     reg_pairs_global = vec([(b1, b2) for b1 in B1, b2 in B2])
#     n_reg_global = length(reg_pairs_global)
#     max_growth = 5.0
#     options.with_reg = true
#     op, best_beta1, best_beta2, best_train_err, states, eval_time, fidx = 
#         find_best_opinf_model(reg_pairs_global, Xhat, Xhat, Xhatdot,
#                             n_train, n_train, max_growth, options,
#                             ds["times"][1:n_train])

#     ## Save results
#     save(joinpath(FILEPATH, "data/results", 
#         "reg_grid_search_r$(r).jld2"), 
#         "beta1", best_beta1, "beta2", best_beta2, 
#         "train_err", best_train_err, "states", states, 
#         "eval_time", eval_time, "final_idx", fidx)

#     ## Save the best model
#     save(joinpath(FILEPATH, "data/models", 
#         "op_stream_r$(r)_lamGS.jld2"), 
#         "op", op)
# else
#     @info "No grid search, using fixed regularization parameters"
#     ## Solve the OpInf problem 
#     options.with_reg = true
#     # beta1 = 4.641588833612782e6
#     # beta1 = 2.1544346900318866e11
#     # beta1 = 5.179474679231202e12
#     # beta2 = 5.179474679231202e12
#     beta1 = 1.0e12
#     beta2 = 1.0e12
#     options.λ = LnL.TikhonovParameter(A=beta1, K=beta1, A2=beta2)
#     op_stream = LnL.opinf(Xhat, options; Xhatdot=Xhatdot)

#     ## Save operators
#     save(joinpath(FILEPATH, "data/models/op_stream_r$(r).jld2"), "op_stream", op_stream)
# end