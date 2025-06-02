"""
3D Channel flow: Compute basis
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using IncrementalSVD
using LinearAlgebra
using ProgressMeter
using SparseArrays
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
DATAPATH = "../../../../../DATA/NREL/3D_CHANNEL"
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"Two-Pass_Streaming-OpInf/3d_channel") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/3d_channel")
fn = "channel_5200_data_0_10000.h5"
datafile = joinpath(DATAPATH, fn)

#==========================================#
## Load struct to read data in HDF5 format 
#==========================================#
include(joinpath(FILEPATH, "datasource.jl"))

#========================#
## Additional functions
#========================#
include(joinpath(FILEPATH, "utils.jl"))

#=============================#
## Load the training dataset
#=============================#
ds = ChannelDataSource(datafile, ["z", "y", "x", "fields", "times"])
Nz, Ny, Nx, n_fields, n = ds.dims

#========================================#
## Compute the mean velocity for shiting
#========================================#
SHIFT_MEAN = false

if SHIFT_MEAN
    mean_file = joinpath(FILEPATH, "data/streaming/mean.jld2")
    if isfile(mean_file)
        @info "Loading existing mean from file"
        xbar = load(mean_file, "xbar")
    else
        @info "Starting mean computation with $(Threads.nthreads()) threads"
        @time begin
            xbar = compute_mean_parallel_threads(ds, (Nz*Ny*Nx*3,); batch_size=100)
        end
        @info "Mean computation complete"
        save(mean_file, "xbar", xbar)
    end
else
    @info "Skipping mean computation, using zero mean"
    xbar = zeros(Nz*Ny*Nx*3)
end

#=========================================================#
## Generate the POD basis using iSVD using all algorithms
#=========================================================#
# (Dry) Run it once with a dummy due to JUlia's JIT compilation
Xdummy = rand(30, 200)
baker = iSVD(x1=Xdummy[:,1], algo=:baker, max_rank=4) 
full_increment!(baker, Xdummy, verbose=true, runtime=true)
# brand = iSVD(x1=Xdummy[:,1], algo=:brand1, reorth_method=:qr, max_rank=4)
# full_increment!(brand, Xdummy, verbose=true, tol=1e-10, runtime=true)
# sketchy = iSVD(algo=:sketchy; m=size(Xdummy,1), n=size(Xdummy,2), r=4, ReduxMap=:Sparse)
# full_increment!(sketchy, Xdummy, verbose=true, runtime=true, dump_all=true)
# svd(Xdummy)

## Settings
rmax = 500
Xall = Array[]

# Execution times 
time_baker = []
time_brand = []
time_sketchy = []

## baker
tmp = @elapsed baker = iSVD(x1=ds[1] - xbar, algo=:baker, max_rank=rmax)
push!(time_baker, tmp)
@showprogress for i in 2:n 
    tmp = @elapsed increment!(baker, ds[i] - xbar)
    push!(time_baker, tmp)
end

# ## brand
# tmp = @elapsed brand = iSVD(x1=ds[1], algo=:brand1, reorth_method=:gramschmidt, max_rank=rmax)
# push!(time_brand, tmp)
# @showprogress for i in 2:n 
#     tmp = @elapsed increment!(brand, ds[i], tol=1e-10)
#     push!(time_brand, tmp)
# end

# ## sketchy
# tmp = @elapsed sketchy = iSVD(algo=:sketchy; m=Nx*Ny*Nz*n_fields, n=n, r=rmax, ReduxMap=:Sparse)
# push!(time_sketchy, tmp)
# X = spzeros(Nx*Ny*Nz*n_fields, n)
# @showprogress for i in 1:(n ÷ 10)
#     idx = 10*(i-1)+1:10*i
#     X[:,idx] .= ds[idx]
#     sketchy.X .+= sketchy.Ξ * X
#     sketchy.Y .+= X * sketchy.Ω'
#     sketchy.Z .+= (sketchy.Φ * X) * sketchy.Ψ'
#     push!(time_sketchy, tmp)
#     fill!(X, 0)
#     dropzeros!(X)
# end
# IncrementalSVD.terminate!(sketchy, false, false)

# #============================================#
# ## Compute the POD basis using the batch SVD 
# #============================================#
# try
#     time_batch = @elapsed F = svd(ds[1:n])
# catch e
#     if isa(e, OutOfMemoryError)
#         @error "Out of memory error during SVD computation. Using randomized SVD."
#     end
#     try
#         time_batch = @elapsed F = rsvd(ds[1:n], rmax, p=10)
#     catch e2
#         @error "Out of memory for randomized SVD as well. We're out of luck for batch methods."
#     end
# end

#=====================================================================#
## Save the POD basis and singular values from the iSVD and batch SVD
#=====================================================================#
bases = Dict(
    "baker" => (iVr=baker.Q[:,1:rmax], iΣr=baker.Σ[1:rmax]),
    # "brand" => (iVr=brand.Q[:,1:rmax], iΣr=brand.Σ[1:rmax]),
    # "sketchy" => (iVr=sketchy.Q[:,1:rmax], iΣr=sketchy.Σ[1:rmax]),
    # "batch" => (Vr=F.U[:,1:rmax], Σr=F.S[1:rmax]),
)
save(joinpath(FILEPATH, "data/streaming/basis.jld2"), bases)

#============================================================#
## Save the runtime of the iSVD algorithms over all streams
#============================================================#
time_baker = reduce(vcat, time_baker)
# time_brand = reduce(vcat, time_brand)
# time_sketchy = reduce(vcat, time_sketchy)
save(
    joinpath(FILEPATH, "data/streaming/basis_runtime.jld2"),
    "baker", time_baker, 
    # "brand", time_brand,  
    # "batch", time_batch,  "sketchy", time_sketchy, 
)

#================================#
## Compute the projection errors
#================================#
# (SEQUENTIAL VERSION)
# proj_error = Dict(
#     "baker" => zeros(rmax),
#     # "brand" => zeros(rmax),
#     # "sketchy" => zeros(rmax),
#     # "batch" => zeros(rmax),
# )
# for i in [rmax]
#     mult = 1
#     tot_norm = 0.0
#     for j in 1:n
#         X = ds[j] - xbar
#         proj_error["baker"][i] += norm(
#             X - bases["baker"].iVr[:,1:i] * bases["baker"].iVr[:,1:i]' * X, 2
#         )
#         tot_norm += norm(X, 2)
#         println("j = $j, i = $i, error = $(proj_error["baker"][i] / tot_norm)")
#     end
#     proj_error["baker"][i] /= tot_norm
#     @info "Projection error for baker at rank $i: $(proj_error["baker"][i])"
#     # proj_error["brand"][i] = norm(X - bases["brand"].iVr[:,1:i] * bases["brand"].iVr[:,1:i]' * X, 2) / norm(X, 2)
#     # proj_error["sketchy"][i] = norm(X - bases["sketchy"].iVr[:,1:i] * bases["sketchy"].iVr[:,1:i]' * X, 2) / norm(X, 2)
#     # proj_error["batch"][i] = norm(X - bases["batch"].Vr[:,1:i] * bases["batch"].Vr[:,1:i]' * X, 2) / norm(X, 2)
# end
# save(joinpath(FILEPATH, "data/projection_errors.jld2"), proj_error)            

# (MULTI-THREAD VERSION)
rspan = 100:100:rmax
# Preallocate the dict and per‐method arrays:
proj_error = Dict(
    "baker"  => zeros(length(rspan)),
    # "brand" => zeros(length(rspan)),
    # "sketchy"=> zeros(length(rspan)),
    # "batch" => zeros(length(rspan)),
)

# Loop over all ranks i = 1:rmax. Inside each i, parallelize over j=1:n.
for (i, r) in enumerate(rspan)
    # Number of threads
    nt = Threads.nthreads()

    # Each thread writes into one slot of these arrays:
    error_per_thread = zeros(nt)
    norm_per_thread  = zeros(nt)

    @threads for j in 1:n
        tid = Threads.threadid()

        # Extract X = ds[j] - xbar
        X = @view ds[j] .- xbar

        # Compute the Baker basis projector: Vr = bases["baker"].iVr[:, 1:r]
        Vr = @view bases["baker"].iVr[:, 1:r]

        # Project X:  P X = Vr * (Vr' * X)
        #    Notice that Vr' * X is a small i×1 vector, so Vr * (…) is O(di)
        PX = Vr * ( (Vr') * X )

        # Accumulate this thread’s partial sums:
        error_per_thread[tid] += norm(X .- PX, 2)
        norm_per_thread[tid]  += norm(X, 2)

        # (Optional) If you really want per‐j printouts, you can do it,
        # but keep in mind printing from multiple threads can interleave.
        # println("Thread $tid ▶ j = $j, i = $i, running error = $(error_per_thread[tid] / norm_per_thread[tid])")
    end

    # Now reduce across threads:
    total_error = sum(error_per_thread)
    total_norm  = sum(norm_per_thread)

    proj_error["baker"][i] = total_error / total_norm
    @info "Projection error for baker at rank $i: $(proj_error["baker"][i])"
end

# Save to disk:
save(joinpath(FILEPATH, "data/projection_errors.jld2"), "proj_error" => proj_error)