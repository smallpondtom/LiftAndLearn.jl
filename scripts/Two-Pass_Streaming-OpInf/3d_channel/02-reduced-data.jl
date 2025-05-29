"""
Generate the reduced data using the reduced basis from the iSVD.
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using LinearAlgebra
using ProgressMeter

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
include(joinpath(FILEPATH, "derivative.jl"))

#=============================#
## Load the training dataset
#=============================#
ds = ChannelDataSource(datafile, ["z", "y", "x", "fields", "times"])
Nz, Ny, Nx, n_fields, n = ds.dims

#===============#
## Input data 
#===============#
U = - 0.001722 * ones(1,n)  

#==========================#
## Load the mean velocity
#==========================#
xbar = load(joinpath(FILEPATH, "data/streaming/mean.jld2"))["xbar"]

#=================#
## Load the bases 
#=================#
basis_file = joinpath(FILEPATH, "data/streaming/basis.jld2")
basis_data = load(basis_file)
iVrmax = basis_data["baker"].iVr  # choose Baker's iSVD basis
rmax = size(iVrmax,2)

#=============================#
## Generate the reduced data
#=============================#
Xhat = Array{Float64}(undef, rmax, n)  # Preallocate the reduced data matrix
Xhatdot = Array{Float64}(undef, rmax, n)  # Preallocate the reduced time derivative matrix

## SEQUENTIAL (SLOW)
# @showprogress for i in 1:n
#     # Get the i-th snapshot
#     x_i = ds[i]  
#     Xhat[:,i] = iVrmax' * x_i  # Project the snapshot onto the basis
    
#     # Compute the i-th time derivative 
#     if i == 1
#         # First snapshot, use forward finite difference
#         dt = ds["times"][5] - ds["times"][1]
#         xdot_i = fwd4(ds[i:i+4], dt/4, true)
#     elseif i == 2
#         # Second snapshot, use forward finite difference but with adjusted stencil
#         dt = ds["times"][5] - ds["times"][1]
#         xdot_i = fwd4(ds[i-1:i+3], dt/4, false)
#     elseif i == n-1
#         # Second to last snapshot, use backward finite difference
#         dt = ds["times"][n] - ds["times"][n-4]
#         xdot_i = bwd4(ds[i-3:i+1], dt/4, false)
#     elseif i == n 
#         # Last snapshot, use backward finite difference with adjusted stencil
#         dt = ds["times"][n] - ds["times"][n-4]
#         xdot_i = bwd4(ds[i-4:i], dt/4, true)
#     else
#         # For all other snapshots, use central finite difference
#         dt = ds["times"][i+2] - ds["times"][i-2]
#         xdot_i = ctd4(ds[i-2:i+2], dt/4)
#     end
#     Xhatdot[:,i] = iVrmax' * xdot_i  # Project the time derivative onto the basis
# end

## PARALLEL (FAST)
# First load times data once (avoid repeated reads)
times = ds["times"]

# Read all snapshots in batches to minimize I/O overhead
const BATCH_SIZE = 100  # Adjust based on available memory
num_batches = ceil(Int, n / BATCH_SIZE)

## Process boundary cases separately (first 2 and last 2 points)
# Handle first two points
for i in 1:2
    if i == 1
        dt = times[5] - times[1]
        snapshots = [ds[j] for j in 1:5] .- xbar  # Adjust for mean velocity
        xdot_i = fwd4(snapshots, dt/4, true)
    else # i == 2
        dt = times[5] - times[1]
        snapshots = [ds[j] for j in 1:5] .- xbar  # Adjust for mean velocity
        xdot_i = fwd4(snapshots, dt/4, false)
    end
    Xhat[:,i] = iVrmax' * ds[i]
    Xhatdot[:,i] = iVrmax' * xdot_i
end

# Handle last two points
for i in (n-1):n
    if i == n-1
        dt = times[n] - times[n-4]
        snapshots = [ds[j] for j in (n-4):n] .- xbar  # Adjust for mean velocity
        xdot_i = bwd4(snapshots, dt/4, false)
    else # i == n
        dt = times[n] - times[n-4]
        snapshots = [ds[j] for j in (n-4):n] .- xbar  # Adjust for mean velocity
        xdot_i = bwd4(snapshots, dt/4, true)
    end
    Xhat[:,i] = iVrmax' * ds[i]
    Xhatdot[:,i] = iVrmax' * xdot_i
end

## Process all other points in parallel batches
@info "Processing $(n-4) snapshots in $num_batches batches using $(Threads.nthreads()) threads"
@time begin
    # Process middle points (3 to n-2) in parallel
    Threads.@threads for batch in 1:num_batches
        start_idx = 3 + (batch-1) * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE - 1, n-2)
        
        # Skip if out of range
        if start_idx > n-2
            continue
        end
        
        # Pre-allocate batch arrays for this thread
        batch_size = end_idx - start_idx + 1
        batch_xhat = Matrix{Float64}(undef, rmax, batch_size)
        batch_xhatdot = Matrix{Float64}(undef, rmax, batch_size)
        
        # Process batch
        for (local_idx, i) in enumerate(start_idx:end_idx)
            # Get 5-point stencil for finite difference
            window_start = i-2
            window_end = i+2
            snapshots = [ds[j] for j in window_start:window_end] .- xbar  # Adjust for mean velocity
            
            # Calculate time step
            dt = times[i+2] - times[i-2]
            
            # Calculate derivative
            xdot_i = ctd4(snapshots, dt/4)
            
            # Store results in thread-local batch arrays
            batch_xhat[:, local_idx] = iVrmax' * snapshots[3]  # middle point is i
            batch_xhatdot[:, local_idx] = iVrmax' * xdot_i
        end
        
        # Copy batch results to global arrays (critical section)
        batch_offset = start_idx - 3
        Xhat[:, start_idx:end_idx] = batch_xhat
        Xhatdot[:, start_idx:end_idx] = batch_xhatdot

        @info "Processed batch $(batch) from $(start_idx) to $(end_idx)"
    end
end

@info "Completed processing $n snapshots"

#===========================#
## Save the reduced data
#===========================#
reduced_data_file = joinpath(FILEPATH, "data/streaming/reduced_data.jld2")
save(reduced_data_file, "Xhat", Xhat, "Xhatdot", Xhatdot, "U", U)