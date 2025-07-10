"""
Generate the reduced data using the reduced basis from the iSVD.
"""

#================#
## Load Packages
#================#
using FileIO
using FLoops
using JLD2
using LinearAlgebra
using ProgressMeter
using BlockDiagonals
using SparseArrays

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
dim_per_field = Nz * Ny * Nx
dPdx = 0.001722
# scale_factors = [sqrt(dPdx), sqrt(dPdx), sqrt(dPdx), dPdx]
scale_factors = [1.0, 0.01, 0.01, dPdx]

#===============#
## Input data 
#===============#
U = dPdx * ones(1,n)  

#==========================#
## Load the mean velocity
#==========================#
xbar = load(joinpath(FILEPATH, "data/streaming/mean.jld2"))["xbar"]
# minmax = load(joinpath(FILEPATH, "data/streaming/minmax.jld2"))["minmax"]
# xbar = minmax["xbar"]
# scale_factors = minmax["scale_factors"]
# minmax = nothing

#=================#
## Load the bases 
#=================#
basis_file = joinpath(FILEPATH, "data/streaming/basis.jld2")
iVrmax = load(basis_file)["bases"]["baker"].iVr

# basis_data = load(basis_file)
# rmax = 1000
# iVrmax = sparse(basis_data["bases"]["baker_fieldwise"].iVr[:,1:rmax])  # choose Baker's iSVD basis
# iVrmax = sparse(basis_data["merged_basis"].iVr[:,1:rmax])  # choose Baker's iSVD basis

# field_results = load(basis_file)["field_results"]
# r = 100
# iVr_u = field_results[1].Q[:, 1:r]  # u-component basis
# iVr_v = field_results[2].Q[:, 1:r]  # v-component basis
# iVr_w = field_results[3].Q[:, 1:r]  # w-component basis
# iVr_p = field_results[4].Q[:, 1:r]  # p-component basis

# iΣr_u = field_results[1].Σ
# iΣr_v = field_results[2].Σ
# iΣr_w = field_results[3].Σ
# iΣr_p = field_results[4].Σ

# field_results = nothing  # Free memory

#=============================#
## Generate the reduced data 
#=============================#
TYPE = "discrete"
Xhat = nothing    # put on global scope just in case
Xhatdot = nothing # put on global scope just in case

if TYPE == "continuous"
    # ===== CONTINUOUS ===== #
    TIME_DERIV_METHOD = 3

    if TIME_DERIV_METHOD == 1
        @info "Sequential using forward, backward, and central finite differences"

        ## METHOD #1) SEQUENTIAL (SLOW) FWD + BWD + CTD
        Xhat = Array{Float64}(undef, rmax, n)  # Preallocate the reduced data matrix
        Xhatdot = Array{Float64}(undef, rmax, n)  # Preallocate the reduced time derivative matrix
        @showprogress for i in 1:n
            # Compute the i-th time derivative 
            if i == 1
                # First snapshot, use forward finite difference
                dt = ds["times"][5] - ds["times"][1]
                x_i_ip4 = scale(ds[i:i+4] .- xbar, dim_per_field, scale_factors)
                xhat_i_ip4 = iVrmax' * x_i_ip4  # reduce
                xhatdot_i = fwd4(xhat_i_ip4, dt/4, true)
                Xhat[:,i] = xhat_i_ip4[:,1]  # Store the first snapshot
            elseif i == 2
                # Second snapshot, use forward finite difference but with adjusted stencil
                dt = ds["times"][5] - ds["times"][1]
                x_im1_ip3 = scale(ds[i-1:i+3] .- xbar, dim_per_field, scale_factors)
                xhat_im1_ip3 = iVrmax' * x_im1_ip3  # reduce
                xhatdot_i = fwd4(xhat_im1_ip3, dt/4, false)
                Xhat[:,i] = xhat_im1_ip3[:,2]  # Store the second snapshot
            elseif i == n-1
                # Second to last snapshot, use backward finite difference
                dt = ds["times"][n] - ds["times"][n-4]
                x_im3_ip1 = scale(ds[i-3:i+1] .- xbar, dim_per_field, scale_factors)
                xhat_im3_ip1 = iVrmax' * x_im3_ip1  # reduce
                xhatdot_i = bwd4(xhat_im3_ip1, dt/4, false)
                Xhat[:,i] = xhat_im3_ip1[:,4]  # Store the second to last snapshot
            elseif i == n 
                # Last snapshot, use backward finite difference with adjusted stencil
                dt = ds["times"][n] - ds["times"][n-4]
                x_im4_i = scale(ds[i-4:i] .- xbar, dim_per_field, scale_factors)
                xhat_im4_i = iVrmax' * x_im4_i  # reduce
                xhatdot_i = bwd4(xhat_im4_i, dt/4, true)
                Xhat[:,i] = xhat_im4_i[:,5]  # Store the last snapshot
            else
                # For all other snapshots, use central finite difference
                dt = ds["times"][i+2] - ds["times"][i-2]
                x_im2_ip2 = scale(ds[i-2:i+2] .- xbar, dim_per_field, scale_factors)
                xhat_im2_ip2 = iVrmax' * x_im2_ip2  # reduce
                xhatdot_i = ctd4(xhat_im2_ip2, dt/4)
                Xhat[:,i] = xhat_im2_ip2[:,3]  # Store the middle snapshot
            end
            Xhatdot[:,i] = xhatdot_i  # Store the time derivative
        end
    elseif TIME_DERIV_METHOD == 2
        @info "Sequential using central finite differences only"

        ## METHOD #2) SEQUENTIAL (SLOW) CTD only
        Xhat = Array{Float64}(undef, rmax, n-4)  # Preallocate the reduced data matrix
        Xhatdot = Array{Float64}(undef, rmax, n-4)  # Preallocate the reduced time derivative matrix
        @showprogress for (ct, i) in enumerate(3:n-2)
            # Get the (i-2)-th to (i+2)-th snapshot
            x_im2_ip2 = scale(ds[i-2:i+2] .- xbar, dim_per_field, scale_factors)
            xhat = iVrmax' * x_im2_ip2  # reduce
            
            # Compute the i-th time derivative using central finite difference
            dt = ds["times"][i+2] - ds["times"][i-2]
            xhatdot_i = ctd4(xhat, dt/4)

            Xhat[:,ct] = xhat[:,3]  # Store the middle snapshot
            Xhatdot[:,ct] = xhatdot_i
        end

    else
        @info "Parallel using forward, backward, and central finite differences"

        ## METHOD #3) PARALLEL (FAST)
        Xhat = Array{Float64}(undef, rmax, n)  # Preallocate the reduced data matrix
        Xhatdot = Array{Float64}(undef, rmax, n)  # Preallocate the reduced time derivative matrix
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
                reduced_snapshots = [
                    iVrmax' * scale(ds[j] .- xbar, dim_per_field, scale_factors)
                    for j in 1:5
                ]  
                xhatdot_i = fwd4(reduced_snapshots, dt/4, true)
            else # i == 2
                dt = times[5] - times[1]
                reduced_snapshots = [
                    iVrmax' * scale(ds[j] .- xbar, dim_per_field, scale_factors)
                    for j in 1:5
                ]
                xhatdot_i = fwd4(reduced_snapshots, dt/4, false)
            end
            Xhat[:,i] = reduced_snapshots[i]
            Xhatdot[:,i] = xhatdot_i
        end

        # Handle last two points
        for i in (n-1):n
            if i == n-1
                dt = times[n] - times[n-4]
                reduced_snapshots = [
                    iVrmax' * scale(ds[j] .- xbar, dim_per_field, scale_factors)
                    for j in (n-4):n
                ] 
                xhatdot_i = bwd4(reduced_snapshots, dt/4, false)
                Xhat[:,i] = reduced_snapshots[4]
            else # i == n
                dt = times[n] - times[n-4]
                reduced_snapshots = [
                    iVrmax' * scale(ds[j] .- xbar , dim_per_field, scale_factors)
                    for j in (n-4):n
                ] 
                xhatdot_i = bwd4(reduced_snapshots, dt/4, true)
                Xhat[:,i] = reduced_snapshots[5]
            end
            Xhatdot[:,i] = xhatdot_i
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
                    reduced_snapshots = [
                        iVrmax' * scale(ds[j] .- xbar, dim_per_field, scale_factors)
                        for j in window_start:window_end
                    ]
                    
                    # Calculate time step
                    dt = times[i+2] - times[i-2]
                    
                    # Calculate derivative
                    xhatdot_i = ctd4(reduced_snapshots, dt/4)
                    
                    # Store results in thread-local batch arrays
                    batch_xhat[:, local_idx] = reduced_snapshots[3] # middle point is i
                    batch_xhatdot[:, local_idx] = xhatdot_i
                end
                
                # Copy batch results to global arrays (critical section)
                batch_offset = start_idx - 3
                Xhat[:, start_idx:end_idx] = batch_xhat
                Xhatdot[:, start_idx:end_idx] = batch_xhatdot

                @info "Processed batch $(batch) from $(start_idx) to $(end_idx)"
            end
        end

        @info "Completed processing $n snapshots"
    end
else
    # for r in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    #     # field_results = load(basis_file)["field_results"]
    #     # iVr_u = field_results[1].Q[:, 1:r]  # u-component basis
    #     # iVr_v = field_results[2].Q[:, 1:r]  # v-component basis
    #     # iVr_w = field_results[3].Q[:, 1:r]  # w-component basis
    #     # iVr_p = field_results[4].Q[:, 1:r]  # p-component basis
    #     # field_results = nothing  # Free memory

    #     # Process snapshots in parallel batches
    #     const BATCH_SIZE = 100  # Adjust based on available memory
    #     num_batches = ceil(Int, n / BATCH_SIZE)
    #     # Xhat = Array{Float64}(undef, r*4, n)  # Preallocate the reduced data matrix
    #     Xhat = Array{Float64}(undef, r, n)  # Preallocate the reduced data matrix

    #     # Compose basis
    #     # iVr = BlockDiagonal([ iVr_u, iVr_v, iVr_w, iVr_p ])
    #     iVr = view(iVrmax, :, 1:r) 

    #     @info "Processing $n snapshots in $num_batches batches using $(Threads.nthreads()) threads"
    #     @time begin
    #         # Use atomic indices to ensure thread safety
    #         batch_indices = collect(1:num_batches)

    #         Threads.@threads for batch in batch_indices
    #             start_idx = (batch-1) * BATCH_SIZE + 1
    #             end_idx = min(start_idx + BATCH_SIZE - 1, n)
                
    #             # Pre-allocate batch array for this thread
    #             batch_size = end_idx - start_idx + 1
    #             # batch_xhat = Matrix{Float64}(undef, r*4, batch_size)
    #             batch_xhat = Matrix{Float64}(undef, r, batch_size)

    #             # Process batch
    #             for (local_idx, i) in enumerate(start_idx:end_idx)
    #                 batch_xhat[:, local_idx] = iVr' * scale(ds[i] .- xbar, dim_per_field, scale_factors)
    #             end
                
    #             # Thread-safe assignment to non-overlapping region
    #             @views Xhat[:, start_idx:end_idx] .= batch_xhat
    #         end

    #         reduced_data_file = joinpath(FILEPATH, "data/streaming/reduced_data_discrete_r$r.jld2")
    #         @info "Saving reduced data for r=$(r) to $reduced_data_file"
    #         save(reduced_data_file, "Xhat", Xhat)
    #     end
    # end

    for r in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        # Process snapshots in parallel batches
        const BATCH_SIZE = 100  # Adjust based on available memory
        num_batches = ceil(Int, n / BATCH_SIZE)
        Xhat = Array{Float64}(undef, r, n)  # Preallocate the reduced data matrix

        # Compose basis
        iVr = view(iVrmax, :, 1:r) 

        # Create a lock for thread-safe data source access
        data_lock = ReentrantLock()

        @info "Processing $n snapshots in $num_batches batches using $(Threads.nthreads()) threads"

        # Use FLoops for better thread management
        @floop ThreadedEx() for batch in 1:num_batches
            start_idx = (batch-1) * BATCH_SIZE + 1
            end_idx = min(start_idx + BATCH_SIZE - 1, n)
            
            # Pre-allocate batch array for this thread
            batch_size = end_idx - start_idx + 1
            batch_xhat = Matrix{Float64}(undef, r, batch_size)

            # Process batch sequentially within each thread
            for (local_idx, i) in enumerate(start_idx:end_idx)
                # Each thread processes its batch independently
                @lock data_lock scaled_snapshot = scale(
                    ds[i] .- xbar, dim_per_field, scale_factors)
                batch_xhat[:, local_idx] = iVr' * scaled_snapshot
            end
            
            # Thread-safe assignment to non-overlapping region
            @views Xhat[:, start_idx:end_idx] .= batch_xhat
        end

        reduced_data_file = joinpath(FILEPATH, "data/streaming/reduced_data_discrete_r$r.jld2")
        @info "Saving reduced data for r=$(r) to $reduced_data_file"
        save(reduced_data_file, "Xhat", Xhat)
    end
end

#===========================#
## Save the reduced data
#===========================#
if TYPE == "continuous"
    reduced_data_file = joinpath(FILEPATH, "data/streaming/reduced_data.jld2")
    save(reduced_data_file, "Xhat", Xhat, "Xhatdot", Xhatdot)
else
    reduced_data_file = joinpath(FILEPATH, "data/streaming/reduced_data_discrete.jld2")
    save(reduced_data_file, "Xhat", Xhat)
end