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
dim_per_field = Nz * Ny * Nx
dPdx = 0.001722
scale_factors = [sqrt(dPdx), sqrt(dPdx), sqrt(dPdx), dPdx]

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
    xbar = 0.0
end

#============================================================#
## Generate the POD basis using specified algorithms
#============================================================#
# Specify which algorithms to run
algorithms = ["baker"]  # Can be extended to ["baker", "brand", "sketchy", "batch"]

# Settings
rmax = 200
bases = Dict()
execution_times = Dict()

# Run each specified algorithm
for algo in algorithms
    @info "Running $algo algorithm..."
    
    if algo == "baker"
        time_data = []

        # Run once with dummy data for JIT compilation
        Xdummy = rand(30, 200)
        baker_dummy = iSVD(x1=Xdummy[:,1], algo=:baker, max_rank=4) 
        full_increment!(baker_dummy, Xdummy, verbose=true, runtime=true)
        
        # Initialize
        tmp = @elapsed baker = iSVD(
            x1=scale(ds[1] .- xbar, dim_per_field, scale_factors), 
            algo=:baker, max_rank=rmax)
        push!(time_data, tmp)
        
        # Incremental updates
        @showprogress for i in 2:n 
            tmp = @elapsed increment!(
                baker, scale(ds[i] .- xbar, dim_per_field, scale_factors))
            push!(time_data, tmp)
        end
        
        # Store results
        bases[algo] = (iVr=baker.Q[:,1:rmax], iΣr=baker.Σ[1:rmax])
        execution_times[algo] = reduce(vcat, time_data)
        
    elseif algo == "brand"
        time_data = []

        # Run once with dummy data for JIT compilation
        Xdummy = rand(30, 200)
        brand = iSVD(x1=Xdummy[:,1], algo=:brand1, reorth_method=:qr, max_rank=4)
        full_increment!(brand, Xdummy, verbose=true, tol=1e-10, runtime=true)
        
        # Initialize
        tmp = @elapsed brand = iSVD(
            x1=scale(ds[1] .- xbar, dim_per_field, scale_factors), 
            algo=:brand1, reorth_method=:gramschmidt, max_rank=rmax)
        push!(time_data, tmp)
        
        # Incremental updates
        @showprogress for i in 2:n 
            tmp = @elapsed increment!(
                brand, scale(ds[i] .- xbar, dim_per_field, scale_factors), tol=1e-10)
            push!(time_data, tmp)
        end
        
        # Store results
        bases[algo] = (iVr=brand.Q[:,1:rmax], iΣr=brand.Σ[1:rmax])
        execution_times[algo] = reduce(vcat, time_data)
        
    elseif algo == "sketchy"
        time_data = []

        # Run once with dummy data for JIT compilation
        Xdummy = rand(30, 200)
        sketchy = iSVD(algo=:sketchy; m=size(Xdummy,1), n=size(Xdummy,2), r=4, ReduxMap=:Sparse)
        full_increment!(sketchy, Xdummy, verbose=true, runtime=true, dump_all=true)
        
        # Initialize
        tmp = @elapsed sketchy = iSVD(
            algo=:sketchy; 
            m=Nz*Ny*Nx*3,  # Excluding pressure field
            n=n, 
            r=rmax, 
            ReduxMap=:Sparse)
        push!(time_data, tmp)
        
        # Process in batches
        X = spzeros(Nz*Ny*Nx*3, n)
        @showprogress for i in 1:(n ÷ 10)
            idx = 10*(i-1)+1:10*i
            X[:,idx] .= [scale(ds[j] .- xbar, dim_per_field, scale_factors) for j in idx]
            sketchy.X .+= sketchy.Ξ * X
            sketchy.Y .+= X * sketchy.Ω'
            sketchy.Z .+= (sketchy.Φ * X) * sketchy.Ψ'
            push!(time_data, tmp)
            fill!(X, 0)
            dropzeros!(X)
        end
        IncrementalSVD.terminate!(sketchy, false, false)
        
        # Store results
        bases[algo] = (iVr=sketchy.Q[:,1:rmax], iΣr=sketchy.Σ[1:rmax])
        execution_times[algo] = reduce(vcat, time_data)
        
    elseif algo == "batch"
        @info "Running batch SVD..."

        # Run once with dummy data for JIT compilation
        Xdummy = rand(30, 200)
        svd(Xdummy)

        try
            time_batch = @elapsed F = svd([scale(ds[j] .- xbar, dim_per_field, scale_factors) for j in 1:n])
            bases[algo] = (Vr=F.U[:,1:rmax], Σr=F.S[1:rmax])
            execution_times[algo] = [time_batch]
        catch e
            if isa(e, OutOfMemoryError)
                @error "Out of memory error during SVD computation. Using randomized SVD."
                try
                    time_batch = @elapsed F = rsvd([scale(ds[j] .- xbar, dim_per_field, scale_factors) for j in 1:n], rmax, p=10)
                    bases[algo] = (Vr=F.U[:,1:rmax], Σr=F.S[1:rmax])
                    execution_times[algo] = [time_batch]
                catch e2
                    @error "Out of memory for randomized SVD as well. Skipping batch method."
                    continue
                end
            else
                @error "Error in batch SVD: $e"
                continue
            end
        end
    else
        @warn "Unknown algorithm: $algo. Skipping."
        continue
    end
    
    @info "Completed $algo algorithm"
end

#=====================================================================#
## Save the POD basis and singular values
#=====================================================================#
save(joinpath(FILEPATH, "data/streaming/basis.jld2"), bases)

#============================================================#
## Save the runtime of the algorithms
#============================================================#
save(joinpath(FILEPATH, "data/streaming/basis_runtime.jld2"), execution_times)
@info "Saved bases for algorithms: $(collect(keys(bases)))"

#================================#
## Compute the projection errors
#================================#
rspan = 100:100:rmax

# Preallocate the dict with per-field errors:
proj_error = Dict(
    "baker" => Dict(
        "u" => zeros(length(rspan)),
        "v" => zeros(length(rspan)),
        "w" => zeros(length(rspan)),
        "p" => zeros(length(rspan)),
        "total" => zeros(length(rspan))  # Keep total for comparison
    )
)

# Loop over all ranks
for (i, r) in enumerate(rspan)
    # Number of threads
    nt = Threads.nthreads()

    # Each thread writes into one slot of these arrays (per field):
    error_per_thread = Dict(
        "u" => zeros(nt),
        "v" => zeros(nt), 
        "w" => zeros(nt),
        "p" => zeros(nt),
        "total" => zeros(nt)
    )
    norm_per_thread = Dict(
        "u" => zeros(nt),
        "v" => zeros(nt),
        "w" => zeros(nt), 
        "p" => zeros(nt),
        "total" => zeros(nt)
    )

    @threads for j in 1:n
        tid = Threads.threadid()

        # Extract full snapshot X = ds[j] - xbar
        X_full = scale(ds[j] .- xbar, dim_per_field, scale_factors)

        # Compute the Baker basis projector: Vr = bases["baker"].iVr[:, 1:r]
        Vr = @view bases["baker"].iVr[:, 1:r]

        # Project full snapshot
        PX_full = Vr * (Vr' * X_full)

        # Extract each field and compute individual errors
        for (field_idx, field_name) in enumerate(ds.fields)
            # Extract field data
            start_idx = (field_idx - 1) * dim_per_field + 1
            end_idx = field_idx * dim_per_field
            
            X_field = @view X_full[start_idx:end_idx]
            PX_field = @view PX_full[start_idx:end_idx]

            # Accumulate field-specific errors
            error_per_thread[field_name][tid] += norm(X_field .- PX_field, 2)
            norm_per_thread[field_name][tid] += norm(X_field, 2)
        end

        # Also compute total error for comparison
        error_per_thread["total"][tid] += norm(X_full .- PX_full, 2)
        norm_per_thread["total"][tid] += norm(X_full, 2)
    end

    # Reduce across threads for each field
    for field_name in [ds.fields; "total"]
        total_error = sum(error_per_thread[field_name])
        total_norm = sum(norm_per_thread[field_name])
        
        proj_error["baker"][field_name][i] = total_error / total_norm
        @info "Projection error for baker at rank $r, field $field_name: $(proj_error["baker"][field_name][i])"
    end
end

# Save to disk:
save(joinpath(FILEPATH, "data/projection_errors.jld2"), "proj_error" => proj_error)