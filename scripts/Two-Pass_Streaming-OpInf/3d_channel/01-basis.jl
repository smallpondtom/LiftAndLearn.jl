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
using BlockDiagonals
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

#================================================================#
## Compute preprocessing parameters (mean and/or minmax scaling)
#================================================================#
COMPUTE_MEAN = true
COMPUTE_MINMAX = true

# Initialize preprocessing variables
xbar = 0.0
scale_factors = [1.0, 0.01, 0.01, dPdx]  # Default scale factors
# scale_factors = [sqrt(dPdx), sqrt(dPdx), sqrt(dPdx), dPdx]

if COMPUTE_MEAN && COMPUTE_MINMAX
    @info "Computing mean and min/max for preprocessing"
    mean_file = joinpath(FILEPATH, "data/streaming/mean.jld2")
    if isfile(mean_file)
        @info "Loading existing mean from file"
        xbar = load(mean_file, "xbar")
    else
        @info "Starting mean computation with $(Threads.nthreads()) threads"
        @time begin
            xbar = compute_mean_parallel_threads(ds, (Nz*Ny*Nx*4,); batch_size=100)
        end
        @info "Mean computation complete"
        save(mean_file, "xbar", xbar)
    end

    minmax_file = joinpath(FILEPATH, "data/streaming/minmax.jld2")
    if isfile(minmax_file)
        @info "Loading existing minmax parameters from file"
        minmax_data = load(minmax_file, "minmax")
        scale_factors = minmax_data["scale_factors"]
    else
        @info "Starting min/max computation with $(Threads.nthreads()) threads"
        @time begin
            x_min, x_max = compute_minmax_parallel_threads(
                ds, (Nz*Ny*Nx*4,), ["u", "v", "w", "p"]; batch_size=100)
        end
        # Compute minmax scaling parameters
        scale_factors = x_max .- x_min
        xshift = reduce(vcat, [xm * ones(Nz*Ny*Nx) for xm in x_min])
        @info "Min/max computation complete"
        save(minmax_file, "minmax", Dict(
            "x_min" => x_min, "x_max" => x_max,
            "xbar" => xshift, "scale_factors" => scale_factors
        ))
    end
elseif COMPUTE_MEAN
    @info "Computing mean for mean-shift preprocessing"
    mean_file = joinpath(FILEPATH, "data/streaming/mean.jld2")
    if isfile(mean_file)
        @info "Loading existing mean from file"
        xbar = load(mean_file, "xbar")
    else
        @info "Starting mean computation with $(Threads.nthreads()) threads"
        @time begin
            xbar = compute_mean_parallel_threads(ds, (Nz*Ny*Nx*4,); batch_size=100)
        end
        @info "Mean computation complete"
        save(mean_file, "xbar", xbar)
    end
elseif COMPUTE_MINMAX
    @info "Computing min/max for minmax shift-and-scale preprocessing"
    minmax_file = joinpath(FILEPATH, "data/streaming/minmax.jld2")
    if isfile(minmax_file)
        @info "Loading existing minmax parameters from file"
        minmax_data = load(minmax_file, "minmax")
        xbar = minmax_data["xbar"]
        scale_factors = minmax_data["scale_factors"]
    else
        @info "Starting min/max computation with $(Threads.nthreads()) threads"
        @time begin
            x_min, x_max = compute_minmax_parallel_threads(
                ds, (Nz*Ny*Nx*4,), ["u", "v", "w", "p"]; batch_size=100)
        end
        # Compute minmax scaling parameters
        scale_factors = x_max .- x_min
        xbar = reduce(vcat, [xm * ones(Nz*Ny*Nx) for xm in x_min])
        @info "Min/max computation complete"
        save(minmax_file, "minmax", Dict(
            "x_min" => x_min, "x_max" => x_max,
            "xbar" => xbar, "scale_factors" => scale_factors
        ))
    end
else
    @info "Skipping preprocessing, using default scaling and zero mean"
    xbar = 0.0
end


##

basis_file = joinpath(FILEPATH, "data/streaming/basis_fieldwise1.jld2")
field_results = load(basis_file)["field_results"]
r = 100
iΣr_u1 = field_results[1].Σ
iΣr_v1 = field_results[2].Σ
iΣr_w1 = field_results[3].Σ
iΣr_p1 = field_results[4].Σ

# #============================================================#
# ## Generate the POD basis using specified algorithms
# #============================================================#
# # Specify which algorithms to run
# algorithms = ["baker"]  # Can be extended to ["baker", "brand", "sketchy", "batch"]

# # Settings
# rmax = 400
# bases = Dict()
# execution_times = Dict()

# # Run each specified algorithm
# for algo in algorithms
#     @info "Running $algo algorithm..."
    
#     if algo == "baker"
#         time_data = []

#         # Run once with dummy data for JIT compilation
#         Xdummy = rand(30, 200)
#         baker_dummy = iSVD(x1=Xdummy[:,1], algo=:baker, max_rank=4) 
#         full_increment!(baker_dummy, Xdummy, verbose=true, runtime=true)
        
#         # Initialize
#         tmp = @elapsed baker = iSVD(
#             x1=scale(ds[1] .- xbar, dim_per_field, scale_factors), 
#             algo=:baker, max_rank=rmax)
#         push!(time_data, tmp)
        
#         # Incremental updates
#         @showprogress for i in 2:n 
#             tmp = @elapsed increment!(
#                 baker, scale(ds[i] .- xbar, dim_per_field, scale_factors))
#             push!(time_data, tmp)
#         end
        
#         # Store results
#         bases[algo] = (iVr=baker.Q[:,1:rmax], iΣr=baker.Σ[1:rmax])
#         execution_times[algo] = reduce(vcat, time_data)
        
#     elseif algo == "brand"
#         time_data = []

#         # Run once with dummy data for JIT compilation
#         Xdummy = rand(30, 200)
#         brand = iSVD(x1=Xdummy[:,1], algo=:brand1, reorth_method=:qr, max_rank=4)
#         full_increment!(brand, Xdummy, verbose=true, tol=1e-10, runtime=true)
        
#         # Initialize
#         tmp = @elapsed brand = iSVD(
#             x1=scale(ds[1] .- xbar, dim_per_field, scale_factors), 
#             algo=:brand1, reorth_method=:gramschmidt, max_rank=rmax)
#         push!(time_data, tmp)
        
#         # Incremental updates
#         @showprogress for i in 2:n 
#             tmp = @elapsed increment!(
#                 brand, scale(ds[i] .- xbar, dim_per_field, scale_factors), tol=1e-10)
#             push!(time_data, tmp)
#         end
        
#         # Store results
#         bases[algo] = (iVr=brand.Q[:,1:rmax], iΣr=brand.Σ[1:rmax])
#         execution_times[algo] = reduce(vcat, time_data)
        
#     elseif algo == "sketchy"
#         time_data = []

#         # Run once with dummy data for JIT compilation
#         Xdummy = rand(30, 200)
#         sketchy = iSVD(algo=:sketchy; m=size(Xdummy,1), n=size(Xdummy,2), r=4, ReduxMap=:Sparse)
#         full_increment!(sketchy, Xdummy, verbose=true, runtime=true, dump_all=true)
        
#         # Initialize
#         tmp = @elapsed sketchy = iSVD(
#             algo=:sketchy; 
#             m=Nz*Ny*Nx*3,  # Excluding pressure field
#             n=n, 
#             r=rmax, 
#             ReduxMap=:Sparse)
#         push!(time_data, tmp)
        
#         # Process in batches
#         X = spzeros(Nz*Ny*Nx*3, n)
#         @showprogress for i in 1:(n ÷ 10)
#             idx = 10*(i-1)+1:10*i
#             X[:,idx] .= [scale(ds[j] .- xbar, dim_per_field, scale_factors) for j in idx]
#             sketchy.X .+= sketchy.Ξ * X
#             sketchy.Y .+= X * sketchy.Ω'
#             sketchy.Z .+= (sketchy.Φ * X) * sketchy.Ψ'
#             push!(time_data, tmp)
#             fill!(X, 0)
#             dropzeros!(X)
#         end
#         IncrementalSVD.terminate!(sketchy, false, false)
        
#         # Store results
#         bases[algo] = (iVr=sketchy.Q[:,1:rmax], iΣr=sketchy.Σ[1:rmax])
#         execution_times[algo] = reduce(vcat, time_data)
        
#     elseif algo == "batch"
#         @info "Running batch SVD..."

#         # Run once with dummy data for JIT compilation
#         Xdummy = rand(30, 200)
#         svd(Xdummy)

#         try
#             time_batch = @elapsed F = svd([scale(ds[j] .- xbar, dim_per_field, scale_factors) for j in 1:n])
#             bases[algo] = (Vr=F.U[:,1:rmax], Σr=F.S[1:rmax])
#             execution_times[algo] = [time_batch]
#         catch e
#             if isa(e, OutOfMemoryError)
#                 @error "Out of memory error during SVD computation. Using randomized SVD."
#                 try
#                     time_batch = @elapsed F = rsvd([scale(ds[j] .- xbar, dim_per_field, scale_factors) for j in 1:n], rmax, p=10)
#                     bases[algo] = (Vr=F.U[:,1:rmax], Σr=F.S[1:rmax])
#                     execution_times[algo] = [time_batch]
#                 catch e2
#                     @error "Out of memory for randomized SVD as well. Skipping batch method."
#                     continue
#                 end
#             else
#                 @error "Error in batch SVD: $e"
#                 continue
#             end
#         end
#     else
#         @warn "Unknown algorithm: $algo. Skipping."
#         continue
#     end
    
#     @info "Completed $algo algorithm"
# end

# #=====================================================================#
# ## Save the POD basis and singular values
# #=====================================================================#
# if isfile(joinpath(FILEPATH, "data/streaming/basis.jld2"))
#     @info "Loading existing basis file to update"
#     existing_data = load(joinpath(FILEPATH, "data/streaming/basis.jld2"))
#     existing_bases = get(existing_data, "bases", Dict())
#     for (algo, basis) in bases
#         if haskey(existing_bases, algo)
#             @info "Updating existing basis for algorithm $algo"
#             existing_bases[algo].iVr = basis.iVr
#             existing_bases[algo].iΣr = basis.iΣr
#         else
#             @info "Adding new basis for algorithm $algo"
#             existing_bases[algo] = basis
#         end
#     end
#     save(joinpath(FILEPATH, "data/streaming/basis.jld2"), "bases", existing_bases)
# else
#     @info "Creating new basis file"
#     save(joinpath(FILEPATH, "data/streaming/basis.jld2"), "bases", bases)
# end

# #============================================================#
# ## Save the runtime of the algorithms
# #============================================================#
# save(joinpath(FILEPATH, "data/streaming/basis_runtime.jld2"), execution_times)
# @info "Saved bases for algorithms: $(collect(keys(bases)))"

# #================================#
# ## Compute the projection errors
# #================================#
# rspan = 100:100:rmax

# # Preallocate the dict with per-field errors:
# proj_error = Dict(
#     "baker" => Dict(
#         "u" => zeros(length(rspan)),
#         "v" => zeros(length(rspan)),
#         "w" => zeros(length(rspan)),
#         "p" => zeros(length(rspan)),
#         "total" => zeros(length(rspan))  # Keep total for comparison
#     )
# )

# # Loop over all ranks
# for (i, r) in enumerate(rspan)
#     # Number of threads
#     nt = Threads.nthreads()

#     # Each thread writes into one slot of these arrays (per field):
#     error_per_thread = Dict(
#         "u" => zeros(nt),
#         "v" => zeros(nt), 
#         "w" => zeros(nt),
#         "p" => zeros(nt),
#         "total" => zeros(nt)
#     )
#     norm_per_thread = Dict(
#         "u" => zeros(nt),
#         "v" => zeros(nt),
#         "w" => zeros(nt), 
#         "p" => zeros(nt),
#         "total" => zeros(nt)
#     )

#     Threads.@threads for j in 1:n
#         tid = Threads.threadid()

#         # Extract full snapshot X = ds[j] - xbar
#         X_full = scale(ds[j] .- xbar, dim_per_field, scale_factors)

#         # Compute the Baker basis projector: Vr = bases["baker"].iVr[:, 1:r]
#         Vr = @view bases["baker"].iVr[:, 1:r]

#         # Project full snapshot
#         PX_full = Vr * (Vr' * X_full)

#         # Extract each field and compute individual errors
#         for (field_idx, field_name) in enumerate(ds.fields)
#             # Extract field data
#             start_idx = (field_idx - 1) * dim_per_field + 1
#             end_idx = field_idx * dim_per_field
            
#             X_field = @view X_full[start_idx:end_idx]
#             PX_field = @view PX_full[start_idx:end_idx]

#             # Accumulate field-specific errors
#             error_per_thread[field_name][tid] += norm(X_field .- PX_field, 2)
#             norm_per_thread[field_name][tid] += norm(X_field, 2)
#         end

#         # Also compute total error for comparison
#         error_per_thread["total"][tid] += norm(X_full .- PX_full, 2)
#         norm_per_thread["total"][tid] += norm(X_full, 2)
#     end

#     # Reduce across threads for each field
#     for field_name in [ds.fields; "total"]
#         total_error = sum(error_per_thread[field_name])
#         total_norm = sum(norm_per_thread[field_name])
        
#         proj_error["baker"][field_name][i] = total_error / total_norm
#         @info "Projection error for baker at rank $r, field $field_name: $(proj_error["baker"][field_name][i])"
#     end
# end

# # Save to disk:
# save(joinpath(FILEPATH, "data/projection_errors.jld2"), proj_error)

#================================#
## Field-wise Baker iSVD with Multi-processing
#================================#
@info "Computing field-wise Baker iSVD with multi-processing..."

using Distributed

# Add worker processes if not already added
if nprocs() == 1
    addprocs(4)  # Add 4 worker processes, adjust based on your system
end

# Load required packages on all workers
@everywhere using IncrementalSVD
@everywhere using LinearAlgebra
@everywhere using ProgressMeter
@everywhere using HDF5
@everywhere using FileIO
@everywhere using JLD2

# Load the datasource module on all workers
@everywhere include(joinpath(@__DIR__, "datasource.jl"))

## Parameters
field_rank = 200  # Rank for each field
field_names = ds.fields  # ["u", "v", "w", "p"]
# field_names = ["u", "v", "w"]
N = dim_per_field

@info "Field-wise Baker iSVD parameters:"
@info "  Fields: $field_names"
@info "  Rank per field: $field_rank"
@info "  Total snapshots: $n"

## Function to compute iSVD for a single field
@everywhere function compute_field_isvd(field_idx, field_name, datafile, n, xbar, 
                                        dPdx, field_rank, N, algo, scalings)

    @info "Worker $(myid()): Computing iSVD for field $field_name (index $field_idx)"

    # Create datasource on worker
    ds_worker = ChannelDataSource(datafile, ["z", "y", "x", "fields", "times"])
    
    # # Apply appropriate scaling
    # if field_name != "p"
    #     # scale_factor = sqrt(dPdx)
    #     scale_factor = 1.0
    # else
    #     # scale_factor = dPdx
    #     scale_factor = 1.0
    # end
    scale_factor = scalings[field_idx]
    
    # Extract and scale first snapshot for initialization
    x1 = ds_worker[field_idx, 1]
    if xbar != 0.0
        field_start = (field_idx - 1) * N + 1
        field_end = field_idx * N
        x1 .-= xbar[field_start:field_end]
    end
    x1 .*= scale_factor
    
    @info "Worker $(myid()): Initializing iSVD for field $field_name"
    
    # Initialize iSVD
    isvd_field = iSVD(x1=x1, algo=algo, max_rank=field_rank, 
                      right_singular_vectors=true)
    
    # Incremental updates
    @info "Worker $(myid()): Running incremental updates for field $field_name"
    for i in 2:n
        # Extract and scale snapshot
        xi = ds_worker[field_idx, i]
        if xbar != 0.0
            field_start = (field_idx - 1) * N + 1
            field_end = field_idx * N
            xi .-= xbar[field_start:field_end]
        end
        xi .*= scale_factor
        
        # Incremental update
        increment!(isvd_field, xi)
        
        # Progress reporting every 1000 snapshots
        if i % 1000 == 0
            @info "Worker $(myid()): Field $field_name completed $i/$n snapshots"
        end
    end
    
    @info "Worker $(myid()): Completed iSVD for field $field_name"
    
    # Return the basis and singular values
    return (
        field_name = field_name,
        Q = isvd_field.Q[:, 1:min(n,field_rank)],
        Σ = isvd_field.Σ[1:min(n,field_rank)],
        W = isvd_field.W[:, 1:min(n,field_rank)],
    )
end

## Run iSVD for each field in parallel
@info "Starting parallel field-wise iSVD computation..."

@time begin
    # Create tasks for each field
    field_tasks = []
    for (field_idx, field_name) in enumerate(field_names)
        task = @spawnat :any compute_field_isvd(
            field_idx, field_name, datafile, n, xbar, 
            dPdx, field_rank, N, :brand1, scale_factors)
        push!(field_tasks, task)
    end
    
    # Wait for all tasks to complete and collect results
    field_results = [fetch(task) for task in field_tasks]
end

@info "Completed parallel field-wise iSVD computation"

# Construct block-diagonal basis matrix
@info "Constructing block-diagonal POD basis..."

total_dim = length(field_names) * N
total_rank = length(field_names) * field_rank

## Create block-diagonal basis matrix using BlockDiagonals.jl
# Sort field results by field names in the order ["u", "v", "w", "p"]
using BlockDiagonals
field_order = field_names
sorted_field_results = []

for field_name in field_order
    # Find the result for this field
    field_result = findfirst(r -> r.field_name == field_name, field_results)
    if field_result !== nothing
        push!(sorted_field_results, field_results[field_result])
    end
end

# Create block-diagonal basis using sorted order
block_diagonal_basis = BlockDiagonal(
    [result.Q for result in sorted_field_results]
)

# Create concatenated singular values in sorted order
block_diagonal_singular_values = vcat([result.Σ for result in sorted_field_results]...)

@info "Created block-diagonal basis using BlockDiagonals.jl with field order: $(field_order)"
@info "Block-diagonal basis size: $(size(block_diagonal_basis))"
@info "Number of singular values: $(length(block_diagonal_singular_values))"

## Store the field-wise results
# Initialize bases dictionary
bases = Dict()
bases["baker_fieldwise"] = (
    iVr = block_diagonal_basis,
    iΣr = block_diagonal_singular_values,
    # field_results = field_results  # Keep individual field results for analysis
)

@info "Block-diagonal basis construction complete"
@info "Total basis dimensions: $(size(block_diagonal_basis))"
@info "Singular values per field:"
for result in field_results
    @info "  $(result.field_name): $(result.Σ[1:min(5, length(result.Σ))])"
end

# ## Merge the singular subspaces into a single one 
# # Merge the singular subspaces into a single one efficiently
# @info "Merging field-wise bases into unified subspace..."

# # Pre-allocate matrices with exact sizes
# Z = Matrix{Float64}(undef, length(field_names) * field_rank, n)
# Vmerge = Matrix{Float64}(undef, total_dim, length(field_names) * field_rank)

# # Build Z matrix and Vmerge simultaneously with fewer allocations
# Threads.@threads for i in eachindex(sorted_field_results)
#     result = sorted_field_results[i]
    
#     # Calculate indices once
#     z_start = (i - 1) * field_rank + 1
#     z_end = i * field_rank
#     v_start = (i - 1) * N + 1
#     v_end = i * N
    
#     # Use views and BLAS operations for efficiency
#     mul!(view(Z, z_start:z_end, :), 
#          Diagonal(@view result.Σ[1:field_rank]), 
#          (@view result.W[:, 1:field_rank])')
    
#     # Store Q block in Vmerge for later use
#     Vmerge[v_start:v_end, z_start:z_end] .= @view result.Q[:, 1:field_rank]
# end

# # Compute SVD of the much smaller Z matrix
# PP, SS, JJ = svd(Z)

# # Final transformation: Vmerge = Q_blocks * PP efficiently
# temp_Vmerge = copy(Vmerge)
# mul!(Vmerge, temp_Vmerge, PP)
# Σmerge = SS

## Save field-wise results
if isfile(joinpath(FILEPATH, "data/streaming/basis_fieldwise.jld2"))
    # Load existing file to update
    @info "Loading existing field-wise basis file"
    existing_data = load(joinpath(FILEPATH, "data/streaming/basis_fieldwise.jld2"))
    existing_bases = get(existing_data, "bases", Dict())
    for (field_name, basis) in bases
        if haskey(existing_bases, field_name)
            @info "Updating existing basis for field $field_name"
            existing_bases[field_name].iVr = basis.iVr
            existing_bases[field_name].iΣr = basis.iΣr
        else
            @info "Adding new basis for field $field_name"
            existing_bases[field_name] = basis
        end
    end
    existing_data["field_results"] = field_results
    existing_data["parameters"] = Dict("field_rank" => field_rank, "total_rank" => total_rank)
    save(joinpath(FILEPATH, "data/streaming/basis_fieldwise.jld2"), existing_data)
else
    @info "Creating new field-wise basis file"
    save(joinpath(FILEPATH, "data/streaming/basis_fieldwise.jld2"), 
        "bases", bases,
        # "merged_basis", (iVr = Vmerge, iΣr = Σmerge),
        # "iVr", block_diagonal_basis,
        # "iΣr", block_diagonal_singular_values,
        "field_results", field_results,
        "parameters", Dict("field_rank" => field_rank, "total_rank" => total_rank))
end

@info "Saved field-wise Baker iSVD results"

#================================#
## Compute projection errors for field-wise basis
#================================#
@info "Computing projection errors for field-wise basis..."

proj_error_file = joinpath(FILEPATH, "data/streaming/proj_error_fieldwise.jld2")
if isfile(proj_error_file)
    @info "Loading existing projection errors from file"
    proj_error = load(proj_error, "proj_error")
else
    @info "Initializing new projection error dictionary"
    proj_error = Dict()
end

##
basis_file = joinpath(FILEPATH, "data/streaming/basis_fieldwise.jld2")
field_results = load(basis_file)["field_results"]

##
rmax = field_rank
iVr_u = field_results[1].Q[:, 1:rmax]  # u-component basis
iVr_v = field_results[2].Q[:, 1:rmax]  # v-component basis
iVr_w = field_results[3].Q[:, 1:rmax]  # w-component basis
iVr_p = field_results[4].Q[:, 1:rmax]  # p-component basis
field_results = nothing  # Free memory

# Test different ranks (multiples of field_rank up to total available)
# fieldwise_rspan = field_rank:field_rank:(length(field_names) * field_rank)
fieldwise_rspan = [50, 100, 150, 200]

# Add field-wise to projection error dictionary
proj_error["baker_fieldwise"] = Dict(
    "u"     => zeros(length(fieldwise_rspan)),
    "v"     => zeros(length(fieldwise_rspan)),
    "w"     => zeros(length(fieldwise_rspan)),
    "p"     => zeros(length(fieldwise_rspan)),
    "total" => zeros(length(fieldwise_rspan)),
)

##
error_per_thread = nothing
norm_per_thread = nothing

##
for (i, r) in enumerate(fieldwise_rspan)
    iVr_u_tmp = @view iVr_u[:, 1:r]
    iVr_v_tmp = @view iVr_v[:, 1:r]
    iVr_w_tmp = @view iVr_w[:, 1:r]
    iVr_p_tmp = @view iVr_p[:, 1:r]
    iVr_tmp = BlockDiagonal([ iVr_u_tmp, iVr_v_tmp, iVr_w_tmp, iVr_p_tmp ])

    # if r <= size(bases["baker_fieldwise"].iVr, 2)
    if r <= size(iVr_u, 2)
        # Number of threads
        nt = Threads.nthreads()

        # Each thread writes into one slot of these arrays (per field):
        error_per_thread = Dict(
            "u" => zeros(nt),
            "v" => zeros(nt), 
            "w" => zeros(nt),
            "p" => zeros(nt),
            "total" => zeros(nt),
            "merge" => zeros(nt) 
        )
        norm_per_thread = Dict(
            "u" => zeros(nt),
            "v" => zeros(nt),
            "w" => zeros(nt), 
            "p" => zeros(nt),
            "total" => zeros(nt),
            "merge" => zeros(nt)
        )

        Threads.@threads for j in 1:n
            tid = Threads.threadid()

            # Extract full snapshot X = ds[j] - xbar
            X_full = ds[j]

            # Shift and scale the full snapshot
            X_full_ss = scale(X_full .- xbar, dim_per_field, scale_factors)

            # Compute the field-wise basis projector
            # Vr = @view bases["baker_fieldwise"].iVr[:, 1:r]

            # Merged basis
            # Vmerge_r = @view Vmerge[:, 1:r]
            # PX_merge = Vmerge_r * (Vmerge_r' l* X_full)

            # Project full snapshot
            PX_full = iVr_tmp * (iVr_tmp' * X_full_ss)
            X_full_ss = nothing

            # Unscale and unshift the projected data
            PX_full = unscale(PX_full, dim_per_field, scale_factors) .+ xbar

            # Extract each field and compute individual errors
            for (field_idx, field_name) in enumerate(field_names)
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
            tmp = norm(X_full, 2)
            norm_per_thread["total"][tid] += tmp

            # Compute merged basis error
            # error_per_thread["merge"][tid] += norm(X_full .- PX_merge, 2)
            # norm_per_thread["merge"][tid] += tmp
        end

        # Reduce across threads for each field
        for field_name in [ds.fields; "total"]
            total_error = sum(error_per_thread[field_name])
            total_norm = sum(norm_per_thread[field_name])
            
            # Map to the correct index in the original rspan if needed
            # if r == field_rank * length(field_names)  # Full rank case
            #     proj_error["baker_fieldwise"][field_name] = total_error / total_norm
            # end

            proj_error["baker_fieldwise"][field_name][i] = total_error / total_norm
            
            @info "Projection error for baker_fieldwise at rank $r, field $field_name: $(total_error / total_norm)"
        end
    end
end

##
# Update saves to include field-wise results
# save(joinpath(FILEPATH, "data/streaming/basis.jld2"), bases)
save(proj_error_file, proj_error)

@info "Field-wise Baker iSVD projection error analysis complete"

# #================================#
# ## Create visualization comparing field-wise vs global approaches
# #================================#
# using CairoMakie

# fig = Figure(size=(1200, 800))

# # Plot singular values comparison
# ax1 = Axis(fig[1,1], yscale=log10, xlabel="Mode Index", ylabel="Singular Value", 
#           title="Singular Values: Global vs Field-wise")

# # Global Baker results
# if haskey(bases, "baker")
#     scatterlines!(ax1, 1:length(bases["baker"].iΣr), bases["baker"].iΣr, 
#                  markersize=3, color=:blue, label="Global Baker")
# end

# # Field-wise results - plot each field separately
# colors = [:red, :green, :orange, :purple]
# for (i, result) in enumerate(field_results)
#     offset = (i-1) * field_rank
#     scatterlines!(ax1, (offset+1):(offset+field_rank), result.Σ, 
#                  markersize=3, color=colors[i], label="Field $(result.field_name)")
# end

# axislegend(ax1)

# # Plot projection errors if available
# if haskey(proj_error, "baker") && haskey(proj_error, "baker_fieldwise")
#     ax2 = Axis(fig[2,1], yscale=log10, xlabel="Rank", ylabel="Projection Error", 
#               title="Projection Error Comparison")
    
#     # Plot global Baker errors
#     lines!(ax2, rspan, proj_error["baker"]["total"], color=:blue, linewidth=2, label="Global Baker")
    
#     # Plot field-wise errors
#     lines!(ax2, [field_rank * length(field_names)], [proj_error["baker_fieldwise"]["total"][end]], 
#            color=:red, marker=:circle, markersize=8, label="Field-wise Baker")
    
#     axislegend(ax2)
# end

# display(fig)
# save(joinpath(FILEPATH, "plots/fieldwise_vs_global_comparison.png"), fig)

# @info "Visualization complete"

# #================================#
# ## Two-level Block-wise SVD with left_ssm and right_ssm merging
# #================================#
# @info "Computing two-level block-wise SVD with left_ssm and right_ssm merging..."

# # Parameters
# Nrow = 128     # Number of row blocks
# Ncol = 20     # Number of column blocks per row
# target_rank = 1000   # Target rank for each small block SVD
# intermediate_rank = 1000  # Rank after column merging (per row block)
# final_rank = 1000   # Final merged rank after row merging

# # Field dimensions
# field_names = ds.fields
# N = dim_per_field
# total_fields = length(field_names)
# total_rows = N * total_fields
# total_cols = length(ds)

# @info "Two-level Block SVD parameters:"
# @info "  Row blocks: $Nrow, Column blocks per row: $Ncol"
# @info "  Target rank per block: $target_rank"
# @info "  Intermediate rank per row: $intermediate_rank" 
# @info "  Final rank: $final_rank"
# @info "  Total matrix size: $total_rows × $total_cols"

# # Function to compute SVD for a single block
# function compute_small_block_svd(row_range, col_range, target_rank)
#     @info "Computing SVD for rows $(first(row_range)):$(last(row_range)), cols $(first(col_range)):$(last(col_range))"
    
#     # Determine which fields are involved
#     field_row_starts = [1; cumsum([N for _ in 1:total_fields-1]) .+ 1]
#     field_row_ends = cumsum([N for _ in 1:total_fields])
    
#     # Extract block data efficiently using the new indexing
#     block_data = zeros(length(row_range), length(col_range))
#     current_row = 1
    
#     for (field_idx, field_name) in enumerate(field_names)
#         field_start = field_row_starts[field_idx]
#         field_end = field_row_ends[field_idx]
        
#         # Check if this field overlaps with our row range
#         if field_start <= last(row_range) && field_end >= first(row_range)
#             # Calculate overlap
#             local_start = max(1, first(row_range) - field_start + 1)
#             local_end = min(N, last(row_range) - field_start + 1)
            
#             if local_start <= local_end
#                 local_range = local_start:local_end
                
#                 # Extract field data for the column range using new efficient indexing
#                 field_data = ds[field_idx, col_range][local_range, :]
                
#                 # Apply scaling
#                 if field_name != "p"
#                     field_data .*= sqrt(dPdx)
#                 else
#                     field_data .*= dPdx
#                 end
                
#                 # Subtract mean if applicable
#                 if xbar != 0.0
#                     global_indices = (field_idx-1)*N .+ local_range
#                     mean_block = xbar[global_indices]
#                     field_data .-= mean_block
#                 end
                
#                 # Place in block_data
#                 rows_to_fill = length(local_range)
#                 block_data[current_row:(current_row + rows_to_fill - 1), :] = field_data
#                 current_row += rows_to_fill
#             end
#         end
#     end
    
#     @info "Block data size: $(size(block_data))"
    
#     # Compute SVD with rank truncation
#     try
#         F = svd(block_data)
#         k = min(target_rank, length(F.S), size(F.U, 2), size(F.Vt, 1))
        
#         @info "SVD computed, truncating to rank $k (from $(length(F.S)) singular values)"
        
#         result = (U=F.U[:, 1:k], S=F.S[1:k], Vt=F.Vt[1:k, :])
        
#         # Clear memory
#         block_data = nothing
#         F = nothing
#         GC.gc()
        
#         return result
#     catch e
#         @error "SVD failed for block rows $(first(row_range)):$(last(row_range)), cols $(first(col_range)):$(last(col_range)): $e"
#         rethrow(e)
#     end
# end

# ## Main two-level processing
# @time begin
#     # Storage for row block SVDs
#     row_block_svds = Vector{NamedTuple}(undef, Nrow)
    
#     # Process each row block
#     for row_block in 1:Nrow
#         @info "Processing row block $row_block of $Nrow"
        
#         # Calculate row range for this block
#         row_block_size = ceil(Int, total_rows / Nrow)
#         row_start = (row_block - 1) * row_block_size + 1
#         row_end = min(row_block * row_block_size, total_rows)
#         row_range = row_start:row_end
        
#         @info "Row block $row_block covers rows $row_start:$row_end"
        
#         # Initialize for column merging within this row block
#         row_merged_U = Matrix{Float64}(undef, 0, 0)
#         row_merged_S = Float64[]
#         row_merged_Vt = Matrix{Float64}(undef, 0, 0)
#         is_first_col_block = true
        
#         # Process column blocks sequentially and merge with left_ssm!
#         for col_block in 1:Ncol
#             @info "  Processing column block $col_block of $Ncol for row block $row_block"
            
#             # Calculate column range for this block
#             col_block_size = ceil(Int, total_cols / Ncol)
#             col_start = (col_block - 1) * col_block_size + 1
#             col_end = min(col_block * col_block_size, total_cols)
#             col_range = col_start:col_end
            
#             try
#                 # Compute SVD for this small block
#                 block_svd = compute_small_block_svd(row_range, col_range, target_rank)
                
#                 if is_first_col_block
#                     # Initialize with first column block
#                     row_merged_U = copy(block_svd.U)
#                     row_merged_S = copy(block_svd.S)
#                     row_merged_Vt = copy(block_svd.Vt)
#                     is_first_col_block = false
#                     @info "  Initialized row block $row_block with first column block: $(length(row_merged_S)) singular values"
#                 else
#                     # Merge with existing result using left_ssm!
#                     @info "  Merging column block $col_block with existing row block result..."
#                     @info "  Current row merged: $(length(row_merged_S)) singular values"
#                     @info "  Block: $(length(block_svd.S)) singular values"
                    
#                     try
#                         max_rank = min(intermediate_rank, length(row_merged_S) + length(block_svd.S))
                        
#                         # Use left_ssm! for column merging
#                         merged_result = left_ssm!(
#                             max_rank,
#                             row_merged_U, block_svd.U,
#                             row_merged_S, block_svd.S,
#                             row_merged_Vt', block_svd.Vt';
#                             γ=1.0,
#                             right_singular_vectors=true
#                         )
                        
#                         # Update merged components
#                         row_merged_U = merged_result.U
#                         row_merged_S = merged_result.S
#                         row_merged_Vt = merged_result.Vt'
                        
#                         @info "  Column merge successful. New row merged: $(length(row_merged_S)) singular values"
                        
#                     catch merge_error
#                         @error "  Column merge failed for row block $row_block, col block $col_block: $merge_error"
#                         rethrow(merge_error)
#                     end
#                 end
                
#                 # Clear block data
#                 block_svd = nothing
#                 GC.gc()
                
#             catch e
#                 @error "Failed to process row block $row_block, col block $col_block: $e"
#                 @info "Continuing with next column block..."
#                 continue
#             end
#         end
        
#         # Store the merged result for this row block
#         row_block_svds[row_block] = (U=row_merged_U, S=row_merged_S, Vt=row_merged_Vt)
#         @info "Completed row block $row_block with $(length(row_merged_S)) singular values"
        
#         # Clear row block data
#         row_merged_U = nothing
#         row_merged_S = nothing  
#         row_merged_Vt = nothing
#         GC.gc()
#     end
    
#     @info "Completed all row blocks. Now merging row blocks with right_ssm!..."
    
#     # Now merge all row blocks using right_ssm!
#     final_merged_S = Float64[]
#     final_merged_Vt = Matrix{Float64}(undef, 0, 0)
#     is_first_row_block = true
    
#     for row_block in 1:Nrow
#         @info "Merging row block $row_block into final result..."
        
#         row_svd = row_block_svds[row_block]
        
#         if is_first_row_block
#             # Initialize with first row block
#             final_merged_S = copy(row_svd.S)
#             final_merged_Vt = copy(row_svd.Vt)
#             is_first_row_block = false
#             @info "Initialized final result with row block 1: $(length(final_merged_S)) singular values"
#         else
#             # Merge with existing result using right_ssm!
#             @info "Current final merged: $(length(final_merged_S)) singular values"
#             @info "Row block: $(length(row_svd.S)) singular values"
            
#             try
#                 max_rank = min(final_rank, length(final_merged_S) + length(row_svd.S))
                
#                 # Use right_ssm! for row merging
#                 final_merged_S_new, final_merged_Vt_new = right_ssm!(
#                     max_rank,
#                     final_merged_S, row_svd.S,
#                     final_merged_Vt', row_svd.Vt';
#                     γ=1.0
#                 )
                
#                 # Update final merged components
#                 final_merged_S = final_merged_S_new
#                 final_merged_Vt = final_merged_Vt_new'
                
#                 @info "Row merge successful. New final merged: $(length(final_merged_S)) singular values"
#                 @info "Spectral decay check - first 5 σ: $(final_merged_S[1:min(5, length(final_merged_S))])"
                
#             catch merge_error
#                 @error "Row merge failed for row block $row_block: $merge_error"
#                 rethrow(merge_error)
#             end
#         end
        
#         # Clear row block data
#         # row_block_svds[row_block] = nothing
#         # GC.gc()
#     end
    
#     @info "Completed two-level merging!"
#     @info "Final number of singular values: $(length(final_merged_S))"
#     @info "Spectral decay - first 20 σ: $(final_merged_S[1:min(20, length(final_merged_S))])"
#     @info "Spectral decay - last 20 σ: $(final_merged_S[max(1, end-19):end])"
# end

# # Store the results
# two_level_singular_values = final_merged_S[1:min(final_rank, length(final_merged_S))]

# ## Save the singular values for analysis
# save(joinpath(FILEPATH, "data/streaming/two_level_block_singular_values.jld2"), 
#      "singular_values", two_level_singular_values,
#      "total_computed", length(final_merged_S),
#      "parameters", Dict("Nrow" => Nrow, "Ncol" => Ncol, "target_rank" => target_rank, 
#                        "intermediate_rank" => intermediate_rank, "final_rank" => final_rank))

# @info "Two-level block-wise SVD computation complete."
# @info "Computed $(length(final_merged_S)) singular values, saved $(length(two_level_singular_values)) for analysis"

# ## Create visualization comparing approaches
# using CairoMakie 
# fig = Figure(size=(1000, 600))

# # Plot both results if available
# ax = Axis(fig[1,1], yscale=log10, xlabel="Singular Value Index", ylabel="Singular Value", 
#          title="Spectral Decay Comparison")

# scatterlines!(ax, 1:length(two_level_singular_values), two_level_singular_values, 
#          markersize=2, color=:red, label="Two-level Block SVD")

# # Add single-level results if they exist
# if @isdefined block_merged_singular_values
#     scatterlines!(ax, 1:min(length(block_merged_singular_values), length(two_level_singular_values)), 
#              block_merged_singular_values[1:min(length(block_merged_singular_values), length(two_level_singular_values))], 
#              markersize=2, color=:blue, label="Single-level Block SVD")
# end

# axislegend(ax)
# display(fig)

# ## Save the comparison plot
# save(joinpath(FILEPATH, "plots/spectral_decay_comparison.png"), fig)

# @info "Two-level block-wise SVD analysis complete"