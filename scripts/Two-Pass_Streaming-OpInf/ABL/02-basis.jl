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
DATAPATH = "../../../../../DATA/NREL/ABL"
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"Two-Pass_Streaming-OpInf/ABL") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/ABL")
fn = "ABL_0_10000.h5"
datafile = joinpath(DATAPATH, fn)

#==========================================#
## Load struct to read data in HDF5 format 
#==========================================#
include(joinpath(FILEPATH, "datasource.jl"))

#========================#
## Additional functions
#========================#
include(joinpath(FILEPATH, "preprocess.jl"))

#=============================#
## Load the training dataset
#=============================#
ds = ChannelDataSource(datafile, ["z", "y", "x", "fields", "times"])
nz, ny, nx, n_fields, n = ds.dims
nxyz = nz * ny * nx
n_test = 2000
n_train = n - n_test

#=============================#
## Load the mean and scalings
#=============================#
means  = load(joinpath(FILEPATH, "data/mean.jld2"))["xbar"]
shifts = load(joinpath(FILEPATH, "data/minmax.jld2"))["minmax"]["shifts"]
scales = load(joinpath(FILEPATH, "data/minmax.jld2"))["minmax"]["scales"]

#============================================================#
## Generate the POD basis using specified algorithms
#============================================================#
# Specify which algorithms to run
algorithms = ["baker"]  # Can be extended to ["baker", "brand", "sketchy", "batch"]

# Settings
rmax = 400
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
            x1=preprocess!(ds[1], means, shifts, scales),
            algo=:baker, max_rank=rmax, right_singular_vectors=true)
        push!(time_data, tmp)
        
        # Incremental updates
        @showprogress for i in 2:n_train
            tmp = @elapsed increment!(
                baker, preprocess!(ds[i], means, shifts, scales)
            )
            push!(time_data, tmp)
        end
        
        # Store results
        bases[algo] = (iVr=baker.Q[:,1:rmax], iΣr=baker.Σ[1:rmax],
                       iW=baker.W[:,1:rmax])
        execution_times[algo] = reduce(vcat, time_data)
        
    elseif algo == "brand"
        time_data = []

        # Run once with dummy data for JIT compilation
        Xdummy = rand(30, 200)
        brand = iSVD(x1=Xdummy[:,1], algo=:brand1, reorth_method=:qr, max_rank=4)
        full_increment!(brand, Xdummy, verbose=true, tol=1e-10, runtime=true)
        
        # Initialize
        tmp = @elapsed brand = iSVD(
            x1=preprocess!(ds[1], means, shifts, scales), 
            algo=:brand1, reorth_method=:gramschmidt, max_rank=rmax,
            right_singular_vectors=true)
        push!(time_data, tmp)
        
        # Incremental updates
        @showprogress for i in 2:n_train
            tmp = @elapsed increment!(
                brand, preprocess!(ds[i], means, shifts, scales), tol=1e-10)
            push!(time_data, tmp)
        end
        
        # Store results
        bases[algo] = (iVr=brand.Q[:,1:rmax], iΣr=brand.Σ[1:rmax],
                       iW=brand.W[:,1:rmax])
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
        X = spzeros(Nz*Ny*Nx*3, n_train)
        @showprogress for i in 1:(n_train ÷ 10)
            idx = 10*(i-1)+1:10*i
            X[:,idx] .= [preprocess!(ds[j], means, shifts, scales) for j in idx]
            sketchy.X .+= sketchy.Ξ * X
            sketchy.Y .+= X * sketchy.Ω'
            sketchy.Z .+= (sketchy.Φ * X) * sketchy.Ψ'
            push!(time_data, tmp)
            fill!(X, 0)
            dropzeros!(X)
        end
        IncrementalSVD.terminate!(sketchy, false, false)
        
        # Store results
        bases[algo] = (iVr=sketchy.Q[:,1:rmax], iΣr=sketchy.Σ[1:rmax],
                       iW=sketchy.W[:,1:rmax])
        execution_times[algo] = reduce(vcat, time_data)
        
    elseif algo == "batch"
        @info "Running batch SVD..."

        # Run once with dummy data for JIT compilation
        Xdummy = rand(30, 200)
        svd(Xdummy)

        try
            time_batch = @elapsed F = svd(
                [
                    preprocess!(ds[j], means, shifts, scales) 
                    for j in 1:n
                ]
            )
            bases[algo] = (Vr=F.U[:,1:rmax], Σr=F.S[1:rmax])
            execution_times[algo] = [time_batch]
        catch e
            if isa(e, OutOfMemoryError)
                @error "Out of memory error during SVD computation. Using randomized SVD."
                try
                    time_batch = @elapsed F = rsvd(
                        [
                            preprocess!(ds[j], means, shifts, scales) 
                            for j in 1:n
                        ], 
                        rmax, p=10
                    )
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
if isfile(joinpath(FILEPATH, "data/bases/basis.jld2"))
    @info "Loading existing basis file to update"
    existing_data = load(joinpath(FILEPATH, "data/bases/basis.jld2"))
    existing_bases = get(existing_data, "bases", Dict())
    for (algo, basis) in bases
        if haskey(existing_bases, algo)
            @info "Updating existing basis for algorithm $algo"
            existing_bases[algo].iVr = basis.iVr
            existing_bases[algo].iΣr = basis.iΣr
        else
            @info "Adding new basis for algorithm $algo"
            existing_bases[algo] = basis
        end
    end
    save(joinpath(FILEPATH, "data/bases/basis.jld2"), "bases", existing_bases)
else
    @info "Creating new basis file"
    save(joinpath(FILEPATH, "data/bases/basis.jld2"), "bases", bases)
end

# #============================================================#
# ## Save the runtime of the algorithms
# #============================================================#
# save(joinpath(FILEPATH, "data/streaming/basis_runtime.jld2"), execution_times)
# @info "Saved bases for algorithms: $(collect(keys(bases)))"

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
proj_error_proc = Dict(
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
    error_per_thread_proc = Dict(
        "u" => zeros(nt),
        "v" => zeros(nt), 
        "w" => zeros(nt),
        "p" => zeros(nt),
        "total" => zeros(nt)
    )
    norm_per_thread_proc = Dict(
        "u" => zeros(nt),
        "v" => zeros(nt),
        "w" => zeros(nt), 
        "p" => zeros(nt),
        "total" => zeros(nt)
    )

    Threads.@threads for j in 1:n
        tid = Threads.threadid()

        # Extract full snapshot X = ds[j] - xbar
        X_full = ds[j]
        X_full_proc = preprocess!(ds[j], means, shifts, scales)

        # Compute the Baker basis projector: Vr = bases["baker"].iVr[:, 1:r]
        Vr = @view bases["baker"].iVr[:, 1:r]

        # Project full snapshot
        PX_full_proc = Vr * (Vr' * X_full_proc)
        PX_full = unprocess!(PX_full_proc, means, shifts, scales)

        # Extract each field and compute individual errors
        for (field_idx, field_name) in enumerate(ds.fields)
            # Extract field data
            start_idx = (field_idx - 1) * nxyz + 1
            end_idx = field_idx * nxyz
            
            X_field = @view X_full[start_idx:end_idx]
            PX_field = @view PX_full[start_idx:end_idx]

            X_field_proc = @view X_full_proc[start_idx:end_idx]
            PX_field_proc = @view PX_full_proc[start_idx:end_idx]

            # Accumulate field-specific errors
            error_per_thread[field_name][tid] += norm(X_field .- PX_field, 2)
            norm_per_thread[field_name][tid] += norm(X_field, 2)
            error_per_thread_proc[field_name][tid] += norm(X_field_proc .- PX_field_proc, 2)
            norm_per_thread_proc[field_name][tid] += norm(X_field_proc, 2)
        end

        # Also compute total error for comparison
        error_per_thread["total"][tid] += norm(X_full .- PX_full, 2)
        norm_per_thread["total"][tid] += norm(X_full, 2)
        error_per_thread_proc["total"][tid] += norm(X_full_proc .- PX_full_proc, 2)
        norm_per_thread_proc["total"][tid] += norm(X_full_proc, 2)
    end

    # Reduce across threads for each field
    for field_name in [ds.fields; "total"]
        total_error = sum(error_per_thread[field_name])
        total_norm = sum(norm_per_thread[field_name])

        total_error_proc = sum(error_per_thread_proc[field_name])
        total_norm_proc = sum(norm_per_thread_proc[field_name])
        
        proj_error["baker"][field_name][i] = total_error / total_norm
        proj_error_proc["baker"][field_name][i] = total_error_proc / total_norm_proc
        @info "Projection error for original data at rank $r, field $field_name: \
                         $(proj_error["baker"][field_name][i])"
        @info "Projection error for preprocessed data at rank $r, field $field_name: \
                         $(proj_error_proc["baker"][field_name][i])"
    end
end

# Save to disk:
save(joinpath(FILEPATH, "data/bases/projection_errors.jld2"), 
     "proj_error", proj_error, "proj_error_proc", proj_error_proc)

# #==============================================#
# ## Field-wise Baker iSVD with Multi-processing
# #==============================================#
# @info "Computing field-wise Baker iSVD with multi-processing..."

# using Distributed

# # Add worker processes if not already added
# if nprocs() == 1
#     addprocs(4)  # Add 4 worker processes, adjust based on your system
# end

# # Load required packages on all workers
# @everywhere using IncrementalSVD
# @everywhere using LinearAlgebra
# @everywhere using ProgressMeter
# @everywhere using HDF5
# @everywhere using FileIO
# @everywhere using JLD2

# # Load the datasource module on all workers
# @everywhere include(joinpath(@__DIR__, "datasource.jl"))
# @everywhere include(joinpath(@__DIR__, "preprocess.jl"))

# ## Parameters
# field_rank = 200  # Rank for each field
# field_names = ds.fields  # ["u", "v", "w", "p"]
# # field_names = ["u", "v", "w"]
# N = nxyz

# @info "Field-wise Baker iSVD parameters:"
# @info "  Fields: $field_names"
# @info "  Rank per field: $field_rank"
# @info "  Total snapshots: $n"

# ## Function to compute iSVD for a single field
# @everywhere function compute_field_isvd(field_idx, field_name, datafile, n, means, 
#                                         dPdx, field_rank, N, algo, shifts, scales)

#     @info "Worker $(myid()): Computing iSVD for field $field_name (index $field_idx)"

#     # Create datasource on worker
#     ds_worker = ChannelDataSource(datafile, ["z", "y", "x", "fields", "times"])
#     scale_factor = scalings[field_idx]
    
#     # Extract and scale first snapshot for initialization
#     x1 = ds_worker[field_idx, 1]
#     x1 = preprocess!(x1, means, shifts, scales)
    
#     @info "Worker $(myid()): Initializing iSVD for field $field_name"
    
#     # Initialize iSVD
#     isvd_field = iSVD(x1=x1, algo=algo, max_rank=field_rank, 
#                       right_singular_vectors=true)
    
#     # Incremental updates
#     @info "Worker $(myid()): Running incremental updates for field $field_name"
#     for i in 2:n
#         # Extract and scale snapshot
#         xi = ds_worker[field_idx, i]
#         xi = preprocess!(xi, means, shifts, scales)
        
#         # Incremental update
#         increment!(isvd_field, xi)
        
#         # Progress reporting every 1000 snapshots
#         if i % 1000 == 0
#             @info "Worker $(myid()): Field $field_name completed $i/$n snapshots"
#         end
#     end
    
#     @info "Worker $(myid()): Completed iSVD for field $field_name"
    
#     # Return the basis and singular values
#     return (
#         field_name = field_name,
#         Q = isvd_field.Q[:, 1:min(n,field_rank)],
#         Σ = isvd_field.Σ[1:min(n,field_rank)],
#         W = isvd_field.W[:, 1:min(n,field_rank)],
#     )
# end

# ## Run iSVD for each field in parallel
# @info "Starting parallel field-wise iSVD computation..."

# @time begin
#     # Create tasks for each field
#     field_tasks = []
#     for (field_idx, field_name) in enumerate(field_names)
#         task = @spawnat :any compute_field_isvd(
#             field_idx, field_name, datafile, n, means, 
#             dPdx, field_rank, N, :brand1, shifts, scales)
#         push!(field_tasks, task)
#     end
    
#     # Wait for all tasks to complete and collect results
#     field_results = [fetch(task) for task in field_tasks]
# end

# @info "Completed parallel field-wise iSVD computation"

# # Construct block-diagonal basis matrix
# @info "Constructing block-diagonal POD basis..."

# total_dim = length(field_names) * N
# total_rank = length(field_names) * field_rank

# ## Create block-diagonal basis matrix using BlockDiagonals.jl
# # Sort field results by field names in the order ["u", "v", "w", "p"]
# using BlockDiagonals
# field_order = field_names
# sorted_field_results = []

# for field_name in field_order
#     # Find the result for this field
#     field_result = findfirst(r -> r.field_name == field_name, field_results)
#     if field_result !== nothing
#         push!(sorted_field_results, field_results[field_result])
#     end
# end

# # Create block-diagonal basis using sorted order
# block_diagonal_basis = BlockDiagonal(
#     [result.Q for result in sorted_field_results]
# )

# # Create concatenated singular values in sorted order
# block_diagonal_singular_values = vcat([result.Σ for result in sorted_field_results]...)

# @info "Created block-diagonal basis using BlockDiagonals.jl with field order: $(field_order)"
# @info "Block-diagonal basis size: $(size(block_diagonal_basis))"
# @info "Number of singular values: $(length(block_diagonal_singular_values))"

# ## Store the field-wise results
# # Initialize bases dictionary
# bases = Dict()
# bases["baker_fieldwise"] = (
#     iVr = block_diagonal_basis,
#     iΣr = block_diagonal_singular_values,
#     # field_results = field_results  # Keep individual field results for analysis
# )

# @info "Block-diagonal basis construction complete"
# @info "Total basis dimensions: $(size(block_diagonal_basis))"
# @info "Singular values per field:"
# for result in field_results
#     @info "  $(result.field_name): $(result.Σ[1:min(5, length(result.Σ))])"
# end

# ## Save field-wise results
# if isfile(joinpath(FILEPATH, "data/streaming/basis_fieldwise.jld2"))
#     # Load existing file to update
#     @info "Loading existing field-wise basis file"
#     existing_data = load(joinpath(FILEPATH, "data/streaming/basis_fieldwise.jld2"))
#     existing_bases = get(existing_data, "bases", Dict())
#     for (field_name, basis) in bases
#         if haskey(existing_bases, field_name)
#             @info "Updating existing basis for field $field_name"
#             existing_bases[field_name].iVr = basis.iVr
#             existing_bases[field_name].iΣr = basis.iΣr
#         else
#             @info "Adding new basis for field $field_name"
#             existing_bases[field_name] = basis
#         end
#     end
#     existing_data["field_results"] = field_results
#     existing_data["parameters"] = Dict("field_rank" => field_rank, "total_rank" => total_rank)
#     save(joinpath(FILEPATH, "data/streaming/basis_fieldwise.jld2"), existing_data)
# else
#     @info "Creating new field-wise basis file"
#     save(joinpath(FILEPATH, "data/streaming/basis_fieldwise.jld2"), 
#         "bases", bases,
#         # "merged_basis", (iVr = Vmerge, iΣr = Σmerge),
#         # "iVr", block_diagonal_basis,
#         # "iΣr", block_diagonal_singular_values,
#         "field_results", field_results,
#         "parameters", Dict("field_rank" => field_rank, "total_rank" => total_rank))
# end

# @info "Saved field-wise Baker iSVD results"

# #==================================================#
# ## Compute projection errors for field-wise basis
# #==================================================#
# @info "Computing projection errors for field-wise basis..."

# proj_error_file = joinpath(FILEPATH, "data/streaming/proj_error_fieldwise.jld2")
# if isfile(proj_error_file)
#     @info "Loading existing projection errors from file"
#     proj_error = load(proj_error, "proj_error")
# else
#     @info "Initializing new projection error dictionary"
#     proj_error = Dict()
# end

# ##
# basis_file = joinpath(FILEPATH, "data/streaming/basis_fieldwise.jld2")
# field_results = load(basis_file)["field_results"]

# ##
# rmax = field_rank
# iVr_u = field_results[1].Q[:, 1:rmax]  # u-component basis
# iVr_v = field_results[2].Q[:, 1:rmax]  # v-component basis
# iVr_w = field_results[3].Q[:, 1:rmax]  # w-component basis
# iVr_p = field_results[4].Q[:, 1:rmax]  # p-component basis
# field_results = nothing  # Free memory

# # Test different ranks (multiples of field_rank up to total available)
# # fieldwise_rspan = field_rank:field_rank:(length(field_names) * field_rank)
# fieldwise_rspan = [50, 100, 150, 200]

# # Add field-wise to projection error dictionary
# proj_error["baker_fieldwise"] = Dict(
#     "u"     => zeros(length(fieldwise_rspan)),
#     "v"     => zeros(length(fieldwise_rspan)),
#     "w"     => zeros(length(fieldwise_rspan)),
#     "p"     => zeros(length(fieldwise_rspan)),
#     "total" => zeros(length(fieldwise_rspan)),
# )
# proj_error_proc["baker_fieldwise"] = Dict(
#     "u"     => zeros(length(fieldwise_rspan)),
#     "v"     => zeros(length(fieldwise_rspan)),
#     "w"     => zeros(length(fieldwise_rspan)),
#     "p"     => zeros(length(fieldwise_rspan)),
#     "total" => zeros(length(fieldwise_rspan)),
# )

# ##
# error_per_thread = nothing
# norm_per_thread = nothing
# error_per_thread_proc = nothing
# norm_per_thread_proc = nothing

# ##
# for (i, r) in enumerate(fieldwise_rspan)
#     iVr_u_tmp = @view iVr_u[:, 1:r]
#     iVr_v_tmp = @view iVr_v[:, 1:r]
#     iVr_w_tmp = @view iVr_w[:, 1:r]
#     iVr_p_tmp = @view iVr_p[:, 1:r]
#     iVr_tmp = BlockDiagonal([ iVr_u_tmp, iVr_v_tmp, iVr_w_tmp, iVr_p_tmp ])

#     # if r <= size(bases["baker_fieldwise"].iVr, 2)
#     if r <= size(iVr_u, 2)
#         # Number of threads
#         nt = Threads.nthreads()

#         # Each thread writes into one slot of these arrays (per field):
#         error_per_thread = Dict(
#             "u" => zeros(nt),
#             "v" => zeros(nt), 
#             "w" => zeros(nt),
#             "p" => zeros(nt),
#             "total" => zeros(nt),
#             "merge" => zeros(nt) 
#         )
#         norm_per_thread = Dict(
#             "u" => zeros(nt),
#             "v" => zeros(nt),
#             "w" => zeros(nt), 
#             "p" => zeros(nt),
#             "total" => zeros(nt),
#             "merge" => zeros(nt)
#         )
#         error_per_threadi_proc = Dict(
#             "u" => zeros(nt),
#             "v" => zeros(nt), 
#             "w" => zeros(nt),
#             "p" => zeros(nt),
#             "total" => zeros(nt),
#             "merge" => zeros(nt) 
#         )
#         norm_per_thread_proc = Dict(
#             "u" => zeros(nt),
#             "v" => zeros(nt),
#             "w" => zeros(nt), 
#             "p" => zeros(nt),
#             "total" => zeros(nt),
#             "merge" => zeros(nt)
#         )

#         Threads.@threads for j in 1:n
#             tid = Threads.threadid()

#             # Extract full snapshot X = ds[j] - xbar
#             X_full = ds[j]

#             # preprocess the full snapshot
#             X_full_proc = preprocess!(X_full, means, shifts, scales)

#             # Project full snapshot
#             PX_full_proc = iVr_tmp * (iVr_tmp' * X_full_ss)

#             # Unscale and unshift the projected data
#             PX_full = unprocess!(PX_full_proc, means, shifts, scales)

#             # Extract each field and compute individual errors
#             for (field_idx, field_name) in enumerate(field_names)
#                 # Extract field data
#                 start_idx = (field_idx - 1) * nxyz + 1
#                 end_idx = field_idx * nxyz
                
#                 X_field = @view X_full[start_idx:end_idx]
#                 X_field_proc = @view X_full_proc[start_idx:end_idx]
#                 PX_field = @view PX_full[start_idx:end_idx]
#                 PX_field_proc = @view PX_full_proc[start_idx:end_idx]

#                 # Accumulate field-specific errors
#                 error_per_thread[field_name][tid] += norm(X_field .- PX_field, 2)
#                 norm_per_thread[field_name][tid] += norm(X_field, 2)
#                 error_per_thread_proc[field_name][tid] += norm(X_field_proc .- PX_field_proc, 2)
#                 norm_per_thread_proc[field_name][tid] += norm(X_field_proc, 2)
#             end

#             # Also compute total error for comparison
#             error_per_thread["total"][tid] += norm(X_full .- PX_full, 2)
#             tmp = norm(X_full, 2)
#             norm_per_thread["total"][tid] += tmp
#             error_per_thread_proc["total"][tid] += norm(X_full_proc .- PX_full_proc, 2)
#             tmp = norm(X_full_proc, 2)
#             norm_per_thread_proc["total"][tid] += tmp
#         end

#         # Reduce across threads for each field
#         for field_name in [ds.fields; "total"]
#             total_error = sum(error_per_thread[field_name])
#             total_norm = sum(norm_per_thread[field_name])

#             total_error_proc = sum(error_per_thread_proc[field_name])
#             total_norm_proc = sum(norm_per_thread_proc[field_name])

#             proj_error["baker_fieldwise"][field_name][i] = total_error / total_norm
#             proj_error_proc["baker_fieldwise"][field_name][i] = total_error / total_norm
            
#             @info "Projection error for original data at rank $r, field $field_name: \
#                    $(total_error / total_norm)"
#             @info "Projection error for preprocessed data at rank $r, field $field_name: \
#                    $(total_error_proc / total_norm_proc)"
            
#         end
#     end
# end

# ##
# save(proj_error_file, proj_error)
# @info "Field-wise Baker iSVD projection error analysis complete"