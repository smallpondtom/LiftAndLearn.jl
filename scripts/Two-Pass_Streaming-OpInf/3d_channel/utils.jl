using Logging
using Distributed

function center!(data::Array{Float64}, means::Vector{Float64})
    @assert length(means) == size(data,1) "Number of means must match number of rows"
    data .-= means
    return data
end

function uncenter!(data::Array{Float64}, means::Vector{Float64})
    @assert length(means) == size(data,1) "Number of means must match number of rows"
    data .+= means
    return data
end

function normalize!(data::Array{Float64}, shifts::Vector{Float64}, scales::Vector{Float64})
    rows = size(data, 1)
    @assert length(shifts) == length(scales) "Number of shifts must match number of scales"
    if length(shifts) == rows && length(scales) == rows
        for i in axes(data, 2)
            data[:, i] .-= shifts
            data[:, i] ./= scales
        end
    else
        dim = row ÷ length(shifts)
        for (i, (sh,sc)) in enumerate(zip(shifts, scales))
            data[dim*(i-1)+1:dim*i, :] .-= sh
            data[dim*(i-1)+1:dim*i, :] ./= sc
        end
    end
    return data
end

function unnormalize!(data::Array{Float64}, shifts::Vector{Float64}, scales::Vector{Float64})
    rows = size(data, 1)
    @assert length(shifts) == length(scales) "Number of shifts must match number of scales"
    if length(shifts) == rows && length(scales) == rows
        for i in axes(data, 2)
            data[:, i] .*= scales
            data[:, i] .+= shifts
        end
    else
        dim = row ÷ length(shifts)
        for (i, (sh,sc)) in enumerate(zip(shifts, scales))
            data[dim*(i-1)+1:dim*i, :] .*= sc
            data[dim*(i-1)+1:dim*i, :] .+= sh
        end
    end
    return data
end

# function scale(data::Array{Float64}, dim::Int, factors::Vector{Float64})
#     @assert length(factors) == div(size(data, 1), dim) "Number of factors 
#         must match number of dimensions"
#     for (i, f) in enumerate(factors)
#         data[dim*(i-1)+1:dim*i, :] ./= f
#     end
#     return data
# end

# function unscale(data::Array{Float64}, dim::Int, factors::Vector{Float64})
#     @assert length(factors) == div(size(data, 1), dim) "Number of factors 
#         must match number of dimensions"
#     for (i, f) in enumerate(factors)
#         data[dim*(i-1)+1:dim*i, :] .*= f
#     end
#     return data
# end

# function minmax_shift_scale!(X, X_min, X_max)
#     shift = X_min / (X_max - X_min)
#     scale = X_max - X_min
#     X .-= X_min
#     X ./= (X_max .- X_min)
#     return shift, scale
# end


# """
#     compute_mean_parallel_threads(ds, dims; batch_size=50)

# Compute the mean of dataset snapshots using parallel processing with threads.

# # Arguments
# - `ds`: A dataset that supports indexing operations to retrieve snapshots
# - `dims`: Dimensions of a single snapshot (Tuple or dimensions)
# - `batch_size`: Number of snapshots to process in each batch (default: 50)

# # Returns
# - `xbar`: The computed mean vector

# # Example
# ```julia
# # Compute mean of 3D velocity field snapshots
# xbar = compute_mean_parallel_threads(ds, (Nz*Ny*Nx*3,))
# ```
# """
# function compute_mean_parallel_threads(ds, dims; batch_size=50)
#     # Get total number of snapshots
#     n = length(ds)
    
#     # Create thread-local accumulators
#     n_threads = Threads.nthreads()
#     thread_sums = [zeros(dims...) for _ in 1:n_threads]
#     thread_counts = zeros(Int, n_threads)

#     # Use batch processing to reduce overhead
#     num_batches = ceil(Int, n / batch_size)
#     batch_results = fill(false, num_batches)  # Track completed batches

#     # Disable logging during parallel execution
#     old_logger = global_logger(NullLogger())
    
#     # Parallel batch processing
#     Threads.@threads for batch in 1:num_batches
#         tid = Threads.threadid()
#         start_idx = (batch-1) * batch_size + 1
#         end_idx = min(batch * batch_size, n)
        
#         # Pre-allocate thread-local storage to reduce GC pressure
#         batch_sum = zeros(eltype(thread_sums[1]), size(thread_sums[1]))
        
#         # Process each snapshot in this batch
#         for i in start_idx:end_idx
#             snapshot = ds[i]  # Get snapshot only once
#             batch_sum .+= snapshot
#             thread_counts[tid] += 1
#         end
        
#         # Update thread sum once per batch
#         thread_sums[tid] .+= batch_sum
#         batch_results[batch] = true
#     end
    
#     # Restore logger
#     global_logger(old_logger)
    
#     # Combine results from all threads
#     xbar = reduce(.+, thread_sums)
#     xbar ./= sum(thread_counts)
    
#     # Report processing summary
#     @info "Processed $(sum(thread_counts)) of $n snapshots ($(count(batch_results)) batches)"
    
#     return xbar
# end


# """
#     compute_mean_multiprocess(ds, dims; batch_size=50, data_path=nothing, init_workers=true)

# Compute the mean of dataset snapshots using distributed processing with multiple processes.

# # Arguments
# - `ds`: A dataset that supports indexing operations to retrieve snapshots
# - `dims`: Dimensions of a single snapshot (Tuple or dimensions)
# - `batch_size`: Number of snapshots to process in each batch (default: 50)
# - `data_path`: Path to the dataset file (required for workers to load their own copy)
# - `init_workers`: Whether to initialize worker processes if not already present (default: true)

# # Returns
# - `xbar`: The computed mean vector

# # Example
# ```julia
# # Compute mean of 3D velocity field snapshots
# xbar = compute_mean_multiprocess(ds, (Nz*Ny*Nx*3,), data_path=datafile)
# ```
# """
# function compute_mean_multiprocess(ds, dims; batch_size=50, data_path=nothing, init_workers=true)
#     # Get total number of snapshots
#     n = length(ds)
    
#     # Add worker processes if needed
#     if init_workers && nprocs() == 1
#         worker_count = min(Sys.CPU_THREADS - 1, 8)  # Limit to reasonable number
#         @info "Adding $worker_count worker processes"
#         addprocs(worker_count)
#     end
#     @info "Running with $(nprocs()) processes ($(nprocs()-1) workers)"
    
#     # Ensure we have data_path
#     if isnothing(data_path)
#         error("data_path is required for multiprocessing to work")
#     end
    
#     # Set up parameters to pass to workers
#     data_file = data_path
#     dim_size = dims
#     n_snapshots = n
    
#     # Define batch function that returns local sum
#     batch_func = @distributed (vcat) for batch in 1:ceil(Int, n/batch_size)
#         start_idx = (batch-1) * batch_size + 1
#         end_idx = min(batch * batch_size, n)
        
#         # Create a new datasource just for this worker/batch
#         local_ds = ChannelDataSource(data_file, ["z", "y", "x", "fields", "times"])
        
#         # Process all snapshots in this batch
#         local_sum = zeros(dim_size...)
#         count = 0
        
#         for i in start_idx:end_idx
#             local_sum .+= local_ds[i]
#             count += 1
#         end
        
#         # Return tuple of sum and count
#         [(local_sum, count)]
#     end
    
#     # Combine results
#     total_sum = zeros(dims...)
#     total_count = 0
    
#     for (local_sum, count) in batch_func
#         total_sum .+= local_sum
#         total_count += count
#     end
    
#     # Compute mean
#     xbar = total_sum ./ total_count
    
#     @info "Processed $total_count of $n snapshots"
#     return xbar
# end

# """
#     compute_minmax_parallel_threads(ds, dims; batch_size=50)

# Compute the minimum and maximum values for each variable field using parallel processing with threads.

# # Arguments
# - `ds`: A dataset that supports indexing operations to retrieve snapshots
# - `dims`: Dimensions of a single snapshot (Tuple or dimensions)
# - `fields`: fields in the dataset (default: ["u", "v", "w", "p"])
# - `batch_size`: Number of snapshots to process in each batch (default: 50)

# # Returns
# - `x_min`: Vector containing minimum values for each spatial location across all fields
# - `x_max`: Vector containing maximum values for each spatial location across all fields

# # Example
# ```julia
# # Compute min/max of 3D channel flow fields [u, v, w, p]
# x_min, x_max = compute_minmax_parallel_threads(ds, (Nz*Ny*Nx*4,))
# ```

# # Notes
# The function assumes the data vector is organized as [u_field; v_field; w_field; p_field]
# where each field has `dim_per_field` spatial locations.
# """
# function compute_minmax_parallel_threads(ds, dims, fields; batch_size=50)
#     # Get total number of snapshots
#     n = length(ds)
    
#     # Create thread-local min/max accumulators
#     n_threads = Threads.nthreads()
#     thread_mins = [fill(Inf, dims...) for _ in 1:n_threads]
#     thread_maxs = [fill(-Inf, dims...) for _ in 1:n_threads]
    
#     # Use batch processing to reduce overhead
#     num_batches = ceil(Int, n / batch_size)
#     batch_results = fill(false, num_batches)  # Track completed batches
    
#     # Disable logging during parallel execution
#     old_logger = global_logger(NullLogger())
    
#     # Parallel batch processing
#     Threads.@threads for batch in 1:num_batches
#         tid = Threads.threadid()
#         start_idx = (batch-1) * batch_size + 1
#         end_idx = min(batch * batch_size, n)
        
#         # Pre-allocate thread-local storage to reduce GC pressure
#         batch_min = fill(Inf, size(thread_mins[1]))
#         batch_max = fill(-Inf, size(thread_maxs[1]))
        
#         # Process each snapshot in this batch
#         for i in start_idx:end_idx
#             snapshot = ds[i]  # Get snapshot only once
            
#             # Update min/max element-wise
#             @inbounds for j in eachindex(snapshot)
#                 val = snapshot[j]
#                 if val < batch_min[j]
#                     batch_min[j] = val
#                 end
#                 if val > batch_max[j]
#                     batch_max[j] = val
#                 end
#             end
#         end
        
#         # Update thread min/max once per batch
#         @inbounds for j in eachindex(thread_mins[tid])
#             if batch_min[j] < thread_mins[tid][j]
#                 thread_mins[tid][j] = batch_min[j]
#             end
#             if batch_max[j] > thread_maxs[tid][j]
#                 thread_maxs[tid][j] = batch_max[j]
#             end
#         end
        
#         batch_results[batch] = true
#     end
    
#     # Restore logger
#     global_logger(old_logger)
    
#     # Combine results from all threads
#     x_min = fill(Inf, dims...)
#     x_max = fill(-Inf, dims...)
    
#     for tid in 1:n_threads
#         @inbounds for j in eachindex(x_min)
#             if thread_mins[tid][j] < x_min[j]
#                 x_min[j] = thread_mins[tid][j]
#             end
#             if thread_maxs[tid][j] > x_max[j]
#                 x_max[j] = thread_maxs[tid][j]
#             end
#         end
#     end
    
#     # Report processing summary
#     @info "Processed $(sum(batch_results) * batch_size) snapshots across $(count(batch_results)) batches"
#     @info "Min/Max computation complete for $(length(x_min)) spatial locations"

#     return x_min, x_max
    
#     # IMPORTANT: 
#     # THE MIN MAX VALUES SHOULD BE FOR EACH ROW NOT FIELDS!!!!

#     # num_fields = length(fields)
#     # x_glob_mins = [0.0 for _ in 1:num_fields]
#     # x_glob_maxs = [0.0 for _ in 1:num_fields]

#     # try
#     #     # Ensure we have at least one field
#     #     @assert num_fields > 0 "No fields provided for min/max computation"
#     #     # Report field-wise statistics if we know the field structure
#     #     if length(dims) == 1 && dims[1] % 4 == 0
#     #         dim_per_field = dims[1] ÷ 4
            
#     #         @info "Field-wise min/max statistics:"
#     #         for (i, field_name) in enumerate(fields)
#     #             start_idx = (i-1) * dim_per_field + 1
#     #             end_idx = i * dim_per_field
#     #             x_glob_mins[i] = minimum(x_min[start_idx:end_idx])
#     #             x_glob_maxs[i] = maximum(x_max[start_idx:end_idx])
#     #             @info "  Field $field_name: min = $(x_glob_mins[i]), max = $(x_glob_maxs[i])"
#     #         end
#     #     end
#     # catch e
#     #     @error "Error in min/max computation: $(e)"
#     #     return x_min, x_max
#     # end

#     # return x_glob_mins, x_glob_maxs
# end
