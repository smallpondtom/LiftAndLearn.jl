using Logging
using Distributed

"""
    compute_mean_parallel_threads(ds, dims; batch_size=50)

Compute the mean of dataset snapshots using parallel processing with threads.

# Arguments
- `ds`: A dataset that supports indexing operations to retrieve snapshots
- `dims`: Dimensions of a single snapshot (Tuple or dimensions)
- `batch_size`: Number of snapshots to process in each batch (default: 50)

# Returns
- `xbar`: The computed mean vector

# Example
```julia
# Compute mean of 3D velocity field snapshots
xbar = compute_mean_parallel_threads(ds, (Nz*Ny*Nx*3,))
```
"""
function compute_mean_parallel_threads(ds, dims; batch_size=50)
    # Get total number of snapshots
    n = length(ds)
    
    # Create thread-local accumulators
    n_threads = Threads.nthreads()
    thread_sums = [zeros(dims...) for _ in 1:n_threads]
    thread_counts = zeros(Int, n_threads)

    # Use batch processing to reduce overhead
    num_batches = ceil(Int, n / batch_size)
    batch_results = fill(false, num_batches)  # Track completed batches

    # Disable logging during parallel execution
    old_logger = global_logger(NullLogger())
    
    # Parallel batch processing
    Threads.@threads for batch in 1:num_batches
        tid = Threads.threadid()
        start_idx = (batch-1) * batch_size + 1
        end_idx = min(batch * batch_size, n)
        
        # Pre-allocate thread-local storage to reduce GC pressure
        batch_sum = zeros(eltype(thread_sums[1]), size(thread_sums[1]))
        
        # Process each snapshot in this batch
        for i in start_idx:end_idx
            snapshot = ds[i]  # Get snapshot only once
            batch_sum .+= snapshot
            thread_counts[tid] += 1
        end
        
        # Update thread sum once per batch
        thread_sums[tid] .+= batch_sum
        batch_results[batch] = true
    end
    
    # Restore logger
    global_logger(old_logger)
    
    # Combine results from all threads
    xbar = reduce(.+, thread_sums)
    xbar ./= sum(thread_counts)
    
    # Report processing summary
    @info "Processed $(sum(thread_counts)) of $n snapshots ($(count(batch_results)) batches)"
    
    return xbar
end


"""
    compute_mean_multiprocess(ds, dims; batch_size=50, data_path=nothing, init_workers=true)

Compute the mean of dataset snapshots using distributed processing with multiple processes.

# Arguments
- `ds`: A dataset that supports indexing operations to retrieve snapshots
- `dims`: Dimensions of a single snapshot (Tuple or dimensions)
- `batch_size`: Number of snapshots to process in each batch (default: 50)
- `data_path`: Path to the dataset file (required for workers to load their own copy)
- `init_workers`: Whether to initialize worker processes if not already present (default: true)

# Returns
- `xbar`: The computed mean vector

# Example
```julia
# Compute mean of 3D velocity field snapshots
xbar = compute_mean_multiprocess(ds, (Nz*Ny*Nx*3,), data_path=datafile)
```
"""
function compute_mean_multiprocess(ds, dims; batch_size=50, data_path=nothing, init_workers=true)
    # Get total number of snapshots
    n = length(ds)
    
    # Add worker processes if needed
    if init_workers && nprocs() == 1
        worker_count = min(Sys.CPU_THREADS - 1, 8)  # Limit to reasonable number
        @info "Adding $worker_count worker processes"
        addprocs(worker_count)
    end
    @info "Running with $(nprocs()) processes ($(nprocs()-1) workers)"
    
    # Ensure we have data_path
    if isnothing(data_path)
        error("data_path is required for multiprocessing to work")
    end
    
    # Set up parameters to pass to workers
    data_file = data_path
    dim_size = dims
    n_snapshots = n
    
    # Define batch function that returns local sum
    batch_func = @distributed (vcat) for batch in 1:ceil(Int, n/batch_size)
        start_idx = (batch-1) * batch_size + 1
        end_idx = min(batch * batch_size, n)
        
        # Create a new datasource just for this worker/batch
        local_ds = ChannelDataSource(data_file, ["z", "y", "x", "fields", "times"])
        
        # Process all snapshots in this batch
        local_sum = zeros(dim_size...)
        count = 0
        
        for i in start_idx:end_idx
            local_sum .+= local_ds[i]
            count += 1
        end
        
        # Return tuple of sum and count
        [(local_sum, count)]
    end
    
    # Combine results
    total_sum = zeros(dims...)
    total_count = 0
    
    for (local_sum, count) in batch_func
        total_sum .+= local_sum
        total_count += count
    end
    
    # Compute mean
    xbar = total_sum ./ total_count
    
    @info "Processed $total_count of $n snapshots"
    return xbar
end