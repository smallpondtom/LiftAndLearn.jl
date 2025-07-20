using Logging

function preprocess!(data::Vector{T}, means::Vector{T}, 
                     shifts::Vector{T}, scales::Vector{T}) where T<:Real
    return scale!(center!(data, means), shifts, scales)
end

function unprocess!(data::Vector{T}, means::Vector{T}, 
                    shifts::Vector{T}, scales::Vector{T}) where T<:Real
    return uncenter!(unscale!(data, shifts, scales), means)
end

function center!(data::Matrix{T}, means::Vector{T}) where T<:Real
    @assert length(means) == size(data,1) "Number of means must match number of rows"
    data .-= means
    return data
end

function uncenter!(data::Matrix{T}, means::Vector{T}) where T<:Real
    @assert length(means) == size(data,1) "Number of means must match number of rows"
    data .+= means
    return data
end

function center!(data::Vector{T}, means::Vector{T}) where T<:Real
    @assert length(means) == length(data) "Number of means must match number of rows"
    data .-= means
    return data
end

function uncenter!(data::Vector{T}, means::Vector{T}) where T<:Real
    @assert length(means) == length(data) "Number of means must match number of rows"
    data .+= means
    return data
end

function scale!(data::Vector{T}, shifts::Vector{T}, scales::Vector{T}) where T<:Real
    rows = length(data)
    @assert length(shifts) == length(scales) "Number of shifts must match number of scales"
    if length(shifts) == rows && length(scales) == rows
        data .-= shifts
        data ./= scales
    else
        dim = row ÷ length(shifts)
        for (i, (sh,sc)) in enumerate(zip(shifts, scales))
            data[dim*(i-1)+1:dim*i] .-= sh
            data[dim*(i-1)+1:dim*i] ./= sc
        end
    end
    return data
end

function scale!(data::Matrix{T}, shifts::Vector{T}, scales::Vector{T}) where T<:Real
    rows = size(data, 1)
    @assert length(shifts) == length(scales) "Number of shifts must match number of scales"
    if length(shifts) == rows && length(scales) == rows
        data .-= shifts
        data ./= scales
    else
        dim = row ÷ length(shifts)
        for (i, (sh,sc)) in enumerate(zip(shifts, scales))
            data[dim*(i-1)+1:dim*i, :] .-= sh
            data[dim*(i-1)+1:dim*i, :] ./= sc
        end
    end
    return data
end

function unscale!(data::Vector{T}, shifts::Vector{T}, scales::Vector{T}) where T<:Real
    rows = length(data)
    @assert length(shifts) == length(scales) "Number of shifts must match number of scales"
    if length(shifts) == rows && length(scales) == rows
        data .*= scales
        data .+= shifts
    else
        dim = row ÷ length(shifts)
        for (i, (sh,sc)) in enumerate(zip(shifts, scales))
            data[dim*(i-1)+1:dim*i] .*= sc
            data[dim*(i-1)+1:dim*i] .+= sh
        end
    end
    return data
end

function unscale!(data::Matrix{T}, shifts::Vector{T}, scales::Vector{T}) where T<:Real
    rows = size(data, 1)
    @assert length(shifts) == length(scales) "Number of shifts must match number of scales"
    if length(shifts) == rows && length(scales) == rows
        for i in axes(data, 2)
            data .*= scales
            data .+= shifts
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


function compute_mean_parallel_threads_fixed(ds, dims; batch_size=50)
    # Get total number of snapshots
    n = length(ds)
    
    # Infer element type from first snapshot
    T = eltype(ds[1])
    
    # Create thread-local accumulators with proper initialization
    n_threads = Threads.nthreads()
    # Use a more robust approach with locks for thread safety
    thread_sums = [zeros(T, dims...) for _ in 1:n_threads]
    
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
        # Make sure to use the correct element type
        batch_sum = zeros(T, dims...)
        
        # Process each snapshot in this batch
        for i in start_idx:end_idx
            snapshot = ds[i]  # Get snapshot only once
            batch_sum .+= snapshot
        end
        
        # Update thread sum once per batch
        # This is safe because each thread only writes to its own slot
        thread_sums[tid] .+= batch_sum
        batch_results[batch] = true
    end
    
    # Restore logger
    global_logger(old_logger)
    
    # Combine results from all threads
    xbar = reduce(.+, thread_sums)
    xbar ./= T(n)  # Divide by the total number of snapshots, not thread counts
    
    # Report processing summary
    @info "Processed $n snapshots ($(count(batch_results)) batches)"
    
    return xbar
end

# Alternative implementation with explicit locking for comparison
function compute_mean_parallel_threads_locked(ds, dims; batch_size=50)
    n = length(ds)
    
    # Infer element type from first snapshot
    T = eltype(ds[1])
    
    # Single shared accumulator with a lock
    result_sum = zeros(T, dims...)
    result_lock = ReentrantLock()
    
    num_batches = ceil(Int, n / batch_size)
    
    # Disable logging during parallel execution
    old_logger = global_logger(NullLogger())
    
    Threads.@threads for batch in 1:num_batches
        start_idx = (batch-1) * batch_size + 1
        end_idx = min(batch * batch_size, n)
        
        # Local accumulator for this batch
        batch_sum = zeros(T, dims...)
        
        for i in start_idx:end_idx
            batch_sum .+= ds[i]
        end
        
        # Thread-safe update
        lock(result_lock) do
            result_sum .+= batch_sum
        end
    end
    
    # Restore logger
    global_logger(old_logger)
    
    return result_sum ./ T(n)
end

function compute_minmax_parallel_threads(ds; batch_size=50, means=nothing)
    # total snapshots
    n = length(ds)
    
    # infer per‐snapshot shape and element type
    first_snap = ds[1]
    T = eltype(first_snap)
    
    # Handle means parameter with proper type inference
    if means === nothing
        adjusted_first_snap = first_snap
        means_vec = zeros(T, size(first_snap))
    else
        # Convert means to proper type if needed
        if eltype(means) != T
            means_vec = convert.(T, means)
        else
            means_vec = means
        end
        adjusted_first_snap = first_snap .- means_vec
    end
    
    snap_dims = size(adjusted_first_snap)
    
    # per‐thread accumulators with proper type
    n_threads = Threads.nthreads()
    thread_mins = [fill(T(Inf), snap_dims...) for _ in 1:n_threads]
    thread_maxs = [fill(T(-Inf), snap_dims...) for _ in 1:n_threads]

    # how many batches?
    num_batches = ceil(Int, n ÷ batch_size) + (n % batch_size == 0 ? 0 : 1)

    Threads.@threads for batch in 1:num_batches
        tid = Threads.threadid()
        start_i = (batch-1) * batch_size + 1
        end_i = min(batch * batch_size, n)

        # local min/max for this batch with proper type
        local_min = fill(T(Inf), snap_dims...)
        local_max = fill(T(-Inf), snap_dims...)

        # scan snapshots in the batch
        for i in start_i:end_i
            snap = ds[i] .- means_vec
            @inbounds local_min .= min.(local_min, snap)
            @inbounds local_max .= max.(local_max, snap)
        end

        # fold into thread‐local
        @inbounds thread_mins[tid] .= min.(thread_mins[tid], local_min)
        @inbounds thread_maxs[tid] .= max.(thread_maxs[tid], local_max)
    end

    # combine across threads with proper type
    x_min = fill(T(Inf), snap_dims...)
    x_max = fill(T(-Inf), snap_dims...)
    for t in 1:n_threads
        @inbounds x_min .= min.(x_min, thread_mins[t])
        @inbounds x_max .= max.(x_max, thread_maxs[t])
    end

    @info "Processed $n snapshots across $num_batches batches."
    return x_min, x_max
end
