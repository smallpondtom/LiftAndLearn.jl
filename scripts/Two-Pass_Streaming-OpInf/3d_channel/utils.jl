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

"""
    compute_minmax_parallel_threads(ds, dims; batch_size=50)

Compute the minimum and maximum values for each variable field using parallel processing with threads.

# Arguments
- `ds`: A dataset that supports indexing operations to retrieve snapshots
- `dims`: Dimensions of a single snapshot (Tuple or dimensions)
- `fields`: fields in the dataset (default: ["u", "v", "w", "p"])
- `batch_size`: Number of snapshots to process in each batch (default: 50)

# Returns
- `x_min`: Vector containing minimum values for each spatial location across all fields
- `x_max`: Vector containing maximum values for each spatial location across all fields

# Example
```julia
# Compute min/max of 3D channel flow fields [u, v, w, p]
x_min, x_max = compute_minmax_parallel_threads(ds, (Nz*Ny*Nx*4,))
```

# Notes
The function assumes the data vector is organized as [u_field; v_field; w_field; p_field]
where each field has `dim_per_field` spatial locations.
"""
function compute_minmax_parallel_threads(ds, dims, fields; batch_size=50)
    # Get total number of snapshots
    n = length(ds)
    
    # Create thread-local min/max accumulators
    n_threads = Threads.nthreads()
    thread_mins = [fill(Inf, dims...) for _ in 1:n_threads]
    thread_maxs = [fill(-Inf, dims...) for _ in 1:n_threads]
    
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
        batch_min = fill(Inf, size(thread_mins[1]))
        batch_max = fill(-Inf, size(thread_maxs[1]))
        
        # Process each snapshot in this batch
        for i in start_idx:end_idx
            snapshot = ds[i]  # Get snapshot only once
            
            # Update min/max element-wise
            @inbounds for j in eachindex(snapshot)
                val = snapshot[j]
                if val < batch_min[j]
                    batch_min[j] = val
                end
                if val > batch_max[j]
                    batch_max[j] = val
                end
            end
        end
        
        # Update thread min/max once per batch
        @inbounds for j in eachindex(thread_mins[tid])
            if batch_min[j] < thread_mins[tid][j]
                thread_mins[tid][j] = batch_min[j]
            end
            if batch_max[j] > thread_maxs[tid][j]
                thread_maxs[tid][j] = batch_max[j]
            end
        end
        
        batch_results[batch] = true
    end
    
    # Restore logger
    global_logger(old_logger)
    
    # Combine results from all threads
    x_min = fill(Inf, dims...)
    x_max = fill(-Inf, dims...)
    
    for tid in 1:n_threads
        @inbounds for j in eachindex(x_min)
            if thread_mins[tid][j] < x_min[j]
                x_min[j] = thread_mins[tid][j]
            end
            if thread_maxs[tid][j] > x_max[j]
                x_max[j] = thread_maxs[tid][j]
            end
        end
    end
    
    # Report processing summary
    @info "Processed $(sum(batch_results) * batch_size) snapshots across $(count(batch_results)) batches"
    @info "Min/Max computation complete for $(length(x_min)) spatial locations"
    
    num_fields = length(fields)
    x_glob_mins = [0.0 for _ in 1:num_fields]
    x_glob_maxs = [0.0 for _ in 1:num_fields]

    try
        # Ensure we have at least one field
        @assert num_fields > 0 "No fields provided for min/max computation"
        # Report field-wise statistics if we know the field structure
        if length(dims) == 1 && dims[1] % 4 == 0
            dim_per_field = dims[1] ÷ 4
            
            @info "Field-wise min/max statistics:"
            for (i, field_name) in enumerate(fields)
                start_idx = (i-1) * dim_per_field + 1
                end_idx = i * dim_per_field
                x_glob_mins[i] = minimum(x_min[start_idx:end_idx])
                x_glob_maxs[i] = maximum(x_max[start_idx:end_idx])
                @info "  Field $field_name: min = $(x_glob_mins[i]), max = $(x_glob_maxs[i])"
            end
        end
    catch e
        @error "Error in min/max computation: $(e)"
        return x_min, x_max
    end

    return x_glob_mins, x_glob_maxs
end

function right_ssm!(selected_rank::Int, 
                    Σ1::AbstractArray{T}, Σ2::AbstractArray{T},
                    W1::AbstractMatrix{T}, W2::AbstractMatrix{T}; 
                    γ::Real=1.0) where {T<:Number}
    # Dimensions
    k1 = length(Σ1)
    k2 = length(Σ2)
    k = k1 + k2
    n = size(W1, 2)  # Number of columns in the right singular vector matrices

    # Check dimensions
    size(W2, 2) == n || throw(DimensionMismatch("W1 and W2 must have the same number of columns"))
    size(W1, 1) == k1 || throw(DimensionMismatch("W1 must have k1 rows"))
    size(W2, 1) == k2 || throw(DimensionMismatch("W2 must have k2 rows"))

    # Convert Σ1, Σ2 to vectors if they are diagonal
    Σ1_vec = (ndims(Σ1) == 1) ? Σ1 : diag(Σ1)
    Σ2_vec = (ndims(Σ2) == 1) ? Σ2 : diag(Σ2)

    # Create combined matrix Z = [γ*Σ1*W1; Σ2*W2] (row-wise concatenation)
    Z = Matrix{T}(undef, k, n)
    @views begin
        Z[1:k1, :] = W1
        Z[k1+1:k, :] = W2
    end

    # Scale rows by singular values
    @views scale_rows!(Z[1:k1, :], γ .* Σ1_vec)
    @views scale_rows!(Z[k1+1:k, :], Σ2_vec)

    # LQ factorization: Z = L * Q where Q is orthogonal
    L = zeros(T, k, k)
    lqf!(Z, L)  # Z gets overwritten with Q, L contains the lower triangular part

    # SVD of L
    _, Σl, Wl = svd(L)

    # Truncate to selected rank
    selected_rank = min(selected_rank, length(Σl), k, n)
    
    # (1) Truncated singular values
    Σmerge = Σl[1:selected_rank]

    # (2) Truncated right singular vectors
    # Z now contains Q from LQ factorization
    Wmerge = Z' * Wl[:, 1:selected_rank]

    return Σmerge, Wmerge
end

function left_ssm!(selected_rank::Int, 
                   V1::AbstractMatrix{T}, V2::AbstractMatrix{T}, 
                   Σ1::AbstractArray{T}, Σ2::AbstractArray{T}, 
                   W1::AbstractMatrix{T}=zeros(T,1,1), W2::AbstractMatrix{T}=zeros(T,1,1);
                   γ::Real=1.0, right_singular_vectors::Bool=false) where {T<:Number}
    # Dimensions
    m = size(V1,1)
    k1 = size(V1,2)
    k2 = size(V2,2)
    k = k1 + k2

    @assert selected_rank ≤ k "selected_rank must be ≤ the sum of ranks of V1 and V2."
    size(V2,1) == m || throw(DimensionMismatch("V1 and V2 must have the same number of rows"))

    # Convert Σ1, Σ2 to vectors if they are diagonal
    Σ1_vec = (ndims(Σ1) == 1) ? Σ1 : diag(Σ1)
    Σ2_vec = (ndims(Σ2) == 1) ? Σ2 : diag(Σ2)

    # Create combined matrix A = [γ*V1*Σ1  V2*Σ2]
    A = Matrix{T}(undef, m, k)
    @views begin
        A[:, 1:k1] = V1
        A[:, k1+1:k] = V2
    end

    # Scale columns
    @views scale_columns!(A[:,1:k1], γ .* Σ1_vec)
    @views scale_columns!(A[:,k1+1:k], Σ2_vec)

    # QR factorization
    R = qrf!(A, zeros(k, k))

    # SVD of R
    # R is (m x (k1+k2)) but typically m ≥ k1+k2 so R's bottom is zero-triangular.
    # We'll get a small SVD:
    if right_singular_vectors
        Vr, Σr, Wr = svd(R)

        # (1) Truncate singular values: Σr[1:selected_rank]
        Σmerge = Σr[1:selected_rank]

        # (2) Truncate (A is the in-place QR factorization of A)
        Vmerge = A * Vr[:, 1:selected_rank]

        # (3) Truncate the right singular vectors if needed
        # Wtilde = BlockDiagonal([W1, W2])
        # Wmerge = Wtilde * Wr[:, 1:selected_rank]
        Wr = Wr[:, 1:selected_rank]
        Wr1 = @view Wr[1:k1, :]
        Wr2 = @view Wr[k1+1:end, :]
        Wmerge = vcat(W1 * Wr1, W2 * Wr2)

        return LinearAlgebra.SVD(Vmerge, Σmerge, Wmerge)
    else
        Vr, Σr, _ = svd(R)

        # (1) Truncate singular values: Σr[1:selected_rank]
        Σmerge = Σr[1:selected_rank]

        # (2) Truncate (A is the in-place QR factorization of A)
        Vmerge = A * Vr[:, 1:selected_rank]

        return LinearAlgebra.SVD(Vmerge, Σmerge, zeros(1,1))
    end
end

@inline function scale_rows!(A::AbstractMatrix, v::AbstractVector)
    @assert size(A, 1) == length(v)
    @inbounds @simd for i in eachindex(v)
        α = v[i]
        for j in axes(A, 2)
            A[i,j] *= α
        end
    end
    return A
end

@inline function scale_columns!(A::AbstractMatrix, Σ::AbstractVector)
    @assert size(A, 2) == length(Σ)
    @inbounds @simd for j in eachindex(Σ)
        σ = Σ[j]
        for i in axes(A,1)
            A[i,j] *= σ
        end
    end
    return A
end

function qrf!(P::AbstractArray{T}, R::AbstractArray{T}) where {T<:Number}
    if issparse(P) # If P is sparse, convert it to dense.
        P = Matrix(P)
    end
    m, b = checksize(P)
    m >= b || throw(DimensionMismatch("Works only for m ≥ b"))
    P, tau = LAPACK.geqrf!(P)
    fill!(R, zero(T))
    @inbounds for j = 1:b, i = 1:j
        R[i,j] = P[i,j]
    end
    LAPACK.orgqr!(P, tau)
    return R
end

function qrf!(P::AbstractArray{<:Number})
    if issparse(P) # If P is sparse, convert it to dense.
        P = Matrix(P)
    end
    m, b = checksize(P)
    m >= b || throw(DimensionMismatch("Works only for m ≥ b"))
    P, tau = LAPACK.geqrf!(P)
    LAPACK.orgqr!(P, tau)
end

function lqf!(P::AbstractArray{T}, L::AbstractArray{T}) where {T<:Number}
    if issparse(P) # If P is sparse, convert it to dense.
        P = Matrix(P)
    end
    m, n = checksize(P)
    n >= m || throw(DimensionMismatch("Works only for n ≥ m"))
    # Compute the LQ factorization of P; gelqf! returns P (with Householder info) and tau.
    P, tau = LAPACK.gelqf!(P)
    # Copy the lower–triangular part of P into L.
    fill!(L, zero(T))
    @inbounds for i = 1:m
        for j = 1:i
            L[i, j] = P[i, j]
        end
    end
    # Generate the orthogonal matrix Q in place (overwriting P).
    LAPACK.orglq!(P, tau)
    return L
end

function lqf!(P::AbstractArray{<:Number})
    if issparse(P) # If P is sparse, convert it to dense.
        P = Matrix(P)
    end
    m, n = checksize(P)
    n >= m || throw(DimensionMismatch("Works only for n ≥ m"))
    P, tau = LAPACK.gelqf!(P)
    LAPACK.orglq!(P, tau)
end

function checksize(A::AbstractArray)
    m, n = nothing, nothing
    try
        m, n = size(A)
    catch e
        if isa(e, BoundsError)
            m, n = length(A), 1
        else
            rethrow(e)
        end
    end
    return m, n
end