export OnePassStreamingOpInf

mutable struct OnePassStreamingOpInf1{T<:Number}
    # iSVD components
    V::AbstractArray{T}
    Σ::AbstractVector{T}
    W::AbstractArray{T}
    
    # Dimensions
    state_dim::Int
    input_dim::Int
    num_of_snapshots::Int
    r_s::Int
    rmax::Int

    # Cache (for efficient implementation)
    Cs::SparseMatrixCSC{T}  # (rmax+1) x (rmax+1)
    q1cache::Vector{T}      # (rmax+1) x 1
    q2cache::Vector{T}      # (rmax+1) x 1
    xperp_cache::Vector{T}  # n x 1

    # Options
    options::LSOpInfOption
end

mutable struct OnePassStreamingOpInf2{T<:Number}
    # iSVD components
    V::AbstractArray{T}
    Σ::AbstractVector{T}
    W::AbstractArray{T}
    P::AbstractArray{T}
    S::AbstractVector{T}
    Q::AbstractArray{T}

    # Dimensions
    state_dim::Int
    input_dim::Int
    r_s::Int
    r_d::Int
    rmax::Int

    # Cache (for efficient implementation)
    Cs::SparseMatrixCSC{T}  # (rmax+1) x (rmax+1)
    Cd::SparseMatrixCSC{T}  # (rmax+1) x (rmax+1)
    q1cache::Vector{T}      # (rmax+1) x 1
    q2cache::Vector{T}      # (rmax+1) x 1
    xperp_cache::Vector{T}  # n x 1

    # Options
    options::LSOpInfOption
end

mutable struct OnePassStreamingOpInfSketchy{T<:Number}
    # SVD components
    V::AbstractArray{T}
    Σ::AbstractVector{T}
    W::AbstractArray{T}

    # Random matrices
    Ξ::AbstractArray{T}          # test matrix for range   (k × m)
    Ω::AbstractArray{T}          # test matrix for corange (k × n)
    Φ::AbstractArray{T}          # test matrix for core    (s × m)
    Ψ::AbstractArray{T}          # test matrix for core    (s × n)

    # Sketches
    Xcorange::AbstractArray{T}   # corange sketch (k × n)
    Xrange::AbstractArray{T}     # range sketch   (m × k)
    Xcore::AbstractArray{T}      # core sketch    (s × s)

    # Innovation
    H::AbstractArray{T}          # current innovation (m × n)

    # Dimensions
    k::Int                       # range sketch size
    s::Int                       # core sketch size
    input_dim::Int               # input dimension
    rmax::Int                    # maximum rank for the iSVDs

    # Counting number of streams 
    num_of_snapshots::Int
    
    # Options
    options::LSOpInfOption
end


function OnePassStreamingOpInf(
    x_init::AbstractVector{T}=T[0];          # Initial state data 
    xdot_init::AbstractVector{T}=T[1],       # Initial derivative data
    options::LSOpInfOption,                  # Standard (Least-Squares) Operator Inference options
    n_state::Int, n_input::Int=0,            # state and input dimensions
    n_snapshots::Int=1,                      # total number of snapshots 
    rank::Int=1,                             # maximum rank for the iSVDs
    finite_diff::Bool=true,                  # finite difference approx. for time derivative
    isvd_method::Symbol=:baker,              # iSVD method (:baker or :sketchy)
    sketch_size::Tuple{Int,Int,Int}=(0,0,0), # (k,s,ζ) for sketchy iSVD
    ) where {T<:Number}

    @assert isvd_method in [:baker, :sketchy] "isvd_method must be :baker or :sketchy"

    n = n_state
    m = n_input
    d = n_snapshots

    if isvd_method == :baker

        @assert(
            length(x_init) == n && !iszero(x_init), 
            "x_init must be initialized if using baker iSVD."
        )

        V = reshape(x_init / norm(x_init), :, 1)
        Σ = [norm(x_init)]
        W = reshape([T(1)], 1, 1)
        P = reshape(xdot_init / norm(xdot_init), :, 1)
        S = [norm(xdot_init)]
        Q = reshape([T(1)], 1, 1)

        # Cache variables
        Cs = spzeros(T, rank+1, rank+1)
        Cd = spzeros(T, rank+1, rank+1)
        q1cache = zeros(T, rank+1)
        q2cache = zeros(T, rank+1)
        xperp_cache = similar(x_init)

        if finite_diff
            return OnePassStreamingOpInf1(
                V, Σ, W, 
                n, m, 1, 1, rank,
                Cs, q1cache, q2cache, xperp_cache,
                options
            )
        else
            return OnePassStreamingOpInf2(
                V, Σ, W, P, S, Q,
                n, m, 1, 1, rank, 
                Cs, Cd, q1cache, q2cache, xperp_cache,
                options
            )
        end
    elseif isvd_method == :sketchy
        k, s, ζ = sketch_size
        @assert k <= s "For sketchy iSVD, k must be less than or equal to s."
        k = iszero(k) ? 4*rank + 1 : k
        s = iszero(s) ? 2*k + 1 : s
        ζ = iszero(ζ) ? min(k, 8) : ζ

        # Initialize the random reduction maps 
        Ξ = sparse_sign_matrix(k, n, ζ)
        Ω = sparse_sign_matrix(k, d, ζ)
        Φ = sparse_sign_matrix(s, n, ζ)
        Ψ = sparse_sign_matrix(s, d, ζ)

        # Initialize the sketches
        Xcorange = zeros(k, d)
        Xrange   = zeros(n, k)
        Xcore    = zeros(s, s)

        return OnePassStreamingOpInfSketchy(
            T[0.0], T[0.0], T[0.0],
            Ξ, Ω, Φ, Ψ,
            Xcorange, Xrange, Xcore,
            spzeros(T, n, d),
            k, s, m, rank, 0,
            options
        )
    else
        error("Unknown isvd_method: $isvd_method. " * 
              "Supported methods are :baker and :sketchy.")
    end
end

function sparse_sign_matrix(k::Int, n::Int, zeta::Int;
                            field::String="real")
    if zeta < 1 || zeta > k
        error("zeta should be between 1 and k.")
    end
    field = lowercase(field)

    # Create indCol: repeat each column index zeta times
    indCol = repeat(1:n, inner=zeta)

    # Initialize indRow
    indRow = Vector{Int}(undef, n * zeta)
    idx = 1
    for _ in 1:n
        rows = sort(sample(1:k, zeta; replace=false))
        indRow[idx:idx+zeta-1] = rows
        idx += zeta
    end

    # Generate values
    if field == "real"
        vals = sign.(randn(n * zeta))
        return sparse(indRow, indCol, vals, k, n)
    elseif field == "complex"
        vals = sign.(randn(n * zeta) .+ im * randn(n * zeta))
        return sparse(indRow, indCol, vals, k, n)
    else
        error("Input 'field' should be 'real' or 'complex'.")
    end
end

function stream!(obj::OnePassStreamingOpInf1, x::AbstractVector{T}; 
                 tol::T=1e-12) where {T<:Real}
    # Reduced dimensions
    r1 = obj.r_s
    rmax = copy(obj.rmax)

    ##
    # --- (1) Snapshot matrix
    ##
    # Double orthogonalization of the new data
    """ naive version
    q1 = obj.V' * x
    xperp = x - obj.V * q1
    q2 = obj.V' * xperp
    xperp = xperp - obj.V * q2
    q = q1 + q2
    p = norm(xperp)
    """
    # Compute C = V' * x
    @views q = obj.q1cache[1:r1]
    mul!(q, obj.V', x)
    # Compute x_perp = x - V*q1
    xperp = obj.xperp_cache
    copy!(xperp, x)
    BLAS.gemv!('N', -one(T), obj.V, q, one(T), xperp)
    # Reorthogonalization
    @views q2 = obj.q2cache[1:r1]
    mul!(q2, obj.V', xperp)
    BLAS.gemv!('N', -one(T), obj.V, q2, one(T), xperp)
    axpy!(one(T), q2, q) # q = q1 + q2

    # QR factorization of the orthogonalized data
    p = [0.0]
    xperp = reshape(xperp, :, 1)
    qrf!(xperp, p)

    # Build the broken arrowhead (sparse) matrix using cache
    (dropzeros! ∘ fill!)(obj.Cs, zero(T))  # reset the cache
    for j in 1:r1
        obj.Cs[j,j] = obj.Σ[j]
        obj.Cs[j,r1+1] = q[j]
    end
    obj.Cs[r1+1,r1+1] = p[1]

    # Use PROPACK svd solver for SparseMatrixCSC type
    Vc, Σc, Wc = nothing, nothing, nothing
    try
        Vc, Σc, Wc, _, _, _ = tsvd(obj.Cs[1:r1+1, 1:r1+1], k=r1+1)
    catch e 
        @warn "PROPACK tsvd failed, increasing kmax"
        try 
            Vc, Σc, Wc, _, _, _ = tsvd(obj.Cs[1:r1+1, 1:r1+1], k=r1+1,
                                        kmax=min(size(obj.Cs))+25)
        catch e
            @error "PROPACK tsvd failed again, using svd instead"
            Vc, Σc, Wc = svd(Matrix(obj.Cs[1:r1+1, 1:r1+1]))
        end
    end

    # Update the iSVD components
    obj.V = hcat(obj.V, xperp) * Vc
    obj.Σ = Σc
    obj.W = [obj.W zeros(size(obj.W,1), 1); zeros(1, r1) 1.0] * Wc
    obj.r_s += 1  # increment the rank

    ##
    # --- (3) Truncate the rank 
    ##
    if obj.r_s > rmax
        obj.V = obj.V[:,1:rmax]
        obj.Σ = obj.Σ[1:rmax]
        obj.W = obj.W[:,1:rmax]
        obj.r_s = rmax
    end

    ##
    # --- (4) Reorthogonalize if necessary
    ##
    reorthogonalize!(obj.V, tol)

    obj.num_of_snapshots += 1  # increment the number of snapshots

    return nothing
end


function stream!(obj::OnePassStreamingOpInf2, x::AbstractVector{T}, 
    xdot::AbstractVector{T}; tol::T=1e-12) where {T<:Real}

    # Reduced dimensions
    r1 = obj.r_s
    r2 = obj.r_d
    rmax = copy(obj.rmax)

    ##
    # --- (1) Snapshot matrix
    ##
    # Double orthogonalization of the new data
    """ naive version
    q1 = obj.V' * x
    xperp = x - obj.V * q1
    q2 = obj.V' * xperp
    xperp = xperp - obj.V * q2
    q = q1 + q2
    p = norm(xperp)
    """
    # Compute C = V' * x
    @views q = obj.q1cache[1:r1]
    mul!(q, obj.V', x)
    # Compute x_perp = x - V*q1
    xperp = obj.xperp_cache
    copy!(xperp, x)
    BLAS.gemv!('N', -one(T), obj.V, q, one(T), xperp)
    # Reorthogonalization
    @views q2 = obj.q2cache[1:r1]
    mul!(q2, obj.V', xperp)
    BLAS.gemv!('N', -one(T), obj.V, q2, one(T), xperp)
    axpy!(one(T), q2, q) # q = q1 + q2

    # QR factorization of the orthogonalized data
    p = [0.0]
    xperp = reshape(xperp, :, 1)
    qrf!(xperp, p)

    # Build the broken arrowhead (sparse) matrix using cache
    (dropzeros! ∘ fill!)(obj.Cs, zero(T))  # reset the cache
    for j in 1:r1
        obj.Cs[j,j] = obj.Σ[j]
        obj.Cs[j,r1+1] = q[j]
    end
    obj.Cs[r1+1,r1+1] = p[1]

    # Use PROPACK svd solver for SparseMatrixCSC type
    Vc, Σc, Wc, _, _, _ = tsvd(obj.Cs[1:r1+1, 1:r1+1], k=r1+1)

    # Update the iSVD components
    obj.V = hcat(obj.V, xperp) * Vc
    obj.Σ = Σc
    obj.W = [obj.W zeros(size(obj.W,1), 1); zeros(1, r1) 1.0] * Wc
    obj.r_s += 1  # increment the rank

    ##
    # --- (2) Derivative matrix
    ##
    """
    q1 = obj.P' * xdot
    xdotperp = xdot - obj.P * q1
    q2 = obj.P' * xdotperp
    xdotperp = xdotperp - obj.P * q2
    q = q1 + q2
    p = norm(xdotperp)
    """
    @views q = obj.q1cache[1:r2]
    mul!(q, obj.P', xdot)
    xdotperp = obj.xperp_cache
    copy!(xdotperp, xdot)
    BLAS.gemv!('N', -one(T), obj.P, q, one(T), xdotperp)
    @views q2 = obj.q2cache[1:r2]
    mul!(q2, obj.P', xdotperp)
    BLAS.gemv!('N', -one(T), obj.P, q2, one(T), xdotperp)
    axpy!(one(T), q2, q)

    p = [0.0]
    xdotperp = reshape(xdotperp, :, 1)
    qrf!(xdotperp, p)

    (dropzeros! ∘ fill!)(obj.Cd, zero(T))
    for j in 1:r2
        obj.Cd[j,j] = obj.S[j]
        obj.Cd[j,r2+1] = q[j]
    end
    obj.Cd[r2+1,r2+1] = p[1]

    Pc, Sc, Qc, _, _, _ = tsvd(obj.Cd[1:r2+1, 1:r2+1], k=r2+1)
    obj.P = hcat(obj.P, xdotperp) * Pc
    obj.S = Sc
    obj.Q = [obj.Q zeros(size(obj.Q,1), 1); zeros(1, r2) 1.0] * Qc
    obj.r_d += 1

    ##
    # --- (3) Truncate the rank 
    ##
    if obj.r_s > rmax
        obj.V = obj.V[:,1:rmax]
        obj.Σ = obj.Σ[1:rmax]
        obj.W = obj.W[:,1:rmax]
        obj.r_s = rmax
    end
    if obj.r_d > rmax
        obj.P = obj.P[:,1:rmax]
        obj.S = obj.S[1:rmax]
        obj.Q = obj.Q[:,1:rmax]
        obj.r_d = rmax
    end

    ##
    # --- (4) Reorthogonalize if necessary
    ##
    reorthogonalize!(obj.V, tol)
    reorthogonalize!(obj.P, tol)

    return nothing
end

function stream!(obj::OnePassStreamingOpInfSketchy, 
                 x::AbstractMatrix{T}) where {T<:Real}

    # Form the innovation
    i = obj.num_of_snapshots + 1  # current stream index
    d = size(x,2)
    obj.H[:, i:i+d-1] = x

    # Update the sketches 
    obj.Xcorange .+= obj.Ξ * H
    obj.Xrange   .+= H * obj.Ω'
    obj.Xcore    .+= (obj.Φ * H) * obj.Ψ'

    # Reset the innovation
    obj.H[:, i:i+d-1] = 0.0
    dropzeros!(obj.H)

    # Increment the number of streams
    obj.num_of_snapshots += d

    return nothing
end

function stream!(obj::OnePassStreamingOpInfSketchy,
                 x::AbstractVector{T}) where {T<:Real}

    i = obj.num_of_snapshots + 1  # current stream index

    # X update - simplified since ν=1, η=1
    mul!(view(obj.Xcorange, :, i), obj.Ξ, x)
    
    # Get the transpose row more efficiently
    Ω_T = obj.Ω'
    
    # Use sparse matrix operations directly instead of iterating
    if isa(Ω_T, SparseMatrixCSC)
        # Direct sparse vector operations
        row_vals = view(Ω_T.nzval, Ω_T.colptr[i]:Ω_T.colptr[i+1]-1)
        row_indices = view(Ω_T.rowval, Ω_T.colptr[i]:Ω_T.colptr[i+1]-1)
        
        # Simplified update since ν=1
        @inbounds for (idx_pos, col_idx) in enumerate(row_indices)
            val = row_vals[idx_pos]
            # Use axpy! for better performance: y = a*x + y
            BLAS.axpy!(val, x, view(obj.Xrange, :, col_idx))
        end
    else
        # Fallback for other sparse types
        row_Ω = Ω_T[i, :]
        @inbounds for (idx, val) in pairs(row_Ω)
            BLAS.axpy!(val, x, view(obj.Xrange, :, idx))
        end
    end
    
    # Pre-compute Φ * x once
    z = obj.Φ * x
    
    # Same optimization for Ψ
    Ψ_T = obj.Ψ'
    
    if isa(Ψ_T, SparseMatrixCSC)
        # Direct sparse vector operations
        row_vals = view(Ψ_T.nzval, Ψ_T.colptr[i]:Ψ_T.colptr[i+1]-1)
        row_indices = view(Ψ_T.rowval, Ψ_T.colptr[i]:Ψ_T.colptr[i+1]-1)
        
        # Simplified update since ν=1
        @inbounds for (idx_pos, col_idx) in enumerate(row_indices)
            val = row_vals[idx_pos]
            BLAS.axpy!(val, z, view(obj.Xcore, :, col_idx))
        end
    else
        # Fallback for other sparse types
        row_Ψ = Ψ_T[i, :]
        @inbounds for (idx, val) in pairs(row_Ψ)
            BLAS.axpy!(val, z, view(obj.Xcore, :, idx))
        end
    end
    
    # Increment the number of streams
    obj.num_of_snapshots += 1
    
    return nothing
end

function compute_stream_operators(obj::OnePassStreamingOpInf1, 
    E::AbstractArray{T}, indices::Tuple{<:Int,<:Int};
    U::AbstractArray{T}=[0.0], rank::Int=obj.rmax) where {T<:Real}

    # Diagonalize the singular values
    Σ = obj.Σ[1:rank]
    Σ_diag = Diagonal(Σ)

    # Extract the appropriate indices
    id1 = indices[1]
    id2 = indices[2]
    W = view(obj.W, id1:id2, 1:rank)

    # Construct the reduce data matrix
    r = rank
    m = obj.input_dim
    d = 0  # total dimension of the data matrix
    if obj.options.optim.nonredundant_operators
        d += sum(i != 0 ? binomial(r+i-1, i) : 0 for i in obj.options.system.state)
    else
        d += sum(i != 0 ? Int(r^i) : 0 for i in obj.options.system.state)
    end
    d += sum(i != 0 ? binomial(m+i-1, i) : 0 for i in obj.options.system.control)
    # d += sum(i != 0 ? binomial(r+i-1, i) * m : 0 for i in obj.options.system.coupled_input)
    d += iszero(obj.options.system.constant) ? 0 : 1
    K = min(obj.num_of_snapshots, id2-id1+1)  # number of snapshots
    D = zeros(T, K, d)  # data matrix
    tmp = 0

    dims = []
    operator_symbols = []

    for i in obj.options.system.state
        if i == 1
            D[:, 1:r] = W * Σ_diag
            push!(dims, r)
            push!(operator_symbols, :A)
            tmp += r
        else
            if obj.options.optim.nonredundant_operators
                ri = binomial(r+i-1, i)
                D[:, tmp+1:tmp+ri] = ⧁(W, i) * Diagonal(⊘(Σ, i))
                push!(dims, ri)
                push!(operator_symbols, Symbol("A$(i)u"))
            else
                ri = Int(r^i)
                D[:, tmp+1:tmp+ri] = ⊖(W, i) * Diagonal(⊗(Σ[:,:], i)[:])
                push!(dims, ri)
                push!(operator_symbols, Symbol("A$(i)"))
            end
            tmp += ri
        end

        if i == 1 && obj.input_dim != 0
            # NOTE: Only works for linear inputs (for now)
            U = fat2tall(U)
            if !iszero(obj.options.system.control)
                D[:, tmp+1:tmp+m] = view(U, id1:id2, :)
                tmp += m
                push!(dims, m)
                push!(operator_symbols, :B)
            end
            # NOTE: Coupled inputs are not implemented yet
        end
    end

    if !iszero(obj.options.system.constant)
        D[:, tmp+1] = ones(K)  # constant term
        push!(dims, 1)
        push!(operator_symbols, :K)
    end

    # Construct the reduced right-hand side matrix
    n_tmp = size(E, 1)
    R = E' * view(obj.W, 1:n_tmp, 1:rank) * Σ_diag

    # compute least squares (pseudo inverse)
    if obj.options.with_reg 
        # Preallocate the Tikhonov weight Matrix
        Γ = spzeros(d)

        # Construct the Tikhonov matrix
        tikhonov_matrix!(Γ, dims, operator_symbols, obj.options.λ)
        Γ = spdiagm(0 => Γ)  # convert to sparse diagonal matrix

        Ot = tikhonov(R, D, Γ;
                      tol=obj.options.tolerance,
                      use_gpu=obj.options.use_gpu,
                      use_normal_form=obj.options.use_normal_equations,
                      use_svd_truncation=obj.options.use_svd_truncation,
                      use_backslash=obj.options.use_backslash,
                      chunk_size=obj.options.chunk_size,
                      max_iterations=obj.options.max_iterations,
                      estimate_memory=obj.options.estimate_memory,
                      preconditioning=obj.options.preconditioning,)
    else
        Ot = standard_least_squares(D, R; 
                                    use_gpu=obj.options.use_gpu, 
                                    use_normal_equations=obj.options.use_normal_equations,
                                    chunk_size=obj.options.chunk_size,
                                    tolerance=obj.options.tolerance,
                                    use_backslash=obj.options.use_backslash,
                                    algorithm=obj.options.algorithm,
                                    estimate_memory=obj.options.estimate_memory)
    end

    # Extract the operators from the operator matrix O
    O = transpose(Ot)

    # Extract the operators
    operators = Operators(O=O)

    # Unpack the operators
    unpack_operators!(operators, O, dims, operator_symbols)

    return operators
end


function compute_stream_operators(obj::OnePassStreamingOpInf1,
    E::AbstractArray{T}, indices::Union{Array{<:Int},UnitRange};
    U::AbstractArray{T}=[0.0], rank::Int=obj.rmax) where {T<:Real}

    # Diagonalize the singular values
    Σ = obj.Σ[1:rank]
    Σ_diag = Diagonal(Σ)

    # Extract the appropriate indices
    W = view(obj.W, indices, 1:rank)

    # Construct the reduce data matrix
    r = rank
    m = obj.input_dim
    d = 0  # total dimension of the data matrix
    if obj.options.optim.nonredundant_operators
        d += sum(i != 0 ? binomial(r+i-1, i) : 0 for i in obj.options.system.state)
    else
        d += sum(i != 0 ? Int(r^i) : 0 for i in obj.options.system.state)
    end
    d += sum(i != 0 ? binomial(m+i-1, i) : 0 for i in obj.options.system.control)
    # d += sum(i != 0 ? binomial(r+i-1, i) * m : 0 for i in obj.options.system.coupled_input)
    d += iszero(obj.options.system.constant) ? 0 : 1
    K = min(obj.num_of_snapshots, length(indices))  # number of snapshots
    D = zeros(T, K, d)  # data matrix
    tmp = 0

    dims = []
    operator_symbols = []

    for i in obj.options.system.state
        if i == 1
            D[:, 1:r] = W * Σ_diag
            push!(dims, r)
            push!(operator_symbols, :A)
            tmp += r
        else
            if obj.options.optim.nonredundant_operators
                ri = binomial(r+i-1, i)
                D[:, tmp+1:tmp+ri] = ⧁(W, i) * Diagonal(⊘(Σ, i))
                push!(dims, ri)
                push!(operator_symbols, Symbol("A$(i)u"))
            else
                ri = Int(r^i)
                D[:, tmp+1:tmp+ri] = ⊖(W, i) * Diagonal(⊗(Σ[:,:], i)[:])
                push!(dims, ri)
                push!(operator_symbols, Symbol("A$(i)"))
            end
            tmp += ri
        end

        if i == 1 && obj.input_dim != 0
            # NOTE: Only works for linear inputs (for now)
            U = fat2tall(U)
            if !iszero(obj.options.system.control)
                D[:, tmp+1:tmp+m] = view(U, indices, :)
                tmp += m
                push!(dims, m)
                push!(operator_symbols, :B)
            end
            # NOTE: Coupled inputs are not implemented yet
        end
    end

    if !iszero(obj.options.system.constant)
        D[:, tmp+1] = ones(K)  # constant term
        push!(dims, 1)
        push!(operator_symbols, :K)
    end

    # Construct the reduced right-hand side matrix
    n_tmp = size(E, 1)
    R = Array(E' * view(obj.W, 1:n_tmp, 1:rank) * Σ_diag)

    # compute least squares (pseudo inverse)
    if obj.options.with_reg 
        # Preallocate the Tikhonov weight Matrix
        Γ = spzeros(d)

        # Construct the Tikhonov matrix
        tikhonov_matrix!(Γ, dims, operator_symbols, obj.options.λ)
        Γ = spdiagm(0 => Γ)  # convert to sparse diagonal matrix
        Ot = tikhonov(R, D, Γ;
                      tol=obj.options.tolerance,
                      use_gpu=obj.options.use_gpu,
                      use_normal_form=obj.options.use_normal_equations,
                      use_svd_truncation=obj.options.use_svd_truncation,
                      use_backslash=obj.options.use_backslash,
                      chunk_size=obj.options.chunk_size,
                      max_iterations=obj.options.max_iterations,
                      estimate_memory=obj.options.estimate_memory,
                      preconditioning=obj.options.preconditioning,)
    else
        Ot = standard_least_squares(D, R; 
                                    use_gpu=obj.options.use_gpu, 
                                    use_normal_equations=obj.options.use_normal_equations,
                                    chunk_size=obj.options.chunk_size,
                                    tolerance=obj.options.tolerance,
                                    use_backslash=obj.options.use_backslash,
                                    algorithm=obj.options.algorithm,
                                    estimate_memory=obj.options.estimate_memory)
    end

    # Extract the operators from the operator matrix O
    O = transpose(Ot)

    # Extract the operators
    operators = Operators(O=O)

    # Unpack the operators
    unpack_operators!(operators, O, dims, operator_symbols)

    return operators
end


function compute_stream_operators(obj::OnePassStreamingOpInf2;
    U::AbstractArray{T}=[0.0], rank::Int=obj.rmax) where {T<:Real}

    # Diagonalize the singular values
    Σ = obj.Σ[1:rank]
    Σ_diag = Diagonal(Σ)
    S_diag = Diagonal(obj.S[1:rank])

    # Assemble the low-rank approximation of the time derivative data
    Xdot_t = obj.Q[:,1:rank]  * S_diag * obj.P[:,1:rank]'
    W = obj.W[:, 1:rank]

    # Construct the reduce data matrix
    r = rank
    m = obj.input_dim
    d = 0  # total dimension of the data matrix
    if obj.options.optim.nonredundant_operators
        d += sum(i != 0 ? binomial(r+i-1, i) : 0 for i in obj.options.system.state)
    else
        d += sum(i != 0 ? Int(r^i) : 0 for i in obj.options.system.state)
    end
    d += sum(i != 0 ? binomial(m+i-1, i) : 0 for i in obj.options.system.control)
    # d += sum(i != 0 ? binomial(r+i-1, i) * m : 0 for i in obj.options.system.coupled_input)
    d += iszero(obj.options.system.constant) ? 0 : 1
    K = size(Xdot_t, 1)  # number of snapshots
    D = zeros(T, K, d)  # data matrix
    tmp = 0

    dims = []
    operator_symbols = []

    for i in obj.options.system.state
        if i == 1
            D[:, 1:r] = W * Σ_diag
            push!(dims, r)
            push!(operator_symbols, :A)
            tmp += r
        else
            if obj.options.optim.nonredundant_operators
                ri = binomial(r+i-1, i)
                D[:, tmp+1:tmp+ri] = ⧁(W, i) * Diagonal(⊘(Σ, i))
                push!(dims, ri)
                push!(operator_symbols, Symbol("A$(i)u"))
            else
                ri = Int(r^i)
                D[:, tmp+1:tmp+ri] = ⊖(W, i) * Diagonal(⊗(Σ, i)[:])
                push!(dims, ri)
                push!(operator_symbols, Symbol("A$(i)"))
            end
            tmp += ri
        end

        if i == 1 && obj.input_dim != 0
            # NOTE: Only works for linear inputs (for now)
            U = fat2tall(U)
            if !iszero(obj.options.system.control)
                D[:, tmp+1:tmp+m] = U
                tmp += m
                push!(dims, m)
                push!(operator_symbols, :B)
            end
            # NOTE: Coupled inputs are not implemented yet
        end
    end

    if !iszero(obj.options.system.constant)
        D[:, tmp+1] = ones(K)  # constant term
        push!(dims, 1)
        push!(operator_symbols, :K)
    end

    # Construct the reduced right-hand side matrix
    R = Xdot_t * obj.V

    # compute least squares (pseudo inverse)
    if obj.options.with_reg 
        # Preallocate the Tikhonov weight Matrix
        Γ = spzeros(d)

        # Construct the Tikhonov matrix
        tikhonov_matrix!(Γ, dims, operator_symbols, obj.options.λ)
        Γ = spdiagm(0 => Γ)  # convert to sparse diagonal matrix

        Ot = tikhonov(R, D, Γ;
                      tol=obj.options.tolerance,
                      use_gpu=obj.options.use_gpu,
                      use_normal_form=obj.options.use_normal_equations,
                      use_svd_truncation=obj.options.use_svd_truncation,
                      use_backslash=obj.options.use_backslash,
                      chunk_size=obj.options.chunk_size,
                      max_iterations=obj.options.max_iterations,
                      estimate_memory=obj.options.estimate_memory,
                      preconditioning=obj.options.preconditioning,)
    else
        Ot = standard_least_squares(D, R; 
                                    use_gpu=obj.options.use_gpu, 
                                    use_normal_equations=obj.options.use_normal_equations,
                                    chunk_size=obj.options.chunk_size,
                                    tolerance=obj.options.tolerance,
                                    use_backslash=obj.options.use_backslash,
                                    algorithm=obj.options.algorithm,
                                    estimate_memory=obj.options.estimate_memory)
    end

    # Extract the operators from the operator matrix O
    O = transpose(Ot)

    # Extract the operators
    operators = Operators(O=O)

    # Unpack the operators
    unpack_operators!(operators, O, dims, operator_symbols)

    return operators
end

function compute_svd_sketchy!(obj::OnePassStreamingOpInfSketchy; 
                             rank::Int=obj.rmax)
    # initial approximation with QR decompositions
    Qs = Matrix(qr(obj.Xrange).Q)
    Ps = Matrix(qr(obj.Xcorange').Q)
    tmp = (obj.Φ * Qs) \ obj.Xcore  # Solve (Φ * Q) * X = Z
    C = tmp / (obj.Ψ * Ps)'         # Solve X = X / (Ψ * P)'

    # Truncate the approximation and compute the singular vectors
    Q, Σ, W = svd(C)
    obj.V = Qs * Q[:,1:rank]
    obj.Σ = Σ[1:rank]
    obj.W = Ps * W[:,1:rank]

    return nothing
end

function compute_stream_operators(obj::OnePassStreamingOpInfSketchy, 
    E::AbstractArray{T}, indices::Tuple{<:Int,<:Int};
    U::AbstractArray{T}=[0.0], rank::Int=obj.rmax) where {T<:Real}

    # Diagonalize the singular values
    Σ = copy(obj.Σ)
    Σ_diag = Diagonal(Σ)

    # Extract the appropriate indices
    id1 = indices[1]
    id2 = indices[2]
    W = view(obj.W, id1:id2, 1:rank)

    # Construct the reduce data matrix
    r = rank
    m = obj.input_dim
    d = 0  # total dimension of the data matrix
    if obj.options.optim.nonredundant_operators
        d += sum(i != 0 ? binomial(r+i-1, i) : 0 for i in obj.options.system.state)
    else
        d += sum(i != 0 ? Int(r^i) : 0 for i in obj.options.system.state)
    end
    d += sum(i != 0 ? binomial(m+i-1, i) : 0 for i in obj.options.system.control)
    # d += sum(i != 0 ? binomial(r+i-1, i) * m : 0 for i in obj.options.system.coupled_input)
    d += iszero(obj.options.system.constant) ? 0 : 1
    K = min(obj.num_of_snapshots, id2-id1+1)  # number of snapshots
    D = zeros(T, K, d)  # data matrix
    tmp = 0

    dims = []
    operator_symbols = []

    for i in obj.options.system.state
        if i == 1
            D[:, 1:r] = W * Σ_diag
            push!(dims, r)
            push!(operator_symbols, :A)
            tmp += r
        else
            if obj.options.optim.nonredundant_operators
                ri = binomial(r+i-1, i)
                D[:, tmp+1:tmp+ri] = ⧁(W, i) * Diagonal(⊘(Σ, i))
                push!(dims, ri)
                push!(operator_symbols, Symbol("A$(i)u"))
            else
                ri = Int(r^i)
                D[:, tmp+1:tmp+ri] = ⊖(W, i) * Diagonal(⊗(Σ[:,:], i)[:])
                push!(dims, ri)
                push!(operator_symbols, Symbol("A$(i)"))
            end
            tmp += ri
        end

        if i == 1 && obj.input_dim != 0
            # NOTE: Only works for linear inputs (for now)
            U = fat2tall(U)
            if !iszero(obj.options.system.control)
                D[:, tmp+1:tmp+m] = view(U, id1:id2, :)
                tmp += m
                push!(dims, m)
                push!(operator_symbols, :B)
            end
            # NOTE: Coupled inputs are not implemented yet
        end
    end

    if !iszero(obj.options.system.constant)
        D[:, tmp+1] = ones(K)  # constant term
        push!(dims, 1)
        push!(operator_symbols, :K)
    end

    # Construct the reduced right-hand side matrix
    n_tmp = size(E, 1)
    R = E' * view(obj.W, 1:n_tmp, :) * Σ_diag

    # compute least squares (pseudo inverse)
    if obj.options.with_reg 
        # Preallocate the Tikhonov weight Matrix
        Γ = spzeros(d)

        # Construct the Tikhonov matrix
        tikhonov_matrix!(Γ, dims, operator_symbols, obj.options.λ)
        Γ = spdiagm(0 => Γ)  # convert to sparse diagonal matrix

        Ot = tikhonov(R, D, Γ;
                      tol=obj.options.tolerance,
                      use_gpu=obj.options.use_gpu,
                      use_normal_form=obj.options.use_normal_equations,
                      use_svd_truncation=obj.options.use_svd_truncation,
                      use_backslash=obj.options.use_backslash,
                      chunk_size=obj.options.chunk_size,
                      max_iterations=obj.options.max_iterations,
                      estimate_memory=obj.options.estimate_memory,
                      preconditioning=obj.options.preconditioning,)
    else
        Ot = standard_least_squares(D, R; 
                                    use_gpu=obj.options.use_gpu, 
                                    use_normal_equations=obj.options.use_normal_equations,
                                    chunk_size=obj.options.chunk_size,
                                    tolerance=obj.options.tolerance,
                                    use_backslash=obj.options.use_backslash,
                                    algorithm=obj.options.algorithm,
                                    estimate_memory=obj.options.estimate_memory)
    end

    # Extract the operators from the operator matrix O
    O = transpose(Ot)

    # Extract the operators
    operators = Operators(O=O)

    # Unpack the operators
    unpack_operators!(operators, O, dims, operator_symbols)

    return operators
end


function compute_stream_operators(obj::OnePassStreamingOpInfSketchy,
    E::AbstractArray{T}, indices::Union{Array{<:Int},UnitRange};
    U::AbstractArray{T}=[0.0], rank::Int=obj.rmax) where {T<:Real}

    # Diagonalize the singular values
    Σ = copy(obj.Σ)
    Σ_diag = Diagonal(Σ)

    # Extract the appropriate indices
    W = view(obj.W, indices, 1:rank)

    # Construct the reduce data matrix
    r = rank
    m = obj.input_dim
    d = 0  # total dimension of the data matrix
    if obj.options.optim.nonredundant_operators
        d += sum(i != 0 ? binomial(r+i-1, i) : 0 for i in obj.options.system.state)
    else
        d += sum(i != 0 ? Int(r^i) : 0 for i in obj.options.system.state)
    end
    d += sum(i != 0 ? binomial(m+i-1, i) : 0 for i in obj.options.system.control)
    # d += sum(i != 0 ? binomial(r+i-1, i) * m : 0 for i in obj.options.system.coupled_input)
    d += iszero(obj.options.system.constant) ? 0 : 1
    K = min(obj.num_of_snapshots, length(indices))  # number of snapshots
    D = zeros(T, K, d)  # data matrix
    tmp = 0

    dims = []
    operator_symbols = []

    for i in obj.options.system.state
        if i == 1
            D[:, 1:r] = W * Σ_diag
            push!(dims, r)
            push!(operator_symbols, :A)
            tmp += r
        else
            if obj.options.optim.nonredundant_operators
                ri = binomial(r+i-1, i)
                D[:, tmp+1:tmp+ri] = ⧁(W, i) * Diagonal(⊘(Σ, i))
                push!(dims, ri)
                push!(operator_symbols, Symbol("A$(i)u"))
            else
                ri = Int(r^i)
                D[:, tmp+1:tmp+ri] = ⊖(W, i) * Diagonal(⊗(Σ[:,:], i)[:])
                push!(dims, ri)
                push!(operator_symbols, Symbol("A$(i)"))
            end
            tmp += ri
        end

        if i == 1 && obj.input_dim != 0
            # NOTE: Only works for linear inputs (for now)
            U = fat2tall(U)
            if !iszero(obj.options.system.control)
                D[:, tmp+1:tmp+m] = view(U, indices, :)
                tmp += m
                push!(dims, m)
                push!(operator_symbols, :B)
            end
            # NOTE: Coupled inputs are not implemented yet
        end
    end

    if !iszero(obj.options.system.constant)
        D[:, tmp+1] = ones(K)  # constant term
        push!(dims, 1)
        push!(operator_symbols, :K)
    end

    # Construct the reduced right-hand side matrix
    n_tmp = size(E,1)
    R = Array(E' * view(obj.W, 1:n_tmp, :) * Σ_diag)

    # compute least squares (pseudo inverse)
    if obj.options.with_reg 
        # Preallocate the Tikhonov weight Matrix
        Γ = spzeros(d)

        # Construct the Tikhonov matrix
        tikhonov_matrix!(Γ, dims, operator_symbols, obj.options.λ)
        Γ = spdiagm(0 => Γ)  # convert to sparse diagonal matrix
        Ot = tikhonov(R, D, Γ;
                      tol=obj.options.tolerance,
                      use_gpu=obj.options.use_gpu,
                      use_normal_form=obj.options.use_normal_equations,
                      use_svd_truncation=obj.options.use_svd_truncation,
                      use_backslash=obj.options.use_backslash,
                      chunk_size=obj.options.chunk_size,
                      max_iterations=obj.options.max_iterations,
                      estimate_memory=obj.options.estimate_memory,
                      preconditioning=obj.options.preconditioning,)
    else
        Ot = standard_least_squares(D, R; 
                                    use_gpu=obj.options.use_gpu, 
                                    use_normal_equations=obj.options.use_normal_equations,
                                    chunk_size=obj.options.chunk_size,
                                    tolerance=obj.options.tolerance,
                                    use_backslash=obj.options.use_backslash,
                                    algorithm=obj.options.algorithm,
                                    estimate_memory=obj.options.estimate_memory)
    end

    # Extract the operators from the operator matrix O
    O = transpose(Ot)

    # Extract the operators
    operators = Operators(O=O)

    # Unpack the operators
    unpack_operators!(operators, O, dims, operator_symbols)

    return operators
end