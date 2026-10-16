export OnePassStreamingOpInf

"""
$(TYPEDEF)

One-pass Streaming OpInf state when the time derivative is approximated by finite differences
(`finite_diff=true`, Baker iSVD).

Only the snapshot matrix is compressed on the fly with an incremental SVD
``\\mathbf{X} \\approx \\mathbf{V}\\boldsymbol{\\Sigma}\\mathbf{W}^\\top``. Because the reduced
snapshots are ``\\hat{\\mathbf{X}}^\\top = \\mathbf{W}\\boldsymbol{\\Sigma}``, the reduced
time-derivative data can be formed afterwards by applying a finite-difference matrix to
``\\mathbf{W}\\boldsymbol{\\Sigma}`` without revisiting the full-order data.
Created by [`OnePassStreamingOpInf`](@ref).

## Fields
- `V::AbstractArray{T}`: left singular vectors (POD basis), ``n \\times r_s``.
- `Σ::AbstractVector{T}`: singular values, length ``r_s``.
- `W::AbstractArray{T}`: right singular vectors, ``K \\times r_s`` (one row per snapshot).
- `state_dim::Int`: full state dimension ``n``.
- `input_dim::Int`: input dimension ``m``.
- `num_of_snapshots::Int`: number of snapshots streamed so far ``K``.
- `r_s::Int`: current rank of the iSVD.
- `rmax::Int`: maximum rank kept by the iSVD (truncation rank).
- `Cs`, `q1cache`, `q2cache`, `xperp_cache`: preallocated work arrays.
- `options::LSOpInfOption`: standard OpInf options defining the model structure.
"""
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

"""
$(TYPEDEF)

One-pass Streaming OpInf state when time-derivative data are streamed alongside the snapshots
(`finite_diff=false`, Baker iSVD).

Both the snapshot matrix ``\\mathbf{X} \\approx \\mathbf{V}\\boldsymbol{\\Sigma}\\mathbf{W}^\\top``
and the derivative matrix ``\\dot{\\mathbf{X}} \\approx \\mathbf{P}\\mathbf{S}\\mathbf{Q}^\\top``
are compressed with incremental SVDs. Created by [`OnePassStreamingOpInf`](@ref).

## Fields
- `V::AbstractArray{T}`, `Σ::AbstractVector{T}`, `W::AbstractArray{T}`: iSVD of the snapshots.
- `P::AbstractArray{T}`, `S::AbstractVector{T}`, `Q::AbstractArray{T}`: iSVD of the derivatives.
- `state_dim::Int`: full state dimension ``n``.
- `input_dim::Int`: input dimension ``m``.
- `r_s::Int`: current rank of the snapshot iSVD.
- `r_d::Int`: current rank of the derivative iSVD.
- `rmax::Int`: maximum rank kept by both iSVDs.
- `Cs`, `Cd`, `q1cache`, `q2cache`, `xperp_cache`: preallocated work arrays.
- `options::LSOpInfOption`: standard OpInf options defining the model structure.
"""
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

"""
$(TYPEDEF)

One-pass Streaming OpInf state based on a randomized sketch of the snapshot matrix
(`isvd_method=:sketchy`).

Each streamed snapshot updates three linear sketches (range, co-range, and core) with sparse
random sign test matrices; the truncated SVD of the snapshot matrix is recovered at the end
with [`compute_svd_sketchy!`](@ref). The total number of snapshots must be known in advance.
Created by [`OnePassStreamingOpInf`](@ref).

## Fields
- `V::AbstractArray{T}`, `Σ::AbstractVector{T}`, `W::AbstractArray{T}`: truncated SVD of the
  snapshot matrix (populated by [`compute_svd_sketchy!`](@ref)).
- `Ξ::AbstractArray{T}`: range test matrix, ``k \\times n``.
- `Ω::AbstractArray{T}`: co-range test matrix, ``k \\times K``.
- `Φ::AbstractArray{T}`: core test matrix, ``s \\times n``.
- `Ψ::AbstractArray{T}`: core test matrix, ``s \\times K``.
- `Xcorange::AbstractArray{T}`: co-range sketch, ``k \\times K``.
- `Xrange::AbstractArray{T}`: range sketch, ``n \\times k``.
- `Xcore::AbstractArray{T}`: core sketch, ``s \\times s``.
- `H::AbstractArray{T}`: sparse innovation buffer, ``n \\times K``.
- `k::Int`: range sketch size.
- `s::Int`: core sketch size.
- `input_dim::Int`: input dimension ``m``.
- `rmax::Int`: target rank.
- `num_of_snapshots::Int`: number of snapshots streamed so far.
- `options::LSOpInfOption`: standard OpInf options defining the model structure.

## References
- Tropp, J. A., Yurtsever, A., Udell, M., and Cevher, V. Streaming low-rank matrix
  approximation with an application to scientific simulation. SIAM Journal on Scientific
  Computing, 41(4):A2430–A2463, 2019.
"""
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


"""
$(SIGNATURES)

Create a one-pass Streaming OpInf object that compresses snapshot data on the fly.

In one-pass Streaming OpInf every full-order snapshot is seen exactly once: it updates a
low-rank (incremental or sketched) SVD of the snapshot matrix and is then discarded. After all
data has been streamed, the POD basis and the reduced data needed by OpInf are read off the
compressed factors and the reduced operators are computed with
[`compute_stream_operators`](@ref).

## Arguments
- `x_init::AbstractVector{T}=Float64[]`: the first snapshot (required, nonzero, length
  `n_state` for `isvd_method=:baker`; ignored for `:sketchy`).

## Keyword Arguments
- `xdot_init::AbstractVector{T}=T[1]`: the first time-derivative snapshot (only used when
  `finite_diff=false`).
- `options::LSOpInfOption`: standard OpInf options defining the model structure.
- `n_state::Int`: full state dimension ``n``.
- `n_input::Int=0`: input dimension ``m``.
- `n_snapshots::Int=1`: total number of snapshots ``K`` (required for `:sketchy`).
- `rank::Int=1`: maximum (truncation) rank of the SVD.
- `finite_diff::Bool=true`: if `true`, only snapshots are streamed and time derivatives are
  later approximated by a finite-difference matrix; if `false`, derivative snapshots are
  streamed too (`:baker` only).
- `isvd_method::Symbol=:baker`: `:baker` (deterministic incremental SVD of Baker et al.) or
  `:sketchy` (randomized streaming sketch of Tropp et al.).
- `sketch_size::Tuple{Int,Int,Int}=(0,0,0)`: `(k, s, ζ)` for `:sketchy` (range sketch size,
  core sketch size, nonzeros per column of the sparse test matrices). Zeros select the defaults
  `k = 4rank + 1`, `s = 2k + 1`, `ζ = min(k, 8)`.

## Returns
- `obj`: an [`OnePassStreamingOpInf1`](@ref) (`:baker`, `finite_diff=true`),
  [`OnePassStreamingOpInf2`](@ref) (`:baker`, `finite_diff=false`), or
  [`OnePassStreamingOpInfSketchy`](@ref) (`:sketchy`).

## Examples
```julia
stream = OnePassStreamingOpInf(X[:, 1]; options=options, n_state=n, n_input=1, rank=10)
for k in 2:size(X, 2)
    LiftAndLearn.stream!(stream, X[:, k])
end
E, idx = LiftAndLearn.finite_diff_matrix(options.data.deriv_type, size(X, 2), Δt)
op = LiftAndLearn.compute_stream_operators(stream, E, idx; U=U)
Vr = stream.V   # POD basis
```

## References
- Baker, C. G., Gallivan, K. A., and Van Dooren, P. Low-rank incremental methods for
  computing dominant singular subspaces. Linear Algebra and its Applications, 436(8), 2012.
- Tropp, J. A., Yurtsever, A., Udell, M., and Cevher, V. Streaming low-rank matrix
  approximation with an application to scientific simulation. SIAM Journal on Scientific
  Computing, 41(4), 2019.
"""
function OnePassStreamingOpInf(
    x_init::AbstractVector{T}=Float64[];     # Initial state data
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

"""
$(SIGNATURES)

Generate a sparse random sign matrix with exactly `zeta` nonzeros (``\\pm 1``) per column,
used as a test matrix by the sketchy streaming SVD. Internal helper.

## Arguments
- `k::Int`: number of rows.
- `n::Int`: number of columns.
- `zeta::Int`: nonzeros per column, ``1 \\le \\zeta \\le k``.

## Keyword Arguments
- `field::String="real"`: `"real"` for ``\\pm 1`` entries or `"complex"` for unit-modulus
  complex entries.

## Returns
- `S::SparseMatrixCSC`: the ``k \\times n`` sparse sign matrix.
"""
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

"""
$(SIGNATURES)

Compute the full SVD of the small sparse "broken arrowhead" matrix arising in an incremental
SVD update. Internal helper.

PROPACK's `tsvd` is tried first; if it fails (e.g., an invariant subspace is found because the
new snapshot lies in the current subspace), the dense LAPACK SVD is used instead.

## Arguments
- `C::AbstractMatrix`: the ``(r+1) \\times (r+1)`` arrowhead matrix.

## Returns
- `(U, S, V)`: left singular vectors, singular values (descending), right singular vectors.
"""
function arrowhead_svd(C::AbstractMatrix)
    k = minimum(size(C))
    try
        U, S, V, _, _, _ = tsvd(C, k=k)
        return U, S, V
    catch
        @debug "PROPACK tsvd failed for the arrowhead matrix; using dense svd instead."
        F = svd(Matrix(C))
        return F.U, F.S, F.V
    end
end


"""
$(SIGNATURES)

Stream one full-order snapshot `x` into a one-pass object (finite-difference variant), updating
its incremental SVD.

The new snapshot is orthogonalized twice against the current basis, the small arrowhead
matrix is diagonalized, the factors ``\\mathbf{V}, \\boldsymbol{\\Sigma}, \\mathbf{W}`` are
updated and truncated to `obj.rmax`, and the basis is re-orthogonalized if needed.

## Arguments
- `obj::OnePassStreamingOpInf1`: the one-pass object (mutated).
- `x::AbstractVector{T}`: the next snapshot (length ``n``).

## Keyword Arguments
- `tol::T=1e-12`: orthogonality tolerance that triggers re-orthogonalization of ``\\mathbf{V}``.

## Returns
- `nothing`; mutates `obj`.
"""
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

    # SVD of the broken arrowhead matrix
    Vc, Σc, Wc = arrowhead_svd(obj.Cs[1:r1+1, 1:r1+1])

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


"""
$(SIGNATURES)

Stream one full-order snapshot `x` and its time derivative `xdot` into a one-pass object,
updating the incremental SVDs of both the snapshot and derivative matrices.

## Arguments
- `obj::OnePassStreamingOpInf2`: the one-pass object (mutated).
- `x::AbstractVector{T}`: the next snapshot (length ``n``).
- `xdot::AbstractVector{T}`: the time derivative at that snapshot (length ``n``).

## Keyword Arguments
- `tol::T=1e-12`: orthogonality tolerance that triggers re-orthogonalization of the bases.

## Returns
- `nothing`; mutates `obj`.
"""
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

    # SVD of the broken arrowhead matrix
    Vc, Σc, Wc = arrowhead_svd(obj.Cs[1:r1+1, 1:r1+1])

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

    Pc, Sc, Qc = arrowhead_svd(obj.Cd[1:r2+1, 1:r2+1])
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

"""
$(SIGNATURES)

Stream a block of consecutive snapshots (columns of `x`) into a sketchy one-pass object,
updating the range, co-range, and core sketches.

## Arguments
- `obj::OnePassStreamingOpInfSketchy`: the one-pass object (mutated).
- `x::AbstractMatrix{T}`: the next ``n \\times b`` block of snapshots.

## Returns
- `nothing`; mutates `obj`.
"""
function stream!(obj::OnePassStreamingOpInfSketchy,
                 x::AbstractMatrix{T}) where {T<:Real}

    # Form the innovation
    i = obj.num_of_snapshots + 1  # current stream index
    d = size(x,2)
    obj.H[:, i:i+d-1] = x

    # Update the sketches
    obj.Xcorange .+= obj.Ξ * obj.H
    obj.Xrange   .+= obj.H * obj.Ω'
    obj.Xcore    .+= (obj.Φ * obj.H) * obj.Ψ'

    # Reset the innovation
    obj.H[:, i:i+d-1] .= zero(T)
    dropzeros!(obj.H)

    # Increment the number of streams
    obj.num_of_snapshots += d

    return nothing
end

"""
$(SIGNATURES)

Stream one snapshot `x` into a sketchy one-pass object, updating the range, co-range, and core
sketches in place (exploiting the sparsity of the test matrices).

## Arguments
- `obj::OnePassStreamingOpInfSketchy`: the one-pass object (mutated).
- `x::AbstractVector{T}`: the next snapshot (length ``n``).

## Returns
- `nothing`; mutates `obj`.
"""
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

"""
$(SIGNATURES)

Compute the reduced operators from a finite-difference one-pass object using the contiguous
snapshot range `indices[1]:indices[2]`.

Same as the `indices::AbstractVector` method but takes the range as a tuple.

## Arguments
- `obj`: the one-pass object after all snapshots have been streamed.
- `E::AbstractArray{T}`: finite-difference matrix of size ``K \\times K'`` such that
  ``\\mathbf{X}\\mathbf{E}`` approximates the time derivatives (see
  [`finite_diff_matrix`](@ref)). For several trajectories use a block-diagonal matrix.
- `indices::Tuple{<:Int,<:Int}`: first and last snapshot index paired with the columns of `E`.

## Keyword Arguments
- `U::AbstractArray{T}=[0.0]`: input data with one row per snapshot (``K \\times m``; an
  ``m \\times K`` matrix is transposed automatically). Only linear inputs (`B`) are supported.
- `rank::Int=obj.rmax`: reduced dimension ``r \\le`` `obj.rmax` used to build the model.

## Returns
- `operators::Operators`: the learned reduced operators (`A`, `B`, `A2u`, `K`, ...) with the
  full operator matrix in `operators.O`.
"""
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


"""
$(SIGNATURES)

Compute the reduced operators from a finite-difference one-pass object using the snapshots at
the positions `indices`.

The reduced snapshots are ``\\hat{\\mathbf{X}}^\\top = \\mathbf{W}\\boldsymbol{\\Sigma}`` and
the reduced time derivatives are ``\\mathbf{E}^\\top\\mathbf{W}\\boldsymbol{\\Sigma}``; the
least-squares (optionally Tikhonov-regularized, per `obj.options`) OpInf problem is then solved.

## Arguments
- `obj`: the one-pass object after all snapshots have been streamed.
- `E::AbstractArray{T}`: finite-difference matrix of size ``K \\times K'`` such that
  ``\\mathbf{X}\\mathbf{E}`` approximates the time derivatives (see
  [`finite_diff_matrix`](@ref)). For several trajectories use a block-diagonal matrix.
- `indices::Union{Array{<:Int},UnitRange}`: snapshot indices (rows of ``\\mathbf{W}``) paired
  with the columns of `E`, e.g. the index range returned by [`finite_diff_matrix`](@ref).

## Keyword Arguments
- `U::AbstractArray{T}=[0.0]`: input data with one row per snapshot (``K \\times m``; an
  ``m \\times K`` matrix is transposed automatically). Only linear inputs (`B`) are supported.
- `rank::Int=obj.rmax`: reduced dimension ``r \\le`` `obj.rmax` used to build the model.

## Returns
- `operators::Operators`: the learned reduced operators (`A`, `B`, `A2u`, `K`, ...) with the
  full operator matrix in `operators.O`.
"""
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


"""
$(SIGNATURES)

Compute the reduced operators from a one-pass object that streamed both snapshots and time
derivatives (`finite_diff=false`).

The reduced snapshots are ``\\mathbf{W}\\boldsymbol{\\Sigma}`` and the reduced time
derivatives are ``\\mathbf{Q}\\mathbf{S}\\mathbf{P}^\\top\\mathbf{V}``, both obtained from the
incremental SVD factors.

## Arguments
- `obj::OnePassStreamingOpInf2`: the one-pass object after streaming.

## Keyword Arguments
- `U::AbstractArray{T}=[0.0]`: input data with one row per snapshot (``K \\times m``; an
  ``m \\times K`` matrix is transposed automatically). Only linear inputs (`B`) are supported.
- `rank::Int=obj.rmax`: reduced dimension ``r \\le`` `obj.rmax` used to build the model.

## Returns
- `operators::Operators`: the learned reduced operators (`A`, `B`, `A2u`, `K`, ...) with the
  full operator matrix in `operators.O`.
"""
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

"""
$(SIGNATURES)

Recover the truncated SVD ``\\mathbf{X} \\approx \\mathbf{V}\\boldsymbol{\\Sigma}\\mathbf{W}^\\top``
of the streamed snapshot matrix from the sketches of a sketchy one-pass object.

## Arguments
- `obj::OnePassStreamingOpInfSketchy`: the one-pass object after streaming (mutated).

## Keyword Arguments
- `rank::Int=obj.rmax`: truncation rank.

## Returns
- `nothing`; sets `obj.V`, `obj.Σ`, and `obj.W`.
"""
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

"""
$(SIGNATURES)

Compute the reduced operators from a sketchy one-pass object using the contiguous snapshot
range `indices[1]:indices[2]`.

Call [`compute_svd_sketchy!`](@ref) first, with the same `rank`.

## Arguments
- `obj`: the one-pass object after all snapshots have been streamed.
- `E::AbstractArray{T}`: finite-difference matrix of size ``K \\times K'`` such that
  ``\\mathbf{X}\\mathbf{E}`` approximates the time derivatives (see
  [`finite_diff_matrix`](@ref)). For several trajectories use a block-diagonal matrix.
- `indices::Tuple{<:Int,<:Int}`: first and last snapshot index paired with the columns of `E`.

## Keyword Arguments
- `U::AbstractArray{T}=[0.0]`: input data with one row per snapshot (``K \\times m``; an
  ``m \\times K`` matrix is transposed automatically). Only linear inputs (`B`) are supported.
- `rank::Int=obj.rmax`: reduced dimension ``r \\le`` `obj.rmax` used to build the model.

## Returns
- `operators::Operators`: the learned reduced operators (`A`, `B`, `A2u`, `K`, ...) with the
  full operator matrix in `operators.O`.
"""
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


"""
$(SIGNATURES)

Compute the reduced operators from a sketchy one-pass object using the snapshots at the
positions `indices`.

Call [`compute_svd_sketchy!`](@ref) first, with the same `rank`.

## Arguments
- `obj`: the one-pass object after all snapshots have been streamed.
- `E::AbstractArray{T}`: finite-difference matrix of size ``K \\times K'`` such that
  ``\\mathbf{X}\\mathbf{E}`` approximates the time derivatives (see
  [`finite_diff_matrix`](@ref)). For several trajectories use a block-diagonal matrix.
- `indices::Union{Array{<:Int},UnitRange}`: snapshot indices (rows of ``\\mathbf{W}``) paired
  with the columns of `E`, e.g. the index range returned by [`finite_diff_matrix`](@ref).

## Keyword Arguments
- `U::AbstractArray{T}=[0.0]`: input data with one row per snapshot (``K \\times m``; an
  ``m \\times K`` matrix is transposed automatically). Only linear inputs (`B`) are supported.
- `rank::Int=obj.rmax`: reduced dimension ``r \\le`` `obj.rmax` used to build the model.

## Returns
- `operators::Operators`: the learned reduced operators (`A`, `B`, `A2u`, `K`, ...) with the
  full operator matrix in `operators.O`.
"""
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

"""
$(SIGNATURES)

Free the random test matrices, sketches, and innovation buffer of a sketchy one-pass object
(after [`compute_svd_sketchy!`](@ref)) to release memory. The SVD factors are kept.

## Arguments
- `obj::OnePassStreamingOpInfSketchy`: the one-pass object (mutated).

## Returns
- `nothing`; mutates `obj`.
"""
function purge!(obj::OnePassStreamingOpInfSketchy)
    obj.Φ = zeros(eltype(obj.Φ), 0, 0)
    obj.Ψ = zeros(eltype(obj.Ψ), 0, 0)
    obj.Ξ = zeros(eltype(obj.Ξ), 0, 0)
    obj.Ω = zeros(eltype(obj.Ω), 0, 0)
    obj.H = zeros(eltype(obj.H), 0, 0)
    obj.Xcore = zeros(eltype(obj.Xcore), 0, 0)
    obj.Xrange = zeros(eltype(obj.Xrange), 0, 0)
    obj.Xcorange = zeros(eltype(obj.Xcorange), 0, 0)
    return nothing
end