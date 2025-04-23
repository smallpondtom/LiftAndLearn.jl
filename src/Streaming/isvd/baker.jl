"""
    Baker

The Baker algorithm is an incremental method for computing the dominant singular subspace of a
matrix. It is based on the incremental SVD algorithm proposed by Baker et al. using the Eigenspace 
Update Algorithm (EUA) [Baker2012], with additional insights from [Zhang2022]. The algorithm 
incrementally updates the singular value decomposition (SVD) of a data matrix as new columns are 
introduced, making it suitable for online or streaming applications where data arrives in sequence.

# Notes
The implementation presented here is modified to:
- Use pre-allocated cache arrays to reduce memory allocations during incremental updates, improving 
  performance.

# Fields
- `V::AbstractMatrix{T}`: The matrix of left singular vectors of size `m x r`, where `m` is the 
  number of rows of the data and `r` is the current rank.
- `Σ::Vector{T}`: The vector of singular values of length `r`.
- `W::AbstractMatrix{T}`: The matrix of the right singular vectors of size `n x r` where `n` is the 
  number of columns of the data and `r` is the current rank.
- `max_rank::Int`: User-selected maximum rank. The algorithm will not exceed this rank and will 
  truncate or filter singular values as necessary to maintain it.
- Caches (for internal computations): To minimize memory allocations and enhance performance, 
  we maintain a set of pre-allocated arrays (caches) reused at every increment step:
    - `Ccache::Vector{T}`: A temporary vector for storing projections of the new column onto the current 
      basis (`V' * x`). Its length is `max_rank+1` to accommodate increments.
    - `C2cache::Vector{T}`: A secondary temporary vector for reorthogonalization corrections.
    - `Vhatcache::Matrix{T}`: A temporary matrix of size `(m x (max_rank+1))` used to store the 
      expanded basis before truncation.
    - `Rhatcache::Matrix{T}`: A temporary `(max_rank+1) x (max_rank+1)` matrix used to form the 
      augmented upper-triangular matrix prior to the small SVD step.
    - `x_perp_tmp::Vector{T}`: A temporary vector of length `m` for storing orthogonalized updates 
      (e.g., `x_perp`).

# References
- [Baker2012] C. G. Baker, K. A. Gallivan, and P. Van Dooren, “Low-rank incremental methods for computing 
  dominant singular subspaces,” Linear Algebra and its Applications, vol. 436, no. 8, pp. 2866–2888, 
  Apr. 2012, doi: 10.1016/j.laa.2011.07.018.
- [Zhang2022] Y. Zhang, “An answer to an open question in the incremental SVD,” Apr. 30, 2022, arXiv: 
  arXiv:2204.05398. doi: 10.48550/arXiv.2204.05398.
"""
mutable struct Baker{T<:Number} <: iSVDAlgorithm
    V::Matrix{T}
    Σ::Vector{T}
    W::Matrix{T}

    max_rank::Int
end

function initialize_baker(x1::AbstractVector{T}; max_rank::Int=0) where {T<:Number}
    m = length(x1)

    # Perform QR decomposition on x1
    Vf, Rf = qr(x1)
    V = Matrix(Vf)
    Σ = [abs(Rf[1])]
    W = Matrix{T}(I, 1, 1)

    if max_rank == 0
        max_rank = m
    end

    return Baker(V, Σ, W, max_rank)
end

function increment!(obj::Baker{T}, x::AbstractVector{T}) where {T<:Number}
    r = size(obj.V,2)
    rmax = obj.max_rank

    q1 = obj.V' * x
    xperp = x - obj.V * q1
    q2 = obj.V' * xperp
    xperp = xperp - obj.V * q2
    q = q1 + q2
    p = norm(xperp)

    p = [p]
    xperp = reshape(xperp, :, 1)
    qrf!(xperp, p)
    p = p[1]

    C = zeros(r+1, r+1)
    for j in 1:r
        C[j,j] = obj.Σ[j]
        C[j,end] = q[j]
    end
    C[end,end] = p

    Vc, Σc, Wc = svd(C)
    obj.V = hcat(obj.V, xperp) * Vc
    obj.Σ = Σc
    obj.W = [obj.W zeros(size(obj.W,1), 1); zeros(1, r) 1.0] * Wc

    if length(obj.Σ) > rmax
        obj.V = obj.V[:,1:rmax]
        obj.Σ = obj.Σ[1:rmax]
        obj.W = obj.W[:,1:rmax]
        r = rmax
    end

    return nothing
end

function full_increment!(obj::Baker{T}, X::AbstractMatrix{T1}; tol::Real=1e-12,
                         verbose::Bool=false, runtime::Bool=false) where {T<:Number, T1<:Number}
    K = size(X, 2)
    if verbose
        p = Progress(K; desc="Incrementing iSVD...")
    end
    t = runtime ? Vector{Float32}(undef, K) : nothing
    i = 0
    for x in eachcol(X)
        if runtime
            t[i+=1] = @elapsed increment!(obj, x)
        else
            increment!(obj, x)
        end
        # Reorthogonalize V (if necessary)
        # INFO: When very small singular values appear, 
        #         reorthogonalization becomes necessary.
        reorthogonalize!(obj.V, tol)
        if verbose
            next!(p)
        end
    end
    if runtime
        return t
    else
        return
    end
end