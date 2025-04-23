"""
    Brand{T<:Number} <: iSVDAlgorithm

An implementation of the incremental Singular Value Decomposition (iSVD) algorithm based on Brand's method for 
online Proper Orthogonal Decomposition (POD). This version dynamically updates the SVD components (V, Σ, and W) without 
forcing a fixed preallocation for them, thereby preserving numerical accuracy, while employing preallocated caches 
for intermediate computations to improve efficiency.

# Fields

- `V::AbstractArray{T}`:  
  The matrix of left singular vectors. This matrix is dynamically resized during updates to reflect the current 
  effective subspace.

- `Σ::Vector{T}`:  
  The vector of singular values associated with the current subspace. It is updated and resized as needed with each 
  incremental update.

- `W::AbstractArray{T}`:  
  The matrix of right singular vectors. This matrix is also dynamically resized during updates.

- `reorth_method::Symbol`:  
  The reorthogonalization method to use. Accepted values are `:gramschmidt` and `:qr`.

- `max_rank::Int`:  
  The maximum allowable rank for the decomposition. When updates cause the rank to exceed this value, the SVD 
  components are truncated to retain only the leading `max_rank` components.

# Caches for Intermediate Computations

To minimize memory allocations during iterative updates, the following caches are preallocated:

- `cache_d::Vector{T}`:  
  A vector used to store the projection coefficients ``d = V^\\top x``. Its length is set to `max_rank`, and only the 
  first _k_ entries (where _k_ is the current rank) are used.

- `cache_Vd::Vector{T}`:  
  A vector for caching the product ``V \\cdot d``. Its length matches the number of rows of `V` (i.e. the dimension 
  of the data).

- `cache_e::Vector{T}`:  
  A vector to hold the residual ``e = x - V \\cdot d`` computed during the update. Its length is equal to the data 
  dimension.

- `cache_Y::Matrix{T}`:  
  A square matrix of size ``(\\mathtt{max\\_rank}+1) \\times (\\mathtt{max\\_rank}+1)`` used for constructing the small 
  update matrix ``Y`` whose singular value decomposition provides the updated singular values and rotation matrix.

- `cache_V_temp::Matrix{T}`:  
  A temporary matrix used for intermediate multiplications during the SVD update. Its dimensions are 
  (number of rows of `V`) x ``(\\mathtt{max\\_rank}+1)``.

- `cache_Vhat::Matrix{T}`:  
  A matrix used to form the augmented basis ``[V \\; e]`` prior to applying the rotation from the SVD of ``Y``. 
  Its size is (number of rows of `V`) x ``(\\mathtt{max\\_rank}+1)``.

# References

- [Brand2002] M. Brand, “Incremental Singular Value Decomposition of Uncertain Data with Missing Values,” ECCV 
  2002.
- [Fareed2018] H. Fareed, et al., “Incremental proper orthogonal decomposition for PDE simulation data,” Computers & 
  Mathematics with Applications, 2018.
- [Zhang2022] Y. Zhang, “An answer to an open question in the incremental SVD,” arXiv, 2022.
"""

mutable struct Brand{T<:Number} <: iSVDAlgorithm
    # SVD components
    V::AbstractArray{T}        # Left singular vectors (dynamically resized)
    Σ::Vector{T}               # Singular values (dynamically resized)
    W::AbstractArray{T}        # Right singular vectors (dynamically resized)

    # Reorthogonalization method
    reorth_method::Symbol

    # Maximum rank
    max_rank::Int
    
    # Caches for intermediate computations
    cache_d::Vector{T}         # Cache for projection coefficients (length max_rank)
    cache_Vd::Vector{T}        # Cache for computing V*d (length = number of rows)
    cache_e::Vector{T}         # Cache for the residual (length = number of rows)
    cache_Y::Matrix{T}         # Cache for the small update matrix Y (size (max_rank+1)×(max_rank+1))
    cache_V_temp::Matrix{T}    # Cache for temporary multiplication (size: m × (max_rank+1))
    cache_Vhat::Matrix{T}      # Cache for the augmented basis (size: m × (max_rank+1))
end

function initialize_brand(x1::AbstractVector{T}; reorth_method::Symbol=:gramschmidt, 
                           max_rank::Int=length(x1)) where {T<:Number}
    @assert !all(x1 .== 0) "x1 must be a nonzero vector."
    @assert reorth_method in [:gramschmidt, :qr] "Invalid reorthogonalization method."

    # Compute initial singular value using the Euclidean norm.
    s = norm(x1)
    Σ = [s]

    # Compute initial V.
    V = x1 / Σ

    # Compute the initial W matrix
    W = Matrix{T}(I, 1, 1)  

    # Determine the dimension (number of rows)
    m = length(x1)

    # Allocate caches based on the maximum rank and m.
    cache_d      = zeros(T, max_rank)  # Will use first k entries where k = current rank.
    cache_Vd     = zeros(T, m)
    cache_e      = zeros(T, m)
    cache_Y      = zeros(T, max_rank+1, max_rank+1)
    cache_V_temp = zeros(T, m, max_rank+1)
    cache_Vhat   = zeros(T, m, max_rank+1)

    return Brand(
      V, Σ, W, reorth_method, max_rank, cache_d, cache_Vd, 
      cache_e, cache_Y, cache_V_temp, cache_Vhat
    )
end

function increment!(obj::Brand{T}, x::AbstractVector{T1}, tol::Real = 1e-12) where {T<:Number, T1<:Number}
    # Dimensions of the current SVD subspace.
    m, k = size(obj.V)
    l = size(obj.W, 1)

    # Use the cached vector 'cache_d' for projection coefficients.
    d = @view obj.cache_d[1:k]   # d has length k.
    mul!(d, obj.V', x)           # d = V' * x

    # Compute V*d and store in the cached vector 'cache_Vd'.
    Vd = @view obj.cache_Vd[1:m]
    mul!(Vd, obj.V, d)

    # Compute the residual: e = x - V*d, storing the result in the cache 'cache_e'.
    copy!(obj.cache_e, x)
    axpy!(-1.0, Vd, obj.cache_e)  # obj.cache_e = x - Vd

    # Compute the Euclidean norm of the residual.
    p = norm(obj.cache_e)
    if p < tol
        p = zero(T)
    else
        # Normalize the residual in place.
        obj.cache_e .= obj.cache_e ./ p
    end

    # Construct the small (k+1)×(k+1) update matrix Y using the cache.
    Y = @view obj.cache_Y[1:(k+1), 1:(k+1)]
    fill!(Y, zero(T))
    # Set the top-left k×k block to be a diagonal matrix with the current singular values.
    for i in 1:k
        Y[i, i] = obj.Σ[i]
    end
    # Set the first k entries of the last column to the coefficients d.
    for i in 1:k
        Y[i, k+1] = d[i]
    end
    # Set the bottom-right element to p.
    Y[k+1, k+1] = p

    # Compute the SVD of Y. (Convert Y to a regular matrix for SVD.)
    Vy, Σy, Wy = svd(Matrix(Y))

    if p < tol
        # Case 1: No rank expansion.
        # Compute the updated V as: V = V * Vy[1:k, 1:k]
        V_temp = @view obj.cache_V_temp[1:m, 1:k]
        mul!(V_temp, obj.V, @view Vy[1:k, 1:k])
        # Update V in-place (the shape remains m×k).
        obj.V .= V_temp

        # Update singular values in place.
        resize!(obj.Σ, k)
        obj.Σ .= Σy[1:k]

        # Compute the update W in-place as: W = W * Wy[1:k, 1:k]
        W_temp = zeros(T, l + 1, k + 1)
        @views W_temp[1:l, 1:k] .= obj.W
        W_temp[l + 1, k + 1] = 1.0  # Set the last element to 1
        obj.W = W_temp * Wy[:, 1:k]
    else
        # Case 2: The new vector increases the rank.
        # Form the augmented matrix Vhat = [V  normalized e] using the cache.
        Vhat = @view obj.cache_Vhat[1:m, 1:(k+1)]
        Vhat[:, 1:k] .= obj.V          # Copy the current V into the first k columns.
        Vhat[:, k+1] .= obj.cache_e     # The last column is the normalized residual.

        # Update V as: V = Vhat * Vy
        V_temp = @view obj.cache_V_temp[1:m, 1:(k+1)]
        mul!(V_temp, Vhat, Vy)
        # Replace V with the updated matrix (reallocating if necessary).
        obj.V = copy(V_temp)

        # Increase the number of singular values by one.
        resize!(obj.Σ, k+1)
        obj.Σ .= Σy

        # Update W (since the num of snapshots increase we cannot use cache)
        W_temp = zeros(T, l + 1, k + 1)
        @views W_temp[1:l, 1:k] .= obj.W
        W_temp[l + 1, k + 1] = 1.0  # Set the last element to 1
        obj.W = W_temp * Wy

        # If the new rank exceeds max_rank, truncate to max_rank.
        if (k+1) > obj.max_rank
            obj.V = obj.V[:, 1:obj.max_rank]
            obj.Σ = obj.Σ[1:obj.max_rank]
            obj.W = obj.W[:, 1:obj.max_rank]  # Resize W to match the new rank.
        end
    end

    return nothing
end

function full_increment!(obj::Brand{T}, X::AbstractArray{T1}; tol::Real = 1e-12, 
                         verbose::Bool = false, runtime::Bool = false) where {T<:Number, T1<:Number}
    # Incremental POD over all columns in X.
    K = size(X, 2)
    if verbose
        p = Progress(K; desc = "Incrementing iSVD...")
    end
    t = runtime ? Vector{Float32}(undef, K) : nothing  # Optionally preallocate timing info.
    i = 0
    for x in eachcol(X)
        if runtime
            t[i += 1] = @elapsed increment!(obj, x, tol)
        else
            increment!(obj, x, tol)
        end

        # Reorthogonalize if necessary (using the standard inner product).
        if obj.reorth_method == :gramschmidt
            reorthogonalize!(obj.V, tol)
        elseif obj.reorth_method == :qr
            if abs(dot(obj.V[:, end], obj.V[:, 1])) > tol
                qrf!(obj.V)
            end
        end
        if verbose
            next!(p)
        end
    end
    return runtime ? t : nothing
end