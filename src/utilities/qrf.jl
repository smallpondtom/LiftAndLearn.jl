"""
    qrf!(P::AbstractArray{T}, R::AbstractArray{T}) where {T<:Number}

Faster QR factorization that returns Q without processing the Householder vectors.

Reference:
https://github.com/JuliaLinearAlgebra/IncrementalSVD.jl/blob/da75cd435ed3f57bc56afab3d2faec7155a9b913/src/IncrementalSVD.jl#L207C1-L217C4
"""
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

"""
    qrf!(P::AbstractArray{<:Number})

Dispatch that computes the QR factorization in place, returning the orthogonal matrix Q.
"""
function qrf!(P::AbstractArray{<:Number})
    if issparse(P) # If P is sparse, convert it to dense.
        P = Matrix(P)
    end
    m, b = checksize(P)
    m >= b || throw(DimensionMismatch("Works only for m ≥ b"))
    P, tau = LAPACK.geqrf!(P)
    LAPACK.orgqr!(P, tau)
end