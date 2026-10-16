"""
$(SIGNATURES)

Compute a thin QR factorization of `P` in place, storing ``\\mathbf{Q}`` in `P` and
``\\mathbf{R}`` in `R`.

Uses LAPACK `geqrf!` followed by `orgqr!`, which forms the explicit thin ``\\mathbf{Q}``
directly (faster than materializing it from a `LinearAlgebra.QR` object). For an
``m \\times b`` input with ``m \\ge b``, on exit `P` holds ``\\mathbf{Q}`` (``m \\times b``, orthonormal
columns) and `R` holds the ``b \\times b`` upper-triangular factor, so that
``\\mathbf{P}_0 = \\mathbf{Q}\\mathbf{R}``.

## Arguments
- `P::AbstractArray{T}`: ``m \\times b`` matrix with ``m \\ge b`` (LAPACK-compatible, e.g. `Matrix{Float64}`).
  Overwritten with the orthonormal factor ``\\mathbf{Q}``.
- `R::AbstractArray{T}`: preallocated matrix of at least ``b \\times b``. It is zero-filled and its
  leading ``b \\times b`` upper triangle is overwritten with ``\\mathbf{R}``.

## Returns
- `R::AbstractArray{T}`: the upper-triangular factor (the same object as the input `R`).

## Notes
- Mutates both `P` (to ``\\mathbf{Q}``) and `R`.
- If `P` is sparse it is first copied to a dense matrix; in that case the caller's `P` is **not**
  overwritten and ``\\mathbf{Q}`` is discarded (only `R` is returned).
- Throws `DimensionMismatch` if ``m < b``.

## References
- Adapted from IncrementalSVD.jl:
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
$(SIGNATURES)

Compute the thin orthonormal factor ``\\mathbf{Q}`` of the QR factorization of `P` in place.

Same as [`qrf!(P, R)`](@ref qrf!) but without extracting ``\\mathbf{R}``: on exit `P` holds the
``m \\times b`` matrix ``\\mathbf{Q}`` with orthonormal columns spanning the column space of the
input.

## Arguments
- `P::AbstractArray{<:Number}`: ``m \\times b`` matrix with ``m \\ge b`` (LAPACK-compatible).
  Overwritten with ``\\mathbf{Q}``.

## Returns
- `Q::AbstractMatrix`: the orthonormal factor (the same object as `P` for dense input).

## Notes
- If `P` is sparse it is first copied to a dense matrix; the caller's `P` is then not modified
  and ``\\mathbf{Q}`` is only available through the return value.
- Throws `DimensionMismatch` if ``m < b``.
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