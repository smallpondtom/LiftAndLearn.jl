"""
$(SIGNATURES)

Compute a numerically robust Givens rotation that annihilates `b`.

The returned cosine `c`, sine `s`, and radius `r` satisfy
```math
\\begin{bmatrix} c & -s \\\\ s & c \\end{bmatrix}
\\begin{bmatrix} a \\\\ b \\end{bmatrix}
=
\\begin{bmatrix} r \\\\ 0 \\end{bmatrix},
\\qquad c^2 + s^2 = 1, \\qquad r = \\sqrt{a^2 + b^2} \\ge 0 .
```
Overflow is avoided by dividing by the larger of ``|a|`` and ``|b|``. Special cases: if
``b = 0`` then ``(c, s, r) = (\\mathrm{sign}(a), 0, |a|)`` (with ``c = 1`` when ``a = 0``), and if
``a = 0`` then ``(c, s, r) = (0, -\\mathrm{sign}(b), |b|)``.

## Arguments
- `a::T`: first entry (the one that is kept), `T<:AbstractFloat`.
- `b::T`: second entry (the one that is zeroed), same type as `a`.

## Returns
- `c::T`: cosine of the rotation.
- `s::T`: sine of the rotation.
- `r::T`: the resulting nonnegative first entry.

## Notes
- The sign convention differs from `LinearAlgebra.givens`; to apply the rotation to two rows
  `x` (row of `a`) and `y` (row of `b`) with `BLAS.rot!`, pass `c` and `-s`.
"""
@inline function givens_rotation(a::T, b::T) where {T<:AbstractFloat}
    zeroT, oneT = zero(T), one(T)
    if b == zeroT
        c = sign(a);
        c = c == zeroT ? oneT : c
        return c, zeroT, abs(a)
    elseif a == zeroT
        return zeroT, -sign(b), abs(b)
    elseif abs(a) > abs(b)
        t = b / a
        u = sign(a) * sqrt(oneT + t * t)
        c = oneT / u
        s = -c * t
        r = a * u
        return c, s, r
    else
        t = a / b
        u = sign(b) * sqrt(oneT + t * t)
        s = -oneT / u
        c = t / u
        r = b * u
        return c, s, r
    end
end


"""
$(SIGNATURES)

Reduce `A` in place to the triangular factor ``\\mathbf{R}`` of its QR factorization using
Givens rotations.

For each column ``i`` the entries below the diagonal are annihilated one at a time by rotating
row ``i`` against row ``j > i`` (see [`givens_rotation`](@ref)), so that on exit
``\\mathbf{A} \\leftarrow \\mathbf{R} = \\mathbf{Q}^\\top \\mathbf{A}_0`` with ``\\mathbf{Q}`` orthogonal.
The orthogonal factor ``\\mathbf{Q}`` is not formed or stored.

## Arguments
- `A::AbstractMatrix{T}`: ``m \\times n`` real matrix (`T<:Real`, must be a BLAS float type such
  as `Float64`, with unit row stride). Overwritten with the upper-triangular (upper-trapezoidal if
  ``m \\ne n``) factor ``\\mathbf{R}``.

## Returns
- `nothing`; mutates `A` in place (``\\mathbf{A}^\\top\\mathbf{A}`` is preserved, i.e.,
  ``\\mathbf{R}^\\top\\mathbf{R} = \\mathbf{A}_0^\\top\\mathbf{A}_0``).

## Notes
- The entries below the diagonal are driven to round-off level by the rotations but are not
  explicitly set to zero; use `triu(A)` if exact zeros are needed.
"""
function qr_givens!(A::AbstractMatrix{T}) where T<:Real
    m,n = size(A)
    for i in 1:min(m,n)
        for j in (i+1):m
            c, s, _ = givens_rotation(A[i,i], A[j,i])
            @turbo BLAS.rot!(
                n - i + 1,
                view(A, i, i:n), stride(A, 2),
                view(A, j, i:n), stride(A, 2),
                c, -s
            )
        end
    end
end
