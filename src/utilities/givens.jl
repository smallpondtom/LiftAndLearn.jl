"""
    givens_rotation(a, b) -> (c, s, r)

Compute a numerically-robust Givens rotation so that

    [  c  -s ] [ a ] = [ r ]
    [  s   c ] [ b ]   [ 0 ]
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
    qr_givens!(A::AbstractMatrix{T}) -> R

Compute the QR factorization of matrix `A` using Givens rotations.
Returns the upper triangular matrix `R` such that `A = QR` in-place of `A`.
"""
function qr_givens!(A::AbstractMatrix{T}) where T<:Real
    m,n = size(A)
    for i in 1:min(m,n)
        for j in (i+1):m
            c, s, _ = givens_rotation(A[i,i], A[j,i])
            @turbo BLAS.rot!(
                n - i + 1,
                view(A, i, i:n), n,
                view(A, j, i:n), n,
                c, -s
            )
        end
    end
end
