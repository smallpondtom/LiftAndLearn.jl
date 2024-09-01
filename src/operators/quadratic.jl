# export dupmat, elimat, commat, symmtzrmat
# export F2H, H2F, F2Hs, squareMatStates, kronMatStates
# export extractF, insert2F, insert2randF, extractH, insert2H
# export Q2H, H2Q, makeQuadOp


"""
    dupmat(n::Integer) → D

Create duplication matrix `D` of dimension `n` [^magnus1980].

## Arguments
- `n`: dimension of the duplication matrix

## Returns
- `D`: duplication matrix

## Examples
```julia-repl
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  3
 3  4

julia> D = LnL.dupmat(2)
4×3 SparseArrays.SparseMatrixCSC{Float64, Int64} with 4 stored entries:
 1.0   ⋅    ⋅ 
  ⋅   1.0   ⋅
  ⋅   1.0   ⋅
  ⋅    ⋅   1.0

julia> D * LnL.vech(A)
4-element Vector{Float64}:
 1.0
 3.0
 3.0
 4.0

julia> a = vec(A)
4-element Vector{Int64}:
 1
 3
 3
 4
```

## References
[^magnus1980] J. R. Magnus and H. Neudecker, “The Elimination Matrix: Some Lemmas and Applications,” 
SIAM. J. on Algebraic and Discrete Methods, vol. 1, no. 4, pp. 422–449, Dec. 1980, doi: 10.1137/0601049.
"""
# function dupmat(n)
#     m = n * (n + 1) / 2
#     nsq = n^2
#     r = 1
#     a = 1
#     v = zeros(nsq)
#     cn = cumsum(n:-1:2)
#     for i = 1:n
#         v[r:(r+i-2)] = (i - n) .+ cn[1:(i-1)]
#         r = r + i - 1

#         v[r:r+n-i] = a:a.+(n-i)
#         r = r + n - i + 1
#         a = a + n - i + 1
#     end
#     D = sparse(1:nsq, v, ones(length(v)), nsq, m)
#     return D
# end


"""
    elimat(m::Integer) → L

Create elimination matrix `L` of dimension `m` [^magnus1980].

##  Arguments
- `m`: dimension of the target matrix

## Return
- `L`: elimination matrix
"""
# function elimat(m)
#     T = tril(ones(m, m)) # Lower triangle of 1's
#     f = findall(x -> x == 1, T[:]) # Get linear indexes of 1's
#     k = m * (m + 1) / 2 # Row size of L
#     m2 = m * m # Colunm size of L
#     x = f + m2 * (0:k-1) # Linear indexes of the 1's within L'

#     row = [mod(a, m2) != 0 ? mod(a, m2) : m2 for a in x]
#     col = [mod(a, m2) != 0 ? div(a, m2) + 1 : div(a, m2) for a in x]
#     L = sparse(row, col, ones(length(x)), m2, k)
#     L = L' # Now transpose to actual L
#     return L
# end


"""
    commat(m::Integer, n::Integer) → K

Create commutation matrix `K` of dimension `m x n` [^magnus1980].

## Arguments
- `m::Integer`: row dimension of the commutation matrix
- `n::Integer`: column dimension of the commutation matrix

## Returns
- `K`: commutation matrix
"""
# function commat(m::Integer, n::Integer)
#     mn = Int(m * n)
#     A = reshape(1:mn, m, n)
#     v = vec(A')
#     K = sparse(1.0I, mn, mn)
#     K = K[v, :]
#     return K
# end


"""
    commat(m::Integer) → K

Dispatch for the commutation matrix of dimensions (m, m)

## Arguments
- `m::Integer`: row and column dimension of the commutation matrix

## Returns
- `K`: commutation matrix
"""
# commat(m::Integer) = commat(m, m)  # dispatch


"""
    symmtzrmat(m::Integer, n::Integer) → N

Create symmetrizer (or symmetric commutation) matrix `N` of dimension `m x n` [^magnus1980].

## Arguments
- `m::Integer`: row dimension of the commutation matrix
- `n::Integer`: column dimension of the commutation matrix

## Returns
- `N`: symmetrizer (symmetric commutation) matrix
"""
# function symmtzrmat(m::Integer, n::Integer)
#     mn = Int(m * n)
#     return 0.5 * (sparse(1.0I, mn, mn) + commat(m, n))
# end


"""
    symmtzrmat(m::Integer) → N

Dispatch for the symmetric commutation matrix of dimensions (m, m)

## Arguments
- `m::Integer`: row and column dimension of the commutation matrix

## Returns
- `N`: symmetric commutation matrix
"""
# symmtzrmat(m::Integer) = symmtzrmat(m, m)  # dispatch


"""
    F2H(F::Union{SparseMatrixCSC,VecOrMat}) → H

Convert the quadratic `F` operator into the `H` operator

## Arguments
- `F`: F matrix

## Returns
- `H`: H matrix
"""
# function F2H(F)
#     n = size(F, 1)
#     Ln = elimat(n)
#     return F * Ln
# end


"""
    H2F(H::Union{SparseMatrixCSC,VecOrMat}) → F

Convert the quadratic `H` operator into the `F` operator

## Arguments
- `H`: H matrix

## Returns
- `F`: F matrix
"""
# function H2F(H)
#     n = size(H, 1)
#     Dn = dupmat(n)
#     return H * Dn
# end


"""
    F2Hs(F::Union{SparseMatrixCSC,VecOrMat}) → Hs

Convert the quadratic `F` operator into the symmetric `H` operator.

This guarantees that the `H` operator is symmetric. The difference from F2H is that
we use the elimination matrix `L` and the symmetric commutation matrix `N` to multiply the `F` matrix.

## Arguments
- `F`: F matrix

## Returns
- `Hs`: symmetric H matrix
"""
# function F2Hs(F)
#     n = size(F, 1)
#     Ln = elimat(n)
#     Nn = symmtzrmat(n)
#     return F * Ln * Nn
# end


"""
    squareMatStates(Xmat::Union{SparseMatrixCSC,VecOrMat}) → Xsq

Generate the `x^[2]` squared state values (corresponding to the `F` matrix) for a 
snapshot data matrix

## Arguments
- `Xmat`: state snapshot matrix

## Returns
- squared state snapshot matrix 
"""
# function squareMatStates(Xmat)
#     function vech_col(X)
#         return X ⊘ X
#     end
#     tmp = vech_col.(eachcol(Xmat))
#     return reduce(hcat, tmp)
# end


"""
    kronMatStates(Xmat::Union{SparseMatrixCSC,VecOrMat}) → Xkron

Generate the kronecker product state values (corresponding to the `H` matrix) for 
a matrix form state data

## Arguments 
- `Xmat`: state snapshot matrix

## Returns
- kronecker product state snapshot matrix
"""
# function kronMatStates(Xmat)
#     function vec_col(X)
#         return X ⊗ X
#     end
#     tmp = vec_col.(eachcol(Xmat))
#     return reduce(hcat, tmp)
# end



