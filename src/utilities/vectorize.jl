"""
    vech(A::AbstractMatrix{T}) → v

Half-vectorization operation. For example half-vectorzation of
```math
A = \\begin{bmatrix}
    a_{11} & a_{12}  \\\\
    a_{21} & a_{22}
\\end{bmatrix}
```
becomes
```math
v = \\begin{bmatrix}
    a_{11} \\\\
    a_{21} \\\\
    a_{22}
\\end{bmatrix}
```

## Arguments
- `A`: matrix to half-vectorize

## Returns
- `v`: half-vectorized form
"""
function vech(A::AbstractMatrix{T}) where {T}
    m = LinearAlgebra.checksquare(A)
    v = Vector{T}(undef, (m * (m + 1)) >> 1)
    k = 0
    for j = 1:m, i = j:m
        @inbounds v[k+=1] = A[i, j]
    end
    return v
end


"""
    invec(r::AbstractArray, m::Int, n::Int) → r

Inverse vectorization.

## Arguments
- `r::AbstractArray`: the input vector
- `m::Int`: the row dimension
- `n::Int`: the column dimension

## Returns
- the inverse vectorized matrix
"""
function invec(r::AbstractArray, m::Int, n::Int)::AbstractArray
    tmp = sparse(reshape(1.0I(n), 1, :))
    return kron(tmp, sparse(1.0I,m,m)) * kron(sparse(1.0I,n,n), r)
end
