export invec

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

