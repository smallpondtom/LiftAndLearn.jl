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
function invec(r::AbstractArray, m::Int, n::Int)::VecOrMat
    tmp = reshape(1.0I(n), 1, :)
    return kron(tmp, 1.0I(m)) * kron(1.0I(n), r)
end

