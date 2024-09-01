export unique_kronecker, ⊘


"""
    CombinationIterator(n::Int, p::Int)

Iterator for generating all combinations with repetition of `n` elements from {1, 2, ..., n}.

## Fields
- `n::Int`: number of elements
- `p::Int`: number of elements in each combination
"""
struct UniqueCombinationIterator
    n::Int
    p::Int
end


"""
    Base.iterate(it::UniqueCombinationIterator, comb::Vector{Int})

Iterate over all combinations with repetition of `n` elements from {1, 2, ..., n}.
"""
function Base.iterate(it::UniqueCombinationIterator, comb::Vector{Int})
    p = it.p
    n = it.n

    # Find the rightmost element that can be incremented
    i = p
    while i > 0
        if comb[i] < n
            comb[i] += 1
            # Fill subsequent positions with the incremented value to 
            # maintain non-decreasing order
            for j in i+1:p
                comb[j] = comb[i]
            end
            return comb, comb
        end
        i -= 1
    end

    # Termination: if we can't increment, we stop
    return nothing
end

function Base.iterate(it::UniqueCombinationIterator)
    # Start with the first combination [1, 1, ..., 1]
    comb = ones(Int, it.p)
    return comb, comb
end

Base.length(it::UniqueCombinationIterator) = binomial(it.n + it.p - 1, it.p)
Base.eltype(it::UniqueCombinationIterator) = Vector{Int}
Base.IteratorSize(::UniqueCombinationIterator) = Base.SizeUnknown()



"""
    unique_kronecker(x::AbstractVector{T}, y::AbstractVector{T}) where T

Unique Kronecker product operation. For example, if

```math
x = y = \\begin{bmatrix}
    1  \\\\
    2
\\end{bmatrix}
```
then
```math
unique_kronecker(x, x) = \\begin{bmatrix}
    1 \\\\
    2 \\\\
    4
\\end{bmatrix}
```

## Arguments
- `x::AbstractVector{T}`: vector to perform the unique Kronecker product
- `y::AbstractVector{T}`: vector to perform the unique Kronecker product

## Returns
- `result`: unique Kronecker product

## Note
This implementation is faster than `unique_kronecker_power` for `p = 2`.
"""
@inline function unique_kronecker(x::AbstractArray{T}, y::AbstractArray{T}) where {T}
    n = length(x)
    m = length(y)
    result = Array{T}(undef, n*(n+1) ÷ 2)
    k = 1
    @inbounds for i in 1:n
        for j in i:m
            result[k] = x[i] * y[j]
            k += 1
        end
    end
    return result
end



"""
    unique_kronecker(x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}) where T

Unique Kronecker product operation for triple Kronecker product.

## Arguments
- `x::AbstractVector{T}`: vector to perform the unique Kronecker product
- `y::AbstractVector{T}`: vector to perform the unique Kronecker product
- `z::AbstractVector{T}`: vector to perform the unique Kronecker product

## Returns
- `result`: unique Kronecker product

## Note
This implementation is faster than `unique_kronecker_power` for `p = 3`.
"""
@inline function unique_kronecker(x::AbstractArray{T}, y::AbstractArray{T}, z::AbstractArray{T}) where {T}
    n = length(x)
    result = Array{T}(undef, n*(n+1)*(n+2) ÷ 6)
    l = 1
    @inbounds for i in 1:n
        for j in i:n
            for k in j:n
                result[l] = x[i] * y[j] * z[k]
                l += 1
            end
        end
    end
    return result
end


"""
    unique_kronecker(x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}, w::AbstractVector{T}) where T

Unique Kronecker product operation for quadruple Kronecker product.

## Arguments
- `x::AbstractVector{T}`: vector to perform the unique Kronecker product
- `y::AbstractVector{T}`: vector to perform the unique Kronecker product
- `z::AbstractVector{T}`: vector to perform the unique Kronecker product
- `w::AbstractVector{T}`: vector to perform the unique Kronecker product

## Returns
- `result`: unique Kronecker product

## Note
This implementation is faster than `unique_kronecker_power` for `p = 4`.
"""
@inline function unique_kronecker(x::AbstractArray{T}, y::AbstractArray{T}, z::AbstractArray{T}, w::AbstractArray{T}) where {T}
    n = length(x)
    result = Array{T}(undef, n*(n+1)*(n+2)*(n+3) ÷ 24)
    d = 1
    @inbounds for i in 1:n
        for j in i:n
            for k in j:n
                for l in k:n
                    result[d] = x[i] * y[j] * z[k] * w[l]
                    d += 1
                end
            end
        end
    end
    return result
end


# """
#     unique_kronecker(x::AbstractVector{T}) where T

# Unique Kronecker product operation (dispatch)

# ## Arguments
# - `x::AbstractVector{T}`: vector to perform the unique Kronecker product

# ## Returns
# - `result`: unique Kronecker product
# """
# @inline function unique_kronecker(x::AbstractArray{T}) where {T}
#     n = length(x)
#     result = Array{T}(undef, n*(n+1) ÷ 2)
#     k = 1
#     @inbounds for i in 1:n
#         for j in i:n
#             result[k] = x[i] * x[j]
#             k += 1
#         end
#     end
#     return result
# end


"""
    unique_kronecker_power(x::AbstractArray{T}, p::Int) where T

Unique Kronecker product operation generalized for power `p`.

## Arguments
- `x::AbstractArray{T}`: vector to perform the unique Kronecker product
- `p::Int`: power of the unique Kronecker product

## Returns
- `result`: unique Kronecker product
"""
@inline function unique_kronecker_power(x::AbstractArray{T}, p::Int) where {T<:Number}
    n = length(x)

    # Calculate the correct number of unique elements
    num_unique_elements = binomial(n + p - 1, p)
    result = Array{T}(undef, num_unique_elements)

    idx = 1
    @inbounds for comb in UniqueCombinationIterator(n, p)
        product = one(T)
        @simd for j in comb
            product *= x[j]
        end
        result[idx] = product
        idx += 1
    end

    if idx != num_unique_elements + 1
        error("Mismatch in expected and actual number of elements filled")
    end

    return result
end


"""
    ⊘(x::AbstractVector{T}, y::AbstractVector{T}) where T

Unique Kronecker product operation

## Arguments
- `x::AbstractVector{T}`: vector to perform the unique Kronecker product
- `y::AbstractVector{T}`: vector to perform the unique Kronecker product

## Returns
- unique Kronecker product
"""
⊘(x::AbstractArray{T}, y::AbstractArray{T}) where {T} = unique_kronecker(x, y)
⊘(x::AbstractArray{T}, y::AbstractArray{T}, z::AbstractArray{T}) where {T} = unique_kronecker(x,y,z)
⊘(x::AbstractArray{T}, y::AbstractArray{T}, z::AbstractArray{T}, w::AbstractArray{T}) where {T} = unique_kronecker(x,y,z,w)


"""
    ⊘(x::AbstractArray{T}...) where {T<:Number}

Generalized Kronecker product operator for multiple vectors.

## Arguments
- `x::AbstractArray{T}...`: one or more vectors to perform the unique Kronecker product

## Returns
- unique Kronecker product of all vectors
"""
⊘(x::AbstractArray{T}, p::Int) where {T<:Number} = 
    p == 2 ? unique_kronecker(x, x) :
    p == 3 ? unique_kronecker(x, x, x) :
    p == 4 ? unique_kronecker(x, x, x, x) :
    unique_kronecker_power(x, p)