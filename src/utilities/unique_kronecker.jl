export Unique_Kronecker, ⊘


"""
    Unique_Kronecker(x::AbstractVector{T}, y::AbstractVector{T}) where T

Unique Kronecker product operation. For example, if

```math
x = y = \\begin{bmatrix}
    1  \\\\
    2
\\end{bmatrix}
```
then
```math
Unique_Kronecker(x, x) = \\begin{bmatrix}
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
"""
@inline function Unique_Kronecker(x::AbstractArray{T}, y::AbstractArray{T}) where {T<:Number}
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
    Unique_Kronecker(x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}) where T

Unique Kronecker product operation for triple Kronecker product.

## Arguments
- `x::AbstractVector{T}`: vector to perform the unique Kronecker product
- `y::AbstractVector{T}`: vector to perform the unique Kronecker product
- `z::AbstractVector{T}`: vector to perform the unique Kronecker product

## Returns
- `result`: unique Kronecker product
"""
@inline function Unique_Kronecker(x::AbstractArray{T}, y::AbstractArray{T}, z::AbstractArray{T}) where {T<:Number}
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
    Unique_Kronecker(x::AbstractVector{T}) where T

Unique Kronecker product operation (dispatch)

## Arguments
- `x::AbstractVector{T}`: vector to perform the unique Kronecker product

## Returns
- `result`: unique Kronecker product
"""
@inline function Unique_Kronecker(x::AbstractArray{T}) where {T<:Number}
    n = length(x)
    result = Array{T}(undef, n*(n+1) ÷ 2)
    k = 1
    @inbounds for i in 1:n
        for j in i:n
            result[k] = x[i] * x[j]
            k += 1
        end
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
⊘(x::AbstractArray{T}, y::AbstractArray{T}) where {T<:Number} = Unique_Kronecker(x, y)
⊘(x::AbstractArray{T}, y::AbstractArray{T}, z::AbstractArray{T}) where {T<:Number} = Unique_Kronecker(x,y,z)
