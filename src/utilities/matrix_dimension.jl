"""
$(SIGNATURES)

Return `A` in "fat" orientation (no more rows than columns), transposing it if necessary.

The shape is determined with [`checksize`](@ref). If ``m \\le n`` (fat or square), `A` is
returned unchanged; if ``m > n`` (tall), the lazy adjoint `A'` is returned. A vector of length
``m`` is treated as an ``m \\times 1`` matrix, so a vector with ``m > 1`` is returned as the
``1 \\times m`` row `A'`, and a length-1 vector is returned unchanged.

## Arguments
- `A::AbstractArray`: input matrix or vector.

## Returns
- `::AbstractArray`: `A` itself or its adjoint `A'` (not a copy).
"""
function tall2fat(A::AbstractArray)
    m, n = checksize(A)
    if m <= n
        return A
    else
        return A'
    end
end

"""
$(SIGNATURES)

Return `A` in "tall" orientation (no fewer rows than columns), transposing it if necessary.

The shape is determined with [`checksize`](@ref). If ``m \\ge n`` (tall or square), `A` is
returned unchanged; if ``m < n`` (fat), the lazy adjoint `A'` is returned. A vector is treated
as an ``m \\times 1`` matrix and is therefore always returned unchanged. This is used, e.g., to
make input matrices ``K \\times m`` (rows = time, columns = inputs) regardless of how the user
passed them.

## Arguments
- `A::AbstractArray`: input matrix or vector.

## Returns
- `::AbstractArray`: `A` itself or its adjoint `A'` (not a copy).
"""
function fat2tall(A::AbstractArray)
    m, n = checksize(A)
    if m >= n
        return A
    else
        return A'
    end
end

"""
$(SIGNATURES)

Return the number of rows and columns of `A`, treating a vector as a single column.

For a matrix this is `size(A)`. For a vector (where `size(A)` has only one entry) it returns
`(length(A), 1)`. For arrays with more than two dimensions only the first two dimensions are
returned.

## Arguments
- `A::AbstractArray`: input matrix or vector.

## Returns
- `m::Int`: number of rows (`length(A)` for a vector).
- `n::Int`: number of columns (`1` for a vector).
"""
function checksize(A::AbstractArray)
    m, n = nothing, nothing
    try
        m, n = size(A)
    catch e
        if isa(e, BoundsError)
            m, n = length(A), 1
        else
            rethrow(e)
        end
    end
    return m, n
end