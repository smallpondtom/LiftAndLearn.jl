"""
    tall2fat(A::AbstractArray)

Convert a tall matrix to a fat matrix by taking the transpose if the number
of rows is less than the number of columns.

## Arguments
- `A::AbstractArray`: input matrix

## Returns
- `A::AbstractArray`: output matrix
"""
function tall2fat(A::AbstractArray)
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

    if m <= n
        return A
    else
        return A'
    end
end
