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
    m, n = checksize(A)
    if m <= n
        return A
    else
        return A'
    end
end
