"""
    fat2tall(A::AbstractArray)

Convert a fat matrix to a tall matrix by taking the transpose if the number 
of rows is less than the number of columns.

## Arguments
- `A::AbstractArray`: input matrix

## Returns
- `A::AbstractArray`: output matrix
"""
function fat2tall(A::AbstractArray)
    m, n = size(A)
    if m >= n
        return A
    else
        return A'
    end
end