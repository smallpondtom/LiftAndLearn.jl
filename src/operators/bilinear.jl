export insert2bilin


"""
    insert2bilin(X::Union{SparseMatrixCSC,VecOrMat}, N::Int, p::Int) → BL

Inserting the values into the bilinear matrix (`N`) for higher dimensions

## Arguments
- `X`: bilinear matrix to insert
- `N`: the larger order

## Returns
- Inserted bilinear matrix
"""
function insert2bilin(X, N, p)
    Ni = size(X, 1)
    BL = zeros(N, N*p)
    for i in 1:p
        idx = (i-1)*N+1
        BL[1:Ni, idx:(idx+Ni-1)] = X[:, (i-1)*Ni+1:i*Ni]
    end
    return BL
end
