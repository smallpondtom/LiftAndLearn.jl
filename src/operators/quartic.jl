# """
#     quartMatStates(Xmat::Union{SparseMatrixCSC,VecOrMat}) → Xquart

# Generate the `x^<4>` cubed state values snapshot data matrix

# ## Arguments
# - `Xmat`: state snapshot matrix

# ## Returns
# - cubed state snapshot matrix
# """
# function cubeMatStates(Xmat)
#     function vech_col(X)
#         return ⊘(X, X, X)
#     end
#     tmp = vech_col.(eachcol(Xmat))
#     return reduce(hcat, tmp)
# end

"""
    kron4MatStates(Xmat::Union{SparseMatrixCSC,VecOrMat}) → Xkron

Generate the 4rd order kronecker product state values (corresponding to the `G` matrix) for 
a matrix form state data

## Arguments 
- `Xmat`: state snapshot matrix

## Returns
- kronecker product state snapshot matrix
"""
# function kron4MatStates(Xmat)
#     function vec_col(x)
#         return x ⊗ x ⊗ x ⊗ x
#     end
#     tmp = vec_col.(eachcol(Xmat))
#     return reduce(hcat, tmp)
# end