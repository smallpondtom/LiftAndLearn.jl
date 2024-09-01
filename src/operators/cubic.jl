export dupmat3, elimat3, symmtzrmat3, G2E, E2G, E2Gs, cubeMatStates, kron3MatStates, makeCubicOp


"""
    elimat3(m::Integer) → L3

Create elimination matrix `L` of dimension `m` for the 3-dim tensor.

## Arguments
- `m::Integer`: dimension of the target matrix

## Returns
- `L3`: elimination matrix
"""
# function elimat3(m::Int)
#     L3 = spzeros(Int, m*(m+1)*(m+2) ÷ 6, m^3)
#     l = 1
#     for i in 1:m
#         ei = [Int(p == i) for p in 1:m]
#         for j in i:m
#             ej = [Int(p == j) for p in 1:m]
#             for k in j:m
#                 ek = [Int(p == k) for p in 1:m]
#                 eijk = ei' ⊗ ej' ⊗ ek'
#                 L3[l, :] = eijk
#                 l += 1
#             end
#         end
#     end
#     return sparse(L3)
# end


"""
    dupmat3(n::Int) → D3

Create duplication matrix `D` of dimension `n` for the 3-dim tensor.

## Arguments
- `n::Int`: dimension of the duplication matrix

## Returns
- `D3`: duplication matrix
"""
# function dupmat3(n::Int)
#     num_unique_elements = div(n*(n+1)*(n+2), 6)
#     D3 = spzeros(Int, n^3, num_unique_elements)
#     l = 1 # Column index for the unique elements
    
#     for i in 1:n
#         for j in i:n
#             for k in j:n
#                 # Initialize the vector for the column of D3
#                 col = spzeros(Int, n^3)
                
#                 # Assign the elements for all permutations
#                 permutations = [
#                     n^2*(i-1) + n*(j-1) + k, # sub2ind([n, n, n], i, j, k)
#                     n^2*(i-1) + n*(k-1) + j, # sub2ind([n, n, n], i, k, j)
#                     n^2*(j-1) + n*(i-1) + k, # sub2ind([n, n, n], j, i, k)
#                     n^2*(j-1) + n*(k-1) + i, # sub2ind([n, n, n], j, k, i)
#                     n^2*(k-1) + n*(i-1) + j, # sub2ind([n, n, n], k, i, j)
#                     n^2*(k-1) + n*(j-1) + i  # sub2ind([n, n, n], k, j, i)
#                 ]
                
#                 # For cases where two or all indices are the same, 
#                 # we should not count permutations more than once.
#                 unique_permutations = unique(permutations)
                
#                 # Set the corresponding entries in the column of D3
#                 # for perm in unique_permutations
#                 #     col[perm] = 1
#                 # end
#                 col[unique_permutations] .+= 1
                
#                 # Assign the column to the matrix D3
#                 D3[:, l] = col
                
#                 # Increment the column index
#                 l += 1
#             end
#         end
#     end
    
#     return sparse(D3)
# end
# function dupmat3(n::Int)
#     num_unique_elements = div(n*(n+1)*(n+2), 6)
#     D3 = spzeros(Int, n^3, num_unique_elements)

#     function elements!(D,i,j,k,l)
#         perms = [
#             n^2*(i-1) + n*(j-1) + k,
#             n^2*(i-1) + n*(k-1) + j,
#             n^2*(j-1) + n*(i-1) + k,
#             n^2*(j-1) + n*(k-1) + i,
#             n^2*(k-1) + n*(i-1) + j,
#             n^2*(k-1) + n*(j-1) + i
#         ]

#         for p in unique(perms)
#             D[p, l] = 1
#         end
#     end

#     # Generate all combinations (i, j, k) with i ≤ j ≤ k
#     combs = [[i, j, k] for i in 1:n for j in i:n for k in j:n]
#     combs = reduce(hcat, combs)
#     elements!.(Ref(D3), combs[1,:], combs[2,:], combs[3,:], 1:num_unique_elements)

#     return sparse(D3)
# end



"""
    symmtzrmat3(n::Int) → N3

Create symmetrizer (or symmetric commutation) matrix `N` of dimension `n` for the 3-dim tensor.

## Arguments
- `n::Int`: row dimension of the commutation matrix

## Returns
- `N3`: symmetrizer (symmetric commutation) matrix
"""
# function symmtzrmat3(n::Int)
#     N3 = spzeros(n^3, n^3)
#     l = 1 # Column index for the unique elements
    
#     for i in 1:n
#         for j in 1:n
#             for k in 1:n
#                 # Initialize the vector for the column of N
#                 col = spzeros(n^3)
                
#                 # Assign the elements for all permutations
#                 permutations = [
#                     n^2*(i-1) + n*(j-1) + k, # sub2ind([n, n, n], i, j, k)
#                     n^2*(i-1) + n*(k-1) + j, # sub2ind([n, n, n], i, k, j)
#                     n^2*(j-1) + n*(i-1) + k, # sub2ind([n, n, n], j, i, k)
#                     n^2*(j-1) + n*(k-1) + i, # sub2ind([n, n, n], j, k, i)
#                     n^2*(k-1) + n*(i-1) + j, # sub2ind([n, n, n], k, i, j)
#                     n^2*(k-1) + n*(j-1) + i  # sub2ind([n, n, n], k, j, i)
#                 ]
                
#                 # For cases where two or all indices are the same, 
#                 # we should not count permutations more than once.
#                 unique_permutations = countmap(permutations)
                
#                 # Set the corresponding entries in the column of N
#                 # for (perm, count) in unique_permutations
#                 #     col[perm] = count / 6
#                 # end
#                 col[(collect ∘ keys)(unique_permutations)] .= values(unique_permutations) ./ 6
                
#                 # Assign the column to the matrix N
#                 N3[:, l] = col
                
#                 # Increment the column index
#                 l += 1
#             end
#         end
#     end
#     return sparse(N3)
# end
# function symmtzrmat3(n::Int)
#     N3 = spzeros(n^3, n^3)

#     function elements!(N,i,j,k,l)
#         perms = [
#             n^2*(i-1) + n*(j-1) + k,
#             n^2*(i-1) + n*(k-1) + j,
#             n^2*(j-1) + n*(i-1) + k,
#             n^2*(j-1) + n*(k-1) + i,
#             n^2*(k-1) + n*(i-1) + j,
#             n^2*(k-1) + n*(j-1) + i
#         ]

#         # For cases where two or all indices are the same, 
#         # we should not count permutations more than once.
#         unique_perms = countmap(perms)

#         # Assign the column to the matrix N
#         for (perm, count) in unique_perms
#             N[perm, l] = count / 6
#         end
#     end

#     # Generate all combinations (i, j, k) with i ≤ j ≤ k
#     combs = [[i, j, k] for i in 1:n for j in 1:n for k in 1:n]
#     combs = reduce(hcat, combs)
#     elements!.(Ref(N3), combs[1,:], combs[2,:], combs[3,:], 1:Int(n^3))
    
#     return sparse(N3)
# end


"""
    G2E(G::Union{SparseMatrixCSC,VecOrMat}) → E

Convert the cubic `G` operator into the `E` operator

## Arguments
- `G`: G matrix

## Returns
- `E`: E matrix
"""
# function G2E(G)
#     n = size(G, 1)
#     D3 = dupmat3(n)
#     return G * D3
# end


"""
    E2G(E::Union{SparseMatrixCSC,VecOrMat}) → G

Convert the cubic `E` operator into the `G` operator

## Arguments
- `E`: E matrix

## Returns
- `G`: G matrix
"""
# function E2G(E)
#     n = size(E, 1)
#     L3 = elimat3(n)
#     return E * L3
# end


"""
    E2Gs(E::Union{SparseMatrixCSC,VecOrMat}) → G

Convert the cubic `E` operator into the symmetric `G` operator

## Arguments
- `E`: E matrix

## Returns
- `G`: symmetric G matrix
"""
# function E2Gs(E)
#     n = size(E, 1)
#     L3 = elimat3(n)
#     N3 = symmtzrmat3(n)
#     return E * L3 * N3
# end



"""
    cubeMatStates(Xmat::Union{SparseMatrixCSC,VecOrMat}) → Xcube

Generate the `x^<3>` cubed state values (corresponding to the `E` matrix) for a
snapshot data matrix

## Arguments
- `Xmat`: state snapshot matrix

## Returns
- cubed state snapshot matrix
"""
# function cubeMatStates(Xmat)
#     function vech_col(X)
#         return ⊘(X, X, X)
#     end
#     tmp = vech_col.(eachcol(Xmat))
#     return reduce(hcat, tmp)
# end




"""
    kron3MatStates(Xmat::Union{SparseMatrixCSC,VecOrMat}) → Xkron

Generate the 3rd order kronecker product state values (corresponding to the `G` matrix) for 
a matrix form state data

## Arguments 
- `Xmat`: state snapshot matrix

## Returns
- kronecker product state snapshot matrix
"""
# function kron3MatStates(Xmat)
#     function vec_col(X)
#         return X ⊗ X ⊗ x
#     end
#     tmp = vec_col.(eachcol(Xmat))
#     return reduce(hcat, tmp)
# end


