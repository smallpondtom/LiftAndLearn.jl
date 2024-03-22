export operators, dupmat, elimat, commat, symmtzrmat, vech, Unique_Kronecker, ⊘
export dupmat3, elimat3, symmtzrmat3, G2E, E2G, E2Gs, cubeMatStates
export F2H, H2F, F2Hs, squareMatStates, kronMatStates, extractF 
export insert2F, insert2randF, extractH, insert2H, insert2bilin
export invec, Q2H, H2Q, makeQuadOp, makeCubicOp

"""
$(TYPEDEF)

Organize the operators of the system in a structure. The operators currently 
supported are up to second order.

## Fields
- `A`: linear state operator
- `B`: linear input operator
- `C`: linear output operator
- `F`: quadratic state operator with no redundancy
- `H`: quadratic state operator with redundancy
- `Q`: quadratic state operator with redundancy in 3-dim tensor form
- `G`: cubic state operator with redundancy
- `E`: cubic state operator with no redundancy
- `K`: constant operator
- `N`: bilinear (state-input) operator
- `f`: nonlinear function operator f(x,u)
"""
Base.@kwdef mutable struct operators
    A::Union{AbstractArray{<:Number},Real} = 0                                           # linear
    B::Union{AbstractArray{<:Number},Real} = 0                                           # control
    C::Union{AbstractArray{<:Number},Real} = 0                                           # output
    H::Union{AbstractArray{<:Number},Real} = 0                                           # quadratic redundant
    F::Union{AbstractArray{<:Number},Real} = 0                                           # quadratic non-redundant
    Q::Union{AbstractArray{<:Number},AbstractArray{<:AbstractArray{<:Number}},Real} = 0  # quadratic (array of 2D square matrices)
    G::Union{AbstractArray{<:Number},Real} = 0                                           # cubic redundant
    E::Union{AbstractArray{<:Number},Real} = 0                                           # cubic non-redundant
    K::Union{AbstractArray{<:Number},Real} = 0                                           # constant
    N::Union{AbstractArray{<:Number},AbstractArray{<:AbstractArray{<:Number}},Real} = 0  # bilinear
    f::Function = x -> x                                                                 # nonlinear function
end


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


"""
    dupmat(n::Integer) → D

Create duplication matrix `D` of dimension `n` [^magnus1980].

## Arguments
- `n`: dimension of the duplication matrix

## Returns
- `D`: duplication matrix

## Examples
```julia-repl
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  3
 3  4

julia> D = LnL.dupmat(2)
4×3 SparseArrays.SparseMatrixCSC{Float64, Int64} with 4 stored entries:
 1.0   ⋅    ⋅ 
  ⋅   1.0   ⋅
  ⋅   1.0   ⋅
  ⋅    ⋅   1.0

julia> D * LnL.vech(A)
4-element Vector{Float64}:
 1.0
 3.0
 3.0
 4.0

julia> a = vec(A)
4-element Vector{Int64}:
 1
 3
 3
 4
```

## References
[^magnus1980] J. R. Magnus and H. Neudecker, “The Elimination Matrix: Some Lemmas and Applications,” 
SIAM. J. on Algebraic and Discrete Methods, vol. 1, no. 4, pp. 422–449, Dec. 1980, doi: 10.1137/0601049.
"""
function dupmat(n)
    m = n * (n + 1) / 2
    nsq = n^2
    r = 1
    a = 1
    v = zeros(nsq)
    cn = cumsum(n:-1:2)
    for i = 1:n
        v[r:(r+i-2)] = (i - n) .+ cn[1:(i-1)]
        r = r + i - 1

        v[r:r+n-i] = a:a.+(n-i)
        r = r + n - i + 1
        a = a + n - i + 1
    end
    D = sparse(1:nsq, v, ones(length(v)), nsq, m)
    return D
end


"""
    elimat(m::Integer) → L

Create elimination matrix `L` of dimension `m` [^magnus1980].

##  Arguments
- `m`: dimension of the target matrix

## Return
- `L`: elimination matrix
"""
function elimat(m)
    T = tril(ones(m, m)) # Lower triangle of 1's
    f = findall(x -> x == 1, T[:]) # Get linear indexes of 1's
    k = m * (m + 1) / 2 # Row size of L
    m2 = m * m # Colunm size of L
    x = f + m2 * (0:k-1) # Linear indexes of the 1's within L'

    row = [mod(a, m2) != 0 ? mod(a, m2) : m2 for a in x]
    col = [mod(a, m2) != 0 ? div(a, m2) + 1 : div(a, m2) for a in x]
    L = sparse(row, col, ones(length(x)), m2, k)
    L = L' # Now transpose to actual L
    return L
end


"""
    commat(m::Integer, n::Integer) → K

Create commutation matrix `K` of dimension `m x n` [^magnus1980].

## Arguments
- `m::Integer`: row dimension of the commutation matrix
- `n::Integer`: column dimension of the commutation matrix

## Returns
- `K`: commutation matrix
"""
function commat(m::Integer, n::Integer)
    mn = Int(m * n)
    A = reshape(1:mn, m, n)
    v = vec(A')
    K = sparse(1.0I, mn, mn)
    K = K[v, :]
    return K
end


"""
    commat(m::Integer) → K

Dispatch for the commutation matrix of dimensions (m, m)

## Arguments
- `m::Integer`: row and column dimension of the commutation matrix

## Returns
- `K`: commutation matrix
"""
commat(m::Integer) = commat(m, m)  # dispatch


"""
    symmtzrmat(m::Integer, n::Integer) → N

Create symmetrizer (or symmetric commutation) matrix `N` of dimension `m x n` [^magnus1980].

## Arguments
- `m::Integer`: row dimension of the commutation matrix
- `n::Integer`: column dimension of the commutation matrix

## Returns
- `N`: symmetrizer (symmetric commutation) matrix
"""
function symmtzrmat(m::Integer, n::Integer)
    mn = Int(m * n)
    return 0.5 * (sparse(1.0I, mn, mn) + commat(m, n))
end


"""
    symmtzrmat(m::Integer) → N

Dispatch for the symmetric commutation matrix of dimensions (m, m)

## Arguments
- `m::Integer`: row and column dimension of the commutation matrix

## Returns
- `N`: symmetric commutation matrix
"""
symmtzrmat(m::Integer) = symmtzrmat(m, m)  # dispatch


"""
    elimat3(m::Integer) → L3

Create elimination matrix `L` of dimension `m` for the 3-dim tensor.

## Arguments
- `m::Integer`: dimension of the target matrix

## Returns
- `L3`: elimination matrix
"""
function elimat3(m::Int)
    L3 = zeros(Int, m*(m+1)*(m+2) ÷ 6, m^3)
    l = 1
    for i in 1:m
        ei = [Int(p == i) for p in 1:m]
        for j in i:m
            ej = [Int(p == j) for p in 1:m]
            for k in j:m
                ek = [Int(p == k) for p in 1:m]
                eijk = ei' ⊗ ej' ⊗ ek'
                L3[l, :] = eijk
                l += 1
            end
        end
    end
    return sparse(L3)
end


"""
    dupmat3(n::Int) → D3

Create duplication matrix `D` of dimension `n` for the 3-dim tensor.

## Arguments
- `n::Int`: dimension of the duplication matrix

## Returns
- `D3`: duplication matrix
"""
function dupmat3(n::Int)
    num_unique_elements = div(n*(n+1)*(n+2), 6)
    D3 = zeros(Int, n^3, num_unique_elements)
    l = 1 # Column index for the unique elements
    
    for i in 1:n
        for j in i:n
            for k in j:n
                # Initialize the vector for the column of D3
                col = zeros(Int, n^3)
                
                # Assign the elements for all permutations
                permutations = [
                    n^2*(i-1) + n*(j-1) + k, # sub2ind([n, n, n], i, j, k)
                    n^2*(i-1) + n*(k-1) + j, # sub2ind([n, n, n], i, k, j)
                    n^2*(j-1) + n*(i-1) + k, # sub2ind([n, n, n], j, i, k)
                    n^2*(j-1) + n*(k-1) + i, # sub2ind([n, n, n], j, k, i)
                    n^2*(k-1) + n*(i-1) + j, # sub2ind([n, n, n], k, i, j)
                    n^2*(k-1) + n*(j-1) + i  # sub2ind([n, n, n], k, j, i)
                ]
                
                # For cases where two or all indices are the same, 
                # we should not count permutations more than once.
                unique_permutations = unique(permutations)
                
                # Set the corresponding entries in the column of D3
                for perm in unique_permutations
                    col[perm] = 1
                end
                
                # Assign the column to the matrix D3
                D3[:, l] = col
                
                # Increment the column index
                l += 1
            end
        end
    end
    
    return sparse(D3)
end



"""
    symmtzrmat3(n::Int) → N3

Create symmetrizer (or symmetric commutation) matrix `N` of dimension `n` for the 3-dim tensor.

## Arguments
- `n::Int`: row dimension of the commutation matrix

## Returns
- `N3`: symmetrizer (symmetric commutation) matrix
"""
function symmtzrmat3(n::Int)
    N3 = zeros(n^3, n^3)
    l = 1 # Column index for the unique elements
    
    for i in 1:n
        for j in 1:n
            for k in 1:n
                # Initialize the vector for the column of N
                col = zeros(n^3)
                
                # Assign the elements for all permutations
                permutations = [
                    n^2*(i-1) + n*(j-1) + k, # sub2ind([n, n, n], i, j, k)
                    n^2*(i-1) + n*(k-1) + j, # sub2ind([n, n, n], i, k, j)
                    n^2*(j-1) + n*(i-1) + k, # sub2ind([n, n, n], j, i, k)
                    n^2*(j-1) + n*(k-1) + i, # sub2ind([n, n, n], j, k, i)
                    n^2*(k-1) + n*(i-1) + j, # sub2ind([n, n, n], k, i, j)
                    n^2*(k-1) + n*(j-1) + i  # sub2ind([n, n, n], k, j, i)
                ]
                
                # For cases where two or all indices are the same, 
                # we should not count permutations more than once.
                unique_permutations = countmap(permutations)
                
                # Set the corresponding entries in the column of N
                for (perm, count) in unique_permutations
                    col[perm] = count / 6
                end
                
                # Assign the column to the matrix N
                N3[:, l] = col
                
                # Increment the column index
                l += 1
            end
        end
    end
    return sparse(N3)
end


"""
    G2E(G::Union{SparseMatrixCSC,VecOrMat}) → E

Convert the cubic `G` operator into the `E` operator

## Arguments
- `G`: G matrix

## Returns
- `E`: E matrix
"""
function G2E(G)
    n = size(G, 1)
    D3 = dupmat3(n)
    return G * D3
end


"""
    E2G(E::Union{SparseMatrixCSC,VecOrMat}) → G

Convert the cubic `E` operator into the `G` operator

## Arguments
- `E`: E matrix

## Returns
- `G`: G matrix
"""
function E2G(E)
    n = size(E, 1)
    L3 = elimat3(n)
    return E * L3
end


"""
    E2Gs(E::Union{SparseMatrixCSC,VecOrMat}) → G

Convert the cubic `E` operator into the symmetric `G` operator

## Arguments
- `E`: E matrix

## Returns
- `G`: symmetric G matrix
"""
function E2Gs(E)
    n = size(E, 1)
    L3 = elimat3(n)
    N3 = symmtzrmat3(n)
    return E * L3 * N3
end


"""
    vech(A::AbstractMatrix{T}) → v

Half-vectorization operation. For example half-vectorzation of
```math
A = \\begin{bmatrix}
    a_{11} & a_{12}  \\\\
    a_{21} & a_{22}
\\end{bmatrix}
```
becomes
```math
v = \\begin{bmatrix}
    a_{11} \\\\
    a_{21} \\\\
    a_{22}
\\end{bmatrix}
```

## Arguments
- `A`: matrix to half-vectorize

## Returns
- `v`: half-vectorized form
"""
function vech(A::AbstractMatrix{T}) where {T}
    m = LinearAlgebra.checksquare(A)
    v = Vector{T}(undef, (m * (m + 1)) >> 1)
    k = 0
    for j = 1:m, i = j:m
        @inbounds v[k+=1] = A[i, j]
    end
    return v
end


"""
    F2H(F::Union{SparseMatrixCSC,VecOrMat}) → H

Convert the quadratic `F` operator into the `H` operator

## Arguments
- `F`: F matrix

## Returns
- `H`: H matrix
"""
function F2H(F)
    n = size(F, 1)
    Ln = elimat(n)
    return F * Ln
end


"""
    H2F(H::Union{SparseMatrixCSC,VecOrMat}) → F

Convert the quadratic `H` operator into the `F` operator

## Arguments
- `H`: H matrix

## Returns
- `F`: F matrix
"""
function H2F(H)
    n = size(H, 1)
    Dn = dupmat(n)
    return H * Dn
end


"""
    F2Hs(F::Union{SparseMatrixCSC,VecOrMat}) → Hs

Convert the quadratic `F` operator into the symmetric `H` operator.

This guarantees that the `H` operator is symmetric. The difference from F2H is that
we use the elimination matrix `L` and the symmetric commutation matrix `N` to multiply the `F` matrix.

## Arguments
- `F`: F matrix

## Returns
- `Hs`: symmetric H matrix
"""
function F2Hs(F)
    n = size(F, 1)
    Ln = elimat(n)
    Nn = symmtzrmat(n)
    return F * Ln * Nn
end


"""
    squareMatStates(Xmat::Union{SparseMatrixCSC,VecOrMat}) → Xsq

Generate the `x^[2]` squared state values (corresponding to the `F` matrix) for a 
snapshot data matrix

## Arguments
- `Xmat`: state snapshot matrix

## Returns
- squared state snapshot matrix 
"""
function squareMatStates(Xmat)
    function vech_col(X)
        return X ⊘ X
    end
    tmp = vech_col.(eachcol(Xmat))
    return reduce(hcat, tmp)
end


"""
    cubeMatStates(Xmat::Union{SparseMatrixCSC,VecOrMat}) → Xcube

Generate the `x^<3>` cubed state values (corresponding to the `E` matrix) for a
snapshot data matrix

## Arguments
- `Xmat`: state snapshot matrix

## Returns
- cubed state snapshot matrix
"""
function cubeMatStates(Xmat)
    function vech_col(X)
        return ⊘(X, X, X)
    end
    tmp = vech_col.(eachcol(Xmat))
    return reduce(hcat, tmp)
end


"""
    kronMatStates(Xmat::Union{SparseMatrixCSC,VecOrMat}) → Xkron

Generate the kronecker product state values (corresponding to the `H` matrix) for 
a matrix form state data

## Arguments 
- `Xmat`: state snapshot matrix

## Returns
- kronecker product state snapshot matrix
"""
function kronMatStates(Xmat)
    function vec_col(X)
        return X ⊗ X
    end
    tmp = vec_col.(eachcol(Xmat))
    return reduce(hcat, tmp)
end


"""
    kron3MatStates(Xmat::Union{SparseMatrixCSC,VecOrMat}) → Xkron

Generate the 3rd order kronecker product state values (corresponding to the `G` matrix) for 
a matrix form state data

## Arguments 
- `Xmat`: state snapshot matrix

## Returns
- kronecker product state snapshot matrix
"""
function kron3MatStates(Xmat)
    function vec_col(X)
        return X ⊗ X ⊗ x
    end
    tmp = vec_col.(eachcol(Xmat))
    return reduce(hcat, tmp)
end


"""
    extractF(F::Union{SparseMatrixCSC,VecOrMat}, r::Int) → F

Extracting the `F` matrix for POD basis of dimensions `(N, r)`

## Arguments
- `F`: F matrix
- `r`: reduced order

## Returns
- extracted `F` matrix
"""
function extractF(F, r)
    N = size(F, 1)
    if 0 < r < N
        xsq_idx = [1 + (N + 1) * (n - 1) - n * (n - 1) / 2 for n in 1:N]
        extract_idx = [collect(x:x+(r-i)) for (i, x) in enumerate(xsq_idx[1:r])]
        idx = Int.(reduce(vcat, extract_idx))
        return F[1:r, idx]
    elseif r <= 0 || N < r
        error("Incorrect dimensions for extraction")
    else
        return F
    end
end


"""
    insertF(Fi::Union{SparseMatrixCSC,VecOrMat}, N::Int) → F

Inserting values into the `F` matrix for higher dimensions

## Arguments
- `Fi`: F matrix to insert
- `N`: the larger order

## Returns
- inserted `F` matrix
"""
function insert2F(Fi, N)
    F = spzeros(N, Int(N * (N + 1) / 2))
    Ni = size(Fi, 1)

    xsq_idx = [1 + (N + 1) * (n - 1) - n * (n - 1) / 2 for n in 1:N]
    insert_idx = [collect(x:x+(Ni-i)) for (i, x) in enumerate(xsq_idx[1:Ni])]
    idx = Int.(reduce(vcat, insert_idx))
    F[1:Ni, idx] = Fi
    return F
end


"""
    insert2randF(Fi::Union{SparseMatrixCSC,VecOrMat}, N::Int) → F

Inserting values into the `F` matrix for higher dimensions

## Arguments
- `Fi`: F matrix to insert
- `N`: the larger order

## Returns
- inserted `F` matrix
"""
function insert2randF(Fi, N)
    F = sprandn(N, Int(N * (N + 1) / 2), 0.8)
    Ni = size(Fi, 1)

    xsq_idx = [1 + (N + 1) * (n - 1) - n * (n - 1) / 2 for n in 1:N]
    insert_idx = [collect(x:x+(Ni-i)) for (i, x) in enumerate(xsq_idx[1:Ni])]
    idx = Int.(reduce(vcat, insert_idx))
    F[1:Ni, idx] = Fi
    return F
end


"""
    extractH(H::Union{SparseMatrixCSC,VecOrMat}, r::Int) → H

Extracting the `H` matrix for POD basis of dimensions `(N, r)`

## Arguments
- `H`: H matrix
- `r`: reduced order

## Returns
- extracted `H` matrix
"""
function extractH(H, r)
    N = size(H, 1)
    if 0 < r < N
        tmp = [(N*i-N+1):(N*i-N+r) for i in 1:r]
        idx = Int.(reduce(vcat, tmp))
        return H[1:r, idx]
    elseif r <= 0 || N < r
        error("Incorrect dimensions for extraction.")
    else
        return H
    end
end


"""
    insertH(Hi::Union{SparseMatrixCSC,VecOrMat}, N::Int) → H

Inserting values into the `H` matrix for higher dimensions

## Arguments
- `Hi`: H matrix to insert
- `N`: the larger order

## Returns
- inserted `H` matrix
"""
function insert2H(Hi, N)
    H = spzeros(N, Int(N^2))
    Ni = size(Hi, 1)

    tmp = [(N*i-N+1):(N*i-N+Ni) for i in 1:Ni]
    idx = Int.(reduce(vcat, tmp))
    H[1:Ni, idx] = Hi
    return H
end


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


"""
    Q2H(Q::AbstractArray) → H

Convert the quadratic `Q` operator into the `H` operator. The `Q` matrix is 
a 3-dim tensor with dimensions `(n x n x n)`. Thus,
    
```math
\\mathbf{Q} = \\begin{bmatrix} 
    \\mathbf{Q}_1 \\\\ 
    \\mathbf{Q}_2 \\\\ 
    \\vdots \\\\ 
    \\mathbf{Q}_n 
\\end{bmatrix}
\\quad \\text{where }~~ \\mathbf{Q}_i \\in \\mathbb{R}^{n \\times n}
```

## Arguments 
- `Q::AbstractArray`: Quadratic matrix in the 3-dim tensor form with dimensions `(n x n x n)`

## Returns
- the `H` quadratic matrix
"""
function Q2H(Q::AbstractArray)
    # The Q matrix should be a 3-dim tensor with dim n
    n = size(Q, 1)

    # Preallocate the sparse matrix of H
    H = spzeros(n, n^2)

    for i in 1:n
        H[i, :] = vec(Q[:, :, i])
    end

    return H
end


"""
    H2Q(H::AbstractArray) → Q

Convert the quadratic `H` operator into the `Q` operator

## Arguments 
- `H::AbstractArray`: Quadratic matrix of dimensions `(n x n^2)`

## Returns
- the `Q` quadratic matrix of 3-dim tensor
"""
function H2Q(H::AbstractArray)
    # The Q matrix should be a 3-dim tensor with dim n
    n = size(H, 1)

    # Preallocate the sparse matrix of H
    Q = Array{Float64}(undef, n, n, n)

    for i in 1:n
        Q[:,:,i] = invec(H[i, :], n, n)
    end

    return Q
end


"""
    makeQuadOp(n::Int, inds::AbstractArray{Tuple{Int,Int,Int}}, vals::AbstractArray{Real}, 
    which_quad_term::Union{String,Char}="H") → H or F or Q

Helper function to construct the quadratic operator from the indices and values. The indices must
be a 1-dimensional array of tuples of the form `(i,j,k)` where `i,j,k` are the indices of the
quadratic term. For example, for the quadratic term ``2.5x_1x_2`` for ``\\dot{x}_3`` would have an 
index of `(1,2,3)` with a value of `2.5`. The `which_quad_term` argument specifies which quadratic
term to construct. Note that the values must be a 1-dimensional array of the same length as the indices.

## Arguments
- `n::Int`: dimension of the quadratic operator
- `inds::AbstractArray{Tuple{Int,Int,Int}}`: indices of the quadratic term
- `vals::AbstractArray{Real}`: values of the quadratic term
- `which_quad_term::Union{String,Char}="H"`: which quadratic term to construct
- `symmetric::Bool=true`: whether to construct the symmetric `H` or `Q` matrix

## Returns
- the quadratic operator
"""
function makeQuadOp(n::Int, inds::AbstractArray{Tuple{Int,Int,Int}}, vals::AbstractArray{<:Real}; 
    which_quad_term::Union{String,Char}="H", symmetric::Bool=true)

    @assert length(inds) == length(vals) "The length of indices and values must be the same."
    Q = zeros(n, n, n)
    for (ind,val) in zip(inds, vals)
        if symmetric
            i, j, k = ind
            if i == j
                Q[ind...] = val
            else
                Q[i,j,k] = val/2
                Q[j,i,k] = val/2
            end
        else
            Q[ind...] = val
        end
    end

    if which_quad_term == "H" || which_quad_term == 'H'
        return Q2H(Q)
    elseif which_quad_term == "F" || which_quad_term == 'F'
        return (H2F ∘ Q2H)(Q)
    elseif which_quad_term == "Q" || which_quad_term == 'Q'
        return Q
    else
        error("The quad term must be either H, F, or Q.")
    end
end


"""
    makeCubicOp(n::Int, inds::AbstractArray{Tuple{Int,Int,Int,Int}}, vals::AbstractArray{Real}, 
    which_cubic_term::Union{String,Char}="G") → G or E

Helper function to construct the cubic operator from the indices and values. The indices must
be a 1-dimensional array of tuples of the form `(i,j,k,l)` where `i,j,k,l` are the indices of the
cubic term. For example, for the cubic term ``2.5x_1x_2x_3`` for ``\\dot{x}_4`` would have an
index of `(1,2,3,4)` with a value of `2.5`. The `which_cubic_term` argument specifies which cubic
term to construct (the redundant or non-redundant operator). Note that the values must be a 
1-dimensional array of the same length as the indices.

## Arguments
- `n::Int`: dimension of the cubic operator
- `inds::AbstractArray{Tuple{Int,Int,Int,Int}}`: indices of the cubic term
- `vals::AbstractArray{Real}`: values of the cubic term
- `which_cubic_term::Union{String,Char}="G"`: which cubic term to construct "G" or "E"
- `symmetric::Bool=true`: whether to construct the symmetric `G` matrix

## Returns
- the cubic operator
"""
function makeCubicOp(n::Int, inds::AbstractArray{Tuple{Int,Int,Int,Int}}, vals::AbstractArray{<:Real};
    which_cubic_term::Union{String,Char}="G", symmetric::Bool=true)

    @assert length(inds) == length(vals) "The length of indices and values must be the same."
    S = zeros(n, n, n, n)
    for (ind,val) in zip(inds, vals)
        if symmetric
            i, j, k, l = ind
            if i == j == k
                S[ind...] = val
            elseif (i == j) && (j != k)
                S[i,j,k,l] = val/3
                S[i,k,j,l] = val/3
                S[k,i,j,l] = val/3
            elseif (i != j) && (j == k)
                S[i,j,k,l] = val/3
                S[j,i,k,l] = val/3
                S[j,k,i,l] = val/3
            elseif (i == k) && (j != k)
                S[i,j,k,l] = val/3
                S[j,i,k,l] = val/3
                S[i,k,j,l] = val/3
            else
                S[i,j,k,l] = val/6
                S[i,k,j,l] = val/6
                S[j,i,k,l] = val/6
                S[j,k,i,l] = val/6
                S[k,i,j,l] = val/6
                S[k,j,i,l] = val/6
            end
        else
            S[ind...] = val
        end
    end

    G = zeros(n, n^3)
    for i in 1:n
        G[i, :] = vec(S[:, :, :, i])
    end

    if which_cubic_term == "G" || which_cubic_term == 'G'
        return G
    elseif which_cubic_term == "E" || which_cubic_term == 'E'
        return G2E(G)
    else
        error("The cubic term must be either G or E.")
    end
end


"""
    fidx(n::Int, j::Int, k::Int) → Int

Auxiliary function for the `F` matrix indexing.

## Arguments 
- `n`: row dimension of the F matrix
- `j`: row index 
- `k`: col index

## Returns
- index corresponding to the `F` matrix
"""
function fidx(n,j,k)
    if j >= k
        return Int((n - k/2)*(k - 1) + j)
    else
        return Int((n - j/2)*(j - 1) + k)
    end
end


"""
    delta(v::Int, w::Int) → Float64

Another auxiliary function for the `F` matrix

## Arguments
- `v`: first index
- `w`: second index

## Returns
- coefficient of 1.0 or 0.5
"""
function delta(v,w)
    return v == w ? 1.0 : 0.5
end

