export dupmat, elimat, commat, symmtzrmat
export F2H, H2F, F2Hs, squareMatStates, kronMatStates
export extractF, insert2F, insert2randF, extractH, insert2H
export invec, Q2H, H2Q, makeQuadOp


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
