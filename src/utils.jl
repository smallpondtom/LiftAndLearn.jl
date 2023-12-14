export operators, dupmat, elimat, commat, nommat, vech
export F2H, H2F, F2Hs, squareMatStates, kronMatStates, extractF 
export insert2F, insert2randF, extractH, insert2H, insert2bilin
export invec, Q2H, H2Q

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
- `K`: constant operator
- `N`: bilinear (state-input) operator
- `f`: nonlinear function operator f(x,u)
"""
Base.@kwdef mutable struct operators
    A::Union{SparseMatrixCSC{Float64,Int64},VecOrMat{Real},Matrix{Float64},Matrix{Any},Int64} = 0
    B::Union{SparseVector{Float64,Int64},SparseMatrixCSC{Float64,Int64},VecOrMat{Real},Matrix{Float64},Matrix{Any},Int64} = 0
    C::Union{SparseVector{Float64,Int64},SparseMatrixCSC{Float64,Int64},VecOrMat{Real},Matrix{Float64},Matrix{Any},Int64} = 0
    F::Union{SparseMatrixCSC{Float64,Int64},VecOrMat{Real},Matrix{Float64},Matrix{Any},Int64} = 0
    H::Union{SparseMatrixCSC{Float64,Int64},VecOrMat{Real},Matrix{Float64},Matrix{Any},Int64} = 0
    Q::Union{AbstractArray,Real} = 0
    K::Union{SparseVector{Float64,Int64},SparseMatrixCSC{Float64,Int64},VecOrMat{Real},Matrix{Float64},Int64} = 0
    N::Union{SparseMatrixCSC{Float64,Int64},AbstractArray,Vector{Matrix{Real}},VecOrMat{Real},Matrix{Float64},Int64} = 0
    f::Function = x -> x
end


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
    nommat(m::Integer, n::Integer) → N

Create symmetric commutation matrix `N` of dimension `m x n` [^magnus1980].

## Arguments
- `m::Integer`: row dimension of the commutation matrix
- `n::Integer`: column dimension of the commutation matrix

## Returns
- `N`: symmetric commutation matrix
"""
function nommat(m::Integer, n::Integer)
    mn = Int(m * n)
    return 0.5 * (sparse(1.0I, mn, mn) + commat(m, n))
end

"""
    nommat(m::Integer) → N

Dispatch for the symmetric commutation matrix of dimensions (m, m)

## Arguments
- `m::Integer`: row and column dimension of the commutation matrix

## Returns
- `N`: symmetric commutation matrix
"""
nommat(m::Integer) = nommat(m, m)  # dispatch


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
    Nn = nommat(n)
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
        return vech(X * X')
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
        return vec(X * X')
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
    invec(r::VecOrMat, m::Int, n::Int) → r

Inverse vectorization.

## Arguments
- `r::AbstractArray`: the input vector
- `m::Int`: the row dimension
- `n::Int`: the column dimension

## Returns
- the inverse vectorized matrix
"""
function invec(r::AbstractArray, m::Int, n::Int)::VecOrMat
    tmp = vec(1.0I(n))'
    return kron(tmp, 1.0I(m)) * kron(1.0I(n), r)
end


"""
    Q2H(Q::Union{Array,VecOrMat}) → H

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
    H2Q(H::Union{Array,VecOrMat,SparseMatrixCSC}) → Q

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

