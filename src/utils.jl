"""
File containing general utility structures and functions for Lift & Learn.
"""


"""
Structure to store the operators of the system

# Fields
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
    N::Union{SparseMatrixCSC{Float64,Int64},Array{Float64},Vector{Matrix{Real}},VecOrMat{Real},Matrix{Float64},Int64} = 0
    f::Function = x -> x
end


"""
Create duplication matrix

# Arguments
- `n`: dimension of the target matrix

# Return
- `D`: duplication matrix
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
Create the elimination matrix

#  Arguments
- `m`: dimension of the target matrix

# Return
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
Commutation matrix
"""
function commat(m::Integer, n::Integer)
    mn = Int(m * n)
    A = reshape(1:mn, m, n)
    v = vec(A')
    K = sparse(1.0I, mn, mn)
    K = K[v, :]
    return K
end
commat(m::Integer) = commat(m, m)  # dispatch


"""
Symmetric commutation matrix.
"""
function nommat(m::Integer, n::Integer)
    mn = Int(m * n)
    return 0.5 * (sparse(1.0I, mn, mn) + commat(m, n))
end
nommat(m::Integer) = nommat(m, m)  # dispatch


"""
Half-vectorization operation

# Arguments
- `A`: matrix to half-vectorize

# Return
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
Convert the quadratic F operator into the H operator

# Arguments
- `F`: F matrix

# Return
- `H`: H matrix
"""
function F2H(F)
    n = size(F, 1)
    Ln = elimat(n)
    return F * Ln
end


"""
Convert the quadratic H operator into the F operator

# Arguments
- `H`: H matrix

# Return
- `F`: F matrix
"""
function H2F(H)
    n = size(H, 1)
    Dn = dupmat(n)
    return H * Dn
end


"""
Convert the quadratic F operator into the symmetric H operator
# Arguments
- `F`: F matrix

# Return
- `Hs`: symmetric H matrix
"""
function F2Hs(F)
    n = size(F, 1)
    Ln = elimat(n)
    Nn = nommat(n)
    return F * Ln * Nn
end


"""
Generate the x^(2) squared state values (corresponding to the F matrix) for a 
matrix form data

# Arguments
- `Xmat`: state matrix

# Return
- squared state matrix 
"""
function squareMatStates(Xmat)
    function vech_col(X)
        return vech(X * X')
    end
    tmp = vech_col.(eachcol(Xmat))
    return reduce(hcat, tmp)
end


"""
Generate the kronecker product state values (corresponding to the H matrix) for 
a matrix form state data

# Arguments 
- `Xmat`: state matrix

# Return
- kronecker product state
"""
function kronMatStates(Xmat)
    function vec_col(X)
        return vec(X * X')
    end
    tmp = vec_col.(eachcol(Xmat))
    return reduce(hcat, tmp)
end


"""
Extracting the F matrix for POD basis of dimensions (N, r)

# Arguments
- `F`: F matrix
- `r`: reduced order

# Return
- extracted F matrix
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
Extracting the H matrix for POD basis of dimensions (N, r)

# Arguments
- `H`: H matrix
- `r`: reduced order

# Return
- extracted H matrix
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
Inverse vectorization.
>>>>>>> energy-preserve
# Arguments
- `r`: the input vector
- `m`: the row dimension
- `n`: the column dimension

# Return
- the inverse vectorized matrix
"""
function invec(r::VecOrMat, m::Int, n::Int)::VecOrMat
    tmp = vec(1.0I(n))'
    return kron(tmp, 1.0I(m)) * kron(1.0I(n), r)
end


"""
Convert the quadratic Q matrix to the H matrix.

# Arguments 
- `Q`: Quadratic matrix in the 3-dim tensor form

# Return
- the H quadratic matrix
"""
function Q2H(Q::Union{Array,VecOrMat})
    # The Q matrix should be a 3-dim tensor with dim n
    n = size(Q, 1)

    # Preallocate the sparse matrix of H
    H = spzeros(n, n^2)

    for i in 1:n
        H[i, :] = vec(Q[i, :, :])
    end

    return H
end


"""
Convert the quadratic H matrix to the Q matrix.

# Arguments 
- `H`: Quadratic matrix of dimensions (n x n^2)

# Return
- the Q quadratic matrix of 3-dim tensor
"""
function H2Q(H::Union{Array,VecOrMat,SparseMatrixCSC})
    # The Q matrix should be a 3-dim tensor with dim n
    n = size(H, 1)

    # Preallocate the sparse matrix of H
    Q = Vector{Matrix{Float64}}(undef, n)

    for i in 1:n
        Q[i] = invec(H[i, :], n, n)
    end

    return Q
end


"""
Auxiliary function for the F matrix indexing.

# Arguments 
- `n`: row dimension of the F matrix
- `j`: row index 
- `k`: col index

# Return
- index corresponding to the F matrix
"""
function fidx(n,j,k)
    if j >= k
        return Int((n - k/2)*(k - 1) + j)
    else
        return Int((n - j/2)*(j - 1) + k)
    end
end


"""
Another auxiliary function for the F matrix

# Arguments
- `v`: first index
- `w`: second index

# Return
- coefficient of 1.0 or 0.5
"""
function delta(v,w)
    return v == w ? 1.0 : 0.5
end

