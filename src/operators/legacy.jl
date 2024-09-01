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
        return eliminate(Q2H(Q), 2)
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

    G = spzeros(n, n^3)
    for i in 1:n
        G[i, :] = vec(S[:, :, :, i])
    end

    if which_cubic_term == "G" || which_cubic_term == 'G'
        return G
    elseif which_cubic_term == "E" || which_cubic_term == 'E'
        return eliminate(G, 3)
    else
        error("The cubic term must be either G or E.")
    end
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
        Q[:,:,i] = reshape(H[i, :], n, n)
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