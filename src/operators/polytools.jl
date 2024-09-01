export dupmat, symmtzrmat, elimat, commat, eliminate, duplicate, duplicate_symmetric
export makePolyOp


"""
    dupmat(n::Int, p::Int) -> Dp::SparseMatrixCSC{Int}

Create a duplication matrix of order `p` for a vector of length `n`.

## Arguments
- `n::Int`: The length of the vector.
- `p::Int`: The order of the duplication matrix, e.g., `p = 2` for x ⊗ x.

## Output
- `Dp::SparseMatrixCSC{Int}`: The duplication matrix of order `p`.

## Example
```julia-repl
julia> dupmat(2,2)
4×3 SparseMatrixCSC{Int64, Int64} with 4 stored entries:
 1  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1
```

## References
[^magnus1980] J. R. Magnus and H. Neudecker, “The Elimination Matrix: Some Lemmas and Applications,” 
SIAM. J. on Algebraic and Discrete Methods, vol. 1, no. 4, pp. 422–449, Dec. 1980, doi: 10.1137/0601049.
"""
function dupmat(n::Int, p::Int)
    # Calculate the number of unique elements in a symmetric tensor of order p
    num_unique_elements = binomial(n + p - 1, p)
    
    # Create the duplication matrix with appropriate size
    Dp = spzeros(Int, n^p, num_unique_elements)

    function elements!(D, l, indices)
        perms = unique([sum((indices[σ[i]] - 1) * n^(p - i) for i in 1:p) + 1 for σ in permutations(1:p)])
        
        @inbounds for p in perms
            D[p, l] = 1
        end
    end
    elements!(D, l, indices...) = elements!(D, l, indices)

    # function generate_nonredundant_combinations(n::Int, p::Int)
    #     if p == 1
    #         return [[i] for i in 1:n]
    #     else
    #         lower_combs = generate_nonredundant_combinations(n, p - 1)
    #         return [vcat(comb, [i]) for comb in lower_combs for i in comb[end]:n]
    #     end
    # end

    # Generate all combinations (i1, i2, ..., ip) with i1 ≤ i2 ≤ ... ≤ ip
    # combs = generate_nonredundant_combinations(n, p)
    combs = with_replacement_combinations(1:n, p) 
    combs = reduce(hcat, combs)
    combs = Vector{eltype(combs)}[eachrow(combs)...]
    
    # # Fill in the duplication matrix using the computed combinations
    elements!.(Ref(Dp), 1:num_unique_elements, combs...)

    return Dp
end


"""
    symmtzrmat(n::Int, p::Int) -> Sp::SparseMatrixCSC{Float64}

Create a symmetrizer matrix of order `p` for a vector of length `n` [^magnus1980].

## Arguments
- `n::Int`: The length of the vector.
- `p::Int`: The order of the symmetrizer matrix, e.g., `p = 2` for x ⊗ x.

## Output
- `Sp::SparseMatrixCSC{Float64}`: The symmetrizer matrix of order `p`.

## Example
```julia-repl
julia> symmtzrmat(2,2)
4×4 SparseMatrixCSC{Float64, Int64} with 6 stored entries:
 1.0   ⋅    ⋅    ⋅ 
  ⋅   0.5  0.5   ⋅
  ⋅   0.5  0.5   ⋅
  ⋅    ⋅    ⋅   1.0
```
"""
function symmtzrmat(n::Int, p::Int)
    # Create the symmetrizer matrix with appropriate size
    np = Int(n^p)
    Sp = spzeros(Float64, np, np)

    function elements!(N, l, indices)
        perms = [sum((indices[σ[i]] - 1) * n^(p - i) for i in 1:p) + 1 for σ in permutations(1:p)]
        
        # For cases where two or all indices are the same, 
        # we should not count permutations more than once.
        unique_perms = countmap(perms)

        # Assign the column to the matrix N
        @inbounds for (perm, count) in unique_perms
            N[perm, l] = count / factorial(p)
        end
    end
    elements!(N, l, indices...) = elements!(N, l, indices)

    function generate_redundant_combinations(n::Int, p::Int)
        iterators = ntuple(_ -> 1:n, p)
        return [collect(product) for product in Iterators.product(iterators...)]
    end

    # Generate all combinations (i1, i2, ..., ip) with i1 ≤ i2 ≤ ... ≤ ip
    combs = generate_redundant_combinations(n, p)
    combs = reduce(hcat, combs)
    combs = Vector{eltype(combs)}[eachrow(combs)...]
    
    # Fill in the duplication matrix using the computed combinations
    elements!.(Ref(Sp), 1:np, combs...)

    return Sp
end


"""
    elimat(n::Int, p::Int) -> Lp::SparseMatrixCSC{Int}

Create an elimination matrix of order `p` for a vector of length `n` [^magnus1980].

## Arguments
- `n::Int`: The length of the vector.
- `p::Int`: The order of the elimination matrix, e.g., `p = 2` for x ⊗ x.

## Output
- `Lp::SparseMatrixCSC{Int}`: The elimination matrix of order `p`.

## Example
```julia-repl
julia> elimat(2,2)
3×4 SparseMatrixCSC{Int64, Int64} with 3 stored entries:
 1  ⋅  ⋅  ⋅
 ⋅  1  ⋅  ⋅
 ⋅  ⋅  ⋅  1
```
"""
function elimat(n::Int, p::Int)
    # Calculate the number of rows in L
    num_rows = binomial(n + p - 1, p)

    # Initialize the output matrix L
    Lp = spzeros(Int, num_rows, n^p)

    # Generate all combinations with repetition of n elements from {1, 2, ..., n}
    combs = with_replacement_combinations(1:n, p) 

    # Fill the matrix L
    for (l, comb) in enumerate(combs)
        v = [1]  # Start with a scalar 1
        @inbounds for d in 1:p
            e = collect(1:n .== comb[d])  # Create the indicator vector
            v = v ⊗ e  # Build the Kronecker product
        end
        Lp[l, :] = v  # Assign the row
    end

    return Lp
end


"""
    commat(m::Int, n::Int) → K

Create commutation matrix `K` of dimension `m x n` [^magnus1980].

## Arguments
- `m::Int`: row dimension of the commutation matrix
- `n::Int`: column dimension of the commutation matrix

## Returns
- `K`: commutation matrix

## Example
```julia-repl
julia> commat(2,2)
4×4 SparseMatrixCSC{Float64, Int64} with 4 stored entries:
 1.0   ⋅    ⋅    ⋅
  ⋅    ⋅   1.0   ⋅
  ⋅   1.0   ⋅    ⋅ 
  ⋅    ⋅    ⋅   1.0
```
"""
function commat(m::Int, n::Int)
    mn = Int(m * n)
    A = reshape(1:mn, m, n)
    v = vec(A')
    K = sparse(1.0I, mn, mn)
    K = K[v, :]
    return K
end


"""
    commat(m::Int) → K

Dispatch for the commutation matrix of dimensions (m, m)

## Arguments
- `m::Int`: row and column dimension of the commutation matrix

## Returns
- `K`: commutation matrix
"""
commat(m::Int) = commat(m, m)  # dispatch


"""
    eliminate(A::AbstractArray, p::Int)

Eliminate the redundant polynomial coefficients in the matrix `A` and return the matrix 
with unique coefficients.

## Arguments
- `A::AbstractArray`: A matrix
- `p::Int`: The order of the polynomial, e.g., `p = 2` for x ⊗ x.

## Returns
- matrix with unique coefficients

## Example
```julia-repl
julia> n = 2; P = rand(n,n); P *= P'; p = vec(P)
4-element Vector{Float64}:
 0.5085988756090203
 0.7704767970682769
 0.7704767970682769
 1.310279680309927

julia> Q = rand(n,n); Q *= Q'; q = vec(Q)
4-element Vector{Float64}:
 0.40940214810208353
 0.2295272821417254
 0.2295272821417254
 0.25503767587483905

julia> A2 = [p'; q']
2×4 Matrix{Float64}:
 0.257622  0.44721   0.0202203  0.247649
 1.55077   0.871029  0.958499   0.650717

julia> eliminate(A2, 2)
2×3 Matrix{Float64}:
 0.508599  1.54095   1.31028
 0.409402  0.459055  0.255038
```
"""
function eliminate(A::AbstractArray, p::Int)
    n = size(A, 1)
    Dp = dupmat(n, p)
    return A * Dp
end


"""
    duplicate(A::AbstractArray)

Duplicate the redundant polynomial coefficients in the matrix `A` with a unique set of coefficients
and return the matrix with redundant coefficients.

## Arguments
- `A::AbstractArray`: A matrix
- `p::Int`: The order of the polynomial, e.g., `p = 2` for x ⊗ x.

## Returns
- matrix with redundant coefficients

## Example
```julia-repl
julia> n = 2; P = rand(n,n); P *= P'; p = vec(P)
4-element Vector{Float64}:
 0.5085988756090203
 0.7704767970682769
 0.7704767970682769
 1.310279680309927

julia> Q = rand(n,n); Q *= Q'; q = vec(Q)
4-element Vector{Float64}:
 0.40940214810208353
 0.2295272821417254
 0.2295272821417254
 0.25503767587483905

julia> A2 = [p'; q']
2×4 Matrix{Float64}:
 0.257622  0.44721   0.0202203  0.247649
 1.55077   0.871029  0.958499   0.650717

julia> D2 = dupmat(2,2)
4×3 SparseMatrixCSC{Int64, Int64} with 4 stored entries:
 1  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1

julia> A2 * D2
2×3 Matrix{Float64}:
 0.508599  1.54095   1.31028
 0.409402  0.459055  0.255038

julia> duplicate(A2 * D2, 2)
2×4 Matrix{Float64}:
 0.508599  1.54095   0.0  1.31028
 0.409402  0.459055  0.0  0.255038
````
"""
function duplicate(A::AbstractArray, p::Int)
    n = size(A, 1)
    Lp = elimat(n, p)
    return A * Lp
end


"""
    duplicate_symmetric(A::AbstractArray, p::Int)

Duplicate the redundant polynomial coefficients in the matrix `A` with a unique set of coefficients
and return the matrix with redundant coefficients which are duplicated symmetrically.
This guarantees that the operator is symmetric. The difference from `duplicate` is that
we use the elimination matrix `Lp` and the symmetric commutation matrix `Sp` to multiply the `A` matrix.

## Arguments
- `A::AbstractArray`: A matrix
- `p::Int`: The order of the polynomial, e.g., `p = 2` for x ⊗ x.

## Returns
- matrix with redundant coefficients duplicated symmetrically

## Example
```julia-repl
julia> n = 2; P = rand(n,n); P *= P'; p = vec(P)
4-element Vector{Float64}:
 0.5085988756090203
 0.7704767970682769
 0.7704767970682769
 1.310279680309927

julia> Q = rand(n,n); Q *= Q'; q = vec(Q)
4-element Vector{Float64}:
 0.40940214810208353
 0.2295272821417254
 0.2295272821417254
 0.25503767587483905

julia> A2 = [p'; q']
2×4 Matrix{Float64}:
 0.257622  0.44721   0.0202203  0.247649
 1.55077   0.871029  0.958499   0.650717

julia> D2 = dupmat(2,2)
4×3 SparseMatrixCSC{Int64, Int64} with 4 stored entries:
 1  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1

julia> A2 * D2
2×3 Matrix{Float64}:
 0.508599  1.54095   1.31028
 0.409402  0.459055  0.255038

julia> duplicate_symmetric(A2 * D2, 2)
2×4 Matrix{Float64}:
 0.508599  0.770477  0.770477  1.31028
 0.409402  0.229527  0.229527  0.255038
````
"""
function duplicate_symmetric(A::AbstractArray, p::Int)
    n = size(A, 1)
    Lp = elimat(n, p)
    Sp = symmtzrmat(n, p)
    return A * Lp * Sp
end


"""
    makePolyOp(n::Int, inds::AbstractArray{<:NTuple{P,<:Int}}, vals::AbstractArray{<:Real}; 
                    nonredundant::Bool=true, symmetric::Bool=true) where P

Helper function to construct the polynomial operator from the indices and values. The indices must
be a 1-dimensional array of, e.g., tuples of the form `(i,j)` where `i,j` are the indices of the
polynomial term. For example, for the polynomial term ``2.5x_1x_2`` for ``\\dot{x}_3`` would have an
index of `(1,2,3)` with a value of `2.5`. The `nonredundant` argument specifies which polynomial
operator to construct (the redundant or non-redundant operator). Note that the values must be a
1-dimensional array of the same length as the indices. The `symmetric` argument specifies whether
to construct the operator with symmetric coefficients.

## Arguments
- `n::Int`: dimension of the polynomial operator
- `inds::AbstractArray{<:NTuple{P,<:Int}}`: indices of the polynomial term
- `vals::AbstractArray{<:Real}`: values of the polynomial term
- `nonredundant::Bool=true`: whether to construct the non-redundant operator
- `symmetric::Bool=true`: whether to construct the symmetric operator

## Returns
- the polynomial operator
"""
function makePolyOp(n::Int, inds::AbstractArray{<:NTuple{P,<:Int}}, vals::AbstractArray{<:Real}; 
                    nonredundant::Bool=true, symmetric::Bool=true) where P
    p = P - 1
    @assert length(inds) == length(vals) "The length of indices and values must be the same."
    # Initialize a p-order tensor of size n^p
    S = zeros(ntuple(_ -> n, p+1))
    
    # Iterate over indices and assign values considering symmetry
    for (ind, val) in zip(inds, vals)
        if symmetric
            element_idx = ind[1:end-1]
            last_idx = ind[end]
            perms = unique(permutations(element_idx))
            contribution = val / length(perms)
            for perm in perms
                S[perm...,last_idx] = contribution
            end
        else
            S[ind...] = val
        end
    end

    # Flatten the p-order tensor into a matrix form with non-unique coefficients
    A = spzeros(n, n^p)
    for i in 1:n
        A[i, :] = vec(S[ntuple(_ -> :, p)..., i])
    end

    if nonredundant
        return eliminate(A, p)
    else
        return A
    end
end


"""
    kron_snapshot_matrix(Xmat::AbstractArray{T}, p::Int) where {T<:Number}

Take the `p`-order Kronecker product of each state of the snapshot matrix `Xmat`.

## Arguments
- `Xmat::AbstractArray{T}`: state snapshot matrix
- `p::Int`: order of the Kronecker product

## Returns
- kronecker product state snapshot matrix
"""
function kron_snapshot_matrix(Xmat::AbstractArray{T}, p::Int) where {T<:Number}
    function kron_timestep(x)
        return x[:,:] ⊗ p  # x has to be AbstractMatrix (Kronecker.jl)
    end
    tmp = kron_timestep.(eachcol(Xmat))
    return reduce(hcat, tmp)
end


"""
    unique_kron_snapshot_matrix(Xmat::AbstractArray{T}, p::Int) where {T<:Number}

Take the `p`-order unique Kronecker product of each state of the snapshot matrix `Xmat`.

## Arguments
- `Xmat::AbstractArray{T}`: state snapshot matrix
- `p::Int`: order of the Kronecker product

## Returns
- unique kronecker product state snapshot matrix
"""
function unique_kron_snapshot_matrix(Xmat::AbstractArray{T}, p::Int) where {T<:Number}
    function unique_kron_timestep(x)
        return ⊘(x, p)
    end
    tmp = unique_kron_timestep.(eachcol(Xmat))
    return reduce(hcat, tmp)
end