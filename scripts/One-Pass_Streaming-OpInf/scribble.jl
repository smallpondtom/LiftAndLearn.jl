using LinearAlgebra
using UniqueKronecker
using Combinatorics
using SparseArrays
using Kronecker

##
"""
    kappa(n, idxs::Vararg{Int})

Compute the lexicographic rank (1-based) of the non-decreasing tuple `idxs...`
with 1 ≤ idxs[1] ≤ … ≤ idxs[p] ≤ n, among all combinations with repetition.
Returns an integer in 1:binomial(n+p-1, p) where p = length(idxs).
"""
function kappa(n::Int, idxs::Vararg{Int})
    p = length(idxs)
    @assert all(1 .<= idxs .<= n) "indices must lie in 1:n"
    @assert issorted(idxs)        "indices must be non-decreasing"

    k = 1
    prev = 1
    for m in 1:p
        im = idxs[m]
        # count how many tuples start with anything in [prev, im-1] at position m
        for t in prev:(im-1)
            k += binomial(n - t + (p-m), p-m)
        end
        prev = im
    end
    return k
end

##
n = 12
for idx in ((1,1,1),(1,1,2),(1,1,3),(1,2,2),(1,2,3),(1,3,3),(2,2,2),(2,2,3),(2,3,3),(3,3,3))
    println(idx, " → ", kappa(n, idx...))
end

##
n = 2
A = rand(n, n, n, n)

##
a = vec(A)

##
ua = dupmat(n, 4)' * a

##
i = rand(1:n)
j = rand(i:n)
k = rand(j:n)
l = rand(k:n)
println(a[n^3*(i-1) + n^2*(j-1) + n*(k-1) + l])

##
println(ua[kappa(n, (i,j,k,l)...)])

## 
"""
    uvec(n, idxs::Vararg{Int}; T=Float64)

Return the “half–vectorization” basis vector u_{i₁…i_p} ∈ ℝ^{binomial(n+p-1,p)},
where `p = length(idxs)` and `1 ≤ idxs[1] ≤ … ≤ idxs[p] ≤ n`.  
The output is of element‐type `T` (default `Float64`).
"""
function uvec(n::Int, idxs::Vararg{Int}; T=Float64)
    # number of modes
    p = length(idxs)
    @assert issorted(idxs) "indices must be non-decreasing"
    @assert all(1 .<= idxs .<= n) "indices must lie in 1:n"

    # dimension of the unique-Kronecker space
    N = binomial(n + p - 1, p)

    # allocate and set the single 1
    u = zeros(T, N)
    u[kappa(n, idxs...)] = one(T)
    return u
end

##
uvec(n, (i,j,k,l)...)


##
"""
    sym_unique_vec(A::AbstractArray{T,N}) where {T,N}

Compute the “symmetric unique vectorization” of an Nth-order tensor A (of size nxnx⋯xn)
by summing over all p!/(multiplicity!) permutations of each entry.  That is, it
returns the same vector as `elimat(n,N) * vec(A)` would.

# Arguments
- `A` : an N-way array with `size(A, k)==n` for all k.
  
# Returns
- `v::Vector{T}` of length `binomial(n+N-1, N)` where each entry
  corresponds to one non-decreasing index‐tuple `(i₁≤…≤i_N)` and
  is the sum of `A[i_{σ(1)},…,i_{σ(N)}]` over all σ∈S_N.
"""
function sym_unique_vec(A::AbstractArray{T,N}) where {T,N}
    n = size(A, 1)
    @assert all(size(A,k)==n for k in 1:N) "A must be cubic"
    total = binomial(n + N - 1, N)
    v = zeros(T, total)
    idx = zeros(Int, N)

    function _recur(depth)
        if depth > N
            # sort the current full index tuple and find its slot
            sorted_idx = sort(idx)
            k = kappa(n, sorted_idx...)
            v[k] += A[idx...]                # accumulate A at the original indices
        else
            for i in 1:n
                idx[depth] = i
                _recur(depth+1)
            end
        end
    end

    _recur(1)
    return v
end

"""
    unique_vec(A::AbstractArray{T,N}) where {T,N}

“Non‐symmetric” half‐vectorization: for each non‐decreasing tuple
(i₁≤…≤i_N) of 1:n, pick A[i₁,…,i_N] exactly once and place it in the slot
that `elimat(n,N)` would select via `kappa`.  I.e.

    unique_vec(A) == elimat(n,N) * vec(A)

even if A is not symmetric.

# Returns
- `v::Vector{T}` of length `binomial(n+N-1, N)`.

"""
function unique_vec(A::AbstractArray{T,N}) where {T,N}
    n = size(A,1)
    @assert all(size(A,k)==n for k in 1:N) "A must be cubic"
    total = binomial(n + N - 1, N)
    v = zeros(eltype(A), total)
    idx = zeros(Int, N)

    # recurse over all non-decreasing idx[1] ≤ … ≤ idx[N]:
    function _recur(start::Int, depth::Int)
        if depth > N
            # place A[idx...] in the slot elimination-matrix would use:
            v[kappa(n, idx...)] = A[idx...]
        else
            for i in start:n
                idx[depth] = i
                _recur(i, depth+1)
            end
        end
    end

    _recur(1, 1)
    return v
end

##
ua2 = sym_unique_vec(A)
println(norm(ua2 - ua) < 1e-12)  

#3
# """
#     elimat_kron_naive(n::Int, p::Int)

# Naïvely build the elimination matrix L_{n,p} by summing over all non-decreasing
# tuples (i₁ ≤ … ≤ i_p):
#   row  = kappa(n, i₁,…,i_p)
#   col  = kron(e_{i_p}', …, e_{i₁}')  # a 1×n^p row-vector
# and placing that row in L.
# """
# function elimat_kron_naive(n::Int, p::Int)
#     R = binomial(n + p - 1, p)
#     C = n^p
#     L = spzeros(Int, R, C)

#     # identity so that E[i,:] is e_i'
#     E = Matrix(I, n, n)

#     for idx in with_replacement_combinations(1:n, p)
#         # determine which row of L this tuple corresponds to
#         r = kappa(n, idx...)

#         # build the 1×n^p row by kron'ing the row-basis vectors
#         v = E[idx[end], :]               # starts with e_{i_p}'
#         for j in reverse(idx[1:end-1])   # then e_{i_{p-1}}', …, e_{i_1}'
#             v = kron(v, E[j, :])
#         end

#         L[r, :] = v
#     end

#     return L
# end

"""
    elimat_kron_naive(n::Int, p::Int)

Builds the elimination matrix L_{n,p} ∈ ℝ^{binomial(n+p-1,p) × n^p}
so that  L * vec_rowmajor(A)  =  vech(A),
where  vec_rowmajor(A)[(i₁−1)*n^(p−1)+…+(i_p−1)+1] = A[i₁,…,i_p].

This matches UniqueKronecker.jl’s `elimat(n,p)`.
"""
function elimat_kron_naive(n::Int, p::Int)
    R = binomial(n + p - 1, p)  # number of unique entries
    C = n^p                     # number of full entries
    L = spzeros(Float64, R, C)

    # identity lets us grab standard row‐vectors
    E = Matrix(I, n, n)
    idx = zeros(Int, p)

    function _recur(start::Int, depth::Int)
        if depth > p
            # figure out which row this tuple lives in
            r = kappa(n, idx...)
            # build the 1×n^p row in row-major order:
            v = E[idx[1],:]
            for d in 2:p
                v = kron(v, E[idx[d], :])
            end
            L[r, :] = v
        else
            for i in start:n
                idx[depth] = i
                _recur(i, depth+1)
            end
        end
    end

    _recur(1, 1)
    return L
end

##
n, p = rand(2:10), rand(2:4)
L = elimat(n, p)
L2 = elimat_kron_naive(n, p)
println(norm(L - L2) < 1e-12)  # should be true


##
X = rand(10, 4)
U,Σ,V = svd(X)

##
L = elimat(10,2)
D = dupmat(4,2)
U2 = L * (U ⊗ U) * D
L = elimat(4,2)
D = dupmat(10,2)
U3 = L * (U ⊗ U)' * D

##
U3 * U2

##