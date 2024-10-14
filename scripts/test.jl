using LiftAndLearn
LnL = LiftAndLearn

using SparseArrays, Combinatorics, LinearAlgebra, StatsBase, Kronecker


##
# struct CombinationIterator
#     n::Int
#     p::Int
# end

# function Base.iterate(it::CombinationIterator, comb::Vector{Int})
#     p = it.p
#     n = it.n

#     # Find the rightmost element that can be incremented
#     i = p
#     while i > 0
#         if comb[i] < n
#             comb[i] += 1
#             # Fill subsequent positions with the incremented value to maintain non-decreasing order
#             for j in i+1:p
#                 comb[j] = comb[i]
#             end
#             return comb, comb
#         end
#         i -= 1
#     end

#     # Termination: if we can't increment, we stop
#     return nothing
# end

# function Base.iterate(it::CombinationIterator)
#     # Start with the first combination [1, 1, ..., 1]
#     comb = ones(Int, it.p)
#     return comb, comb
# end

# Base.length(it::CombinationIterator) = binomial(it.n + it.p - 1, it.p)
# Base.eltype(it::CombinationIterator) = Vector{Int}
# Base.IteratorSize(::CombinationIterator) = Base.SizeUnknown()

# @inline function Unique_Kronecker_Power(x::AbstractArray{T}, p::Int) where {T<:Number}
#     n = length(x)

#     # Preallocate the resulting vector
#     num_unique_elements = binomial(n + p - 1, p)
#     result = Array{T}(undef, num_unique_elements)

#     idx = 1
#     @inbounds for comb in CombinationIterator(n, p)
#         product = one(T)
#         @simd for j in comb
#             product *= x[j]
#         end
#         result[idx] = product
#         idx += 1
#     end

#     return result
# end





##

n = 7
x = rand(n)
p = 5

y = LnL.Unique_Kronecker_Power(x, p)

##
# z = x ⊘ x ⊘ x
z = ⊘(x,p)

norm(y - z)

## 
if p == 2
    z = Unique_Kronecker(x,x)
elseif p == 3
    z = Unique_Kronecker(x,x,x)
elseif p == 4
    z = Unique_Kronecker(x,x,x,x)
end

norm(y - z)

##

function generate_combinations1(n::Int, p::Int)
    if p == 1
        return [[i] for i in 1:n]
    else
        lower_combs = generate_combinations1(n, p - 1)
        return [vcat(comb, [i]) for comb in lower_combs for i in comb[end]:n]
    end
end

function generate_combinations2(n::Int, p::Int)
    iterators = ntuple(_ -> 1:n, p)
    return [collect(product) for product in Iterators.product(iterators...)]
end

##
n = 4
p = 2
foo = generate_combinations1(n, p)
bar = generate_combinations2(n, p)
# Generate all combinations with repetition of n elements from {1, 2, ..., n}
combs = collect(with_replacement_combinations(1:n, p))

## 

function dupmatN(n::Int, p::Int)
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

    function generate_combinations(n::Int, p::Int)
        if p == 1
            return [[i] for i in 1:n]
        else
            lower_combs = generate_combinations(n, p - 1)
            return [vcat(comb, [i]) for comb in lower_combs for i in comb[end]:n]
        end
    end

    # Generate all combinations (i1, i2, ..., ip) with i1 ≤ i2 ≤ ... ≤ ip
    combs = generate_combinations(n, p)
    combs = reduce(hcat, combs)
    combs = Vector{eltype(combs)}[eachrow(combs)...]
    
    # # Fill in the duplication matrix using the computed combinations
    elements!.(Ref(Dp), 1:num_unique_elements, combs...)

    return Dp
end



##

n = 5
D2 = LnL.dupmat(n)
D2_ = dupmatN(n, 2)
e1 = norm(D2 - D2_)
@show e1

D3 = LnL.dupmat3(n)
D3_ = dupmatN(n, 3)
e2 = norm(D3 - D3_)
@show e2


##

function symmtzrmatN(n::Int, p::Int)
    # Create the symmetrizer matrix with appropriate size
    np = Int(n^p)
    Np = spzeros(Float64, np, np)

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

    function generate_combinations(n::Int, p::Int)
        iterators = ntuple(_ -> 1:n, p)
        return [collect(product) for product in Iterators.product(iterators...)]
    end

    # Generate all combinations (i1, i2, ..., ip) with i1 ≤ i2 ≤ ... ≤ ip
    combs = generate_combinations(n, p)
    combs = reduce(hcat, combs)
    combs = Vector{eltype(combs)}[eachrow(combs)...]
    
    # # Fill in the duplication matrix using the computed combinations
    elements!.(Ref(Np), 1:np, combs...)

    return Np
end


##

n = 5
N2 = LnL.symmtzrmat(n)
N2_ = symmtzrmatN(n, 2)
e1 = norm(N2 - N2_)
@show e1

N3 = LnL.symmtzrmat3(n)
N3_ = symmtzrmatN(n, 3)
e2 = norm(N3 - N3_)
@show e2

##


function elimat4(m::Int)
    # Calculate the number of rows in L
    num_rows = div(m*(m+1)*(m+2)*(m+3), 24)
    
    # Initialize the output matrix L
    L = zeros(Int, num_rows, m^4)
    
    d = 1
    @inbounds for i in 1:m
        for j in i:m
            for k in j:m
                for l in k:m
                    el = (1:m .== l) .|> Int
                    ek = (1:m .== k) .|> Int
                    ej = (1:m .== j) .|> Int
                    ei = (1:m .== i) .|> Int
                    eijkl = kron(kron(kron(ei, ej), ek), el)
                    L[d, :] = eijkl'
                    d += 1
                end
            end
        end
    end
    
    return L
end

function elimat5(m::Int)
    # Calculate the number of rows in L
    num_rows = div(m*(m+1)*(m+2)*(m+3)*(m+4), 120)
    
    # Initialize the output matrix L
    L = zeros(Int, num_rows, m^5)
    
    d = 1
    @inbounds for i in 1:m
        for j in i:m
            for k in j:m
                for l in k:m
                    for p in l:m
                        ep = (1:m .== p) .|> Int
                        el = (1:m .== l) .|> Int
                        ek = (1:m .== k) .|> Int
                        ej = (1:m .== j) .|> Int
                        ei = (1:m .== i) .|> Int
                        eijklp = kron(kron(kron(kron(ei, ej), ek), el), ep)
                        L[d, :] = eijklp'
                        d += 1
                    end
                end
            end
        end
    end
    
    return L
end



function elimatN(n::Int, p::Int)
    # Calculate the number of rows in L
    num_rows = binomial(n + p - 1, p)

    # Initialize the output matrix L
    L = spzeros(Int, num_rows, n^p)

    # Generate all combinations with repetition of n elements from {1, 2, ..., n}
    combs = with_replacement_combinations(1:n, p) 

    # Fill the matrix L
    for (l, comb) in enumerate(combs)
        v = [1]  # Start with a scalar 1
        @inbounds for d in 1:p
            e = collect(1:n .== comb[d])  # Create the indicator vector
            v = v ⊗ e  # Build the Kronecker product
        end
        L[l, :] = v  # Assign the row
    end

    return L
end

##

n = 7
L2 = LnL.elimat(n)
L2_ = elimatN(n, 2)
e1 = norm(L2 - L2_)
@show e1

L3 = LnL.elimat3(n)
L3_ = elimatN(n, 3)
e2 = norm(L3 - L3_)
@show e2

L4 = elimat4(n)
L4_ = elimatN(n, 4)
e3 = norm(L4 - L4_)
@show e3

L5 = elimat5(n)
L5_ = elimatN(n, 5)
e4 = norm(L5 - L5_)
@show e4

##
n = 3
N = 1000
e = zeros(N)

L4 = elimatN(n, 4)
D4 = dupmatN(n, 4)

for i in 1:N
    P = rand(n,n)
    P = P*P'
    Q = rand(n,n)
    Q = Q*Q'
    p = vec(P)
    q = vec(Q)
    qp = q ⊗ p

    x = rand(n)
    x4 = vec(x ⊗ x ⊗ x ⊗ x)

    s1 = qp' * x4
    s2 = (qp' * D4) * (L4 * x4)

    e[i] = norm(s1 - s2)
end

@show mean(e)

##

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

##

n = 3
p = 2
ind = [(1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2)]
val = [1.0, 2.0, 3.0, 4.0]

F = LnL.makeQuadOp(n, ind, val, which_quad_term="F")
A = LnL.makePolyOp(n, ind, val, nonredundant=true, symmetric=true)

##
n = 2
p = 3
ind = [(1, 1, 1, 1), (1, 1, 2,1), (1, 2, 2,2), (2, 2, 2,2)]
val = [1.0, 2.0, 3.0, 4.0]

E = LnL.makeCubicOp(n, ind, val, which_cubic_term="E")
A = LnL.makePolyOp(n, ind, val, nonredundant=true, symmetric=true)
