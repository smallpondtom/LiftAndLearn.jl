using LinearAlgebra
using Random
import LiftAndLearn as LnL

##
X = rand(5,10)
U, S, V = svd(X)

##
isvd = LnL.initialize_brand(X[:,1]; max_rank=5)
K = size(X, 2)
for i in 2:K
    LnL.increment!(isvd, X[:,i], 1e-8)
end

##
norm(S - isvd.Σ) / norm(S)

##
sqrt(sum(abs2, 1 .- svdvals(U' * isvd.V)))

##
sqrt(sum(abs2, 1 .- svdvals(V' * isvd.W)))

##
opnorm(X - isvd.V * Diagonal(isvd.Σ) * isvd.W') / opnorm(X)


##
U2, S2, V2 = unko(X, 5)


##
norm(S - S2) / norm(S)

##
sqrt(sum(abs2, 1 .- svdvals(U' * U2)))

##
sqrt(sum(abs2, 1 .- svdvals(V' * V2)))

##
opnorm(X - U2 * Diagonal(S2) * V2') / opnorm(X)


##
sqrt(sum(abs2, 1 .- svdvals(isvd.W' * V2)))

##

function unko(X, rmax)
    n, K = size(X)

    x1 = X[:,1] 
   
    V = x1 / norm(x1)
    Σ = norm(x1)
    W = 1.0
    r1 = 1  

    for i in 2:K 
        xi = X[:,i]

        q1 = V' * xi
        xperp = xi - V * q1
        q2 = V' * xperp
        xperp = xperp - V * q2
        q = q1 + q2
        p = norm(xperp)

        p = [p]
        xperp = reshape(xperp, :, 1)
        qrf!(xperp, p)
        p = p[1]

        C = zeros(r1+1, r1+1)
        for j in 1:r1
            C[j,j] = Σ[j]
            C[j,end] = q[j]
        end
        C[end,end] = p

        Vc, Σc, Wc = svd(C)
        V = hcat(V, xperp) * Vc
        Σ = Σc
        W = [W zeros(size(W,1), 1); zeros(1, r1) 1.0] * Wc
        r1 += 1

        if r1 > rmax
            V = V[:,1:rmax]
            Σ = Σ[1:rmax]
            W = W[:,1:rmax]
            r1 = rmax
        end
    end

    return LinearAlgebra.SVD{Float64}(V, Σ, W')
end


##
function qrf!(P::AbstractArray{T}, R::AbstractArray{T}) where {T<:Number}
    m, b = checksize(P)
    m >= b || throw(DimensionMismatch("Works only for m > b"))
    P, tau = LAPACK.geqrf!(P)
    fill!(R, zero(T))
    @inbounds for j = 1:b, i = 1:j
        R[i,j] = P[i,j]
    end
    LAPACK.orgqr!(P, tau)
    return R
end

function qrf!(P::AbstractArray{<:Number})
    m, b = checksize(P)
    m >= b || throw(DimensionMismatch("Works only for m > b"))
    P, tau = LAPACK.geqrf!(P)
    LAPACK.orgqr!(P, tau)
end

function checksize(A::AbstractArray)
    m, n = nothing, nothing
    try
        m, n = size(A)
    catch e
        if isa(e, BoundsError)
            m, n = length(A), 1
        else
            rethrow(e)
        end
    end
    return m, n
end

##

using LinearAlgebra
using LinearMaps
using Kronecker: ⊗
using BenchmarkTools
using UniqueKronecker

##
n, K = 3, 4
X = rand(n,K)
U,S,V = svd(X)

##
X2 = X ⦼ X

##
U2, S2, V2 = svd(X2)


##
L2 = elimat(n, 2)
D2 = dupmat(n, 2)
N2 = symmtzrmat(n ,2)

##

U3 =  L2 * (U ⊗ U) * D2
S3 = Diagonal(S ⊘ S)
V3 = V ⧁ V
X3 = U3 * S3 * V3'

##
X4 = L2 * (U ⊗ U)' * D2 * X2

##
D2' * (U ⊗ U)' * L2' * D2' * (U ⊗ U) * L2'

##
L2 * (U ⊗ U) * D2 * L2 * (U ⊗ U)' * D2

##
Xhat = U' * X
X5 = Xhat ⦼ Xhat

##
X3 = X ⊙ X ⊙ X

##
U3 = U ⊗ U ⊗ U
S3 = S ⊗ S ⊗ S
S3 = Diagonal(S3[:])
V3 = V ⊖ V ⊖ V
U3 * S3 * V3'

##
X3 = ⦼(X, 3)

##
L3 = elimat(n, 3)
D3 = dupmat(n, 3)
U3 = Matrix(U ⊗ U ⊗ U)
S3 = Diagonal(⊘(S, 3))
V3 = ⧁(V, 3)
(L3 * U3 * D3) * S3 * V3'

##

# ##
# U2 = U ⦼ U


# ##
# D2 = dupmat(3,2)
# U2 = L2 * (U ⊗ U)

# ##
# U2' * U2

# ##
# U2 * U2'

# ## 
# U2 = U ⊗ U 

# ##
# U2' * U2

# ##
# D2' * D2

# ##
# L2 = elimat(n, 2)

# ##
# L2' * L2

# ##

# L2 * L2'

# ##
# S2 = symmtzrmat(3,2)
# S2' * S2

# ##
# u2 = U[:,1] ⊘ U[:,1]

# ##
# u2' * u2

# ##
# u2 = D2 * u2

# ##
# u2' * u2

##
# ##
# # function khatri_rao1(A)
# #     n = size(A, 2)
# #     # Compute the Kronecker product for each column with itself
# #     # C = hcat([kron(A[:, i], A[:, i]) for i in 1:n]...)
# #     C = hcat(map(kronecker, eachcol(A), eachcol(A))...)
# #     return C
# # end

# # khatri_rao2(A::AbstractMatrix, B::AbstractMatrix) = Matrix(hcat(map(LinearMap∘kronecker, eachcol(A), eachcol(B))...))
# # khatri_rao2(A::AbstractMatrix) = khatri_rao2(A, A)
# # ⊙(A::AbstractMatrix, B::AbstractMatrix) = khatri_rao2(A, B)
# # ⊙(A::AbstractMatrix) = khatri_rao2(A, A)

# ##
# function khatri_rao(A::AbstractMatrix, B::AbstractMatrix)
#     return hcat(map(kronecker, eachcol(A), eachcol(B))...)
# end
# khatri_rao(A::AbstractMatrix) = khatri_rao(A, A)
# ⊙(A::AbstractMatrix, B::AbstractMatrix) = khatri_rao(A, B)
# ⊙(A::AbstractMatrix) = khatri_rao(A, A)

# function khatri_rao(mats::AbstractMatrix...)
#     L = length(mats)
#     L ≥ 1 || throw(ArgumentError("need at least one matrix"))
#     nc = size(mats[1], 2)
#     for M in mats
#         size(M, 2) == nc || throw(ArgumentError("all matrices must have the same number of columns"))
#     end
#     if L == 1
#         return khatri_rao(mats[1])
#     elseif L == 2
#         return khatri_rao(mats[1], mats[2])
#     else
#         cols = [ reduce(kronecker, (M[:, j] for M in mats)) for j in 1:nc ]
#     end
#     return hcat(cols...)
# end

# ⊙(mats::AbstractMatrix...) = khatri_rao(mats...)

# function khatri_rao(A::AbstractMatrix, d::Integer)
#     d ≥ 1 || throw(ArgumentError("d must be at least 1"))
#     n = size(A, 2)
#     cols = [ reduce(kron, ntuple(_->A[:, j], d)) for j in 1:n ]
#     return hcat(cols...)
# end

# ⊙(A::AbstractMatrix, d::Integer) = khatri_rao(A, d)

# ##

# """
#     unique_khatri_rao(A::AbstractMatrix, B::AbstractMatrix)

# Column-wise unique Kronecker (Khatri-Rao) of A and B.
# """
# function unique_khatri_rao(A::AbstractMatrix, B::AbstractMatrix)
#     size(A,2) == size(B,2) ||
#       throw(ArgumentError("matrices must have same number of columns"))
#     return hcat(map(unique_kronecker, eachcol(A), eachcol(B))...)
# end

# unique_khatri_rao(A::AbstractMatrix) = unique_khatri_rao(A, A)

# """
#     unique_khatri_rao(mats::AbstractMatrix...)

# Generalized column-wise unique Kronecker of any number of matrices.
# """
# function unique_khatri_rao(mats::AbstractMatrix...)
#     L = length(mats)
#     L ≥ 1 || throw(ArgumentError("need at least one matrix"))
#     nc = size(mats[1], 2)
#     for M in mats
#         size(M,2) == nc ||
#           throw(ArgumentError("all matrices must have the same number of columns"))
#     end
#     if L == 1
#         return unique_khatri_rao(mats[1])
#     elseif L == 2
#         return unique_khatri_rao(mats[1], mats[2])
#     else
#         cols = [ reduce(unique_kronecker, (M[:,j] for M in mats)) for j in 1:nc ]
#         return hcat(cols...)
#     end
# end

# """
#     unique_khatri_rao(A::AbstractMatrix, d::Integer)

# Raise each column of A to the unique-Kronecker power d.
# """
# function unique_khatri_rao(A::AbstractMatrix, d::Integer)
#     d ≥ 1 || throw(ArgumentError("d must be at least 1"))
#     nc = size(A,2)
#     cols = [ unique_kronecker_power(A[:,j], d) for j in 1:nc ]
#     return hcat(cols...)
# end

# # operator aliases
# ⦼(A::AbstractMatrix, B::AbstractMatrix)    = unique_khatri_rao(A, B)
# ⦼(mats::AbstractMatrix...)                 = unique_khatri_rao(mats...)
# ⦼(A::AbstractMatrix, d::Integer)           = unique_khatri_rao(A, d)

# ##

# function face_split1(A)
#     m = size(A,1)
#     C = vcat([kronecker(A[i, :], A[i, :])' for i in 1:m]...)
#     return C
# end

# face_split(A::AbstractMatrix, B::AbstractMatrix) = vcat(map(transpose ∘ kronecker, eachrow(A), eachrow(B))...)
# face_split(A::AbstractMatrix) = face_split(A, A)
# ⊖(A::AbstractMatrix, B::AbstractMatrix) = face_split(A, B)
# ⊖(A::AbstractMatrix) = face_split(A, A)


# ## 
# # U2 = khatri_rao1(U)
# # U2' * U2

# @benchmark khatri_rao1(U)

# ## 
# @benchmark khatri_rao2(U)

# ##
# U3 = face_split(U)
# U3 * U3'

# ##
# V2 = khatri_rao(V)
# V2' * V2

# ##
# V3 = face_split(V)
# V3 * V3'

# ## 
# U2 = kron(U, U)
# U2' * U2

# ##
# khatri_rao(U)