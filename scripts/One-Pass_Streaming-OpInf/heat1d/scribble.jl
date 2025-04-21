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
Xhat = U' * X
X5 = Xhat ⦼ Xhat

##


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