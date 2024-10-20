"""
$(TYPEDEF)

Inverse QR Decomposition Recursive Least-Squares (iQRRLS) cache struct to solve for DO = R.
"""
# mutable struct iQRRLSCache{T<:Real}
#     O::Array{T,2}           # Operator matrix
#     Psq::Array{T,2}         # Square-root inverse covariance matrix (lower triangular)
#     K::Array{T,2}           # Kalman gain matrix
#     ξpre::Array{T,1}        # A priori error vector
#     ξpost::Array{T,1}       # A posteriori error vector
#     C::T                    # Conversion factor (scalar)
#     J::T                    # Cost (scalar)
#     γ::T                    # Regularization term
#     λ::T                    # Forgetting factor

#     # Preallocated temporary variables
#     A::Array{T,2}           # Temporary matrix for QR factorization
#     u::Array{T,1}           # Temporary vector for computations
#     temp_dO::Array{T,1}     # Temporary vector for d * O
#     temp_Ke::Array{T,2}     # Temporary matrix for K * ξpre'
# end

mutable struct iQRRLSCache{T<:Real}
    O::Array{T,2}           # Operator matrix (N x n)
    Psq::Array{T,2}         # Square-root inverse covariance matrix (lower triangular, N x N)
    K::Array{T,2}           # Kalman gain matrix (N x n)
    ξpre::Array{T,1}        # A priori error vector (1 x n)
    ξpost::Array{T,1}       # A posteriori error vector (1 x n)
    C::T                    # Conversion factor (scalar)
    J::T                    # Cost (scalar)
    γ::T                    # Regularization term
    λ::T                    # Forgetting factor

    # Preallocated temporary variables
    A::Array{T,2}           # Temporary matrix for QR factorization ((N+1) x (N+1))
    u::Array{T,1}           # Temporary vector for computations (N x 1)
    temp_dO::Array{T,1}     # Temporary vector for d * O (1 x n)
    temp_Ke::Array{T,2}     # Temporary matrix for K * ξpre (N x n)
end


"""
Inverse QR Decomposition Recursive Least-Squares (iQRRLS) algorithm.

This function updates the operator inference state within the `iQRRLSCache` struct,
performing computations in-place and minimizing memory allocations.

# Arguments:
- `obj::iQRRLSCache`: iQRRLSCache object containing the state and preallocated variables.
- `d::AbstractArray`: Data vector (features) at the current time step.
- `r::AbstractArray`: Response vector (targets) at the current time step.

# Note
The function updates the following fields in `obj`:
- `O`, `Psq`, `K`, `ξpre`, `ξpost`, `C`, `J`.
"""
function iqrrls!(obj::iQRRLSCache{T}, d::AbstractMatrix{T}, r::AbstractMatrix{T}) where T<:Real
    # d: 1 x N (row vector)
    # r: 1 x n (row vector)
    N = size(d, 2)  # Number of features
    n = size(r, 2)  # Residual dimension (state dimension)
    λsq = sqrt(obj.λ)

    # Ensure temporary variables are correctly sized
    @assert size(obj.A) == (N+1, N+1)
    @assert length(obj.u) == N
    @assert size(obj.temp_dO) == (1, n)
    @assert size(obj.temp_Ke) == (N, n)

    # Compute the A matrix
    # A = [1                 zeros(1, N);
    #      Psq' * d' / λsq   Psq' / λsq]
    # Initialize A
    A = obj.A
    A .= 0
    A[1,1] = T(1)

    # Compute u = (Psq' * d') / λsq
    # d': N x 1
    mul!(obj.u, obj.Psq', d', T(1)/λsq, T(0))  # obj.u: N x 1

    # Set A[2:end, 1] = u
    @views copyto!(A[2:end, 1], obj.u)

    # Compute Psq_scaled = Psq' / λsq and set A[2:end, 2:end] = Psq_scaled
    @views mul!(A[2:end, 2:end], obj.Psq', LinearAlgebra.I, T(1)/λsq, T(0))

    # Perform in-place QR factorization of A
    F = qr!(A)  # QR factorization in-place; A is overwritten
    R = F.R  # Upper triangular matrix R

    # Extract Csq_inv and gCsq_inv
    Csq_inv = R[1,1]
    @views gCsq_inv = R[1,2:end]

    # Update Psq: Psq = (R[2:end, 2:end])', ensuring it's lower triangular
    @views copyto!(obj.Psq, R[2:end, 2:end]')
    obj.Psq .= LowerTriangular(obj.Psq)

    # Compute K = (gCsq_inv / Csq_inv)'
    obj.K .= (gCsq_inv ./ Csq_inv)'

    # Compute ξpre = r - d * O
    # d: 1 x N, O: N x n, d * O: 1 x n
    mul!(obj.temp_dO, d, obj.O, T(1), T(0))  # temp_dO: 1 x n
    obj.ξpre .= r .- obj.temp_dO  # ξpre: 1 x n

    # Update O: O += K * ξpre
    # K: N x n, ξpre: 1 x n (broadcasted), K * ξpre': N x n
    obj.O .+= obj.K .* obj.ξpre  # Element-wise multiplication and accumulation

    # Compute ξpost = r - d * O
    mul!(obj.temp_dO, d, obj.O, T(1), T(0))  # temp_dO: 1 x n
    obj.ξpost .= r .- obj.temp_dO  # ξpost: 1 x n

    # Update conversion factor C and cost J
    obj.C = T(1) / (Csq_inv^2)
    # Since ξpre and ξpost are 1 x n row vectors, compute dot product
    obj.J = obj.λ * obj.J + dot(vec(obj.ξpre), vec(obj.ξpost))

    return nothing
end


# """
# Inverse QR Decomposition Recursive Least-Squares (iQRRLS) algorithm.

# This function updates the operator inference state within the `iQRRLSCache` struct,
# performing computations in-place and minimizing memory allocations.

# # Arguments:
# - `obj::iQRRLSCache`: iQRRLSCache object containing the state and preallocated variables.
# - `d::AbstractArray`: Data vector (features) at the current time step.
# - `r::AbstractArray`: Response vector (targets) at the current time step.

# # Note
# The function updates the following fields in `obj`:
# - `O`, `Psq`, `K`, `ξpre`, `ξpost`, `C`, `J`.
# """
# function iqrrls!(obj::iQRRLSCache{T}, d::AbstractVector{T}, r::AbstractVector{T}) where T<:Real
#     dim = length(d)
#     n = length(r)
#     λsq = sqrt(obj.λ)

#     # Ensure temporary variables are correctly sized
#     @assert size(obj.A) == (dim+1, dim+1)
#     @assert length(obj.u) == dim
#     @assert length(obj.temp_dO) == n
#     @assert size(obj.temp_Ke) == (dim, n)

#     # Compute the A matrix
#     # A = [1               zeros(1, dim);
#     #      Psq' * d / λsq      Psq' / λsq]
#     # Initialize A
#     A = obj.A
#     A .= 0
#     A[1,1] = T(1)

#     # Compute u = (Psq' * d) / λsq
#     mul!(obj.u, obj.Psq', d, T(1)/λsq, T(0))

#     # Set A[2:end, 1] = u
#     @views copyto!(A[2:end, 1], obj.u)

#     # Compute Psq_scaled = Psq' / λsq and set A[2:end, 2:end] = Psq_scaled
#     @views mul!(A[2:end, 2:end], obj.Psq', LinearAlgebra.I, T(1)/λsq, T(0))

#     # Perform in-place QR factorization of A
#     F = qr!(A)  # QR factorization in-place; A is overwritten
#     R = F.R  # Upper triangular matrix R

#     # Extract Csq_inv and gCsq_inv
#     Csq_inv = R[1,1]
#     @views gCsq_inv = R[1,2:end]

#     # Update Psq: Psq = (R[2:end, 2:end])', ensuring it's lower triangular
#     @views copyto!(obj.Psq, R[2:end, 2:end]')
#     obj.Psq .= LowerTriangular(obj.Psq)

#     # Compute K = (gCsq_inv / Csq_inv)'
#     obj.K .= (gCsq_inv ./ Csq_inv)'  # transpose to make it a column vector

#     # Compute ξpre = r - O' * d
#     mul!(obj.temp_dO, obj.O', d, T(1), T(0))
#     obj.ξpre .= r .- obj.temp_dO

#     # Update O: O += K * ξpre'
#     mul!(obj.temp_Ke, obj.K, obj.ξpre', T(1), T(0))
#     obj.O .+= obj.temp_Ke

#     # Compute ξpost = r - O' * d
#     mul!(obj.temp_dO, obj.O', d, T(1), T(0))
#     obj.ξpost .= r .- obj.temp_dO

#     # Update conversion factor C and cost J
#     obj.C = T(1) / (Csq_inv^2)
#     obj.J = obj.λ * obj.J + dot(obj.ξpre, obj.ξpost)

#     return nothing
# end

"""
iQRRLS

P2_km1: is actually the square-root of the inverse of the correlation matrix
"""
# function iQRRLS(d_k::AbstractArray{T}, r_k::AbstractArray{T}, O_km1::AbstractArray{T},
#                 P2_km1::AbstractArray{T}, d::Int) where T<:Real
#     # Prearray
#     A_k = [1 zeros(1,d); P2_km1'*d_k' P2_km1']  # note: it's actually the transpose

#     # Compute postarray using QR factorization
#     _, B_k = qr(A_k)  

#     # Extract the square-root of the conversion factor and 
#     # the Kalman gain matrix multiplied by square-root of the conversion factor
#     α2_k_inv = B_k[1,1]
#     gα2_k_inv = B_k[1,2:end]  # becomes a column vector after slicing
#     P2_k = B_k[2:end, 2:end]'  # make sure it's lower triangular

#     # Compute the next operator matrix and Kalman gain matrix
#     K_k = gα2_k_inv * (α2_k_inv)^(-1)
#     O_k = O_km1 + K_k * (r_k - d_k * O_km1)
#     return O_k, P2_k, K_k
# end

# function iQRRLS(obj::iQRRLSCache, d::AbstractArray{T}, r::AbstractArray{T}, dim::Int) where T<:Real
#     # Prearray 
#     λsq = sqrt(obj.λ)
#     A = [1 zeros(1,dim); obj.Psq'*d'/λsq obj.Psq'/λsq]  # note: it's actually the transpose

#     # Compute postarray using QR factorization
#     B = qr(A).R 

#     # Extract the square-root of the conversion factor and
#     # the Kalman gain matrix multiplied by square-root of the conversion factor
#     Csq_inv = B[1,1]
#     gCsq_inv = B[1,2:end]  # becomes a column vector after slicing
#     obj.Psq = B[2:end, 2:end]'  # make sure it's lower triangular

#     # Compute the next operator matrix and Kalman gain matrix
#     obj.K = gCsq_inv / (Csq_inv)
#     obj.ξpre = r - d * obj.O
#     obj.O = obj.O + obj.K * obj.ξpre
#     obj.ξpost = r - d * obj.O
#     obj.C = 1 / Csq_inv^2
#     obj.J = obj.λ * obj.J + obj.ξpre' * obj.ξpost
# end

