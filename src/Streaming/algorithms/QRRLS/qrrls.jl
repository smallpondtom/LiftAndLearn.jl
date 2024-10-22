"""
$(TYPEDEF)

QR Decomposition Recursive Least-Squares (QRRLS) cache struct to solve for DO = R.
"""
# mutable struct QRRLSCache{T<:Real}
#     O::Array{T,2}        # Operator matrix
#     P::Array{T,2}        # Inverse covariance matrix
#     K::Array{T,2}        # Kalman gain matrix
#     Φsq::Array{T,2}      # Square-root covariance matrix (upper triangular)
#     q::Array{T,2}        # Auxiliary matrix
#     ξpre::Array{T,1}     # A priori error vector
#     ξpost::Array{T,1}    # A posteriori error vector
#     C::T                 # Conversion factor (scalar)
#     J::T                 # Cost (scalar)
#     γ::T                 # Regularization term
#     λ::T                 # Forgetting factor

#     # Preallocated temporary variables
#     A::Array{T,2}        # Temporary matrix for QR factorization
#     temp_dO::Array{T,1}  # Temporary vector for d * O
#     temp_Kd::Array{T,2}  # Temporary matrix for P * d
#     temp_dP::Array{T,2}  # Temporary matrix for d' * P
# end
mutable struct QRRLSCache{T<:Real}
    O::Array{T,2}           # Operator matrix (N x n)
    P::Array{T,2}           # Inverse covariance matrix (N x N)
    K::Array{T,2}           # Kalman gain matrix (N x n)
    Φsq::AbstractArray{T,2} # Square-root covariance matrix (upper triangular, N x N)
    q::Array{T,2}           # Auxiliary matrix (N x n)
    ξpre::Array{T,2}        # A priori error vector (1 x n)
    ξpost::Array{T,2}       # A posteriori error vector (1 x n)
    C::T                    # Conversion factor (scalar)
    J::T                    # Cost (scalar)
    γ::T                    # Regularization term
    λ::T                    # Forgetting factor

    # Preallocated temporary variables
    A::Array{T,2}           # Temporary matrix for QR factorization ((N + n + 1) x (N + n + 1))
    temp_dO::Array{T,2}     # Temporary vector for d * O (1 x n)
    temp_Kd::Array{T,2}     # Temporary matrix for P * d' (N x 1)
end

function qrrls!(obj::QRRLSCache{T}, d::AbstractMatrix{T}, r::AbstractMatrix{T}) where T<:Real
    # d: 1 x N (row vector)
    # r: 1 x n (row vector)
    N = size(d, 2)  # Number of features
    n = size(r, 2)  # residual dimension (state dimension)
    λsq = sqrt(obj.λ)

    # Ensure temporary variables are correctly sized
    @assert size(obj.A) == (N + n + 1, N + n + 1)
    @assert size(obj.temp_dO) == (1, n)
    @assert size(obj.temp_Kd) == (N, 1)

    # Compute the a priori error: ξpre = r - d * O
    mul!(obj.temp_dO, d, obj.O, T(1), T(0))  # temp_dO: 1 x n
    obj.ξpre .= r .- obj.temp_dO             # ξpre: 1 x n

    # Construct the augmented matrix A for QR factorization
    # A = [λsq * Φsq   λsq * q   zeros(N, 1);
    #      d           r         1]
    A = obj.A
    A .= 0  # Reset A to zero
    # Top-left block: λsq * Φsq
    @views mul!(A[1:N, 1:N], λsq, obj.Φsq)
    # Top-right block: λsq * q
    @views copyto!(A[1:N, N+1:N+n], λsq * obj.q)
    # Bottom row: [d; r; 1]
    @views A[N+1, 1:N] .= d[1, :]
    @views A[N+1, N+1:N+n] .= r[1, :]
    A[N+1, N+n+1] = T(1)

    # Perform in-place QR factorization of A (we want the R matrix)
    F = qr!(A)  # A is overwritten

    # Extract Φsq (upper triangular) and q
    obj.Φsq .= @views A[1:N, 1:N]
    obj.q .= @views A[1:N, N+1:N+n]
    Csq = A[N+n+1, N+n+1]
    obj.C = Csq^2

    # Update operator matrix O by solving Φsq * O = q
    # Since Φsq is upper triangular, use back substitution
    obj.O .= UpperTriangular(obj.Φsq) \ obj.q

    # Compute the a posteriori error: ξpost = r - d * O
    mul!(obj.temp_dO, d, obj.O, T(1), T(0))  # temp_dO: 1 x n
    obj.ξpost .= r .- obj.temp_dO            # ξpost: 1 x n

    # Update the cost J: J = λ * J + ξpre * ξpost'
    obj.J = obj.λ * obj.J + dot(vec(obj.ξpre), vec(obj.ξpost))

    # Update inverse covariance matrix P
    # temp_Kd = P * d'
    mul!(obj.temp_Kd, obj.P, d', T(1), T(0))  # temp_Kd: N x 1
    denom = T(1) + (d * obj.temp_Kd)[1,1] / obj.λ  # Scalar

    # Update P in-place
    BLAS.syr!('U', -1.0 / (obj.λ * denom), obj.temp_Kd[:,1], obj.P)
    obj.P ./= obj.λ  # P = P / λ

    # Ensure symmetry of P
    for i in 1:N, j in i+1:N
        obj.P[j, i] = obj.P[i, j]
    end

    # Update Kalman gain K: K = (P * d') * (C / λ)
    obj.K .= obj.temp_Kd  # K = P * d'
    obj.K .*= obj.C / obj.λ  # K = K * (C / λ)

    return nothing
end

# """
# QR Decomposition Recursive Least Squares (QRRLS) algorithm.

# This function updates the operator inference state within the `QRRLSCache` struct,
# performing computations in-place and minimizing memory allocations.

# # Arguments:
# - `obj::QRRLSCache`: QRRLSCache object containing the state and preallocated variables.
# - `d::AbstractArray`: Data vector (features) at the current time step (column vector).
# - `r::AbstractArray`: Residual vector (targets) at the current time step (column vector).

# # Note
# The function updates the following fields in `obj`:
# - `O`, `Φsq`, `q`, `P`, `K`, `ξpre`, `ξpost`, `C`, `J`.
# """
# function qrrls!(obj::QRRLSCache{T}, d::AbstractVector{T}, r::AbstractVector{T}) where T<:Real
#     dim = length(d)
#     n = length(r)
#     λsq = sqrt(obj.λ)

#     # Ensure temporary variables are correctly sized
#     @assert size(obj.A) == (dim + n + 1, dim + n + 1)
#     @assert length(obj.temp_dO) == n

#     # Compute the a priori error: ξpre = r - O' * d
#     mul!(obj.temp_dO, obj.O', d, T(1), T(0))  # temp_dO = O' * d
#     obj.ξpre .= r .- obj.temp_dO              # ξpre = r - temp_dO

#     # Construct the augmented matrix A for QR factorization
#     # A = [λsq * Φsq  λsq * q    zeros(dim, 1);
#     #      d'         r'         1]
#     A = obj.A
#     A .= 0  # Reset A to zero
#     # Top-left block: λsq * Φsq
#     @views mul!(A[1:dim, 1:dim], λsq, obj.Φsq)
#     # Top-right block: λsq * q
#     @views copyto!(A[1:dim, dim+1:dim+n], λsq * obj.q)
#     # Bottom row: [d'; r'; 1]
#     @views A[dim+1, 1:dim] .= d'
#     @views A[dim+1, dim+1:dim+n] .= r'
#     A[dim+1, dim+n+1] = T(1)

#     # Perform in-place QR factorization of A
#     F = qr!(A)  # A is overwritten

#     # Extract Φsq (upper triangular) and q
#     obj.Φsq .= @views A[1:dim, 1:dim]
#     obj.q .= @views A[1:dim, dim+1:dim+n]
#     Csq = A[dim+1, dim+n+1]
#     obj.C = Csq^2

#     # Update operator matrix O by solving Φsq * O = q
#     # Since Φsq is upper triangular, use back substitution
#     obj.O .= LinearAlgebra.UpperTriangular(obj.Φsq) \ obj.q

#     # Compute the a posteriori error: ξpost = r - O' * d
#     mul!(obj.temp_dO, obj.O', d, T(1), T(0))
#     obj.ξpost .= r .- obj.temp_dO

#     # Update the cost J: J = λ * J + ξpre' * ξpost
#     obj.J = obj.λ * obj.J + dot(obj.ξpre, obj.ξpost)

#     # Update inverse covariance matrix P
#     # P = (P / λ) - (P * d * d' * P) / (λ^2 * denom)
#     # denom = 1 + (d' * P * d) / λ
#     # Compute temp_Kd = P * d
#     mul!(obj.temp_Kd, obj.P, d)
#     denom = T(1) + dot(d, obj.temp_Kd) / obj.λ

#     # Update P in-place
#     BLAS.syr!('U', -1.0 / (obj.λ^2 * denom), obj.temp_Kd, obj.P)
#     obj.P .= obj.P / obj.λ  # P = P / λ

#     # Ensure symmetry of P
#     for i in 1:dim, j in i+1:dim
#         obj.P[j, i] = obj.P[i, j]
#     end

#     # Update Kalman gain K: K = (P * d) * (C / λ)
#     obj.K .= obj.temp_Kd  # K = P * d (already computed)
#     obj.K .*= obj.C / obj.λ  # K = K * (C / λ)

#     return nothing
# end



# """
# QRRLS
# """
# function qrrls(d_k::AbstractArray{T}, r_k::AbstractArray{T}, Φ_km1::AbstractArray{T}, 
#                q_km1::AbstractArray{T}, d::Int, r::Int) where T<:Real
#     # Prearray
#     A_k = [Φ_km1' q_km1; d_k r_k]  # note: it's actually the transpose

#     # Compute postarray using QR factorization
#     qr!(A_k)  # in-place QR factorization (B_k = A_k)

#     # Extract the inverse covariance matrix and auxiliary matrix
#     Φ_km1 = A_k[1:d, 1:d]  # keep it upper triangular here
#     q_km1 = A_k[1:d, d+1:d+r] 

#     # Compute the next operator matrix with inverse of upper triangular matrix
#     O_k = Φ_km1 \ q_km1   # (backslash inverse) automatically does backward substitution
#     # O_k = copy(q_km1)
#     # backsub!(Φ_km1', O_k)  # (backward subtitution) transpose to make upper triangular

#     # Compute the inverse covariance matrix and Kalman gain matrix
#     P_k = (Φ_km1'*Φ_km1) \ I   # Φ_km1 is still upper triangular
#     K_k = P_k * d_k'
#     return O_k, Φ_km1', q_km1, P_k, K_k
# end

# function qrrls!(obj::QRRLSCache, d::AbstractArray{T}, r::AbstractArray{T}) where T<:Real
#     dim = length(d)
#     n = length(r)
#     λsq = sqrt(obj.λ)

#     # Compute the a priori error
#     obj.ξpre = r - d * obj.O

#     # Prearray
#     A = [λsq*obj.Φsq' λsq*obj.q 0; d r 1]  # note: it's actually the transpose

#     # Compute postarray using QR factorization
#     qr!(A)  # in-place QR factorization (B = A)

#     # Extract the inverse covariance matrix and auxiliary matrix
#     obj.Φsq = A[1:dim, 1:dim]  # keep it upper triangular here
#     obj.q = A[1:dim, dim+1:dim+n] 
#     Csq = A[end, end] 
#     obj.C = Csq^2

#     # Compute the next operator matrix with inverse of upper triangular matrix
#     obj.O = obj.Φ \ obj.q   # (backslash inverse) automatically does backward substitution

#     # Compute the a posteriori error
#     obj.ξpost = r - d * obj.O

#     # Compute the cost
#     obj.J = obj.λ * obj.J + obj.ξpre' * obj.ξpost

#     # Compute the inverse covariance matrix and Kalman gain matrix
#     obj.K = obj.P * d' * obj.C / obj.λ
#     denom = 1 + d' * obj.P * d / obj.λ
#     obj.P = obj.P / obj.λ - obj.P * d * d' * obj.P / denom / obj.λ^2

#     return nothing
# end


# function backsub!(U::Matrix{T}, x::Vector{T}) where T<:Real
#     n = length(x)
#     # Backward substitution for U*x = y
#     @inbounds for i = n:-1:1
#         x[i] /= U[i, i]
#         for j = 1:i-1
#             x[j] -= A[j, i] * x[i]
#         end
#     end
# end


# function backsub!(U::Matrix{T}, X::Matrix{T}) where T<:Real
#     n = size(X,1)
    
#     # Ensure the dimensions match
#     if size(U, 1) != n || size(U, 2) != n
#         error("Dimensions of U and X do not match")
#     end
    
#     # vectorized backward substitution for U*X = Y
#     @inbounds for i in n:-1:1
#         X[i, :] ./= U[i, i]
#         X[1:i-1, :] .-= U[1:i-1, i] .* X[i, :]'
#     end
# end

