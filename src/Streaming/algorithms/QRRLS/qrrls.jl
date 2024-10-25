"""
$(TYPEDEF)

QR Decomposition Recursive Least-Squares (QRRLS) cache struct to solve for DO = R.
"""
@with_kw mutable struct QRRLSCache{T<:Real}
    N::Int                                        # Number of features (total dimension of operators)
    n::Int                                        # Number of outputs (residual dimension)
    O::Array{T,2} = zeros(T,N,n)          # Operator matrix (N x n)
    P::Array{T,2}                                 # Inverse correlation matrix (N x N)
    K::Array{T,2} = zeros(T,N,1)          # Kalman gain matrix (N x 1)
    Φsq::AbstractArray{T,2}                       # Square-root correlation matrix (upper triangular, N x N)
    q::Array{T,2} = zeros(T,N,n)          # Auxiliary matrix (N x n)
    ξpre::Array{T,2} = zeros(T,1,n)       # A priori error vector (1 x n)
    ξpost::Array{T,2} = zeros(T,1,n)      # A posteriori error vector (1 x n)
    C::T = zero(T)                                # Conversion factor (scalar)
    J::T = zero(T)                                # Cost (scalar)
    γ::T                                          # Regularization term
    λ::T                                          # Forgetting factor

    # Preallocated temporary variables
    A::Array{T,2} = zeros(T,N+n+1,N+n+1)  # Temporary matrix for QR factorization ((N + n + 1) x (N + n + 1))
    temp_dO::Array{T,2} = zeros(T,1,n)    # Temporary vector for d * O (1 x n)
    temp_Kd::Array{T,2} = zeros(T,N,1)    # Temporary matrix for P * d' (N x 1)
end


function qrrls!(obj::QRRLSCache{T}, d::AbstractArray{T}, r::AbstractArray{T}) where T<:Real
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

    # Update inverse correlation matrix P
    # obj.P .= (obj.Φsq' * obj.Φsq) \ I  # P = (Φsq' * Φsq)^-1
    # temp_Kd = P * d'
    mul!(obj.temp_Kd, obj.P, d', T(1), T(0))  # temp_Kd: N x 1
    denom = T(1) + (d * obj.temp_Kd)[1,1] / obj.λ  # Scalar

    # Update P in-place
    BLAS.syr!('U', -1.0 / (obj.λ * denom), obj.temp_Kd[:,1], obj.P)
    obj.P ./= obj.λ  # P = P / λ

    # Ensure symmetry of P
    @inbounds for i in 1:N, j in i+1:N
        obj.P[j, i] = obj.P[i, j]
    end

    # Update Kalman gain K: K = (P * d') * (C / λ)
    obj.K .= obj.temp_Kd  # K = P * d'
    obj.K .*= obj.C / obj.λ  # K = K * (C / λ)

    return nothing
end


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