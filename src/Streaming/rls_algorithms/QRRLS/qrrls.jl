"""
$(TYPEDEF)

QR Decomposition Recursive Least-Squares (QRRLS) cache struct to solve for DO = R.
"""
mutable struct QRRLSCache{T<:Real}
    N::Int                              # Number of features (total dimension of operators)
    n::Int                              # Number of outputs (residual dimension)
    λ::T                                # Forgetting factor
    O::Matrix{T}                        # Operator matrix (N x n)
    P::Symmetric{T,Matrix{T}}           # Inverse correlation matrix (N x N)
    K::Matrix{T}                        # Kalman gain matrix (N x 1)
    Φsq::UpperTriangular{T,Matrix{T}}   # Square-root correlation matrix (lower triangular, N x N)
    q::Matrix{T}                        # Auxiliary matrix (N x n)
    ξpre::Matrix{T}                     # A priori error vector (1 x n)
    ξpost::Matrix{T}                    # A posteriori error vector (1 x n)
    C::T                                # Conversion factor (scalar)
    J::T                                # Cost (scalar)

    # Preallocated temporary variables
    A::Matrix{T}                        # Temporary matrix for QR factorization ((N + n + 1) x (N + n + 1))
    temp_dO::Matrix{T}                  # Temporary vector for d * O (1 x n)
    temp_Kd::Matrix{T}                  # Temporary matrix for P * d' (N x 1)

    mthd::Symbol                        # method used for updates, default is :qr
end


"""
Constructor: initialize all fields
"""
function QRRLSCache{T}(;N::Int=1, n::Int=1, λ::T=one(T), 
                        P::AbstractMatrix{T}=Matrix{T}(I, N, N),
                        Φsq::AbstractMatrix=Matrix{T}(I, N, N),
                        method::Symbol=:qr) where T<:Real
    λ       = T(λ)
    P_T     = convert(AbstractMatrix{T}, P)
    Φsq_T   = convert(AbstractMatrix{T}, Φsq)

    O       = zeros(T,N,n)
    P       = Symmetric(P_T)
    K       = zeros(T, N, 1)
    Φsq     = UpperTriangular(Φsq_T)  
    q       = zeros(T, N, n)
    ξpre    = zeros(T, 1, n)
    ξpost   = zeros(T, 1, n)
    C       = zero(T)
    J       = zero(T)
    A       = zeros(T, N+n+1, N+n+1) 
    temp_dO = zeros(T, 1, n)
    temp_Kd = zeros(T, N, 1)
    method  = method in (:givens, :qr) ? method : :qr
    return QRRLSCache{T}(N, n, λ, O, P, K, Φsq, q, ξpre,
                         ξpost, C, J, A, temp_dO, temp_Kd, method)
end


function qrrls!(obj::QRRLSCache{T}, d::AbstractArray{T}, 
                r::AbstractArray{T}) where T<:Real
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
    fill!(A, 0.0)  # Reset A to zero
    # Top-left block: λsq * Φsq
    @views mul!(A[1:N, 1:N], λsq, obj.Φsq)
    # Top-right block: λsq * q
    @views copyto!(A[1:N, N+1:N+n], λsq * obj.q)
    # Bottom row: [d; r; 1]
    @views A[N+1, 1:N] .= d[1, :]
    @views A[N+1, N+1:N+n] .= r[1, :]
    A[N+1, N+n+1] = T(1)

    # Perform in-place QR factorization of A (we want the R matrix)
    if obj.mthd == :qr
        LAPACK.geqrf!(A)  
    else # Givens rotations
        qr_givens!(A)
    end

    # Extract Φsq (upper triangular) and q
    obj.Φsq .= @views A[1:N, 1:N]
    obj.q .= @views A[1:N, N+1:N+n]
    Csq = A[N+1, N+n+1]
    obj.C = Csq^2

    # Update operator matrix O by solving Φsq * O = q
    # Since Φsq is upper triangular, use back substitution
    # obj.O .= UpperTriangular(obj.Φsq) \ obj.q
    obj.O .= obj.Φsq \ obj.q

    # Compute the a posteriori error: ξpost = r - d * O
    mul!(obj.temp_dO, d, obj.O, T(1), T(0))  # temp_dO: 1 x n
    obj.ξpost .= r .- obj.temp_dO            # ξpost: 1 x n

    # Update the cost J: J = λ * J + ξpre * ξpost'
    obj.J = obj.λ * obj.J + dot(vec(obj.ξpre), vec(obj.ξpost))

    # Update inverse correlation matrix P
    # obj.P .= (obj.Φsq' * obj.Φsq) \ I  # P = (Φsq' * Φsq)^-1
    # temp_Kd = P * d'
    mul!(obj.temp_Kd, obj.P, d', T(1), T(0))  # temp_Kd: N x 1
    denom = T(1) + dot(d, obj.temp_Kd) / obj.λ  # Scalar

    # Update Kalman gain K: K = (P * d') * (C / λ)
    obj.K .= obj.temp_Kd  # K = P * d'
    obj.K .*= obj.C / obj.λ  # K = K * (C / λ)

    # Update P in-place
    BLAS.syr!('U', -1.0 / (obj.λ * denom), obj.temp_Kd[:,1], obj.P.data)
    BLAS.scal!(1/obj.λ, obj.P.data)  # Scale P by 1/λ

    # Ensure symmetry of P
    # @inbounds for i in 1:N, j in i+1:N
    #     obj.P[j, i] = obj.P[i, j]
    # end

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