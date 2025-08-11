"""
$(TYPEDEF)

QR Decomposition Recursive Least-Squares (QRRLS) cache struct to solve for DO = R.
"""
mutable struct QRRLSCache{T<:AbstractFloat}
    N::Int                                              # Number of features (total dimension of operators)
    n::Int                                              # Number of outputs (residual dimension)
    λ::T                                                # Forgetting factor
    O::Union{Matrix{T},CUDA.CuArray{T,2}}               # Operator matrix (N x n)
    P::Union{Symmetric{T,<:AbstractMatrix{T}}}          # Inverse correlation matrix (N x N)
    K::Union{Matrix{T},CUDA.CuArray{T,2}}               # Kalman gain matrix (N x 1)
    Φsq::Union{UpperTriangular{T,<:AbstractMatrix{T}}}  # Square-root correlation matrix (upper triangular, N x N)
    q::Union{Matrix{T},CUDA.CuArray{T,2}}               # Auxiliary matrix (N x n)
    ξpre::Union{Matrix{T},CUDA.CuArray{T,2}}            # A priori error vector (1 x n)
    ξpost::Union{Matrix{T},CUDA.CuArray{T,2}}           # A posteriori error vector (1 x n)
    C::T                                                # Conversion factor (scalar)
    J::T                                                # Cost (scalar)

    # Preallocated temporary variables
    A::Union{Matrix{T},CUDA.CuArray{T,2}}               # Temporary matrix for QR factorization ((N + n + 1) x (N + n + 1))
    temp_dO::Union{Matrix{T},CUDA.CuArray{T,2}}         # Temporary vector for d * O (1 x n)
    temp_Kd::Union{Matrix{T},CUDA.CuArray{T,2}}         # Temporary matrix for P * d' (N x 1)

    mthd::Symbol                                        # method used for updates, default is :qr
    use_gpu::Bool                                       # run on GPU
    tau                                                 # workspace for cuSOLVER.geqrf!
end


"""
Constructor: initialize all fields
"""
function QRRLSCache{T}(;N::Int=1, n::Int=1, λ::T=one(T),
                        P::AbstractMatrix{T}=Matrix{T}(I, N, N),
                        Φsq::AbstractMatrix=Matrix{T}(I, N, N),
                        method::Symbol=:qr,
                        use_gpu::Bool=false) where T<:AbstractFloat
    λ     = T(λ)
    method = method in (:givens, :qr) ? method : :qr

    if use_gpu
        O        = CUDA.zeros(T, N, n)
        P_dat    = CUDA.CuArray(P)
        P_wrapped= Symmetric(P_dat)
        K        = CUDA.zeros(T, N, 1)
        Φsq_dat  = CUDA.CuArray(Φsq)
        Φsq_up   = UpperTriangular(Φsq_dat)
        q        = CUDA.zeros(T, N, n)
        ξpre     = CUDA.zeros(T, 1, n)
        ξpost    = CUDA.zeros(T, 1, n)
        A        = CUDA.zeros(T, N+n+1, N+n+1)
        temp_dO  = CUDA.zeros(T, 1, n)
        temp_Kd  = CUDA.zeros(T, N, 1)
        C        = zero(T)
        J        = zero(T)
        tau      = CUDA.CuArray{T,1}(undef, N+n+1)

        @assert method == :qr "Givens rotations not implemented for GPU."

        return QRRLSCache{T}(N, n, λ, O, P_wrapped, K, Φsq_up, q, ξpre, ξpost, C, J,
                             A, temp_dO, temp_Kd, method, true, tau)
    else
        O        = zeros(T, N, n)
        P_wrapped= Symmetric(Matrix{T}(P))
        K        = zeros(T, N, 1)
        Φsq_up   = UpperTriangular(Matrix{T}(Φsq))
        q        = zeros(T, N, n)
        ξpre     = zeros(T, 1, n)
        ξpost    = zeros(T, 1, n)
        C        = zero(T)
        J        = zero(T)
        A        = zeros(T, N+n+1, N+n+1)
        temp_dO  = zeros(T, 1, n)
        temp_Kd  = zeros(T, N, 1)
        tau      = nothing
        return QRRLSCache{T}(N, n, λ, O, P_wrapped, K, Φsq_up, q, ξpre, ξpost, C, J,
                             A, temp_dO, temp_Kd, method, false, tau)
    end
end


# Public entry point: route to CPU/GPU and method
function qrrls!(obj::QRRLSCache{T}, d, r) where T<:AbstractFloat
    return obj.use_gpu ? qrrls_step_gpu!(obj, d, r) : qrrls_step!(obj, d, r)
end


# CPU implementation (existing logic, with minor fixes)
function qrrls_step!(obj::QRRLSCache{T}, d::AbstractArray{T},
                   r::AbstractArray{T}) where T<:AbstractFloat
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
    fill!(A, T(0))  # Reset A to zero
    # Top-left block: λsq * Φsq
    @views A[1:N, 1:N] .= λsq .* obj.Φsq
    # Top-right block: λsq * q
    @views A[1:N, N+1:N+n] .= λsq .* obj.q
    # Bottom row: [d; r; 1]
    @views A[N+1, 1:N] .= d[1, :]
    @views A[N+1, N+1:N+n] .= r[1, :]
    A[N+1, N+n+1] = T(1)

    # Perform in-place QR factorization of A (we want the R matrix)
    if obj.mthd == :qr
        qr!(A)  # This is slightly faster than LAPACK.geqrf!
        # LAPACK.geqrf!(A)  
    else # Givens rotations
        qrrls_givens_fast!(A)
    end

    # Extract Φsq (upper triangular) and q
    obj.Φsq .= @views A[1:N, 1:N]
    obj.q   .= @views A[1:N, N+1:N+n]
    Csq = A[N+1, N+n+1]
    obj.C = Csq^2

    # Update operator matrix O by solving Φsq * O = q
    obj.O .= obj.Φsq \ obj.q

    # Compute the a posteriori error: ξpost = r - d * O
    mul!(obj.temp_dO, d, obj.O, T(1), T(0))  # temp_dO: 1 x n
    obj.ξpost .= r .- obj.temp_dO            # ξpost: 1 x n

    # Update the cost J: J = λ * J + ξpre ⋅ ξpost
    obj.J = obj.λ * obj.J + dot(vec(obj.ξpre), vec(obj.ξpost))

    # Update inverse correlation matrix P
    mul!(obj.temp_Kd, obj.P, d', T(1), T(0))  # temp_Kd: N x 1
    denom = T(1) + dot(d, obj.temp_Kd) / obj.λ  # Scalar

    # Update Kalman gain K: K = (P * d') * (C / λ)
    obj.K .= obj.temp_Kd
    obj.K .*= obj.C / obj.λ

    # Update P in-place (upper triangle)
    BLAS.syr!('U', -one(T) / (obj.λ * denom), obj.temp_Kd[:,1], obj.P.data)
    BLAS.scal!(one(T)/obj.λ, obj.P.data)  # Scale P by 1/λ

    return nothing
end


# GPU implementation (CUDA/cuSOLVER)
function qrrls_step_gpu!(obj::QRRLSCache{T}, d::CUDA.CuArray{T,2},
                       r::CUDA.CuArray{T,2}) where T<:Union{Float32,Float64}
    N, n = obj.N, obj.n
    λsq = sqrt(obj.λ)
    A = obj.A

    # a priori error: ξpre = r - d * O
    mul!(obj.temp_dO, d, obj.O, one(T), zero(T))
    obj.ξpre .= r .- obj.temp_dO

    # Build augmented A on device
    fill!(A, zero(T))
    @views A[1:N, 1:N]         .= λsq .* obj.Φsq
    @views A[1:N, N+1:N+n]     .= λsq .* obj.q
    @views A[N+1, 1:N]         .= d[1, :]
    @views A[N+1, N+1:N+n]     .= r[1, :]
    A[N+1:N+1, N+n+1:N+n+1] = one(T)

    # QR factorization (in-place, geqrf!)
    CUSOLVER.geqrf!(A, obj.tau)

    # Extract Φsq, q, and C
    obj.Φsq .= @views A[1:N, 1:N]
    obj.q   .= @views A[1:N, N+1:N+n]
    Csq      = A[N+1, N+n+1]
    obj.C    = Csq^2

    # Solve Φsq * O = q (triangular solve on GPU)
    obj.O .= obj.Φsq \ obj.q

    # a posteriori error
    mul!(obj.temp_dO, d, obj.O, one(T), zero(T))
    obj.ξpost .= r .- obj.temp_dO

    # cost
    obj.J = obj.λ * obj.J + dot(vec(obj.ξpre), vec(obj.ξpost))

    # P and K updates (device)
    mul!(obj.temp_Kd, obj.P, d', one(T), zero(T))    # temp_Kd = P * d'
    denom = one(T) + dot(d, obj.temp_Kd) / obj.λ     # host scalar

    obj.K .= obj.temp_Kd
    obj.K .*= obj.C / obj.λ

    # Symmetric rank-1 update (full form, then scale), all on device:
    # P ← (P - (temp_Kd*temp_Kd') / (λ*denom)) / λ
    obj.P.data .-= (one(T) / (obj.λ * denom)) .* (obj.temp_Kd * obj.temp_Kd')
    obj.P.data .*= one(T) / obj.λ

    return nothing
end


function qrrls_givens_fast!(A::AbstractMatrix{T}) where {T<:AbstractFloat}
    """
    In-place Givens rotation applied to the first column.
    Modifies A directly and returns it.
    """
    np1, nprp1 = size(A)
    n = np1 - 1
    r = nprp1 - np1
    
    # Apply Givens rotations 
    for j in 1:n
        c, s, r = givens_rotation(A[j, j], A[np1, j])
        
        # Apply rotation directly to rows 1 and j
        @inbounds @simd for k in j:nprp1
            ajk = A[j, k]
            ank = A[np1, k]
            A[j, k] = c * ajk - s * ank
            A[np1, k] = s * ajk + c * ank
        end
    end

    # Apply negation and ensure upper triangular structure
    @inbounds @simd for j in 1:nprp1
        # Negate upper triangular part
        for i in 1:min(j, n)
            A[i, j] = -A[i, j]
        end
        # Zero out lower triangular part (except last row)
        for i in (j+1):n
            A[i, j] = zero(T)
        end
    end
    
    # Ensure zero for last row in 1:n columns
    @inbounds @simd for j in 1:n
        A[np1, j] = 0.0
    end
    
    return A
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