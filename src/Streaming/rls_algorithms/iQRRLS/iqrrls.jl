"""
Cache for inverse-QR RLS via square-root updates, with vectorized loops.
Maintains lower-triangular Psq = Cholesky factor of P^{-1}. With GPU support.
"""
mutable struct iQRRLSCache{T<:AbstractFloat}
    N::Int                     # number of features
    n::Int                     # output dimension
    λ::T                       # forgetting factor

    O::Union{Matrix{T},CuArray{T,2}}                    # operator matrix (N×n)
    Psq::Union{LowerTriangular{T,<:AbstractMatrix{T}}}  # sqrt inv-cov
    u::Union{Vector{T},CuArray{T,1}}                    # temp vector (N)
    K::Union{Matrix{T},CuArray{T,2}}                    # Kalman gain (N×1)
    ξpre::Union{Matrix{T},CuArray{T,2}}                 # a priori error (1×n)
    ξpost::Union{Matrix{T},CuArray{T,2}}                # a posteriori error (1×n)
    A::Union{Matrix{T},CuArray{T,2}}                    # temp QR matrix ((N+1)×(N+1))
    temp_dO::Union{Matrix{T},CuArray{T,2}}              # temp for d*O (1×n)
    temp_Ke::Union{Matrix{T},CuArray{T,2}}              # temp for K*ξpre (N×n)

    C::T                       # conversion factor
    J::T                       # cumulative cost
    mthd::Symbol               # :qr or :givens
    use_gpu::Bool              # flag to run on GPU
    tau                        # workspace for cuSOLVER.geqrf! (length N+1)
end

"""
Constructor: initialize all fields and wrap Psq via LowerTriangular.
"""
function iQRRLSCache{T}(;
    N::Int=1,
    n::Int=1,
    λ::T=one(T),
    Psq::AbstractMatrix=Matrix{T}(I, N, N),
    method::Symbol=:qr,
    use_gpu::Bool=false
) where T<:AbstractFloat
    λ = T(λ)
    # allocate on GPU or CPU
    if use_gpu
        O       = CUDA.zeros(T, N, n)
        Psq_dat = CUDA.CuArray(Psq)
        Psq     = LowerTriangular(Psq_dat)
        u       = CUDA.zeros(T, N)
        K       = CUDA.zeros(T, N, 1)
        ξpre    = CUDA.zeros(T, 1, n)
        ξpost   = CUDA.zeros(T, 1, n)
        A       = CUDA.zeros(T, N+1, N+1)
        temp_dO = CUDA.zeros(T, 1, n)
        temp_Ke = CUDA.zeros(T, N, n)
        tau     = CUDA.CuArray{T,1}(undef, N+1)
    else
        O       = zeros(T, N, n)
        Psq     = LowerTriangular(Matrix{T}(Psq))
        u       = zeros(T, N)
        K       = zeros(T, N, 1)
        ξpre    = zeros(T, 1, n)
        ξpost   = zeros(T, 1, n)
        A       = zeros(T, N+1, N+1)
        temp_dO = zeros(T, 1, n)
        temp_Ke = zeros(T, N, n)
        tau     = nothing
    end
    C = zero(T)
    J = zero(T)
    method = method in (:qr, :givens) ? method : :qr

    @assert (method == :qr && use_gpu == true) "Givens rotations not implemented for GPU."

    return iQRRLSCache{T}(N, n, λ,
        O, Psq, u, K, ξpre, ξpost, A, temp_dO, temp_Ke,
        C, J, method, use_gpu, tau
    )
end


function iqrrls!(obj::iQRRLSCache{T}, d::AbstractArray{T}, 
                 r::AbstractArray{T}) where T<:AbstractFloat
    return obj.use_gpu ? iqrrls_step_gpu!(obj, d, r) : iqrrls_step!(obj, d, r)
end


"""
iqrrls! - Perform one rank-1 iQRRLS update using QR factorization or Givens.

Note: This is slow when `N` is large, as it uses O(N^3) operations for QR.
"""
function iqrrls_step!(obj::iQRRLSCache{T}, d::AbstractArray{T}, 
                    r::AbstractArray{T}) where T<:Real
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

    # Compute the lower-triangular A matrix
    # A = [1                 zeros(1, N);
    #      Psq' * d' / λsq   Psq' / λsq]
    # Initialize A
    A = obj.A
    A .= 0
    A[1,1] = T(1)

    # Compute u = (Psq' * d') / λsq
    # d': N x 1
    # transposing Psq seems off but it's correct for `mul!` semantics
    mul!(obj.u, obj.Psq', vec(d), T(1)/λsq, T(0))  # obj.u: N x 1

    # Set A[2:end, 1] = u
    @views copyto!(A[2:end, 1], obj.u)

    # Compute Psq_scaled = Psq' / λsq and set A[2:end, 2:end] = Psq_scaled
    # transposing Psq seems off but it's correct for `mul!` semantics
    @views mul!(A[2:end, 2:end], obj.Psq', LinearAlgebra.I, T(1)/λsq, T(0))

    # Perform in-place QR factorization of A without storing Q
    if obj.mthd == :qr
        qr!(A) # this is slightly faster
        # LAPACK.geqrf!(A)  
    else # Givens rotations
        iqrrls_givens_fast!(A)
    end

    # Extract Csq_inv and gCsq_inv
    Csq_inv = A[1,1]
    @views gCsq_inv = A[1,2:end]

    # Update Psq: Psq = (A[2:end, 2:end])', ensuring it's lower triangular
    @views copyto!(obj.Psq, tril(A[2:end, 2:end]'))

    # Compute K = (gCsq_inv / Csq_inv)'
    obj.K[:,1] .= gCsq_inv ./ Csq_inv

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


function iqrrls_step_gpu!(obj::iQRRLSCache{T}, d::CuArray{T,2}, 
                        r::CuArray{T,2}) where T<:Union{Float32,Float64}
    N, n = obj.N, obj.n
    invλ = one(T)/sqrt(obj.λ)
    A = obj.A

    # 1) Build A = [1      0
    #               Psq'/√λ Psq'/√λ ]
    fill!(A, zero(T))
    A[1:1,1:1] .= one(T)
    mul!(obj.u, obj.Psq', vec(d), invλ, zero(T))             # u = Psq' * d'/√λ
    A[2:N+1, 1] .= obj.u                                     # first column
    mul!(view(A,2:N+1,2:N+1), obj.Psq', LinearAlgebra.I, invλ, zero(T))

    # 2) GPU QR factorization (in-place)
    CUSOLVER.geqrf!(A, obj.tau) # A ↦ R in upper, Q info in lower+tau

    # 3) Extract scalars and update Psq
    Csq_inv = A[1:1,1:1]
    gCsq_inv = view(A, 1, 2:N+1)
    copyto!(obj.Psq.data, tril(transpose(view(A,2:N+1,2:N+1))))

    # 4) Kalman gain
    obj.K[:,1] .= gCsq_inv ./ Csq_inv

    # 5) compute ξpre = r - d*O
    mul!(obj.temp_dO, d, obj.O, one(T), zero(T))
    obj.ξpre .= r .- obj.temp_dO

    # 6) update O
    obj.O .+= obj.K .* obj.ξpre

    # 7) compute ξpost = r - d*O
    mul!(obj.temp_dO, d, obj.O, one(T), zero(T))
    obj.ξpost .= r .- obj.temp_dO

    # 8) conversion factor & cost
    obj.C = one(T)/Array((Csq_inv[1:1,1:1]))[1]^2
    obj.J = obj.λ * obj.J + dot(vec(obj.ξpre), vec(obj.ξpost))

    return nothing
end


function iqrrls_givens_fast!(A::AbstractMatrix{T}) where {T<:AbstractFloat}
    """
    In-place Givens rotation applied to the first column.
    Modifies A directly and returns it.
    """
    np1 = size(A, 1)
    n = size(A, 2)
    
    # Apply Givens rotations from bottom to top
    for j in np1:-1:2
        c, s, r = givens_rotation(A[1, 1], A[j, 1])
        
        # Apply rotation directly to rows 1 and j
        @inbounds @simd for k in 1:n
            a1k = A[1, k]
            ajk = A[j, k]
            A[1, k] = c * a1k - s * ajk
            A[j, k] = s * a1k + c * ajk
        end
    end
    
    # Negate and apply upper triangular structure
    @inbounds @simd for i in 1:np1
        for j in 1:n
            if i <= j
                A[i, j] = -A[i, j]
            else
                A[i, j] = zero(T)
            end
        end
    end
    A[np1, np1] *= -1.0 # Fix the last diagonal entry for odd n
    return A
end