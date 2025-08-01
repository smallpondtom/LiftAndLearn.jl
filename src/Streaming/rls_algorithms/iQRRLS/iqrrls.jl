"""
Cache for inverse-QR RLS via square-root updates, with vectorized loops.
Maintains lower-triangular Psq = Cholesky factor of P^{-1}.
"""
# mutable struct iQRRLSCache{T<:Real}
#     N::Int                             # number of features
#     n::Int                             # dimension of response
#     λ::T                               # forgetting factor
#     O::Matrix{T}                       # operator matrix (N×n)
#     Psq::LowerTriangular{T,Matrix{T}}  # lower-triangular sqrt-inverse correlation (N×N)
#     u::Array{T}                        # temporary (N)
#     K::Matrix{T}                       # Kalman gain (N×1)
#     ξpre::Matrix{T}                    # a priori error (1×n)
#     ξpost::Matrix{T}                   # a posteriori error (1×n)
#     A::Matrix{T}                       # temporary matrix for QR fact. ((N+1)×(N+1)) or NxN
#     temp_dO::Union{T,Matrix{T}}        # temporary for d * O (1×n)
#     temp_Ke::Matrix{T}                 # temporary for K * ξpre (N×n)
#     C::T                               # Conversion factor (scalar)
#     J::T                               # cost
#     mthd::Symbol                       # method used for updates, default is :qr
# end


# -----------------------------------------------------------------------------
# Extended iQRRLSCache with GPU support
# -----------------------------------------------------------------------------
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
# function iQRRLSCache{T}(;N::Int=1, n::Int=1, λ::T=one(T), 
#                          Psq::AbstractMatrix=Matrix{T}(I, N, N),
#                          method::Symbol=:qr) where T<:Real
#     λ       = T(λ)
#     Psq_T   = convert(AbstractMatrix{T}, Psq)

#     O       = zeros(T,N,n)
#     Psq     = LowerTriangular(Psq_T)
#     u       = zeros(T, N, 1)
#     K       = zeros(T, N, 1)
#     ξpre    = zeros(T, 1, n)
#     ξpost   = zeros(T, 1, n)
#     A       = method == :qr ? zeros(T, N+1, N+1) : zeros(T, N, N)
#     temp_dO = zeros(T, 1, n)
#     temp_Ke = zeros(T, N, n)
#     C       = zero(T)
#     J       = zero(T)
#     method  = method in (:givens, :qr) ? method : :qr
#     return iQRRLSCache{T}(N, n, λ, O, Psq, u, K, ξpre,  
#                           ξpost, A, temp_dO, temp_Ke, C, J, method)
# end


# -----------------------------------------------------------------------------
# GPU‐aware constructor
# -----------------------------------------------------------------------------
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
    return iQRRLSCache{T}(N, n, λ,
        O, Psq, u, K, ξpre, ξpost, A, temp_dO, temp_Ke,
        C, J, method, use_gpu, tau
    )
end


# -----------------------------------------------------------------------------
# Dispatch: CPU or GPU branch
# -----------------------------------------------------------------------------
function iqrrls!(obj::iQRRLSCache{T}, d::AbstractArray{T}, 
                 r::AbstractArray{T}) where T<:AbstractFloat
    if obj.mthd == :givens
        return iqrrls_givens!(obj, d, r)
    else
        return obj.use_gpu ? iqrrls_qr_gpu!(obj, d, r) : iqrrls_qr!(obj, d, r)
    end
end


# function iqrrls!(obj::iQRRLSCache{T}, d::AbstractArray{T}, 
#                  r::AbstractArray{T}) where T<:Real
#     if obj.mthd == :givens
#         return iqrrls_givens!(obj, d, r)
#     else
#         return iqrrls_qr!(obj, d, r)
#     end
# end


"""
Perform one rank-1 iQRRLS update via vectorized loops and BLAS calls (O(N^2)).
This algorithm takes advantage of the lower-triangular structure of Psq
and performs the update in-place, minimizing memory allocations and computation
time.
"""
function iqrrls_givens!(obj::iQRRLSCache{T}, d::AbstractArray{T}, 
                        r::AbstractArray{T}) where T<:Real

    # d: 1 x N (row vector)
    # r: 1 x n (row vector)
    @assert length(d)==obj.N && length(r)==obj.n
    N = obj.N
    
    # 1) Compute the scaling 1/sqrt(λ)
    invλ = one(T)/sqrt(obj.λ)

    # 2) mat-vec Psq * d' / √λ (triangular kernel)
    mul!(obj.u, obj.Psq', vec(d), invλ, zero(T))

    # 3) Create a working copy of Psq scaled by 1/√λ
    # We need to work with Psq'/√λ, not Psq itself
    mul!(obj.A, obj.Psq', LinearAlgebra.I, invλ, zero(T))

    # 4) annihilate sub-diagonal via Givens rotation, inner loop vectorized
    # We need to simulate the QR decomposition of B without forming B
    # B has the structure: [1, 0, 0, ..., 0]
    #                      [u[1], Psq0[1,1]/√λ, Psq0[1,2]/√λ, ..., Psq0[1,N]/√λ]
    #                      [u[2], Psq0[2,1]/√λ, Psq0[2,2]/√λ, ..., Psq0[2,N]/√λ]
    #                      [...]
    #                      [u[N], Psq0[N,1]/√λ, Psq0[N,2]/√λ, ..., Psq0[N,N]/√λ]
    #
    # Track the first row elements as we go
    first_row = zeros(T, N+1)
    first_row[1] = one(T)       

    # Process column 1: eliminate u[i] for i = 1:N
    for i in 1:N
        if abs(obj.u[i]) > eps(T)
            # Apply Givens rotation between row 1 and row i+1
            c, s, _ = givens_rotation(first_row[1], obj.u[i])
            
            # Update first row element
            old_first = first_row[1]
            first_row[1] = c * old_first - s * obj.u[i]
            
            # Use BLAS.rot! for the remaining elements of first_row and 
            # row i of Psq_work
            if N > 0
                @turbo BLAS.rot!(
                    N,
                    view(first_row, 2:N+1), 1,  # first_row[2:end] with stride 1
                    view(obj.A, i, 1:N), N,     # Psq_work[i, :] with stride N
                    c, -s
                )
            end
            obj.u[i] = zero(T)  # This element is now eliminated
        end
    end

    # Now perform QR on the remaining NxN block (obj.A) using BLAS.rot!
    for i in 1:N
        for j in (i+1):N
            if abs(obj.A[j, i]) > eps(T)
                c, s, _ = givens_rotation(obj.A[i, i], obj.A[j, i])
                
                # Use BLAS.rot! for columns i:N of rows i and j
                @turbo BLAS.rot!(
                    N - i + 1,
                    view(obj.A, i, i:N), N,  # row i, columns i:N
                    view(obj.A, j, i:N), N,  # row j, columns i:N
                    c, -s
                )
            end
        end
    end
    
    # Update obj.Psq with the lower triangular part of the transpose
    copyto!(obj.Psq.data, tril(obj.A'))

    # 5) conversion factor C = 1/("first row first element"^2)
    obj.C = one(T) / first_row[1]^2

    # 6) Kalman gain K = first_row[2:end]/first_row[1]
    obj.K[:,1] .= first_row[2:end] / first_row[1]

    # 7) Compute ξpre = r - d * O
    # d: 1 x N, O: N x n, d * O: 1 x n
    mul!(obj.temp_dO, d, obj.O, T(1), T(0))  # temp_dO: 1 x n
    obj.ξpre .= r .- obj.temp_dO  # ξpre: 1 x n

    # 8) Update O: O += K * ξpre
    # K: N x n, ξpre: 1 x n (broadcasted), K * ξpre': N x n
    obj.O .+= obj.K .* obj.ξpre  # Element-wise multiplication and accumulation

    # 9) Compute ξpost = r - d * O
    mul!(obj.temp_dO, d, obj.O, T(1), T(0))  # temp_dO: 1 x n
    obj.ξpost .= r .- obj.temp_dO  # ξpost: 1 x n

    # 10) Since ξpre and ξpost are 1 x n row vectors, compute dot product
    obj.J = obj.λ * obj.J + dot(vec(obj.ξpre), vec(obj.ξpost))
    
    return nothing
end


# function iqrrls_givens!(obj::iQRRLSCache{T}, d::AbstractArray{T}, 
#                         r::AbstractArray{T}) where T<:Real
#     # d: 1 x N (row vector)
#     # r: 1 x n (row vector)
#     N = size(d, 2)  # Number of features
#     n = size(r, 2)  # Residual dimension (state dimension)
#     λsq = sqrt(obj.λ)

#     # Ensure temporary variables are correctly sized
#     @assert size(obj.A) == (N+1, N+1)
#     @assert length(obj.u) == N
#     @assert size(obj.temp_dO) == (1, n)
#     @assert size(obj.temp_Ke) == (N, n)

#     # Compute the lower-triangular A matrix
#     # A = [1                 zeros(1, N);
#     #      Psq' * d' / λsq   Psq' / λsq]
#     # Initialize A
#     A = obj.A
#     A .= 0
#     A[1,1] = T(1)

#     # Compute u = (Psq' * d') / λsq
#     # d': N x 1
#     # transposing Psq seems off but it's correct for `mul!` semantics
#     mul!(obj.u, obj.Psq', vec(d), T(1)/λsq, T(0))  # obj.u: N x 1

#     # Set A[2:end, 1] = u
#     @views copyto!(A[2:end, 1], obj.u)

#     # Compute Psq_scaled = Psq' / λsq and set A[2:end, 2:end] = Psq_scaled
#     # transposing Psq seems off but it's correct for `mul!` semantics
#     @views mul!(A[2:end, 2:end], obj.Psq', LinearAlgebra.I, T(1)/λsq, T(0))

#     # Perform in-place QR factorization of A without storing Q
#     qr_givens!(A) 

#     # Extract Csq_inv and gCsq_inv
#     Csq_inv = A[1,1]
#     @views gCsq_inv = A[1,2:end]

#     # Update Psq: Psq = (A[2:end, 2:end])', ensuring it's lower triangular
#     @views copyto!(obj.Psq, tril(A[2:end, 2:end]'))

#     # Compute K = (gCsq_inv / Csq_inv)'
#     obj.K[:,1] .= gCsq_inv ./ Csq_inv

#     # Compute ξpre = r - d * O
#     # d: 1 x N, O: N x n, d * O: 1 x n
#     mul!(obj.temp_dO, d, obj.O, T(1), T(0))  # temp_dO: 1 x n
#     obj.ξpre .= r .- obj.temp_dO  # ξpre: 1 x n

#     # Update O: O += K * ξpre
#     # K: N x n, ξpre: 1 x n (broadcasted), K * ξpre': N x n
#     obj.O .+= obj.K .* obj.ξpre  # Element-wise multiplication and accumulation

#     # Compute ξpost = r - d * O
#     mul!(obj.temp_dO, d, obj.O, T(1), T(0))  # temp_dO: 1 x n
#     obj.ξpost .= r .- obj.temp_dO  # ξpost: 1 x n

#     # Update conversion factor C and cost J
#     obj.C = T(1) / (Csq_inv^2)
#     # Since ξpre and ξpost are 1 x n row vectors, compute dot product
#     obj.J = obj.λ * obj.J + dot(vec(obj.ξpre), vec(obj.ξpost))

#     return nothing

# end



"""
iqrrls_qr! - Perform one rank-1 iQRRLS update using QR factorization.

Note: This is slow when `N` is large, as it uses O(N^3) operations.
"""
function iqrrls_qr!(obj::iQRRLSCache{T}, d::AbstractArray{T}, 
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
    qr!(A) # this is slightly faster
    # LAPACK.geqrf!(A)  

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



# -----------------------------------------------------------------------------
# GPU‐accelerated QR‐based update
# -----------------------------------------------------------------------------
function iqrrls_qr_gpu!(obj::iQRRLSCache{T}, d::CuArray{T,2}, 
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

