"""
$(TYPEDEF)

Maintain the state for a Recursive Least-Squares (RLS) update
of an (N x n)-dimensional operator matrix `O`, forgetting factor λ,
and optional regularization Γ (scalar or N x N). 
"""
mutable struct RLSCache{T<:Real}
    N::Int                           # number of features
    M::Int                           # number of data points (or block size)
    n::Int                           # number of outputs
    λ::T                             # forgetting factor

    O::Union{Matrix{T},CUDA.CuArray{T,2}}            # N × n operator
    P::Union{Symmetric{T,<:AbstractMatrix{T}}}       # N × N inverse‐covariance (symmetric)
    K::Union{Matrix{T},CUDA.CuArray{T,2}}            # N × M Kalman gain (M=1 or block size)
    ξpre::Union{Matrix{T},CUDA.CuArray{T,2}}         # M × n a‐priori error
    ξpost::Union{Matrix{T},CUDA.CuArray{T,2}}        # M × n a‐posteriori error
    C::Union{Matrix{T},CUDA.CuArray{T,2}}            # M × M intermediate (covariance)
    J::Union{Matrix{T},CUDA.CuArray{T,2}}            # M × M cost

    # temporaries (pre‐allocated)
    u::Union{Vector{T},CUDA.CuArray{T,1}}            # length N
    temp_DP::Union{Matrix{T},CUDA.CuArray{T,2}}      # M × N
    temp_PD::Union{Matrix{T},CUDA.CuArray{T,2}}      # N × M
    temp_update::Union{Matrix{T},CUDA.CuArray{T,2}}  # N × N
    temp_Ke::Union{Matrix{T},CUDA.CuArray{T,2}}      # N × n

    use_gpu::Bool
end


"""
RLSCache constructor

Set use_gpu=true to allocate on the GPU.
"""
function RLSCache{T}(; N::Int=1, M::Int=1, n::Int=1, λ::T=one(T),
                      P::AbstractMatrix=Matrix{T}(I, N, N),
                      use_gpu::Bool=false) where T<:Real
    λ   = T(λ)

    if use_gpu
        P_dat     = CUDA.CuArray{T}(P)
        P_wrap    = Symmetric(P_dat, :U)

        O         = CUDA.zeros(T, N, n)
        K         = CUDA.zeros(T, N, M)
        ξpre      = CUDA.zeros(T, M, n)
        ξpost     = CUDA.zeros(T, M, n)
        C         = CUDA.zeros(T, M, M)
        J         = CUDA.zeros(T, M, M)

        u         = CUDA.zeros(T, N)
        temp_DP   = CUDA.zeros(T, M, N)
        temp_PD   = CUDA.zeros(T, N, M)
        temp_up   = CUDA.zeros(T, N, N)
        temp_Ke   = CUDA.zeros(T, N, n)

        return RLSCache{T}(
            N, M, n, λ, O, P_wrap, K,
            ξpre, ξpost, C, J, u,
            temp_DP, temp_PD, temp_up, temp_Ke,
            true
        )
    else
        P_T       = convert(AbstractMatrix{T}, P)
        O         = zeros(T, N, n)
        P_wrap    = Symmetric(P_T, :U)
        K         = zeros(T, N, M)
        ξpre      = zeros(T, M, n)
        ξpost     = zeros(T, M, n)
        C         = zeros(T, M, M)
        J         = zeros(T, M, M)

        u         = zeros(T, N)
        temp_DP   = zeros(T, M, N)
        temp_PD   = zeros(T, N, M)
        temp_up   = zeros(T, N, N)
        temp_Ke   = zeros(T, N, n)

        return RLSCache{T}(
            N, M, n, λ, O, P_wrap, K,
            ξpre, ξpost, C, J, u,
            temp_DP, temp_PD, temp_up, temp_Ke,
            false
        )
    end
end


# Public entry point: route to CPU/GPU
function rls!(obj::RLSCache{T}, D, R, Q) where T<:Real
    return obj.use_gpu ? rls_gpu!(obj, D, R, Q) : rls_cpu!(obj, D, R, Q)
end


"""
    rls_cpu!(obj::RLSCache{T}, D::AbstractArray{T}, R::AbstractArray{T}, 
             Q::Union{T,AbstractMatrix{T}}) where T<:Real

CPU implementation (original logic).
"""
function rls_cpu!(obj::RLSCache{T}, D::AbstractArray{T}, R::AbstractArray{T},
                  Q::Union{T,AbstractMatrix{T}}) where T<:Real
    M = size(D,1) # M data points

    # 1) a‐priori error: ξpre = R – D*O
    obj.ξpre .= R
    mul!(obj.ξpre, D, obj.O, -one(T), one(T))

    # 2) update Kalman gain and inverse‐covariance P 
    if M == 1
        # == rank‐1 update ==
        d = view(D,1,:)
        mul!(obj.u, obj.P, d, one(T), zero(T))  # u = P * d'
        denom = (isa(Q, Number) ? T(Q) : Q[1,1]) * obj.λ + dot(d,obj.u)
        obj.C[1,1] = inv(denom)

        # Kalman gain K[:,1] = (P*d') * (1/Q) * C
        if isa(Q, Number)
            mul!(view(obj.K, :, 1), obj.P, d, obj.C[1,1]/T(Q), zero(T))
        else
            mul!(view(obj.K, :, 1), obj.P, d, obj.C[1,1], zero(T))
            obj.K .*= inv(Q)
        end

        # P ← (P – u*uᵀ/denom) / λ via BLAS
        BLAS.syr!('U', -one(T)/denom, obj.u, obj.P.data)
        BLAS.scal!(one(T)/obj.λ, obj.P.data)

    else
        # == block update ==
        # temp_DP = D * P / λ
        mul!(obj.temp_DP, D, obj.P, one(T)/obj.λ, zero(T))  # M×N

        # C = temp_DP * Dᵀ  + Q
        mul!(obj.C, obj.temp_DP, D', one(T), zero(T))   # M×M
        obj.C .+= Q

        # factor C once
        F = cholesky(Symmetric(obj.C, :U))

        # temp_PD = P * Dᵀ
        mul!(obj.temp_PD, obj.P, D', one(T), zero(T))   # N×M

        # K = (temp_PD/λ) * inv(C) via two triangular solves
        BLAS.scal!(one(T)/obj.λ, obj.temp_PD)
        BLAS.trsm!('L','U','N','N', one(T), F.L, obj.temp_PD) # solve L * X = temp_PD
        BLAS.trsm!('U','U','T','N', one(T), F.U, obj.temp_PD) # solve Uᵀ * X = prev
        copy!(obj.K, obj.temp_PD)                             # N×M

        # P ← (P – K * Kᵀ) / λ   (obj.temp_PD holds K)
        BLAS.syrk!('U', 'N', -one(T), obj.temp_PD, one(T), obj.P.data)
        BLAS.scal!(one(T)/obj.λ, obj.P.data)
    end

    # 3) update operator: O += K * ξpre
    mul!(obj.temp_Ke, obj.K, obj.ξpre, one(T), zero(T))
    obj.O .+= obj.temp_Ke

    # 4) a‐posteriori error: ξpost = R – D*O
    obj.ξpost .= R
    mul!(obj.ξpost, D, obj.O, -one(T), one(T))

    # 5) update cost: J = λ*J + ξpre*ξpostᵀ
    obj.J .*= obj.λ
    obj.J .+= obj.ξpre * obj.ξpost'

    return nothing
end


"""
GPU implementation using CUDA/CuArrays.
D, R must be CuArray{T,2}. Q may be scalar T or CuArray{T,2} (MxM).
"""
function rls_gpu!(obj::RLSCache{T}, D::CUDA.CuArray{T,2}, R::CUDA.CuArray{T,2},
                  Q::Union{T,CUDA.CuArray{T,2}}) where T<:Union{Float32,Float64}
    M = size(D,1)

    # 1) a‐priori error: ξpre = R – D*O
    obj.ξpre .= R
    mul!(obj.ξpre, D, obj.O, -one(T), one(T))

    if M == 1
        # == rank‐1 update ==
        d = D[1,:]                                   # Extract as CuArray, not view
        mul!(obj.u, obj.P.data, d, one(T), zero(T))  # Use .data to get underlying CuArray

        # denom on host (scalar)
        q11 = isa(Q, Number) ? T(Q) : Array(Q)[1,1]
        denom = q11 * obj.λ + dot(d, obj.u)
        obj.C[1:1,1:1] = inv(denom)

        # K[:,1] - avoid view, use direct assignment
        K_col = obj.P.data * d * (obj.C[1:1,1:1] / (isa(Q, Number) ? q11 : one(T)))
        obj.K[:,1] .= K_col
        
        if !isa(Q, Number)
            obj.K .*= inv(Q)   # Q is 1×1; inv(Q) is safe
        end

        # P ← (P – u*uᵀ/denom) / λ  (all on device)
        obj.P.data .-= (one(T)/denom) .* (obj.u * obj.u')
        obj.P.data .*= one(T)/obj.λ

    else
        # == block update ==
        # temp_DP = D * P / λ
        mul!(obj.temp_DP, D, obj.P.data, one(T)/obj.λ, zero(T))  # Use .data

        # C = temp_DP * Dᵀ + Q
        mul!(obj.C, obj.temp_DP, D', one(T), zero(T))       # M×M
        obj.C .+= Q

        # cholesky factorization on GPU
        F = cholesky(Symmetric(obj.C, :U))

        # temp_PD = P * Dᵀ
        mul!(obj.temp_PD, obj.P.data, D', one(T), zero(T))       # Use .data
        obj.temp_PD .*= one(T)/obj.λ

        # Solve C \ temp_PD in-place to get K
        ldiv!(F, obj.temp_PD)                               # temp_PD = C \ temp_PD
        copy!(obj.K, obj.temp_PD)                           # N×M

        # P ← (P – K*Kᵀ) / λ
        obj.P.data .-= obj.temp_PD * obj.temp_PD'
        obj.P.data .*= one(T)/obj.λ
    end

    # 3) O += K * ξpre
    mul!(obj.temp_Ke, obj.K, obj.ξpre, one(T), zero(T))
    obj.O .+= obj.temp_Ke

    # 4) ξpost = R – D*O
    obj.ξpost .= R
    mul!(obj.ξpost, D, obj.O, -one(T), one(T))

    # 5) J = λ*J + ξpre*ξpostᵀ
    obj.J .*= obj.λ
    obj.J .+= obj.ξpre * obj.ξpost'

    return nothing
end


"""
Variable-regularization RLS.

WARNING: Experimental, not yet tested.
"""
function vrrls!(obj::RLSCache{T}, D::AbstractMatrix{T}, R::AbstractMatrix{T},
                Q::Union{Real,AbstractMatrix{T}}, γ_k::Real, γ_km1::Real) where T<:Number
    M, N = size(D)
    n = size(R, 2)
    
    # Compute a priori error: obj.ξpre = R - D * obj.O
    mul!(obj.ξpre, D, obj.O, -1.0, 1.0)
    obj.ξpre .+= R

    # Update inverse correlation matrix P_k with variable regularization
    if M == 1  # Rank-1 update
        # Compute u = P * D'
        u = similar(obj.P, N)
        mul!(u, obj.P, @view(D[1, :])', 1.0, 0.0)  # u = P * D'

        # Compute denominator: denom = Q + D * u
        denom = Q + dot(@view(D[1, :]), u)  # scalar

        # Compute conversion factor: obj.C[1] = 1 / denom
        obj.C[1] = 1 / denom

        # Compute T = P - (u * uᵗ) / denom
        T_matrix = copy(obj.P)
        BLAS.syr!('U', -1/denom, u, T_matrix)
        for i in 1:N, j in i+1:N
            T_matrix[j, i] = T_matrix[i, j]
        end

        # Update P with variable regularization: P = (I - (γ_k - γ_km1) * T) * T
        scalar = γ_k - γ_km1
        temp_P = similar(obj.P)
        mul!(temp_P, T_matrix, T_matrix)
        obj.P .= T_matrix .- scalar .* temp_P

        # Compute Kalman gain: K = obj.P * Dᵗ / Q
        if isa(Q, Number)
            Q_inv = 1 / Q
            mul!(obj.K, obj.P, D', Q_inv, 0.0)
        else
            Q_inv = Q \ I
            temp_D = @view(D[1, :])'
            mul!(obj.K, obj.P, temp_D)
            mul!(obj.K, obj.K, Q_inv)
        end
    else  # Block (rank-M) update
        # Compute S = Q + D * obj.P * Dᵗ
        temp_DP = similar(D, M, N)
        mul!(temp_DP, D, obj.P)  # temp_DP = D * P
        S = similar(temp_DP, M, M)
        mul!(S, temp_DP, D', 1.0, 0.0)  # S = temp_DP * D'

        if isa(Q, Number)
            @. S += Q
        else
            S .+= Q
        end

        # Compute conversion factor: obj.C = inv(S)
        obj.C = S \ I  # obj.C is M x M matrix

        # Compute T = P - P * Dᵗ * obj.C * D * P
        temp_PD = similar(obj.P, N, M)
        mul!(temp_PD, obj.P, D')  # temp_PD = P * D'
        T_matrix = copy(obj.P)
        temp_update = similar(obj.P, N, N)
        mul!(temp_update, temp_PD, obj.C)
        mul!(temp_update, temp_update, temp_PD', 1.0, 0.0)
        T_matrix .-= temp_update

        # Update P with variable regularization: P = (I - (γ_k - γ_km1) * T) * T
        scalar = γ_k - γ_km1
        temp_P = similar(obj.P)
        mul!(temp_P, T_matrix, T_matrix)
        obj.P .= T_matrix .- scalar .* temp_P

        # Compute Kalman gain: K = obj.P * Dᵗ * obj.C
        mul!(obj.K, obj.P, D')
        mul!(obj.K, obj.K, obj.C)
    end

    # Update obj.O: obj.O += obj.K * obj.ξpre
    temp_Ke = similar(obj.O)
    mul!(temp_Ke, obj.K, obj.ξpre)
    obj.O .+= temp_Ke

    # Compute a posteriori error: obj.ξ = R - D * obj.O
    mul!(obj.ξpost, D, obj.O, -1.0, 1.0)
    obj.ξpost .+= R

    # Update the cost obj.J: obj.J += ξpre' * ξpost
    obj.J = obj.λ * obj.J + sum(obj.ξpre .* obj.ξpost)  # For matrix e and ξ

    # Update regularization term
    obj.Γ = γ_k

    return nothing
end