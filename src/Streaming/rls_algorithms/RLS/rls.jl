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

    O::Matrix{T}                     # N × n operator
    P::Symmetric{T,Matrix{T}}        # N × N inverse‐covariance (symmetric)
    K::Matrix{T}                     # N × M Kalman gain (M=1 or block size)
    ξpre::Matrix{T}                  # M × n a‐priori error
    ξpost::Matrix{T}                 # M × n a‐posteriori error
    C::Matrix{T}                     # M × M intermediate (covariance)
    J::Matrix{T}                     # M × M cost

    # temporaries (pre‐allocated)
    u::Vector{T}                     # length N
    temp_DP::Matrix{T}               # M × N
    temp_PD::Matrix{T}               # N × M
    temp_update::Matrix{T}           # N × N
    temp_Ke::Matrix{T}               # N × n
end


"""
RLSCache constructor
"""
function RLSCache{T}(;N::Int=1, M::Int=1, n::Int=1, λ::T=one(T), 
                      P::AbstractMatrix=Matrix{T}(I, N, N)) where T<:Real
    λ        = T(λ)
    P_T      = convert(AbstractMatrix{T}, P)

    O        = zeros(T,N,n)
    P        = Symmetric(P_T, :U)
    K        = zeros(T,N,M)
    ξpre     = zeros(T,M,n)
    ξpost    = zeros(T,M,n)
    C        = zeros(T,M,M)
    J        = zeros(T,M,M)

    u        = zeros(T,N)
    temp_DP  = zeros(T,M,N)
    temp_PD  = zeros(T,N,M)
    temp_up  = zeros(T,N,N)
    temp_Ke  = zeros(T,N,n)

    return RLSCache{T}(
        N, M, n, λ, O, P, K, 
        ξpre, ξpost, C, J, u,
        temp_DP, temp_PD, temp_up, temp_Ke)
end


"""
    rls!(obj::RLSCache{T}, D::AbstractArray{T}, R::AbstractArray{T}, 
         Q::Union{T,AbstractMatrix{T}}) where T<:Real

Perform one or block RLS update:
- If `D` is M x N with M>1, does block update of size M.
- If `D` is 1 x N, does rank-1 update.

# Arguments:
- `obj`: RLSCache object containing the state and preallocated variables.
- `D`: Data matrix (M x N), where each row is a data point.
- `R`: Response matrix (M x n), where each row corresponds to the output for a data point.
- `Q`: Noise covariance (scalar or matrix).

# Note:
`R` is M x n; `Q` is scalar or M x M noise covariance.
"""
function rls!(obj::RLSCache{T}, D::AbstractArray{T}, R::AbstractArray{T},
              Q::Union{T,AbstractMatrix{T}}) where T<:Real
    M = size(D,1) # M data points

    # 1) a‐priori error: ξpre = R – D*O
    obj.ξpre .= R
    mul!(obj.ξpre, D, obj.O, -1.0, 1.0)

    # 2) update Kalman gain and inverse‐covariance P 
    if M == 1
        # == rank‐1 update ==
        d = view(D,1,:)
        mul!(obj.u, obj.P, d, 1.0, 0.0)  # u = P * d'
        denom = (isa(Q, Number) ? Q : Q[1,1]) * obj.λ + dot(d,obj.u)
        obj.C[1,1] = inv(denom)

        # Kalman gain K[:,1] = (P*d') * (1/Q) * C
        if isa(Q, Number)
            mul!(view(obj.K, :, 1), obj.P, d, obj.C[1,1]/Q, 0.0)
        else
            # Q is 1×1 matrix
            mul!(view(obj.K, :, 1), obj.P, d, obj.C[1,1], 0.0)
            obj.K .*= inv(Q)
        end

        # P ← (P – u*uᵀ/denom) / λ via BLAS
        # NOTE: 
        # 1) syr only updates the upper triangular part of the matrix
        # 2) P is set to `Symmetric` so symmetry is ensured
        BLAS.syr!('U', -1.0/denom, obj.u, obj.P.data)
        BLAS.scal!(1/obj.λ, obj.P.data)

    else
        # == block update ==
        # temp_DP = D * P / λ
        mul!(obj.temp_DP, D, obj.P, 1/obj.λ, 0.0)  # M×N

        # C = temp_DP * Dᵀ  + Q
        mul!(obj.C, obj.temp_DP, D', 1.0, 0.0)   # M×M
        obj.C .+= Q

        # factor C once
        F = cholesky(Symmetric(obj.C, :U))

        # temp_PD = P * Dᵀ
        mul!(obj.temp_PD, obj.P, D', 1.0, 0.0)   # N×M

        # K = (temp_PD/λ) * inv(C) via two triangular solves
        BLAS.scal!(1/obj.λ, obj.temp_PD)
        BLAS.trsm!('L','U','N','N', 1.0, F.L, obj.temp_PD) # solve L * X = temp_PD
        BLAS.trsm!('U','U','T','N', 1.0, F.U, obj.temp_PD) # solve Uᵀ * X = prev
        copy!(obj.K, obj.temp_PD)                          # N×M

        # P ← (P – K * (D*P)) / λ
        # note D*P = (temp_DP)
        BLAS.syrk!('U', 'N', -1.0, obj.temp_PD, 1.0, obj.P.data) # P -= K * Kᵀ
        BLAS.scal!(1/obj.λ, obj.P.data)

    end

    # 3) update operator: O += K * ξpre
    mul!(obj.temp_Ke, obj.K, obj.ξpre, 1.0, 0.0)
    obj.O .+= obj.temp_Ke

    # 4) a‐posteriori error: ξpost = R – D*O
    obj.ξpost .= R
    mul!(obj.ξpost, D, obj.O, -1.0, 1.0)

    # 5) update cost: J = λ*J + ξpre*ξpostᵀ
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