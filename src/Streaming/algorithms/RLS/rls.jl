"""
$(TYPEDEF)

Recursive Least-Squares (RLS) cache struct to solve for DO = R.
"""
mutable struct RLSCache{T<:Number}
    O::AbstractArray{T}
    P::AbstractArray{T}
    K::AbstractArray{T}
    e::AbstractArray{T}
    ξ::AbstractArray{T}
    C::AbstractArray{T}
    J::Real
    γ::Real
    λ::Real
end

mutable struct RLSOpInf{T<:Number} <: StreamingOpInf
    # # State and input
    # O::AbstractArray{T}   # operator matrix
    # P::AbstractArray{T}   # inverse correlation matrix (state)
    # K::AbstractArray{T}   # Kalman gain matrix (state)

    # # Output
    # Y::AbstractArray{T}   # output matrix
    # Py::AbstractArray{T}  # inverse correlation matrix (output)
    # Ky::AbstractArray{T}  # Kalman gain matrix (output)

    # # Error, conversion factor, and cost
    # e::AbstractArray{T}   # a priori error vector
    # ξ::AbstractArray{T}   # a posteriori error vector
    # C::AbstractArray{T}   # conversion factor
    # J::T                  # cost (scalar)

    # # Regularization terms (state and output)
    # γs::Real
    # γy::Real

    # # Forgetting factor
    # λ::Real

    cache::RLSCache{T}

    # Dimensions
    dims::Dict{Symbol,Int}

    # Termination settings
    termination_settings::Dict{Symbol,Any}

    # Options
    options::LSOpInfOption         # Standard (Least-Squares) Operator Inference options
    variable_regularization::Bool  # variable regularization flag
    initial_step::Bool             # Flag for initial step when γs is zero
end

"""
$(SIGNATURES)

Regularized Least-Squares (RLS) algorithm for the Operator Inference problem:

```math
\\Vert \\mathbf{R} - \\mathbf{D}\\mathbf{O} \\Vert_F^2
```

where `\\mathbf{R}` is the output matrix, `\\mathbf{D}` is the data matrix, and
`\\mathbf{O}` is the operator matrix.
"""
function rls!(obj::RLSCache{T}, D::AbstractMatrix{T}, R::AbstractMatrix{T}, Q::Union{Real,AbstractMatrix{T}}) where T<:Number
    M, N = size(D)
    n = size(R, 2)

    # Compute a priori error: obj.e = R - D * obj.O (before updating obj.O)
    mul!(obj.e, D, obj.O, -1.0, 1.0)  # obj.e = R - D * obj.O
    obj.e .+= R

    # Update inverse covariance matrix P_k
    if M == 1  # Rank-1 update
        # Compute u = P * D'
        u = similar(obj.P, N)
        mul!(u, obj.P, @view(D[1, :])', 1.0, 0.0)  # u = P * D'

        # Compute denominator: denom = Q + D * u
        denom = Q + dot(@view(D[1, :]), u) / obj.λ # scalar

        # Compute conversion factor: C = 1 / denom
        obj.C[1] = 1 / denom  # obj.C is scalar in rank-1 case

        # Update P: P = P - (P * Dᵗ * D * P) / denom
        # This is equivalent to: P = P - (u * uᵗ) / denom
        BLAS.syr!('U', -1/denom/obj.λ, u, obj.P)
        obj.P ./= obj.λ

        # Ensure symmetry of obj.P
        for i in 1:N, j in i+1:N
            obj.P[j, i] = obj.P[i, j]
        end

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
        # Compute S = Q + D * P * Dᵗ
        temp_DP = similar(D, M, N)
        mul!(temp_DP, D, obj.P)  # temp_DP = D * P
        S = similar(temp_DP, M, M)
        mul!(S, temp_DP, D', 1.0, 0.0)  # S = temp_DP * D'
        S ./= obj.λ

        if isa(Q, Number)
            @. S += Q
        else
            S .+= Q
        end

        # Compute conversion factor: obj.C = inv(S)
        obj.C = S \ I  # obj.C is M x M matrix

        # Update P: P = P - P * Dᵗ * inv(S) * D * P
        temp_PD = similar(obj.P, N, M)
        mul!(temp_PD, obj.P, D')  # temp_PD = P * D'
        temp_update = similar(obj.P, N, N)
        mul!(temp_update, temp_PD, obj.C)
        mul!(temp_update, temp_update, temp_PD', 1.0, 0.0)
        obj.P .-= temp_update / obj.λ
        obj.P ./= obj.λ

        # Ensure symmetry of obj.P
        for i in 1:N, j in i+1:N
            obj.P[j, i] = obj.P[i, j]
        end

        # Compute Kalman gain: K = obj.P * Dᵗ * obj.C
        mul!(obj.K, obj.P, D')
        mul!(obj.K, obj.K, obj.C)
    end

    # Update obj.O: obj.O += obj.K * obj.e
    temp_Ke = similar(obj.O)
    mul!(temp_Ke, obj.K, obj.e)
    obj.O .+= temp_Ke

    # Compute a posteriori error: obj.ξ = R - D * obj.O (after updating obj.O)
    mul!(obj.ξ, D, obj.O, -1.0, 1.0)  # obj.ξ = R - D * obj.O
    obj.ξ .+= R

    # Update the cost obj.J: obj.J += e' * ξ
    obj.J += sum(obj.e .* obj.ξ)  # For matrix e and ξ

    return nothing  # No need to return values
end


"""
Variable-regularization RLS.
"""
function vrrls!(obj::RLSCache{T}, D::AbstractMatrix{T}, R::AbstractMatrix{T},
                Q::Union{Real,AbstractMatrix{T}}, γ_k::Real, γ_km1::Real) where T<:Number
    M, N = size(D)
    n = size(R, 2)

    # Compute a priori error: obj.e = R - D * obj.O
    mul!(obj.e, D, obj.O, -1.0, 1.0)
    obj.e .+= R

    # Update inverse covariance matrix P_k with variable regularization
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

    # Update obj.O: obj.O += obj.K * obj.e
    temp_Ke = similar(obj.O)
    mul!(temp_Ke, obj.K, obj.e)
    obj.O .+= temp_Ke

    # Compute a posteriori error: obj.ξ = R - D * obj.O
    mul!(obj.ξ, D, obj.O, -1.0, 1.0)
    obj.ξ .+= R

    # Update the cost obj.J: obj.J += e' * ξ
    obj.J += sum(obj.e .* obj.ξ)  # For matrix e and ξ

    # Update regularization term
    obj.γs = γ_k

    return nothing
end