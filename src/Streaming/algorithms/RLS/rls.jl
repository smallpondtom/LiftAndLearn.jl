"""
$(TYPEDEF)

Recursive Least-Squares (RLS) cache struct to solve for DO = R.
"""
# mutable struct RLSCache{T<:Number}
#     O::AbstractArray{T}
#     P::AbstractArray{T}
#     K::AbstractArray{T}
#     ξpre::AbstractArray{T}
#     ξpost::AbstractArray{T}
#     C::AbstractArray{T}
#     J::Real
#     γ::Real
#     λ::Real
# end
mutable struct RLSCache{T<:Real}
    O::Array{T,2}         # Operator matrix (N x n)
    P::Array{T,2}         # Inverse covariance matrix (N x N)
    K::Array{T,2}         # Kalman gain matrix (N x M)
    ξpre::Array{T,2}      # A priori error matrix (M x n)
    ξpost::Array{T,2}     # A posteriori error matrix (M x n)
    C::Array{T,2}         # Conversion factor (M x M)
    J::T                  # Cost (scalar)
    γ::T                  # Regularization term
    λ::T                  # Forgetting factor

    # Preallocated temporary variables
    u::Array{T,1}         # For rank-1 update (N x 1)
    temp_DP::Array{T,2}   # Temporary matrix D * P (M x N)
    temp_PD::Array{T,2}   # Temporary matrix P * D' (N x M)
    temp_update::Array{T,2} # Temporary matrix (N x N)
    temp_Ke::Array{T,2}   # For updating O (N x n)
end


"""
Recursive Least Squares (RLS) algorithm for the Operator Inference problem.

This function updates the operator inference state within the `RLSCache` struct,
performing computations in-place and minimizing memory allocations.

# Arguments:
- `obj`: RLSCache object containing the state and preallocated variables.
- `D`: Data matrix (M x N), where each row is a data point.
- `R`: Response matrix (M x n), where each row corresponds to the output for a data point.
- `Q`: Noise covariance (scalar or matrix).

# Notes:
The function updates the following fields in `obj`:
- `O`, `P`, `K`, `ξpre`, `ξpost`, `C`, `J`.
"""
function rls!(obj::RLSCache{T}, D::AbstractMatrix{T}, R::AbstractMatrix{T}, 
              Q::Union{Real, AbstractMatrix{T}}) where T<:Real
    M, N = size(D)   # M: number of data points, N: number of features
    n = size(R, 2)   # n: residual dimension (state dimemsion)

    # Compute a priori error: ξpre = R - D * O
    mul!(obj.ξpre, D, obj.O, -1.0, 1.0)  # ξpre = R - D * O
    # No need to add R since mul! already computes ξpre = -D*O + 1*ξpre
    # So we add R to ξpre
    obj.ξpre .+= R

    # Update inverse covariance matrix P_k
    if M == 1  # Rank-1 update
        # Compute u = P * D'
        # D[1, :] is 1 x N, D[1, :]' is N x 1
        mul!(obj.u, obj.P, D[1, :]', 1.0, 0.0)  # obj.u: N x 1

        # Compute denominator: denom = Q + D * u / λ
        denom = Q + (D[1, :] * obj.u)[1] / obj.λ  # scalar

        # Compute conversion factor: C = 1 / denom
        obj.C[1,1] = 1 / denom  # obj.C is 1 x 1 in rank-1 case

        # Update P: P = (P - (u * uᵗ) / denom / λ) / λ
        BLAS.syr!('U', -1.0 / denom / obj.λ, obj.u, obj.P)
        obj.P ./= obj.λ

        # Ensure symmetry of P
        for i in 1:N, j in i+1:N
            obj.P[j, i] = obj.P[i, j]
        end

        # Compute Kalman gain: K = P * Dᵗ / Q / λ
        if isa(Q, Number)
            Q_inv = 1 / Q
            mul!(obj.K, obj.P, D[1, :]', Q_inv / obj.λ, 0.0)  # K: N x 1
        else
            Q_inv = Q \ I
            mul!(obj.K, obj.P, D[1, :]', 1.0 / obj.λ, 0.0)
            mul!(obj.K, Q_inv, obj.K)
        end
    else  # Block (rank-M) update
        # Compute temp_DP = D * P / λ
        mul!(obj.temp_DP, D, obj.P, 1 / obj.λ, 0.0)  # temp_DP: M x N

        # Compute S = Q + (D * P * Dᵗ) / λ
        mul!(obj.C, obj.temp_DP, D', 1.0, 0.0)  # C: M x M

        if isa(Q, Number)
            @. obj.C += Q
        else
            obj.C .+= Q
        end

        # Compute inverse of S
        S_inv = obj.C \ I

        # Compute temp_PD = P * D'
        mul!(obj.temp_PD, obj.P, D', 1.0, 0.0)  # temp_PD: N x M

        # Update P: P = (P - temp_PD * S_inv * temp_PDᵗ) / λ
        mul!(obj.temp_update, temp_PD, S_inv)
        mul!(obj.temp_update, obj.temp_update, temp_PD', 1.0 / obj.λ, 0.0)
        obj.P .-= obj.temp_update
        obj.P ./= obj.λ

        # Ensure symmetry of P
        for i in 1:N, j in i+1:N
            obj.P[j, i] = obj.P[i, j]
        end

        # Compute Kalman gain: K = P * Dᵗ * S_inv / λ
        mul!(obj.K, obj.P, D', 1.0 / obj.λ, 0.0)
        mul!(obj.K, obj.K, S_inv)
    end

    # Update O: O += K * ξpre
    mul!(obj.temp_Ke, obj.K, obj.ξpre, 1.0, 0.0)  # temp_Ke: N x n
    obj.O .+= obj.temp_Ke

    # Compute a posteriori error: ξpost = R - D * O
    mul!(obj.ξpost, D, obj.O, -1.0, 1.0)  # ξpost = R - D * O
    obj.ξpost .+= R

    # Update the cost J: J = λ * J + sum(ξpre .* ξpost)
    obj.J = obj.λ * obj.J + sum(obj.ξpre .* obj.ξpost)

    return nothing
end



# """
# $(SIGNATURES)

# Regularized Least-Squares (RLS) algorithm for the Operator Inference problem:

# ```math
# \\Vert \\mathbf{R} - \\mathbf{D}\\mathbf{O} \\Vert_F^2
# ```

# where `\\mathbf{R}` is the output matrix, `\\mathbf{D}` is the data matrix, and
# `\\mathbf{O}` is the operator matrix.
# """
# function rls!(obj::RLSCache{T}, D::AbstractMatrix{T}, R::AbstractMatrix{T}, Q::Union{Real,AbstractMatrix{T}}) where T<:Number
#     M, N = size(D)
#     n = size(R, 2)

#     # Compute a priori error: obj.ξpre = R - D * obj.O (before updating obj.O)
#     mul!(obj.ξpre, D, obj.O, -1.0, 1.0)  # obj.ξpre = R - D * obj.O
#     obj.ξpre .+= R

#     # Update inverse covariance matrix P_k
#     if M == 1  # Rank-1 update
#         # Compute u = P * D'
#         u = similar(obj.P, N)
#         mul!(u, obj.P, @view(D[1, :])', 1.0, 0.0)  # u = P * D'

#         # Compute denominator: denom = Q + D * u
#         denom = Q + dot(@view(D[1, :]), u) / obj.λ # scalar

#         # Compute conversion factor: C = 1 / denom
#         obj.C[1] = 1 / denom  # obj.C is scalar in rank-1 case

#         # Update P: P = P - (P * Dᵗ * D * P) / denom
#         # This is equivalent to: P = P - (u * uᵗ) / denom
#         BLAS.syr!('U', -1/denom/obj.λ, u, obj.P)
#         obj.P ./= obj.λ

#         # Ensure symmetry of obj.P
#         for i in 1:N, j in i+1:N
#             obj.P[j, i] = obj.P[i, j]
#         end

#         # Compute Kalman gain: K = obj.P * Dᵗ / Q
#         if isa(Q, Number)
#             Q_inv = 1 / Q
#             mul!(obj.K, obj.P, D', Q_inv, 0.0)
#         else
#             Q_inv = Q \ I
#             temp_D = @view(D[1, :])'
#             mul!(obj.K, obj.P, temp_D)
#             mul!(obj.K, obj.K, Q_inv)
#         end
#     else  # Block (rank-M) update
#         # Compute S = Q + D * P * Dᵗ
#         temp_DP = similar(D, M, N)
#         mul!(temp_DP, D, obj.P)  # temp_DP = D * P
#         S = similar(temp_DP, M, M)
#         mul!(S, temp_DP, D', 1.0, 0.0)  # S = temp_DP * D'
#         S ./= obj.λ

#         if isa(Q, Number)
#             @. S += Q
#         else
#             S .+= Q
#         end

#         # Compute conversion factor: obj.C = inv(S)
#         obj.C = S \ I  # obj.C is M x M matrix

#         # Update P: P = P - P * Dᵗ * inv(S) * D * P
#         temp_PD = similar(obj.P, N, M)
#         mul!(temp_PD, obj.P, D')  # temp_PD = P * D'
#         temp_update = similar(obj.P, N, N)
#         mul!(temp_update, temp_PD, obj.C)
#         mul!(temp_update, temp_update, temp_PD', 1.0, 0.0)
#         obj.P .-= temp_update / obj.λ
#         obj.P ./= obj.λ

#         # Ensure symmetry of obj.P
#         for i in 1:N, j in i+1:N
#             obj.P[j, i] = obj.P[i, j]
#         end

#         # Compute Kalman gain: K = obj.P * Dᵗ * obj.C
#         mul!(obj.K, obj.P, D')
#         mul!(obj.K, obj.K, obj.C)
#     end

#     # Update obj.O: obj.O += obj.K * obj.ξpre
#     temp_Ke = similar(obj.O)
#     mul!(temp_Ke, obj.K, obj.ξpre)
#     obj.O .+= temp_Ke

#     # Compute a posteriori error: obj.ξpost = R - D * obj.O (after updating obj.O)
#     mul!(obj.ξpost, D, obj.O, -1.0, 1.0)  # obj.ξpost = R - D * obj.O
#     obj.ξpost .+= R

#     # Update the cost obj.J: obj.J += ξpre' * ξpost
#     obj.J = obj.λ * obj.J + sum(obj.ξpre .* obj.ξpost)  # For matrix e and ξ

#     return nothing  # No need to return values
# end


"""
Variable-regularization RLS.
"""
function vrrls!(obj::RLSCache{T}, D::AbstractMatrix{T}, R::AbstractMatrix{T},
                Q::Union{Real,AbstractMatrix{T}}, γ_k::Real, γ_km1::Real) where T<:Number
    M, N = size(D)
    n = size(R, 2)

    # Compute a priori error: obj.ξpre = R - D * obj.O
    mul!(obj.ξpre, D, obj.O, -1.0, 1.0)
    obj.ξpre .+= R

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
    obj.γs = γ_k

    return nothing
end