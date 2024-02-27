export streaming_OpInf


"""
$(TYPEDEF)

Streaming Operator Inference
"""
mutable struct streaming_OpInf
    D_k::AbstractArray{Real,2}                  # data matrix
    R_k::AbstractArray{Real,2}                  # residual matrix (derivative)
    P_k::AbstractArray{Real,2}                  # projection matrix
    O_k::AbstractArray{Real,2}                  # operator matrix (objective)
    K_k::Union{AbstractArray{Real,2}, Nothing}  # gain matrix
    Q_k::Union{AbstractArray{Real,2}, Nothing}  # covariance matrix
    update!::Function
end

function streaming_OpInf(D_k::AbstractArray{Real,2}, R_k::AbstractArray{Real,2}, 
                            Q_k::Union{AbstractArray{Real,2}, Nothing}=nothing)
    K, n = size(D_k)
    @assert K > n "Should be a tall matrix for an overdetermined least squares problem."
    Q_k = isnothing(Q_k) ? sparse(Matrix(1.0I, K, K)) : Q_k
    Q_k_inv = isnothing(Q_k) ? sparse(Matrix(1.0I, K, K)) : Q_k \ I

    P_k = (D_k' * Q_k_inv * D_k) \ I
    O_k = P_k * D_k' * Q_k_inv * R_k
    K_k = nothing
    return streaming_OpInf(D_k, R_k, P_k, O_k, K_k, Q_k, update!)
end


"""
$(SIGNATURES)

Update the streaming operator inference with new data.
"""
function update!(D_kp1::AbstractArray, R_kp1::AbstractArray, Q_kp1::Union{AbstractArray, Nothing}=nothing)
    if !isnothing(Q_kp1)
        P_k = streaming_OpInf.P_k
        streaming_OpInf.P_k -= P_k * D_kp1' * (I + D_kp1 * P_k * D_kp1') \ D_kp1 * P_k
        streaming_OpInf.K_k = streaming_OpInf.P_k * D_kp1'
        streaming_OpInf.O_k += streaming_OpInf.K_k * (R_kp1 - D_kp1 * streaming_OpInf.O_k)
    else
        streaming_OpInf.Q_k = Q_kp1
        Q_k_inv = Q_kp1 \ I

        P_k = streaming_OpInf.P_k
        streaming_OpInf.P_k -= P_k * D_kp1' * (Q_kp1 + D_kp1 * P_k * D_kp1') \ D_kp1 * P_k
        streaming_OpInf.K_k = streaming_OpInf.P_k * D_kp1' * Q_k_inv
        streaming_OpInf.O_k += streaming_OpInf.K_k * (R_kp1 - D_kp1 * streaming_OpInf.O_k)
    end
end


"""
$(SIGNATURES)

Update the streaming operator inference with new data for multiple data matrices.
"""
function update!(D_kp1::AbstractArray{AbstractArray}, R_kp1::AbstractArray{AbstractArray},
                    Q_kp1::Union{AbstractArray{AbstractArray}, Nothing}=nothing)
    N = length(D_kp1)
    if !isnothing(Q_kp1)
        for i in 1:N
            update!(D_kp1[i], R_kp1[i], Q_kp1[i])
        end
    else
        for i in 1:N
            update!(D_kp1[i], R_kp1[i])
        end
    end
end
