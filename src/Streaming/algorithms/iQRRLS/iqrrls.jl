"""
$(TYPEDEF)

Inverse QR Decomposition Recursive Least-Squares (iQRRLS) cache struct to solve for DO = R.
"""
mutable struct iQRRLSCache{T<:Number}
    O::AbstractArray{T}
    Psq::AbstractArray{T}
    K::AbstractArray{T}
    e::AbstractArray{T}
    ξ::AbstractArray{T}
    C::AbstractArray{T}
    J::Real
    γ::Real
    λ::Real
end

mutable struct iQRRLSOpInf{T<:Number} <: StreamingOpInf
    # # State and input
    # O::AbstractArray{T}     # operator matrix
    # Psq::AbstractArray{T}   # square-root of inverse correlation matrix (state)
    # K::AbstractArray{T}     # Kalman gain matrix
    
    # # Output
    # Y::AbstractArray{T}     # output matrix
    # Psqy::AbstractArray{T}  # square-root of inverse correlation matrix (output)
    # Ky::AbstractArray{T}    # Kalman gain matrix (output)

    # # Regularization terms (state and output)
    # γs::Real
    # γy::Real

    # # Forgetting factor
    # λ::Real

    cache::iQRRLSCache{T}

    # Dimensions
    dims::Dict{Symbol,Int}

    # Options
    options::LSOpInfOption         # Standard (Least-Squares) Operator Inference options
end


"""
iQRRLS

P2_km1: is actually the square-root of the inverse of the correlation matrix
"""
function iQRRLS(d_k::AbstractArray{T}, r_k::AbstractArray{T}, O_km1::AbstractArray{T},
                P2_km1::AbstractArray{T}, d::Int) where T<:Real
    # Prearray
    A_k = [1 zeros(1,d); P2_km1'*d_k' P2_km1']  # note: it's actually the transpose

    # Compute postarray using QR factorization
    _, B_k = qr(A_k)  

    # Extract the square-root of the conversion factor and 
    # the Kalman gain matrix multiplied by square-root of the conversion factor
    α2_k_inv = B_k[1,1]
    gα2_k_inv = B_k[1,2:end]  # becomes a column vector after slicing
    P2_k = B_k[2:end, 2:end]'  # make sure it's lower triangular

    # Compute the next operator matrix and Kalman gain matrix
    K_k = gα2_k_inv * (α2_k_inv)^(-1)
    O_k = O_km1 + K_k * (r_k - d_k * O_km1)
    return O_k, P2_k, K_k
end