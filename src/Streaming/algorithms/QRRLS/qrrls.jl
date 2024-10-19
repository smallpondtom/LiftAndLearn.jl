"""
$(TYPEDEF)

QR Decomposition Recursive Least-Squares (QRRLS) cache struct to solve for DO = R.
"""
mutable struct QRRLSCache{T<:Number}
    O::AbstractArray{T}
    P::AbstractArray{T}
    K::AbstractArray{T}
    Φ::AbstractArray{T}
    q::AbstractArray{T}
    e::AbstractArray{T}
    ξ::AbstractArray{T}
    C::AbstractArray{T}
    J::Real
    γ::Real
    λ::Real
end

mutable struct QRRLSOpInf{T<:Number} <: StreamingOpInf
    # # State and input
    # O::AbstractArray{T}   # operator matrix
    # P::AbstractArray{T}   # inverse correlation matrix (state)
    # K::AbstractArray{T}   # Kalman gain matrix (state)
    # Φ::AbstractArray{T}   # correlation matrix
    # q::AbstractArray{T}   # auxiliary matrix

    # # Output
    # Y::AbstractArray{T}   # output matrix
    # Py::AbstractArray{T}  # inverse correlation matrix (output)
    # Ky::AbstractArray{T}  # Kalman gain matrix (output)
    # Φy::AbstractArray{T}  # correlation matrix
    # qy::AbstractArray{T}  # auxiliary matrix

    # # Regularization terms (state and output)
    # γs::Real
    # γy::Real

    # # Forgetting factor
    # λ::Real

    cache::QRRLSCache{T}

    # Dimensions
    dims::Dict{Symbol,Int}

    # Options
    options::LSOpInfOption         # Standard (Least-Squares) Operator Inference options
end

"""
QRRLS
"""
function qrrls(d_k::AbstractArray{T}, r_k::AbstractArray{T}, Φ_km1::AbstractArray{T}, 
               q_km1::AbstractArray{T}, d::Int, r::Int) where T<:Real
    # Prearray
    A_k = [Φ_km1' q_km1; d_k r_k]  # note: it's actually the transpose

    # Compute postarray using QR factorization
    qr!(A_k)  # in-place QR factorization (B_k = A_k)

    # Extract the inverse covariance matrix and auxiliary matrix
    Φ_km1 = A_k[1:d, 1:d]  # keep it upper triangular here
    q_km1 = A_k[1:d, d+1:d+r] 

    # Compute the next operator matrix with inverse of upper triangular matrix
    O_k = Φ_km1 \ q_km1   # (backslash inverse) automatically does backward substitution
    # O_k = copy(q_km1)
    # backsub!(Φ_km1', O_k)  # (backward subtitution) transpose to make upper triangular

    # Compute the inverse covariance matrix and Kalman gain matrix
    P_k = (Φ_km1'*Φ_km1) \ I   # Φ_km1 is still upper triangular
    K_k = P_k * d_k'
    return O_k, Φ_km1', q_km1, P_k, K_k
end


function backsub!(U::Matrix{T}, x::Vector{T}) where T<:Real
    n = length(x)
    # Backward substitution for U*x = y
    @inbounds for i = n:-1:1
        x[i] /= U[i, i]
        for j = 1:i-1
            x[j] -= A[j, i] * x[i]
        end
    end
end


function backsub!(U::Matrix{T}, X::Matrix{T}) where T<:Real
    n = size(X,1)
    
    # Ensure the dimensions match
    if size(U, 1) != n || size(U, 2) != n
        error("Dimensions of U and X do not match")
    end
    
    # vectorized backward substitution for U*X = Y
    @inbounds for i in n:-1:1
        X[i, :] ./= U[i, i]
        X[1:i-1, :] .-= U[1:i-1, i] .* X[i, :]'
    end
end

