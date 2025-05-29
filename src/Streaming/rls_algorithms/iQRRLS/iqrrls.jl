"""
$(TYPEDEF)

Inverse QR Decomposition Recursive Least-Squares (iQRRLS) cache struct to solve for DO = R.
"""
@with_kw mutable struct iQRRLSCache{T<:Real}
    N::Int                               # Number of features (total dimension of operators)
    n::Int                               # Number of outputs (residual dimension)
    O::Array{T,2} = zeros(T,N,n)         # Operator matrix (N x n)
    Psq::AbstractArray{T,2}              # Square-root inverse correlation matrix (lower triangular, N x N)
    K::Array{T,2} = zeros(T,N,1)         # Kalman gain matrix (N x 1)
    ξpre::Array{T,2} = zeros(T,1,n)      # A priori error vector (1 x n)
    ξpost::Array{T,2} = zeros(T,1,n)     # A posteriori error vector (1 x n)
    C::T = zero(T)                       # Conversion factor (scalar)
    J::T = zero(T)                       # Cost (scalar)
    λ::T                                 # Forgetting factor

    # Preallocated temporary variables
    A::Array{T,2} = zeros(T,N+1,N+1)     # Temporary matrix for QR factorization ((N+1) x (N+1))
    u::Array{T,2} = zeros(T,N,1)         # Temporary vector for computations (N x 1)
    temp_dO::Array{T,2} = zeros(T,1,n)   # Temporary vector for d * O (1 x n)
    temp_Ke::Array{T,2} = zeros(T,N,n)   # Temporary matrix for K * ξpre (N x n)

    # Update counter
    # counter::Int = 0
end


"""
Inverse QR Decomposition Recursive Least-Squares (iQRRLS) algorithm.

This function updates the operator inference state within the `iQRRLSCache` struct,
performing computations in-place and minimizing memory allocations.

# Arguments:
- `obj::iQRRLSCache`: iQRRLSCache object containing the state and preallocated variables.
- `d::AbstractArray`: Data vector (features) at the current time step.
- `r::AbstractArray`: Response vector (targets) at the current time step.

# Note
The function updates the following fields in `obj`:
- `O`, `Psq`, `K`, `ξpre`, `ξpost`, `C`, `J`.
"""
function iqrrls!(obj::iQRRLSCache{T}, d::AbstractArray{T}, r::AbstractArray{T}) where T<:Real
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

    # Compute the A matrix
    # A = [1                 zeros(1, N);
    #      Psq' * d' / λsq   Psq' / λsq]
    # Initialize A
    A = obj.A
    A .= 0
    A[1,1] = T(1)

    # Compute u = (Psq' * d') / λsq
    # d': N x 1
    mul!(obj.u, obj.Psq', d', T(1)/λsq, T(0))  # obj.u: N x 1

    # Set A[2:end, 1] = u
    @views copyto!(A[2:end, 1], obj.u)

    # Compute Psq_scaled = Psq' / λsq and set A[2:end, 2:end] = Psq_scaled
    @views mul!(A[2:end, 2:end], obj.Psq', LinearAlgebra.I, T(1)/λsq, T(0))

    # Perform in-place QR factorization of A
    F = qr!(A)  # QR factorization in-place; A is overwritten
    R = F.R  # Upper triangular matrix R

    # Extract Csq_inv and gCsq_inv
    Csq_inv = R[1,1]
    @views gCsq_inv = R[1,2:end]

    # Update Psq: Psq = (R[2:end, 2:end])', ensuring it's lower triangular
    @views copyto!(obj.Psq, R[2:end, 2:end]')
    obj.Psq .= LowerTriangular(obj.Psq)

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