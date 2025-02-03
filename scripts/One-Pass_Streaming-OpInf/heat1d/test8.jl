using LinearAlgebra

"""
    kaczmarz_vanilla(D, b; maxiter=10_000, tol=1e-6)

Solve the least-squares problem `D*x ≈ b` using the cyclic (vanilla) Kaczmarz method.

# Arguments
- `D::AbstractMatrix{T}`: A (K x d) matrix.
- `b::AbstractVector{T}`: A length-K vector.
- `maxiter::Int`: Maximum number of row-updates (default = 10_000).
- `tol::Real`: Residual tolerance for early stopping (default = 1e-6).

# Returns
- `x::Vector{Float64}`: Approximate solution of `D*x = b`.
"""
function kaczmarz_vanilla(D, b; maxiter=10_000, tol=1e-6)
    @assert size(D, 1) == length(b) "Matrix D and vector b must have compatible dimensions."
    
    K, d = size(D)
    x = zeros(eltype(D), d)  # initial guess

    updates = 0
    while updates < maxiter
        for i in 1:K
            updates += 1
            if updates > maxiter
                break
            end
            Di = @view D[i, :]
            r  = b[i] - Di ⋅ x
            denom = Di ⋅ Di
            x .+= (r / denom) .* Di
        end

        # Check residual norm
        if norm(D * x - b) < tol
            break
        end
    end

    return x
end

"""
    kaczmarz_matrix_vanilla_columns(D, R; maxiter=10_000, tol=1e-6)

Solve min ||R - D*O||_F by solving each column of O independently with the
vanilla Kaczmarz method.

# Arguments
- `D::AbstractMatrix{T}`: A (K x d) matrix.
- `R::AbstractMatrix{T}`: A (K x r) matrix.
- `maxiter::Int`: Maximum number of row-updates for each column's solve (default = 10_000).
- `tol::Real`: Residual tolerance for each column's solve (default = 1e-6).

# Returns
- `O::Matrix{Float64}`: A (d x r) matrix, the column-wise solution.
"""
function kaczmarz_matrix_vanilla_columns(D, R; maxiter=10_000, tol=1e-6)
    @assert size(D, 1) == size(R, 1) "D and R must have the same number of rows."
    K, d = size(D)
    K2, r = size(R)
    @assert K == K2

    # We'll solve for each column of O separately
    O = zeros(eltype(D), d, r)
    for j in 1:r
        # Solve D * O[:, j] ≈ R[:, j]
        O[:, j] = kaczmarz_vanilla(D, R[:, j]; maxiter=maxiter, tol=tol)
    end
    return O
end


using LinearAlgebra, Distributions

"""
    kaczmarz_randomized(D, b; maxiter=10_000, tol=1e-6)

Solve the least-squares problem D*x ≈ b using the randomized Kaczmarz method.
Samples row i with probability proportional to ||D[i,:]||^2.

# Arguments
- `D::AbstractMatrix{T}`: A (K x d) matrix.
- `b::AbstractVector{T}`: A length-K vector.
- `maxiter::Int`: Maximum number of iterations (default = 10_000).
- `tol::Real`: Residual tolerance (default = 1e-6).

# Returns
- `x::Vector{Float64}`
"""
function kaczmarz_randomized(D, b; maxiter=10_000, tol=1e-6)
    @assert size(D, 1) == length(b)

    K, d = size(D)
    x = zeros(eltype(D), d)

    # Precompute row norms for sampling
    row_norms_sq = [(@views dot(D[i, :], D[i, :])) for i in 1:K]
    p = row_norms_sq ./ sum(row_norms_sq)
    row_dist = Categorical(p)

    for iter in 1:maxiter
        i = rand(row_dist)
        Di = @view D[i, :]
        r  = b[i] - Di ⋅ x
        denom = row_norms_sq[i]
        x .+= (r / denom) .* Di

        # Optional: check residual every iteration
        if norm(D*x - b) < tol
            break
        end
    end
    return x
end


"""
    kaczmarz_matrix_randomized_columns(D, R; maxiter=10_000, tol=1e-6)

Solve min ||R - D*O||_F by solving each column of O separately with 
the randomized Kaczmarz method.

# Arguments
- `D::AbstractMatrix{T}`: (K x d)
- `R::AbstractMatrix{T}`: (K x r)
- `maxiter::Int`: maximum row-iterations per column
- `tol::Real`: residual tolerance per column

# Returns
- `O::Matrix{Float64}`: (d x r), each column solved with randomized Kaczmarz
"""
function kaczmarz_matrix_randomized_columns(D, R; maxiter=10_000, tol=1e-6)
    @assert size(D, 1) == size(R, 1)
    K, d = size(D)
    K2, r = size(R)
    @assert K == K2

    O = zeros(eltype(D), d, r)
    for j in 1:r
        # solve D*O[:, j] ≈ R[:, j] with randomized Kaczmarz
        O[:, j] = kaczmarz_randomized(D, R[:, j]; maxiter=maxiter, tol=tol)
    end
    return O
end


using Random
Random.seed!(123)

K, d, r = 200, 50, 10
D = randn(K, d)
O_true = randn(d, r)
R = D * O_true .+ 0.01*randn(K, r)

O_est = kaczmarz_matrix_vanilla_columns(D, R, maxiter=10_000, tol=1e-9)
O_est_rand = kaczmarz_matrix_randomized_columns(D, R, maxiter=10_000, tol=1e-9)

println("Frobenius error = ", norm(D*O_est - R))
println("Frobenius error (randomized) = ", norm(D*O_est_rand - R))

