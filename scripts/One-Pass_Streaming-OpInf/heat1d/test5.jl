using LinearAlgebra
using Distributions

"""
    kaczmarz(A, b; maxiter=10_000, tol=1e-6)

Solve the least-squares problem `A*x ≈ b` using the Randomized Kaczmarz method.

# Arguments
- `A::AbstractMatrix{T}`: An m-by-n matrix (Float64 or similar).
- `b::AbstractVector{T}`: A vector of length m.
- `maxiter::Int`: Maximum number of iterations (default 10_000).
- `tol::Real`: Residual tolerance for early stopping (default 1e-6).

# Returns
- `x::Vector{Float64}`: Approximate solution to `A*x = b`.

# Notes
- Randomized Kaczmarz samples rows with probability proportional to the squared row-norm.
- The update equation for the row i is:
  x ← x + (b[i] - A[i,:] ⋅ x) / (‖A[i,:]‖²) * A[i,:]

"""
function kaczmarz(A, b; maxiter=10_000, tol=1e-6)
    @assert size(A, 1) == length(b) "Matrix A and vector b must have compatible dimensions."
    
    m, n = size(A)
    # Initial guess
    x = zeros(eltype(A), n)

    # Precompute row norms (squared) for weighted sampling
    row_norms_sq = Array{Float64}(undef, m)
    for i in 1:m
        row_norms_sq[i] = @views dot(A[i, :], A[i, :])
    end

    # Probability for each row i (proportional to the row-norm squared)
    row_probs = row_norms_sq ./ sum(row_norms_sq)

    # A discrete distribution to sample rows
    row_dist = Categorical(row_probs)

    # Optional: We won't check the residual every iteration (for speed);
    # we can check only every k steps or rely solely on maxiter.
    # For demonstration, we'll check each time—tweak for performance if needed.
    for iter in 1:maxiter
        # Randomly pick a row index according to row_probs
        i = rand(row_dist)
        Ai = @view A[i, :]

        # Compute the residual component for row i
        ri = b[i] - dot(Ai, x)
        
        # Update x in place
        α = ri / row_norms_sq[i]
        @inbounds @simd for j in 1:n
            x[j] += α * Ai[j]
        end

        # Check convergence by residual norm every iteration
        if norm(A*x - b) < tol
            break
        end
    end

    return x
end


"""
    kaczmarz_vanilla(A, b; maxiter=10_000, tol=1e-6)

Solve the least-squares problem `A*x ≈ b` using the (cyclic) Kaczmarz method.

# Arguments
- `A::AbstractMatrix{T}`: An m-by-n matrix (Float64 or similar).
- `b::AbstractVector{T}`: A vector of length m.
- `maxiter::Int`: Maximum number of row-updates to perform (default = 10_000).
- `tol::Real`: Residual tolerance for early stopping (default = 1e-6).

# Returns
- `x::Vector{Float64}`: Approximate solution to `A*x = b`.

# Notes
- The *vanilla* or *cyclic* Kaczmarz method processes rows in a fixed order: 1,2,...,m,1,2,...
- Each row-update is:
  x ← x + (b[i] - A[i,:] ⋅ x) / (‖A[i,:]‖²) * A[i,:]
- One "sweep" is typically one pass over all rows from 1 to m.
- We stop if the residual norm ‖A*x - b‖ is below `tol`, or if `maxiter` updates are done.
"""
function kaczmarz_vanilla(A, b; maxiter=10_000, tol=1e-6)
    @assert size(A, 1) == length(b) "Matrix A and vector b must have compatible dimensions."
    
    m, n = size(A)
    # Initial guess for x
    x = zeros(eltype(A), n)

    # Counters
    updates = 0

    # Iterate until we reach maxiter row updates or achieve desired tolerance
    while updates < maxiter
        # A single "sweep" over all rows
        for i in 1:m
            updates += 1
            if updates > maxiter
                break
            end
            Ai = @view A[i, :]   # row i
            # Residual for row i
            r = b[i] - dot(Ai, x)
            # Update x
            α = r / dot(Ai, Ai)
            @inbounds @simd for j in 1:n
                x[j] += α * Ai[j]
            end
        end

        # Check stopping condition by residual norm
        # (In practice, you might want to check less frequently for performance reasons.)
        if norm(A * x - b) < tol
            break
        end
    end

    return x
end

# Example usage
using Random

# Seed RNG for reproducible results
Random.seed!(0)

# Construct a random system A*x = b
m, n = 1000, 200
A = randn(m, n)
x_true = randn(n)
b = A * x_true + 0.01*randn(m)  # add slight noise

# Solve using Kaczmarz
x_est = kaczmarz_vanilla(A, b, maxiter=100_000, tol=1e-6)
x_est_rand = kaczmarz(A, b, maxiter=100_000, tol=1e-6)

# Check the error
println("‖x_est - x_true‖ = ", norm(x_est - x_true))
println("‖x_est_rand - x_true‖ = ", norm(x_est_rand - x_true))
