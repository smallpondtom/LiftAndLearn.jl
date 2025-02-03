using LinearAlgebra

"""
    kaczmarz_matrix_vanilla(D, R; maxiter=10_000, tol=1e-6)

Solve min ||R - D*O||_F for O via the cyclic Kaczmarz method.

# Arguments
- `D::AbstractMatrix{T}`: A (K x d) matrix.
- `R::AbstractMatrix{T}`: A (K x r) matrix.
- `maxiter::Int`: Maximum number of row-updates (default = 10_000).
- `tol::Real`: Frobenius norm tolerance for early stopping (default = 1e-6).

# Returns
- `O::Matrix{Float64}`: (d x r) matrix approximating the solution of D*O = R.

# Notes
- Processes rows i in order: i = 1..K, then wraps back to i=1, ...
- Update for row i: O ← O + (1 / ||D[i,:]||^2) * D[i,:]' * (R[i,:] - D[i,:]*O).
"""
function kaczmarz_matrix_vanilla(D, R; maxiter=10_000, tol=1e-6)
    @assert size(D,1) == size(R,1) "D and R must have the same number of rows."
    K, d = size(D)
    _, r  = size(R)

    # Initialize O
    O = zeros(eltype(D), d, r)

    # Count how many row updates we have done
    updates = 0

    while updates < maxiter
        # One sweep over all rows
        for i in 1:K
            updates += 1
            if updates > maxiter
                break
            end
            # Row i of D: shape (1 x d)
            Di = transpose(@view D[i, :])
            # Row i of R: shape (1 x r)
            Ri = transpose(@view R[i, :])

            # residual row: (1 x r)
            res_i = Ri - Di * O

            # row norm-squared (scalar)
            denom = dot(Di, Di)
            # Update O: rank-1 update
            # (d x 1) * (1 x r) = (d x r)
            O .+= (1/denom) * (Di' * res_i)
        end

        # Check residual on the full matrix, occasionally or every sweep
        # ||D*O - R||_F
        if norm(D*O - R) < tol
            break
        end
    end

    return O
end

using LinearAlgebra
using Distributions

"""
    kaczmarz_matrix_randomized(D, R; maxiter=10_000, tol=1e-6)

Solve min ||R - D*O||_F for O via the randomized Kaczmarz method.

# Arguments
- `D::AbstractMatrix{T}`: A (K x d) matrix.
- `R::AbstractMatrix{T}`: A (K x r) matrix.
- `maxiter::Int`: Maximum number of row-updates (default = 10_000).
- `tol::Real`: Frobenius norm tolerance for early stopping (default = 1e-6).

# Returns
- `O::Matrix{Float64}`: (d x r) matrix approximating the solution of D*O = R.

# Notes
- Each iteration picks row i w.p. proportional to ||D[i,:]||^2.
- The update for row i is: O ← O + (1 / ||D[i,:]||^2) * D[i,:]'*(R[i,:] - D[i,:]*O).
"""
function kaczmarz_matrix_randomized(D, R; maxiter=10_000, tol=1e-6)
    @assert size(D,1) == size(R,1) "D and R must have the same number of rows."
    K, d = size(D)
    _, r  = size(R)

    # Initialize O
    O = zeros(eltype(D), d, r)

    # Precompute row norms squared
    row_norms_sq = [(@views dot(D[i,:], D[i,:])) for i in 1:K]
    total_norm   = sum(row_norms_sq)
    p            = row_norms_sq ./ total_norm

    # Build a discrete distribution for row selection
    row_dist = Categorical(p)

    for iter in 1:maxiter
        # pick a random row index i
        i = rand(row_dist)
        Di = transpose(@view D[i, :])
        Ri = transpose(@view R[i, :])

        # row residual
        res_i = Ri - Di * O
        # update
        denom = row_norms_sq[i]
        O .+= (1/denom) * (Di' * res_i)

        # Optional check of global residual every iteration (can be expensive)
        # In practice, one might check less frequently for speed.
        if norm(D*O - R) < tol
            break
        end
    end

    return O
end


using Random
Random.seed!(123)

K, d, r = 200, 50, 10
D = randn(K, d)
O_true = randn(d, r)
R = D*O_true + 0.01*randn(K, r)

O_est = kaczmarz_matrix_vanilla(D, R, maxiter=100_000, tol=1e-6)
O_est_rand = kaczmarz_matrix_randomized(D, R, maxiter=100_000, tol=1e-6)
println("Frobenius error in the solution = ", norm(D*O_est - R))
println("Frobenius error in the randomized solution = ", norm(D*O_est_rand - R))
