"""
$(SIGNATURES)

Fill the diagonal entries of the Tikhonov matrix ``\\boldsymbol{\\Gamma}`` block by block
from the regularization parameters `λ`.

The entries of `Γ` are split into consecutive blocks of lengths `dims`; the block of
operator symbol `s` is set to `getproperty(λ, s)`. For polynomial state operators the
first two characters of the symbol are used, so `:A2` and `:A2u` both take `λ.A2`.

## Arguments
- `Γ::AbstractArray`: vector of length `sum(dims)` holding the diagonal of ``\\boldsymbol{\\Gamma}`` (mutated).
- `dims::AbstractArray`: number of columns of the data matrix belonging to each operator.
- `operator_symbols::AbstractArray`: operator symbols in the same order as `dims` (e.g. `[:A, :B, :A2u]`).
- `λ::TikhonovParameter`: regularization parameter of each operator.

## Returns
- `nothing`; mutates `Γ` in place.

## Notes
- Symbols without a matching `TikhonovParameter` field (e.g. `:N2` for higher-order
  state-input coupling) raise an error.
"""
function tikhonov_matrix!(Γ::AbstractArray, dims::AbstractArray, operator_symbols::AbstractArray, 
                         λ::TikhonovParameter)
    si = 1
    for (d, symbol) in zip(dims, operator_symbols)
        symbol_str  = string(symbol)
        if (length(symbol_str) >= 2) && ('A' in symbol_str)
            # lambda has fields λ.A2, λ.A3, etc. for polynomial operators
            # and not λ.A2u, λ.A3u, etc.
            Γ[si:si+d-1] .= getproperty(λ, Symbol(symbol_str[1:2]))
        else
            Γ[si:si+d-1] .= getproperty(λ, symbol)
        end
        si += d
    end
end

"""
$(TYPEDEF)

Configuration of the Tikhonov-regularized least-squares solver used by `tikhonov`.
Internal helper.

Construct with the keyword constructor `TikhonovSolver{T}(; kwargs...)`, whose keywords
and defaults are the fields listed below.

## Fields
- `tolerance::T=1e-12`: absolute tolerance of the iterative solvers and relative singular-value cutoff of the SVD-truncation solver.
- `use_gpu::Bool=false`: request GPU acceleration.
- `use_normal_form::Bool=false`: allow the normal-form solver.
- `use_svd_truncation::Bool=false`: allow the SVD-truncation solver.
- `use_backslash::Bool=false`: use the backslash operator (forced to `false` when `use_gpu=true`).
- `chunk_size::Int=1000`: number of right-hand-side columns solved per batch.
- `max_iterations::Int=1000`: maximum number of iterations of the iterative solvers.
- `preconditioning::Bool=false`: use diagonal preconditioning (only in `solve_iterative_normal_form`).
"""
struct TikhonovSolver{T<:Real}
    tolerance::T
    use_gpu::Bool
    use_normal_form::Bool
    use_svd_truncation::Bool
    use_backslash::Bool
    chunk_size::Int
    max_iterations::Int
    preconditioning::Bool
    
    function TikhonovSolver{T}(; tolerance::T=1e-12,
                              use_gpu::Bool=false, use_normal_form::Bool=false,
                              use_svd_truncation::Bool=false, 
                              use_backslash::Bool=false,
                              chunk_size::Int=1000,
                              max_iterations::Int=1000,
                              preconditioning::Bool=false) where T<:Real
        use_backslash = use_gpu ? false : use_backslash
        new{T}(tolerance, use_gpu, use_normal_form, use_svd_truncation, 
               use_backslash, chunk_size, max_iterations, preconditioning)
    end
end

"""
$(SIGNATURES)

Construct a `TikhonovSolver{Float64}`; the keywords are the fields of `TikhonovSolver`.
Internal helper.
"""
TikhonovSolver(; kwargs...) = TikhonovSolver{Float64}(; kwargs...)

"""
$(TYPEDEF)

Abstract supertype of the Tikhonov solution methods selected by `select_optimal_method`.
Internal helper.
"""
abstract type RegularizationMethod end

"""
$(TYPEDEF)

Tag for the augmented-system method ``[\\mathbf{A}; \\boldsymbol{\\Gamma}^{1/2}]\\mathbf{x} = [\\mathbf{b}; \\mathbf{0}]``
(see `solve_augmented_system`). Internal helper.
"""
struct AugmentedSystem <: RegularizationMethod end

"""
$(TYPEDEF)

Tag for the normal-form method ``(\\mathbf{A}^\\top\\mathbf{A} + \\boldsymbol{\\Gamma})\\mathbf{x} = \\mathbf{A}^\\top\\mathbf{b}``
(see `solve_normal_form`). Internal helper.
"""
struct NormalForm <: RegularizationMethod end

"""
$(TYPEDEF)

Tag for the SVD-truncation method (see `solve_svd_truncation`). Internal helper.
"""
struct SVDTruncation <: RegularizationMethod end

"""
$(SIGNATURES)

Move `arrays` to the GPU if `use_gpu` is `true` and a functional GPU backend (Metal.jl on
macOS, CUDA.jl otherwise) is available. Internal helper.

## Arguments
- `arrays...`: arrays to transfer.

## Keyword Arguments
- `use_gpu::Bool=false`: whether GPU computation is requested.

## Returns
- `(arrays..., gpu_active)`: the (possibly GPU) arrays followed by a `Bool` that is `true`
  if the arrays were moved to the GPU; falls back to the CPU with a warning otherwise.
"""
function setup_gpu_computation(arrays...; use_gpu::Bool=false)
    if !use_gpu
        return arrays..., false
    end
    
    if Sys.isapple()
        if Metal.functional()
            @info "Using Metal.jl for GPU acceleration"
            gpu_arrays = map(arr -> Metal.MetalArray(arr), arrays)
            return gpu_arrays..., true
        else
            @warn "Metal.jl not functional. Falling back to CPU"
            return arrays..., false
        end
    else
        if CUDA.functional()
            @info "Using CUDA.jl for GPU acceleration"
            gpu_arrays = map(arr -> CUDA.CuArray(arr), arrays)
            return gpu_arrays..., true
        else
            @warn "CUDA not functional. Falling back to CPU"
            return arrays..., false
        end
    end
end

"""
$(SIGNATURES)

Compute a square-root factor ``\\boldsymbol{\\Gamma}^{1/2}`` of the Tikhonov matrix, used as
the lower block of the augmented system in `solve_augmented_system`. Internal helper.

## Arguments
- `Γ::AbstractMatrix{T}`: Tikhonov matrix of size ``n \\times n``.
- `method::Symbol=:cholesky`: `:cholesky` (upper Cholesky factor ``\\mathbf{U}`` with
  ``\\mathbf{U}^\\top\\mathbf{U} = \\boldsymbol{\\Gamma}``, used if `Γ` is positive definite),
  `:eigen` (``\\mathbf{V}\\boldsymbol{\\Lambda}^{1/2}`` from a symmetric eigendecomposition,
  which satisfies ``\\mathbf{S}\\mathbf{S}^\\top = \\boldsymbol{\\Gamma}``) or anything else for
  the element-wise square root.

## Returns
- `Γsq::AbstractMatrix`: the square-root factor.

## Notes
- A diagonal `Γ` (the default built by `tikhonov_matrix!`) is factored exactly by the
  element-wise square root, which preserves sparsity.
- If `Γ` is not positive definite (e.g. it has zero diagonal entries) or the Cholesky
  factorization fails, the element-wise square root `sqrt.(Γ)` is returned, which is
  exact only for diagonal `Γ`.
"""
function compute_regularization_matrix(Γ::AbstractMatrix{T}, 
                                       method::Symbol=:cholesky) where T
    if isdiag(Γ)
        # Diagonal Tikhonov matrices (the default built by `tikhonov_matrix!`): the
        # element-wise square root is the exact factor
        return sqrt.(Γ)
    elseif method === :cholesky && isposdef(Γ)
        # Most efficient for positive definite matrices
        try
            chol_factor = cholesky(Γ)
            # Handle both sparse and dense Cholesky factors
            if isa(chol_factor.U, SparseArrays.CHOLMOD.FactorComponent)
                return Matrix(chol_factor.U)  # Convert sparse factor to dense
            else
                return chol_factor.U  # Already a regular matrix
            end
        catch
            # Fall back to element-wise sqrt if Cholesky fails
            @warn "Cholesky factorization failed, using element-wise sqrt"
            return sqrt.(Γ)
        end
    elseif method === :eigen
        # For general symmetric matrices
        F = eigen(Hermitian(Γ))
        λ_sqrt = sqrt.(max.(F.values, zero(T)))
        return F.vectors * Diagonal(λ_sqrt)
    else
        # Fallback to element-wise sqrt (less efficient)
        return sqrt.(Γ)
    end
end

"""
$(SIGNATURES)

Solve the Tikhonov problem through the augmented least-squares system
``[\\mathbf{A}; \\boldsymbol{\\Gamma}^{1/2}]\\mathbf{x} \\approx [\\mathbf{b}; \\mathbf{0}]``.
Internal helper.

``\\boldsymbol{\\Gamma}^{1/2}`` is obtained from `compute_regularization_matrix`. The
system is solved with backslash (sparse augmented matrix on the CPU) if the GPU is
active or `solver.use_backslash` is `true`, otherwise with `solve_least_squares_batch`.
On `OutOfMemoryError` it falls back to `solve_normal_form`.

## Arguments
- `A::AbstractMatrix{T}`: data matrix of size ``m \\times n``.
- `b::AbstractArray{T}`: right-hand side of size ``m \\times p``.
- `Γ::AbstractMatrix{T}`: Tikhonov matrix of size ``n \\times n``.
- `solver::TikhonovSolver{T}`: solver configuration.

## Returns
- `x::AbstractMatrix{T}`: regularized solution of size ``n \\times p``.
"""
function solve_augmented_system(A::AbstractMatrix{T}, b::AbstractArray{T}, 
                               Γ::AbstractMatrix{T}, solver::TikhonovSolver{T}) where T
    m, n = size(A)
    _, p = size(b)
    
    # Compute regularization matrix square root efficiently
    Γsq = compute_regularization_matrix(Γ, :cholesky)
    
    # Setup GPU computation if requested
    A_gpu, b_gpu, Γsq_gpu, gpu_active = setup_gpu_computation(
                                            A, b, Γsq; use_gpu=solver.use_gpu)
    
    # Construct augmented system
    m_reg = size(Γsq_gpu, 1)
    Atilde = solver.use_backslash ? vcat(A_gpu, sparse(Γsq_gpu)) : vcat(A_gpu, Γsq_gpu)
    btilde = vcat(b_gpu, zeros(T, m_reg, p))
    
    try
        if gpu_active || solver.use_backslash
            # GPU solve using built-in backslash (most efficient)
            O = Atilde \ btilde
            return Array(O)
        else
            # CPU solve with batching for multiple RHS
            @info "Using batch least squares solve for augmented system"
            return solve_least_squares_batch(Atilde, btilde, solver.chunk_size)
        end
    catch e
        if isa(e, OutOfMemoryError)
            @warn "Out of memory in augmented system. Switching to normal form."
            return solve_normal_form(A, b, Γ, solver)
        else
            rethrow(e)
        end
    end
end

"""
$(SIGNATURES)

Solve the Tikhonov problem through the normal form
``(\\mathbf{A}^\\top\\mathbf{A} + \\boldsymbol{\\Gamma})\\mathbf{x} = \\mathbf{A}^\\top\\mathbf{b}``.
Internal helper.

The normal-form matrix is solved with backslash if the GPU is active or
`solver.use_backslash` is `true`; otherwise the matrix-free iterative solver
`solve_iterative_tikhonov` is used.

## Arguments
- `A::AbstractMatrix{T}`: data matrix of size ``m \\times n``.
- `b::AbstractArray{T}`: right-hand side of size ``m \\times p``.
- `Γ::AbstractMatrix{T}`: Tikhonov matrix of size ``n \\times n``.
- `solver::TikhonovSolver{T}`: solver configuration.

## Returns
- `x::AbstractMatrix{T}`: regularized solution of size ``n \\times p``.
"""
function solve_normal_form(A::AbstractMatrix{T}, b::AbstractArray{T}, 
                          Γ::AbstractMatrix{T}, solver::TikhonovSolver{T}) where T
    m, n = size(A)
    _, p = size(b)
    
    # Setup GPU computation
    A_gpu, b_gpu, Γ_gpu, gpu_active = setup_gpu_computation(A, b, Γ; use_gpu=solver.use_gpu)
    
    # Compute normal form components
    AtA = A_gpu' * A_gpu
    Atb = A_gpu' * b_gpu
    M = AtA + Γ_gpu
    
    try
        if gpu_active || solver.use_backslash
            O = M \ Atb
            return Array(O)
        else
            # Use iterative solver for CPU with large problems
            # return solve_iterative_normal_form(M, Atb, solver)
            return solve_iterative_tikhonov(A, b, Γ, solver)
        end
    catch e
        if isa(e, OutOfMemoryError)
            @warn "Out of memory in normal form. Switching to iterative approach."
            return solve_iterative_tikhonov(A, b, Γ, solver)
        else
            rethrow(e)
        end
    end
end

"""
$(SIGNATURES)

Solve the assembled normal-form system ``\\mathbf{M}\\mathbf{x} = \\mathbf{A}^\\top\\mathbf{b}``
column by column with LinearSolve.jl, optionally with the diagonal (Jacobi-type)
preconditioner ``\\mathrm{diag}(\\mathbf{M})^{-1/2}``. Internal helper.

## Arguments
- `M::AbstractMatrix{T}`: normal-form matrix ``\\mathbf{A}^\\top\\mathbf{A} + \\boldsymbol{\\Gamma}`` of size ``n \\times n``.
- `Atb::AbstractArray{T}`: right-hand side ``\\mathbf{A}^\\top\\mathbf{b}`` of size ``n \\times p``.
- `solver::TikhonovSolver{T}`: solver configuration (`preconditioning`, `tolerance`, `max_iterations`).

## Returns
- `x::AbstractMatrix{T}`: solution of size ``n \\times p``.

## Notes
- Not called by the default solve paths (its call in `solve_normal_form` is commented out).
"""
function solve_iterative_normal_form(M::AbstractMatrix{T}, Atb::AbstractArray{T}, 
                                     solver::TikhonovSolver{T}) where T
    n, p = size(Atb)
    O = similar(Atb)
    
    # Use diagonal preconditioning
    if solver.preconditioning
        P = Diagonal(1 ./ sqrt.(diag(M)))
    end 

    # Solve for each column
    ls = nothing
    for i in 1:p
        if i == 1
            prob = LinearProblem(M, Atb[:, i])
            if solver.preconditioning
                ls = init(prob, Pl=P)
            else
                ls = init(prob)
            end
        else
            ls.b = Atb[:, i]
        end
        sol = solve!(ls; maxiters=solver.max_iterations, abstol=solver.tolerance)
        O[:, i] .= sol.u
    end
    
    return O
end

"""
$(SIGNATURES)

Solve ``(\\mathbf{A}^\\top\\mathbf{A} + \\boldsymbol{\\Gamma})\\mathbf{x} = \\mathbf{A}^\\top\\mathbf{b}``
column by column with an iterative LinearSolve.jl solver, applying the normal-form
operator matrix-free (``\\mathbf{A}^\\top\\mathbf{A}`` is never formed). Internal helper.

## Arguments
- `A::AbstractMatrix{T}`: data matrix of size ``m \\times n``.
- `b::AbstractArray{T}`: right-hand side of size ``m \\times p``.
- `Γ::AbstractMatrix{T}`: Tikhonov matrix of size ``n \\times n``.
- `solver::TikhonovSolver{T}`: solver configuration (`tolerance`, `max_iterations`).

## Returns
- `x::AbstractMatrix{T}`: regularized solution of size ``n \\times p``.
"""
function solve_iterative_tikhonov(A::AbstractMatrix{T}, 
                                  b::AbstractArray{T}, 
                                  Γ::AbstractMatrix{T}, 
                                  solver::TikhonovSolver{T}) where T
    m, n = size(A)
    _, p = size(b)
    
    # # Create linear operator for (A'*A + Γ) without storing the full matrix
    # op = let A = A, Γ = Γ
    #     function matvec!(y, x, p, t)
    #         # y = (A'*A + Γ) * x
    #         temp = A * x
    #         mul!(y, A', temp)
    #         y .+= Γ * x
    #     end
    #     FunctionOperator{T}(matvec!, n, n; issymmetric=true)
    # end

    # Create linear operator for (A'*A + Γ) without storing the full matrix
    function matvec!(y, x, p, t)
        # y = (A'*A + Γ) * x
        temp = A * x
        mul!(y, A', temp)
        y .+= Γ * x
    end
    # SciMLOperators ≥ 1.0 calls in-place operators as op(w, v, u, p, t)
    matvec!(w, v, u, p, t) = matvec!(w, v, p, t)
    
    # Create prototype arrays for input and output
    input_prototype = zeros(T, n)
    output_prototype = zeros(T, n)
    
    # Use the correct FunctionOperator constructor with prototype arrays
    op = FunctionOperator(matvec!, input_prototype, output_prototype; 
                         isinplace=true, issymmetric=true)
    
    # Compute A' * b
    Atb = A' * b
    
    # Solve iteratively
    O = similar(A, n, p)
    ls = nothing
    
    for i in 1:p
        if i == 1
            prob = LinearProblem(op, Atb[:, i])
            ls = init(prob)
        else
            ls.b = Atb[:, i]
        end
        sol = solve!(ls; maxiters=solver.max_iterations, abstol=solver.tolerance)
        O[:, i] .= sol.u
    end
    
    return O
end


"""
$(SIGNATURES)

Solve ``\\min_{\\mathbf{x}} \\|\\mathbf{A}\\mathbf{x} - \\mathbf{b}\\|_2^2 + \\lambda^2\\|\\mathbf{x}\\|_2^2``
with the SVD ``\\mathbf{A} = \\mathbf{U}\\boldsymbol{\\Sigma}\\mathbf{V}^\\top`` as
``\\mathbf{x} = \\mathbf{V}\\,\\mathrm{diag}(\\sigma_i/(\\sigma_i^2+\\lambda^2))\\,\\mathbf{U}^\\top\\mathbf{b}``.
Internal helper.

## Arguments
- `A::AbstractMatrix{T}`: data matrix of size ``m \\times n``.
- `b::AbstractArray{T}`: right-hand side of size ``m \\times p``.
- `λ::T`: regularization parameter; note that it is squared, so pass ``\\sqrt{\\gamma}`` for ``\\boldsymbol{\\Gamma} = \\gamma\\mathbf{I}``.
- `solver::TikhonovSolver{T}`: solver configuration; if `solver.tolerance > 0` the residual norm is logged.

## Returns
- `x::AbstractMatrix{T}`: regularized solution of size ``n \\times p``.
"""
function solve_svd_scalar_regularization(A::AbstractMatrix{T}, 
                                         b::AbstractArray{T}, 
                                         λ::T,
                                         solver::TikhonovSolver{T}) where T
    try
        # Compute SVD of A
        U, σ, V = svd(A)
        
        # Compute regularized inverse of singular values
        σ_inv = σ ./ (σ.^2 .+ λ^2)
        
        # Efficient computation: V * (σ_inv .* (U' * b))
        # This is equivalent to: V * Diagonal(σ_inv) * U' * b
        x = V * (σ_inv .* (U' * b))
        
        # Optionally compute residual for diagnostics
        if solver.tolerance > 0  # Use tolerance flag to indicate if residual is needed
            b_estimate = A * x
            residual = norm(b - b_estimate)
            @info "L2 regularized solution residual: $residual"
        end
        
        return x
        
    catch e
        if isa(e, OutOfMemoryError)
            @warn "Out of memory in SVD scalar regularization. Falling back to normal form."
            # Fall back to normal form with scalar regularization
            n = size(A, 2)
            Γ_scalar = λ * I(n)
            return solve_normal_form(A, b, Γ_scalar, solver)
        else
            rethrow(e)
        end
    end
end

"""
$(SIGNATURES)

Check whether the Tikhonov matrix is a scalar multiple of the identity,
``\\boldsymbol{\\Gamma} = \\gamma\\mathbf{I}`` (diagonal entries equal up to a relative tolerance of `1e-14`).
Internal helper.

## Arguments
- `Γ::AbstractMatrix{T}`: Tikhonov matrix.

## Returns
- `is_scalar::Bool`: `true` if `Γ` is diagonal with equal diagonal entries.
- `γ::T`: the common diagonal value if `is_scalar`, otherwise `zero(T)`.
"""
function is_scalar_regularization(Γ::AbstractMatrix{T}) where T
    n = size(Γ, 1)
    
    # Check if it's diagonal
    if !isdiag(Γ)
        return false, zero(T)
    end
    
    # Get diagonal values
    diag_vals = diag(Γ)
    
    # Check if all diagonal values are equal (within tolerance)
    first_val = diag_vals[1]
    tol = 1e-14 * abs(first_val)  # Relative tolerance
    
    if all(abs.(diag_vals .- first_val) .<= tol)
        return true, first_val
    else
        return false, zero(T)
    end
end

"""
$(SIGNATURES)

Solve the Tikhonov problem with an SVD of the normal-form matrix and truncation of small
singular values, for (nearly) rank-deficient problems. Internal helper.

If ``\\boldsymbol{\\Gamma} = \\gamma\\mathbf{I}`` with ``\\gamma > 0``, the problem is
delegated to `solve_svd_scalar_regularization`. Otherwise the SVD of
``\\mathbf{M} = \\mathbf{A}^\\top\\mathbf{A} + \\boldsymbol{\\Gamma}`` is computed; singular values
below `solver.tolerance * σ_max` are discarded (truncated pseudoinverse) and, if none are
discarded, ``\\mathbf{M}\\mathbf{x} = \\mathbf{A}^\\top\\mathbf{b}`` is solved directly. On
`OutOfMemoryError` it falls back to `LAPACK.gelsd!` on the unregularized problem.

## Arguments
- `A::AbstractMatrix{T}`: data matrix of size ``m \\times n``.
- `b::AbstractArray{T}`: right-hand side of size ``m \\times p``.
- `Γ::AbstractMatrix{T}`: Tikhonov matrix of size ``n \\times n``.
- `solver::TikhonovSolver{T}`: solver configuration (`tolerance`).

## Returns
- `x::AbstractMatrix{T}`: regularized solution of size ``n \\times p``.
"""
function solve_svd_truncation(A::AbstractMatrix{T}, 
                              b::AbstractArray{T}, 
                              Γ::AbstractMatrix{T}, 
                              solver::TikhonovSolver{T}) where T
    # Check if Γ is scalar regularization
    is_scalar, λ = is_scalar_regularization(Γ)
    
    if is_scalar && λ > 0
        @info "Detected scalar regularization (Γ = $(λ)I). Using efficient SVD method."
        return solve_svd_scalar_regularization(A, b, sqrt(λ), solver)
    end

    try 
        n = size(A,2)
        p = size(b,2)

        # Convert Adjoint to contiguous matrix if needed
        b = b isa Adjoint ? Matrix(b) : b
        
        # Pre-allocate matrices for BLAS operations
        M = Matrix{T}(undef, n, n)
        rhs = Matrix{T}(undef, n, p)
        
        # Use BLAS level 3 operations
        LinearAlgebra.BLAS.gemm!('T', 'N', one(T), A, A, zero(T), M)
        M .+= Γ
        LinearAlgebra.BLAS.gemm!('T', 'N', one(T), A, b, zero(T), rhs)
        
        # Compute SVD (iterating an SVD object yields U, S, V -- not Vt)
        U, S, V = svd(M)
        
        # Determine effective rank using relative tolerance
        σ_max = S[1]
        effective_rank = count(σ -> σ > solver.tolerance * σ_max, S)
        
        if effective_rank < length(S)
            @info "Rank deficient system detected. Effective rank: \
                    $effective_rank/$(length(S))"
            
            # Stable pseudoinverse computation
            inv_S = zeros(T, length(S))
            inv_S[1:effective_rank] = 1 ./ S[1:effective_rank]
            
            # Efficient computation: V * (inv_S .* (U' * rhs))
            return V * (inv_S .* (U' * rhs))
        else
            # Full rank, use standard solve
            return M \ rhs
        end
    catch e
        if isa(e, OutOfMemoryError)
            @warn "Out of memory. Using LAPACK.gelsd! routine."
            # Convert to contiguous matrices for LAPACK
            A = A isa Adjoint ? Matrix(A) : A
            b = b isa Adjoint ? Matrix(b) : b
            x, rank = LAPACK.gelsd!(A, b, solver.tolerance)
            if rank < size(A, 2)
                @info "Rank deficient system detected. Effective rank: \
                        $rank/$(size(A,2))"
            end
            return x
        else
            rethrow(e)
        end
    end
end

"""
$(SIGNATURES)

Solve the least-squares problem ``\\mathbf{A}\\mathbf{x} \\approx \\mathbf{b}`` with backslash,
processing the right-hand-side columns in chunks of `chunk_size`. Internal helper.

## Arguments
- `A::AbstractMatrix{T}`: coefficient matrix of size ``m \\times n``.
- `b::AbstractArray{T}`: right-hand side of size ``m \\times p``.
- `chunk_size::Int`: number of right-hand-side columns per chunk.

## Returns
- `x::AbstractMatrix{T}`: least-squares solution of size ``n \\times p``.
"""
function solve_least_squares_batch(A::AbstractMatrix{T}, b::AbstractArray{T}, 
                                   chunk_size::Int) where T
    m, n = size(A)
    _, p = size(b)
    
    O = similar(A, n, p)
    
    # Process in chunks
    for start_idx in 1:chunk_size:p
        end_idx = min(start_idx + chunk_size - 1, p)
        chunk_range = start_idx:end_idx
        
        # Solve for current chunk
        O[:, chunk_range] = A \ view(b, :, chunk_range)
    end
    
    return O
end

"""
$(SIGNATURES)

Select the Tikhonov solution method from the solver flags and the problem size.
Internal helper.

The first matching rule is used:
1. `SVDTruncation()` if `solver.use_svd_truncation` and the condition estimate exceeds `1e12`;
2. `NormalForm()` if `solver.use_normal_form`, ``m > 2n`` and ``p < 100``;
3. `AugmentedSystem()` if `solver.use_backslash`;
4. otherwise `nothing` (the iterative solver `solve_iterative_tikhonov`).

## Arguments
- `A::AbstractMatrix{T}`: data matrix of size ``m \\times n``.
- `b::AbstractArray{T}`: right-hand side of size ``m \\times p``.
- `Γ::AbstractMatrix{T}`: Tikhonov matrix (unused).
- `solver::TikhonovSolver{T}`: solver configuration.

## Returns
- `method::Union{RegularizationMethod,Nothing}`: the selected method tag.
- `method_symbol::Symbol`: `:svd`, `:normal`, `:augmented` or `:auto`.

## Notes
- The "condition estimate" is `norm(A, 2) / norm(A, -2)`, i.e. a ratio of entry-wise
  vector norms (Frobenius norm over the ``p=-2`` entry norm), not the spectral condition
  number. It is infinite if `A` has a zero entry.
"""
function select_optimal_method(A::AbstractMatrix{T}, 
                               b::AbstractArray{T}, 
                               Γ::AbstractMatrix{T}, 
                               solver::TikhonovSolver{T}) where T
    m, n = size(A)
    _, p = size(b)
    
    # Estimate condition number roughly
    condition_estimate = norm(A, 2) / norm(A, -2)  # Approximate condition number
    
    if solver.use_svd_truncation && condition_estimate > 1e12
        @info "Using SVD truncation method due to high condition number: \
                             $condition_estimate"
        return SVDTruncation(), :svd
    elseif solver.use_normal_form && (m > 2n && p < 100)
        @info "Using normal form method for large overdetermined system"
        return NormalForm(), :normal
    elseif solver.use_backslash
        @info "Using augmented system method for general case"
        return AugmentedSystem(), :augmented
    else
        @info "Using iterative Tikhonov solver for memory efficiency"
        return nothing, :auto
    end
end

"""
$(SIGNATURES)

Solve the Tikhonov-regularized least-squares problem

```math
\\min_{\\mathbf{x}} \\|\\mathbf{A}\\mathbf{x} - \\mathbf{b}\\|_2^2 + \\mathbf{x}^\\top\\boldsymbol{\\Gamma}\\mathbf{x}
= \\|\\mathbf{A}\\mathbf{x} - \\mathbf{b}\\|_2^2 + \\|\\boldsymbol{\\Gamma}^{1/2}\\mathbf{x}\\|_2^2
```

column by column for all right-hand sides in `b`. In Operator Inference (OpInf),
`A` is the data matrix ``\\mathbf{D}``, `b` the derivative data ``\\mathbf{R}^\\top`` and the
solution is ``\\mathbf{O}^\\top``.

The solution method is chosen by `select_optimal_method`: SVD truncation
(`use_svd_truncation`), normal form (`use_normal_form`), augmented system
(`use_backslash`), or, by default, a matrix-free iterative solver. On
`OutOfMemoryError` it falls back to the iterative solver.

## Arguments
- `b::AbstractArray{T}`: right-hand side of size ``m \\times p``.
- `A::AbstractArray{T}`: coefficient (data) matrix of size ``m \\times n``.
- `Γ::AbstractMatrix{T}`: Tikhonov matrix ``\\boldsymbol{\\Gamma}`` of size ``n \\times n`` (symmetric positive semidefinite).

## Keyword Arguments
- `tol::Real=1e-12`: absolute tolerance of the iterative solvers and relative singular-value cutoff of the SVD-truncation solver.
- `use_gpu::Bool=false`: use the GPU (CUDA.jl or Metal.jl) if functional. This forces `use_backslash=false`, so the augmented-system solver is not selected and only the normal-form solver runs on the GPU.
- `use_normal_form::Bool=false`: allow the normal-form solver (used if ``m > 2n`` and ``p < 100``).
- `use_svd_truncation::Bool=false`: allow the SVD-truncation solver (used if the condition estimate exceeds `1e12`).
- `use_backslash::Bool=false`: use backslash in the augmented-system and normal-form solvers.
- `chunk_size::Int=1000`: number of right-hand-side columns per batch.
- `max_iterations::Int=1000`: maximum number of iterations of the iterative solvers.
- `estimate_memory::Bool=false`: log an estimate of the memory usage (see `estimate_tikhonov_memory`).
- `preconditioning::Bool=false`: stored in the solver configuration (see `TikhonovSolver`).

## Returns
- `x::AbstractMatrix{T}`: regularized solution of size ``n \\times p``.

## Notes
- The iterative solvers stop at an absolute tolerance `tol`, so their accuracy is lower
  than that of the direct (backslash) solvers.
"""
function tikhonov(b::AbstractArray{T}, A::AbstractArray{T}, Γ::AbstractMatrix{T};
                  tol::Real=1e-12, use_gpu::Bool=false, use_normal_form::Bool=false,
                  use_svd_truncation::Bool=false, use_backslash::Bool=false,
                  chunk_size::Int=1000, max_iterations::Int=1000,
                  estimate_memory::Bool=false, 
                  preconditioning::Bool=false) where T
    
    # Input validation
    size(A, 1) == size(b, 1) || throw(DimensionMismatch(
        "A and b must have same number of rows"))
    size(A, 2) == size(Γ, 1) == size(Γ, 2) || throw(DimensionMismatch(
        "A columns must match Γ dimensions"))
    
    # Create solver configuration
    solver = TikhonovSolver{T}(; tolerance=T(tol), use_gpu, use_normal_form,
                                 use_svd_truncation, use_backslash,  
                                 chunk_size, max_iterations, preconditioning)
    
    # Select optimal method
    method, method_symbol = select_optimal_method(A, b, Γ, solver)

    # Estimate memory if requested
    if estimate_memory
        mem_estimate = estimate_tikhonov_memory(A, b, Γ, method_symbol)
        @info "Estimated memory usage for $method: $(mem_estimate / (1024^2)) MB"
    end
    
    # Solve using selected method
    try
        if method isa SVDTruncation
            return solve_svd_truncation(A, b, Γ, solver)
        elseif method isa NormalForm
            return solve_normal_form(A, b, Γ, solver)
        elseif method isa AugmentedSystem
            return solve_augmented_system(A, b, Γ, solver)
        else 
            return solve_iterative_tikhonov(A, b, Γ, solver)
        end
    catch e
        if isa(e, OutOfMemoryError)
            @warn "Out of memory with selected method. Falling back to iterative solver."
            return solve_iterative_tikhonov(A, b, Γ, solver)
        else
            rethrow(e)
        end
    end
end

# # Convenience method for automatic type promotion
# tikhonov(b::AbstractArray, A::AbstractArray, Γ::AbstractMatrix; kwargs...) = 
#     tikhonov(promote(b, A, Γ)...; kwargs...)

"""
$(SIGNATURES)

Convenience method of `tikhonov` for inputs with different element types or `Adjoint`
wrappers: materializes adjoints, promotes `b`, `A` and `Γ` to a common element type and
calls the typed method with the same keyword arguments.

## Arguments
- `b::AbstractArray`: right-hand side of size ``m \\times p``.
- `A::AbstractArray`: coefficient (data) matrix of size ``m \\times n``.
- `Γ::AbstractMatrix`: Tikhonov matrix of size ``n \\times n``.

## Keyword Arguments
- `kwargs...`: forwarded to the typed `tikhonov` method.

## Returns
- `x::AbstractMatrix`: regularized solution of size ``n \\times p``.
"""
function tikhonov(b::AbstractArray, A::AbstractArray, Γ::AbstractMatrix; kwargs...)
    # Convert Adjoint to Matrix to avoid promotion issues
    b_matrix = b isa Adjoint ? Matrix(b) : b
    A_matrix = A isa Adjoint ? Matrix(A) : A  
    Γ_matrix = Γ isa Adjoint ? Matrix(Γ) : Γ
    
    # Promote element types to common type
    T = promote_type(eltype(b_matrix), eltype(A_matrix), eltype(Γ_matrix))
    
    # Convert to common element type
    b_promoted = convert(AbstractArray{T}, b_matrix)
    A_promoted = convert(AbstractArray{T}, A_matrix)
    Γ_promoted = convert(AbstractMatrix{T}, Γ_matrix)
    
    return tikhonov(b_promoted, A_promoted, Γ_promoted; kwargs...)
end

"""
$(SIGNATURES)

Compute the Tikhonov solutions ``(\\mathbf{A}^\\top\\mathbf{A} + \\gamma\\mathbf{I})^{-1}\\mathbf{A}^\\top\\mathbf{b}``
for each scalar parameter ``\\gamma`` in `Γ_vals`, reusing ``\\mathbf{A}^\\top\\mathbf{A}`` and
``\\mathbf{A}^\\top\\mathbf{b}``.

## Arguments
- `b::AbstractArray{T}`: right-hand side matrix of size ``m \\times p``.
- `A::AbstractArray{T}`: coefficient (data) matrix of size ``m \\times n``.
- `Γ_vals::AbstractVector{T}`: scalar regularization parameters ``\\gamma``.

## Keyword Arguments
- `kwargs...`: accepted but ignored.

## Returns
- `solutions::Vector{Matrix{T}}`: one ``n \\times p`` solution per entry of `Γ_vals`.

## Notes
- `b` must be a matrix; a vector `b` produces vector solutions that cannot be stored in the
  `Vector{Matrix{T}}` result.
"""
function tikhonov_regularization_path(b::AbstractArray{T}, A::AbstractArray{T}, 
                                     Γ_vals::AbstractVector{T}; kwargs...) where T
    solutions = Vector{Matrix{T}}()
    
    # Pre-compute A'*A and A'*b for efficiency
    AtA = A' * A
    Atb = A' * b
    
    for γ in Γ_vals
        Γ = γ * I(size(A, 2))
        M = AtA + Γ
        push!(solutions, M \ Atb)
    end
    
    return solutions
end

"""
$(SIGNATURES)

Estimate the memory (in bytes) required by the Tikhonov solution `method`.

The base memory is `sizeof(T)*(m*n + m*p + n*n)`; `:augmented` adds
`sizeof(T)*(m+n)*(n+p)`, `:normal` adds `sizeof(T)*(n*n + n*p)`, `:svd` adds
`sizeof(T)*3*n*n`, and any other symbol returns the base memory.

## Arguments
- `A::AbstractMatrix{T}`: coefficient (data) matrix of size ``m \\times n``.
- `b::AbstractArray{T}`: right-hand side of size ``m \\times p``.
- `Γ::AbstractMatrix{T}`: Tikhonov matrix (unused).
- `method::Symbol=:auto`: `:augmented`, `:normal`, `:svd` or `:auto`.

## Returns
- `mem::Integer`: estimated memory in bytes.
"""
function estimate_tikhonov_memory(A::AbstractMatrix{T}, b::AbstractArray{T}, 
                                 Γ::AbstractMatrix{T}, method::Symbol=:auto) where T
    m, n = size(A)
    _, p = size(b)
    
    base_memory = sizeof(T) * (m * n + m * p + n * n)
    
    if method === :augmented
        # Additional memory for augmented system
        aug_memory = sizeof(T) * (m + n) * (n + p)
        return base_memory + aug_memory
    elseif method === :normal
        # Additional memory for normal equations
        normal_memory = sizeof(T) * (n * n + n * p)
        return base_memory + normal_memory
    elseif method === :svd
        # Additional memory for SVD
        svd_memory = sizeof(T) * (n * n + 2 * n * n)  # U, S, V storage
        return base_memory + svd_memory
    else
        return base_memory
    end
end

"""
$(SIGNATURES)

Select the scale ``\\gamma`` of the Tikhonov matrix ``\\boldsymbol{\\Gamma} = \\gamma\\boldsymbol{\\Gamma}_{\\text{base}}``
by generalized cross-validation (GCV) over `n_params` log-spaced values in ``[10^{-8}, 10^{2}]``.

For each ``\\gamma`` the GCV score
``K\\,\\|\\mathbf{A}\\mathbf{x}_\\gamma - \\mathbf{b}\\|^2 / (K - \\mathrm{tr}(\\mathbf{A}(\\mathbf{A}^\\top\\mathbf{A}+\\boldsymbol{\\Gamma})^{-1}\\mathbf{A}^\\top))^2``
is computed, where ``K`` is the length of `b`.

## Arguments
- `A::AbstractMatrix{T}`: coefficient (data) matrix of size ``K \\times n``.
- `b::AbstractVector{T}`: right-hand-side vector of length ``K``.
- `Γ_base::AbstractMatrix{T}`: base Tikhonov matrix of size ``n \\times n``.

## Keyword Arguments
- `method::Symbol=:gcv`: selection criterion; only `:gcv` is supported.
- `n_params::Int=50`: number of candidate values of ``\\gamma``.

## Returns
- `γ_opt::Float64`: the candidate with the smallest GCV score.
- `scores::Vector{T}`: GCV score of each candidate.

## Notes
- Throws an `ArgumentError` for any `method` other than `:gcv`.
"""
function optimize_regularization_parameter(A::AbstractMatrix{T}, 
                                           b::AbstractVector{T}, 
                                           Γ_base::AbstractMatrix{T}; 
                                           method::Symbol=:gcv, 
                                           n_params::Int=50) where T
    
    # Generate parameter range
    γ_range = exp10.(range(-8, 2, length=n_params))
    
    if method === :gcv
        # Generalized Cross Validation
        gcv_scores = zeros(T, n_params)
        
        for (i, γ) in enumerate(γ_range)
            Γ = γ * Γ_base
            M = A' * A + Γ
            x = M \ (A' * b)
            
            # Compute GCV score
            residual = A * x - b
            trace_term = tr(A * (M \ A'))
            gcv_scores[i] = (norm(residual)^2 / (length(b) - trace_term)^2) * length(b)
        end
        
        optimal_idx = argmin(gcv_scores)
        return γ_range[optimal_idx], gcv_scores
    else
        throw(ArgumentError("Only GCV method is currently supported"))
    end
end


#==============================================================================#
#==================== Old code left for deubbging purposes ====================#
#==============================================================================#
# This is the old version of the tikhonov function. It is kept here for reference.
# Will be archived in the future.
# function tikhonov(b::AbstractArray, A::AbstractArray, Γ::AbstractMatrix, tol::Real; flag::Bool=false)
#     if flag
#         # Ag = A' * A + Γ' * Γ  # This is if || Γ*O ||_F is desired
#         Ag = A' * A + Γ         # This is if || Γ^{1/2}*O ||_2 is desired
#         Ag_svd = svd(Ag)
#         sing_idx = findfirst(Ag_svd.S .< tol)

#         # If singular values are nearly singular, truncate at a certain threshold
#         # and fill in the rest with zeros
#         if sing_idx !== nothing
#             @warn "Rank difficient, rank = $(sing_idx), tol = $(Ag_svd.S[sing_idx]).\n"
#             foo = [1 ./ Ag_svd.S[1:sing_idx-1]; zeros(length(Ag_svd.S[sing_idx:end]))]
#             bar = Ag_svd.Vt' * Diagonal(foo) * Ag_svd.U'
#             return bar * (A' * b)
#         else
#             @info "No singular values below the threshold. Fall back to standard solve."
#         end
#     end

#     Γsq = sqrt.(Γ)
#     Atilde = vcat(A, Γsq)
#     btilde = vcat(b, zeros(size(Γsq, 1), size(b, 2)))
#     return Atilde \ btilde
#     # return (A' * A + Γ) \ (A' * b)    # This is if || Γ^{1/2}*O ||_2 is desired

#     # return (A' * A + Γ' * Γ) \ (A' * b)  # This is if || Γ*O ||_F is desired
# end


# """
#     tikhonov(b::AbstractArray, A::AbstractArray, Γ::AbstractMatrix, tol::Real;
#         tol_flag::Bool=false, use_gpu::Bool=false, use_backslash::Bool=true)

# Tikhonov regression to solve the operator inference problem.

# ## Features
# - This function solves the regression problem using the Tikhonov regularization method.
# - The function uses the LinearSolve.jl package to solve the regression problem.
# - The function uses a manual SVD-based truncation if the singular values are below the tolerance `tol`.
#   To enable this feature, set `tol_flag=true`. This is generally not recommended.
# - If the direct solve fails due to memory issues, it switches to the iterative approach (Krylov GMRES).
# - The function also supports GPU acceleration using CUDA.jl (Windows/Linux) or Metal.jl (Apple M series).
#   To enable GPU acceleration, set `use_gpu=true`.
# - The function also supports using the backslash operator for the regression solve (default: true).
#   To enable this feature, set `use_backslash=true`.

# ## Arguments
# - `b::AbstractArray`: right hand side of the regression problem 
# - `A::AbstractArray`: left hand side of the regression problem 
# - `Γ::AbstractMatrix`: Tikhonov matrix 
# - `tol::Real`: tolerance for the singular values 
# - `tol_flag::Bool`: flag for the tolerance (not recommended)
# - `use_gpu::Bool`: flag for GPU acceleration
# - `use_backslash::Bool`: flag for using the backslash operator

# ## Returns
# - regression solution
# """
# function tikhonov(b::AbstractArray, A::AbstractArray, Γ::AbstractMatrix, tol::Real;
#                    tol_flag::Bool=false, use_gpu::Bool=false, use_backslash::Bool=true)
#     # If the tolerance flag is set, perform SVD-based truncation where the singular 
#     # values are below the tolerance level are truncated to zero and the rest are 
#     # filled with zeros. This is a manual way to truncate the singular values.
#     let 
#         # if tol_flag
#         #     try
#         #         # Define the key quantities
#         #         M = A' * A + Γ     # (desired norm: || Γ^(1/2)*O ||_2)
#         #         bhat = A' * b

#         #         # Perform SVD-based truncation if singular values are below tol
#         #         M_svd = svd(M)
#         #         sing_idx = findfirst(M_svd.S .< tol)
#         #         if sing_idx !== nothing
#         #             @warn "Rank deficient, rank = $(sing_idx), tol = $(M_svd.S[sing_idx])."
#         #             invSV = [1 ./ M_svd.S[1:sing_idx-1]; zeros(length(M_svd.S[sing_idx:end]))]
#         #             pinvM = M_svd.Vt' * Diagonal(invSV) * M_svd.U'
#         #             return pinvM * bhat
#         #         else
#         #             @info "No singular values below the threshold. Fall back to standard solve."
#         #         end
#         #     catch e
#         #         if isa(e, OutOfMemoryError)
#         #             @warn string("OutOfMemory encountered when `with_tol=true`. Switching to backslash methods vector. ", 
#         #                          "Truncation based on singular values will no longer be performed.")
#         #         else
#         #             rethrow(e)
#         #         end
#         #     end 
#         # end

#         if tol_flag
#             try
#                 # Define the key quantities
#                 M = A' * A + Γ     # (desired norm: || Γ^(1/2)*O ||_2)
#                 bhat = A' * b

#                 # Use a more numerically stable SVD approach
#                 M_svd = svd(M)
                
#                 # Find the effective rank using relative tolerance
#                 max_sv = M_svd.S[1]
#                 effective_rank = count(s -> s > tol * max_sv, M_svd.S)
                
#                 if effective_rank < length(M_svd.S)
#                     @warn "Rank deficient, effective rank = $effective_rank/$(length(M_svd.S)), relative tol = $(tol)."
                    
#                     # More stable pseudoinverse using only significant singular values
#                     inv_S = zeros(length(M_svd.S))
#                     inv_S[1:effective_rank] = 1 ./ M_svd.S[1:effective_rank]
                    
#                     # Compute pseudoinverse more efficiently
#                     # pinvM = V * Diagonal(inv_S) * U'
#                     # pinvM * bhat = V * (inv_S .* (U' * bhat))
#                     return M_svd.V * (inv_S .* (M_svd.U' * bhat))
#                 else
#                     @info "No singular values below the threshold. Fall back to standard solve."
#                 end
#             catch e
#                 if isa(e, OutOfMemoryError)
#                     @warn string("OutOfMemory encountered when `with_tol=true`. Switching to backslash methods. ", 
#                                 "Truncation based on singular values will no longer be performed.")
#                 else
#                     rethrow(e)
#                 end
#             end 
#         end
#     end

#     # Γsq = sqrt.(Γ)
#     # Atilde = vcat(A, Γsq)
#     # btilde = vcat(b, zeros(size(Γsq, 1), size(b, 2)))
#     # return Atilde \ btilde

#     # Tikhonov regularization with augumented matrix
#     # [A; Γ^(1/2)] * O = [b; 0]
#     # O = ([A; Γ^(1/2)]^⊤ [A; Γ^(1/2)])^(-1) [A; Γ^(1/2)] [b; 0]
#     Γsq = sqrt.(Γ)
#     if use_gpu  # GPU
#         if Sys.isapple()
#             @info "GPU computation requested on macOS. Using Metal.jl."
#             # Safety net: Ensure that a Metal device (expected for M series) is available
#             metal_devs = Metal.devices()
#             has_compatible = any(dev -> occursin("Apple M", string(dev)), metal_devs)
#             @assert has_compatible "Metal.jl is only available for Apple M series GPUs."
#             # Construct the augmented matrix with the Tikhonov matrix
#             Atilde = Metal.CuArray(vcat(A, Γsq))
#             btilde = Metal.CuArray(vcat(b, zeros(size(Γsq, 1), size(b, 2))))
#         else
#             if CUDA.has_cuda()
#                 @info "GPU computation requested. Using CUDA.jl."
#                 # Construct the augmented matrix with the Tikhonov matrix
#                 Atilde = CUDA.CuArray(vcat(A, Γsq))
#                 btilde = CUDA.CuArray(vcat(b, zeros(size(Γsq, 1), size(b, 2))))
#             else
#                 @warn "CUDA GPU not available on this machine. Falling back to CPU"
#                 use_gpu = false
#             end
#         end
#     else  # CPU
#         Atilde = vcat(A, Γsq)
#         btilde = vcat(b, zeros(size(Γsq, 1), size(b, 2)))
#     end

#     # Solve using the backslash operator
#     # Works for CUDA/Metal as well
#     if use_backslash || use_gpu
#         try 
#             if use_gpu
#                 O = Atilde \ btilde  # Operator matrix solution
#                 return Array(O)
#             else
#                 return Atilde \ btilde  # Operator matrix solution
#             end
#         catch e 
#             if isa(e, OutOfMemoryError)
#                 @warn "OutOfMemory with backslash least squares solve. Switching to LinearSolve.jl approach."
#             elseif isa(e,  SparseArrays.CHOLMOD.CHOLMODException)
#                 @warn "Sparse array CHOLMOD encountered out of memory. Switching to LinearSolve.jl approach."
#             else
#                 rethrow(e)
#             end
#             @assert !use_gpu "Disable `use_gpu` to switch to LinearSolve.jl approach."
#         end
#     end

#     # Solve using LinearSolve.jl 
#     O = similar(A, size(A, 2), size(b, 2))
#     try
#         # Try solving the least squares problem directly:
#         # Finds O such that D*O ≈ Rt.
#         # ATTENTION: LinearSolve.jl works for only vector right-hand side
#         # so we need to solve for each column of Rt separately in a loop.
#         ls = nothing
#         for i in axes(btilde, 2)  
#             if i == 1
#                 prob = LinearSolve.LinearProblem(Atilde, view(btilde, :, i))
#                 ls = LinearSolve.init(prob)
#             else # reuse the linear problem
#                 ls.b .= view(btilde, :, i)
#             end
#             sol = LinearSolve.solve(ls)
#             O[:,i] .= sol.u
#         end
#     catch e
#         if isa(e, OutOfMemoryError) || isa(e,  SparseArrays.CHOLMOD.CHOLMODException)
#             @warn "OutOfMemory in direct least squares solve. Switching to memory-efficient vector version."
#             # Solve normal equations: (D' * D + Γ) x = D' * Rt.
#             n = size(A, 2)
#             op = let  # construct a linear operator to reduce memory usage
#                 f = (u,p,t) -> Atilde' * (Atilde * u)
#                 f = (du,u,p,t) -> (mul!(du,Atilde,u); du .= Atilde' * du)
#                 SciMLOperators.FunctionOperator(f, spzeros(n), spzeros(n))
#             end
#             ls2 = nothing
#             for i in axes(btilde, 2)
#                 if i == 1
#                     prob2 = LinearSolve.LinearProblem(op, view(btilde, :, i))
#                     ls2 = LinearSolve.init(prob2)
#                 else
#                     ls2.b .= view(btilde, :, i)
#                 end
#                 sol2 = LinearSolve.solve(ls2, LinearSolve.KrylovJL_GMRES())
#                 O[:,i] .= sol2.u
#             end
#         else
#             rethrow(e)
#         end
#     end
#     return O  # Operator matrix solution
# end
