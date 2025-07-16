"""
    tikhonov_matrix!(Γ::AbstractArray, dims::Dict, options::AbstractOption)

Construct the Tikhonov matrix

## Arguments
- `Γ::AbstractArray`: Tikhonov matrix (pass by reference)
- `options::AbstractOption`: options for the operator inference set by the user

## Returns
- `Γ`: Tikhonov matrix (pass by reference)
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
    TikhonovSolver

High-performance Tikhonov regularized least squares solver with multiple 
optimization strategies.
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

TikhonovSolver(; kwargs...) = TikhonovSolver{Float64}(; kwargs...)

"""
    RegularizationMethod

Abstract type for different regularization approaches.
"""
abstract type RegularizationMethod end

struct AugmentedSystem <: RegularizationMethod end
struct NormalForm <: RegularizationMethod end
struct SVDTruncation <: RegularizationMethod end

"""
    setup_gpu_computation(arrays...; use_gpu::Bool)

Efficiently setup GPU computation with proper error handling.
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
    compute_regularization_matrix(Γ::AbstractMatrix, method::Symbol=:cholesky)

Compute the square root of the regularization matrix using the most efficient method.
"""
function compute_regularization_matrix(Γ::AbstractMatrix{T}, 
                                       method::Symbol=:cholesky) where T
    if method === :cholesky && isposdef(Γ)
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
    solve_augmented_system(A, b, Γ, solver)

Solve using the augmented system approach: [A; Γ^(1/2)] * x = [b; 0]
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
    solve_normal_form(A, b, Γ, solver)

Solve using normal form: (A'*A + Γ) * x = A' * b
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
            return solve_iterative_normal_form(M, Atb, solver)
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
    solve_iterative_normal_form(M, Atb, solver)

Solve normal form using iterative methods with preconditioning.
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
            prob = LinearProblem(M, view(Atb, :, i))
            if solver.preconditioning
                ls = init(prob, KrylovJL_CG(), Pl=P)
            else
                ls = init(prob, KrylovJL_CG())
            end
        else
            ls.b = view(Atb, :, i)
        end
        sol = solve!(ls; maxiters=solver.max_iterations, abstol=solver.tolerance)
        O[:, i] .= sol.u
    end
    
    return O
end

"""
    solve_iterative_tikhonov(A, b, Γ, solver)

Memory-efficient iterative solver using linear operators.
"""
function solve_iterative_tikhonov(A::AbstractMatrix{T}, 
                                  b::AbstractArray{T}, 
                                  Γ::AbstractMatrix{T}, 
                                  solver::TikhonovSolver{T}) where T
    m, n = size(A)
    _, p = size(b)
    
    # Create linear operator for (A'*A + Γ) without storing the full matrix
    op = let A = A, Γ = Γ
        function matvec!(y, x, p, t)
            # y = (A'*A + Γ) * x
            temp = A * x
            mul!(y, A', temp)
            y .+= Γ * x
        end
        LinearOperator{T}(matvec!, n, n; ismutating=true, issymmetric=true)
    end
    
    # Compute A' * b
    Atb = A' * b
    
    # Solve iteratively
    O = similar(A, n, p)
    ls = nothing
    
    for i in 1:p
        if i == 1
            prob = LinearProblem(op, view(Atb, :, i))
            ls = init(prob, KrylovJL_CG())
        else
            ls.b = view(Atb, :, i)
        end
        sol = solve!(ls; maxiters=solver.max_iterations, abstol=solver.tolerance)
        O[:, i] .= sol.u
    end
    
    return O
end

"""
    solve_svd_truncation(A, b, Γ, solver)

Solve using SVD with intelligent truncation for rank-deficient problems.
"""
function solve_svd_truncation(A::AbstractMatrix{T}, 
                              b::AbstractArray{T}, 
                              Γ::AbstractMatrix{T}, 
                              solver::TikhonovSolver{T}) where T
    # Form normal equations matrix
    M = A' * A + Γ
    rhs = A' * b
    
    # Compute SVD
    U, S, Vt = svd(M)
    
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
        return Vt' * (inv_S .* (U' * rhs))
    else
        # Full rank, use standard solve
        return M \ rhs
    end
end

"""
    solve_least_squares_batch(A, b, chunk_size)

Solve least squares with batching for memory efficiency.
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
    select_optimal_method(A, b, Γ, solver)

Automatically select the most efficient solution method based on problem characteristics.
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
    elseif solver.use_normal_form || (m > 2n && p < 100)
        @info "Using normal form method for large overdetermined system"
        return NormalForm(), :normal
    else
        @info "Using augmented system method for general case"
        return AugmentedSystem(), :augmented
    end
end

"""
    tikhonov(b::AbstractArray, A::AbstractArray, Γ::AbstractMatrix; kwargs...)

High-performance Tikhonov regularized least squares solver.

Solves the regularized least squares problem:
    minimize ||A*x - b||₂² + ||Γ^(1/2)*x||₂²

## Arguments
- `b::AbstractArray`: Right-hand side vector/matrix (m × p)
- `A::AbstractArray`: Coefficient matrix (m × n)  
- `Γ::AbstractMatrix`: Regularization matrix (n × n)
- `tol::Real`: Tolerance for SVD truncation (default: 1e-12)
- `use_gpu::Bool`: Enable GPU acceleration (default: false)
- `use_normal_form::Bool`: Force normal equations method (default: false)
- `use_svd_truncation::Bool`: Enable SVD-based truncation (default: false)
- `chunk_size::Int`: Batch size for multiple RHS (default: 1000)
- `max_iterations::Int`: Maximum iterations for iterative methods (default: 1000)
- `estimate_memory::Bool`: Estimate memory usage (default: false)

## Returns
- Solution matrix x (n × p)
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
        else  # AugmentedSystem
            return solve_augmented_system(A, b, Γ, solver)
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

# Convenience method for automatic type promotion
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
    tikhonov_regularization_path(b, A, Γ_vals; kwargs...)

Compute Tikhonov solutions for multiple regularization parameters efficiently.
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
    estimate_tikhonov_memory(A, b, Γ, method)

Estimate memory requirements for different Tikhonov methods.
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
    optimize_regularization_parameter(A, b, Γ_base; method=:gcv, n_params=50)

Find optimal regularization parameter using cross-validation or GCV.
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
"""
This is the old version of the tikhonov function. It is kept here for reference.
Will be archived in the future.
"""
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
