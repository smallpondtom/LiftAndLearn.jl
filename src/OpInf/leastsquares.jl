"""
$(TYPEDEF)

Configuration of the least-squares solver for ``\\mathbf{D}\\mathbf{O} \\approx \\mathbf{R}^\\top``
used by `standard_least_squares`. Internal helper.

Construct with the keyword constructor `LeastSquaresSolver{T}(; kwargs...)`, whose keywords
and defaults are the fields listed below.

## Fields
- `use_gpu::Bool=false`: request GPU acceleration.
- `use_normal_equations::Bool=false`: use the matrix-free normal-equations solver.
- `chunk_size::Int=1000`: number of right-hand-side columns solved per batch.
- `tolerance::T=1e-12`: absolute tolerance of the iterative solver.
- `algorithm::Union{Function,Nothing}=nothing`: LinearSolve.jl algorithm (`nothing` = LinearSolve.jl default).
- `max_iterations::Int=1000`: maximum number of iterations of the iterative solver.
"""
struct LeastSquaresSolver{T}
    use_gpu::Bool
    use_normal_equations::Bool
    chunk_size::Int
    tolerance::T
    algorithm::Union{Function,Nothing}
    max_iterations::Int
    
    function LeastSquaresSolver{T}(; 
        use_gpu::Bool=false, use_normal_equations::Bool=false, 
        chunk_size::Int=1000, tolerance::T=1e-12, 
        algorithm::Union{Function,Nothing}=nothing,
        max_iterations::Int=1000) where T

        new{T}(
            use_gpu, use_normal_equations, chunk_size, 
            tolerance, algorithm, max_iterations
        )
    end
end

"""
$(SIGNATURES)

Construct a `LeastSquaresSolver{Float64}`; the keywords are the fields of
`LeastSquaresSolver`. Internal helper.
"""
LeastSquaresSolver(; kwargs...) = LeastSquaresSolver{Float64}(; kwargs...)

"""
$(SIGNATURES)

Move `D` and `Rt` to the GPU if `use_gpu` is `true` and a functional GPU backend
(Metal.jl on macOS, CUDA.jl otherwise) is available. Internal helper.

## Arguments
- `D::AbstractArray{T}`: data matrix.
- `Rt::AbstractArray{T}`: right-hand-side matrix.
- `use_gpu::Bool`: whether GPU computation is requested.

## Returns
- `D_out`: `D` (as a GPU array if the GPU is active).
- `Rt_out`: `Rt` (as a GPU array if the GPU is active).
- `gpu_active::Bool`: `true` if the arrays were moved to the GPU; falls back to the CPU with a warning otherwise.
"""
function setup_gpu_arrays(D::AbstractArray{T}, Rt::AbstractArray{T}, 
                          use_gpu::Bool) where T
    if !use_gpu
        return D, Rt, false
    end
    
    if Sys.isapple()
        if Metal.functional()
            @info "Using Metal.jl for GPU acceleration"
            return Metal.MetalArray(D), Metal.MetalArray(Rt), true
        else
            @warn "Metal.jl not functional. Falling back to CPU"
            return D, Rt, false
        end
    else
        if CUDA.functional()
            @info "Using CUDA.jl for GPU acceleration"
            return CUDA.CuArray(D), CUDA.CuArray(Rt), true
        else
            @warn "CUDA not functional. Falling back to CPU"
            return D, Rt, false
        end
    end
end

"""
$(SIGNATURES)

Solve ``\\mathbf{D}\\mathbf{O} \\approx \\mathbf{R}^\\top`` column by column with a direct
LinearSolve.jl solver, reusing the factorization of `D` for every right-hand side.
Internal helper.

## Arguments
- `D::AbstractArray{T}`: data matrix of size ``K \\times d``.
- `Rt::AbstractArray{T}`: right-hand-side matrix of size ``K \\times p``.
- `solver::LeastSquaresSolver{T}`: solver configuration (`algorithm`, `chunk_size`).

## Returns
- `O::AbstractMatrix{T}`: least-squares solution of size ``d \\times p``.
"""
function solve_direct_batch(D::AbstractArray{T}, Rt::AbstractArray{T}, 
                            solver::LeastSquaresSolver{T}) where T
    m, n = size(D)
    _, p = size(Rt)
    
    # Pre-allocate output matrix
    O = similar(D, n, p)
    
    # Process in chunks to manage memory
    chunk_size = min(solver.chunk_size, p)
    
    # Pre-allocate factorization for reuse
    prob = LinearProblem(D, view(Rt, :, 1))
    ls = init(prob, solver.algorithm)
    
    for start_idx in 1:chunk_size:p
        end_idx = min(start_idx + chunk_size - 1, p)
        chunk_range = start_idx:end_idx
        
        # Solve for current chunk
        for (local_idx, global_idx) in enumerate(chunk_range)
            if global_idx == 1
                # First solve - already initialized
                sol = solve!(ls)
            else
                # Reuse factorization, just update RHS
                ls.b .= view(Rt, :, global_idx)
                sol = solve!(ls)
            end
            O[:, global_idx] .= sol.u
        end
    end
    
    return O
end

"""
$(SIGNATURES)

Solve the normal equations ``\\mathbf{D}^\\top\\mathbf{D}\\mathbf{O} = \\mathbf{D}^\\top\\mathbf{R}^\\top``
column by column with an iterative LinearSolve.jl solver, applying
``\\mathbf{D}^\\top\\mathbf{D}`` matrix-free (it is never formed). Internal helper.

## Arguments
- `D::AbstractArray{T}`: data matrix of size ``K \\times d``.
- `Rt::AbstractArray{T}`: right-hand-side matrix of size ``K \\times p``.
- `solver::LeastSquaresSolver{T}`: solver configuration (`algorithm`, `tolerance`, `max_iterations`).

## Returns
- `O::AbstractMatrix{T}`: solution of size ``d \\times p``.

## Notes
- Accuracy is limited by the iterative solver tolerance and by the squared condition number of `D`.
"""
function solve_normal_equations(D::AbstractArray{T}, Rt::AbstractArray{T}, 
                                solver::LeastSquaresSolver{T}) where T
    m, n = size(D)
    _, p = size(Rt)
    
    # Compute D'*Rt once (more efficient than computing D'*D)
    DtRt = D' * Rt
    
    # Pre-allocate output
    O = similar(D, n, p)

    # Preallocate intermediate matrix 
    tmp = zeros(T, m)

    # Create efficient operator for D'*D without storing the full matrix
    function matvec!(y, x, p, t)
        mul!(tmp, D, x)      # y = D*x
        mul!(y, D', tmp)     # y = D'*(D*x) = D'*D*x
    end

    # Create prototype arrays for input and output
    input_prototype = zeros(T, n)
    output_prototype = zeros(T, n)
    
    # Use the correct FunctionOperator constructor
    op = FunctionOperator(matvec!, input_prototype, output_prototype; 
                          isinplace=true, issymmetric=true)
    
    # Solve normal equations for each column
    ls = nothing
    for i in 1:p
        if i == 1
            prob = LinearProblem(op, view(DtRt, :, i))
            ls = init(prob, solver.algorithm) 
        else
            ls.b .= view(DtRt, :, i)
        end
        sol = solve!(ls; maxiters=solver.max_iterations, abstol=solver.tolerance)
        O[:, i] .= sol.u
    end
    
    return O
end

"""
$(SIGNATURES)

Solve ``\\mathbf{D}\\mathbf{O} \\approx \\mathbf{R}^\\top`` with a single QR decomposition
``\\mathbf{D} = \\mathbf{Q}\\mathbf{R}`` and batched triangular solves over chunks of
right-hand-side columns. Internal helper.

## Arguments
- `D::AbstractArray{T}`: data matrix of size ``K \\times d`` (assumed full column rank).
- `Rt::AbstractArray{T}`: right-hand-side matrix of size ``K \\times p``.
- `solver::LeastSquaresSolver{T}`: solver configuration (`chunk_size`).

## Returns
- `O::AbstractMatrix{T}`: least-squares solution of size ``d \\times p``.
"""
function solve_qr_batch(D::AbstractArray{T}, Rt::AbstractArray{T}, 
                        solver::LeastSquaresSolver{T}) where T
    m, n = size(D)
    _, p = size(Rt)
    
    # Compute QR decomposition once
    Q, R = qr(D)
    
    # Pre-allocate output
    O = similar(D, n, p)
    
    # Solve R*O = Q'*Rt efficiently
    QtRt = Q' * Rt

    # Extract the upper triangular part that matches R's dimensions
    # R is n×n, so we need the first n rows of QtRt
    QtRt_reduced = view(QtRt, 1:n, :)
    
    # Batch solve triangular systems
    chunk_size = min(solver.chunk_size, p)
    
    for start_idx in 1:chunk_size:p
        end_idx = min(start_idx + chunk_size - 1, p)
        chunk_range = start_idx:end_idx
        
        # Solve triangular system for chunk
        O[:, chunk_range] .= R \ view(QtRt_reduced, :, chunk_range)
    end
    
    return O
end

"""
$(SIGNATURES)

Solve the least-squares problem ``\\min_{\\mathbf{O}} \\|\\mathbf{D}\\mathbf{O} - \\mathbf{R}^\\top\\|_F``
with automatic selection of the solution method.

With ``\\mathbf{D}`` of size ``m \\times n`` and ``\\mathbf{R}^\\top`` of size ``m \\times p``,
the method is chosen in this order:
1. backslash `D \\ Rt` if the GPU is active or `use_backslash=true`;
2. matrix-free normal equations (`solve_normal_equations`) if `use_normal_equations=true`
   and ``m > 3n``;
3. QR decomposition with batched triangular solves (`solve_qr_batch`) if ``p >`` `chunk_size`;
4. otherwise a direct LinearSolve.jl solve reusing the factorization (`solve_direct_batch`).

If an `OutOfMemoryError` occurs, the solver falls back to the normal-equations method on
the CPU.

## Arguments
- `D::AbstractArray{T}`: data matrix ``\\mathbf{D}`` of size ``m \\times n``.
- `Rt::AbstractArray{T}`: right-hand-side matrix ``\\mathbf{R}^\\top`` of size ``m \\times p``.

## Keyword Arguments
- `use_gpu::Bool=false`: use the GPU (CUDA.jl or Metal.jl) if functional; the backslash solve is used on the GPU.
- `use_normal_equations::Bool=false`: use the normal-equations method for strongly overdetermined systems (``m > 3n``).
- `chunk_size::Int=400`: number of right-hand-side columns per batch; also the threshold on ``p`` above which the QR method is used.
- `tolerance::Real=1e-12`: absolute tolerance of the iterative (normal-equations) solver.
- `use_backslash::Bool=false`: solve with the backslash operator.
- `algorithm::Union{Function,Nothing}=nothing`: LinearSolve.jl algorithm for the direct and normal-equations solves (`nothing` = LinearSolve.jl default).
- `estimate_memory::Bool=false`: log an estimate of the memory usage (see `estimate_memory_usage`).
- `max_iterations::Int=1000`: maximum number of iterations of the iterative solver.

## Returns
- `O::AbstractMatrix{T}`: solution of size ``n \\times p`` such that ``\\mathbf{D}\\mathbf{O} \\approx \\mathbf{R}^\\top``.

## Notes
- `LSOpInfOption` uses different defaults (`use_backslash=true`, `chunk_size=1000`) when this function is called from `opinf`.
"""
function standard_least_squares(D::AbstractArray{T}, Rt::AbstractArray{T}; 
                                use_gpu::Bool=false,
                                use_normal_equations::Bool=false,
                                chunk_size::Int=400,
                                tolerance::Real=1e-12,
                                use_backslash::Bool=false,
                                algorithm::Union{Function,Nothing}=nothing,
                                estimate_memory::Bool=false,
                                max_iterations::Int=1000) where T
    
    # Input validation
    size(D, 1) == size(Rt, 1) || throw(DimensionMismatch(
        "D and Rt must have same number of rows"))
    
    # Create solver configuration
    solver = LeastSquaresSolver{T}(; 
        use_gpu, use_normal_equations, chunk_size, 
        tolerance, algorithm, max_iterations
    )

    # Materialize lazy Adjoint/Transpose wrappers for the LinearSolve.jl-based paths
    # (LinearSolve.jl's default algorithm fails on a rectangular Adjoint matrix)
    if !use_backslash
        D = D isa Union{Adjoint,Transpose} ? Matrix(D) : D
        Rt = Rt isa Union{Adjoint,Transpose} ? Matrix(Rt) : Rt
    end

    # Setup GPU arrays if requested
    D_compute, Rt_compute, gpu_active = setup_gpu_arrays(D, Rt, use_gpu)

    if estimate_memory
        mem_usage = estimate_memory_usage(D_compute, Rt_compute)
        @info "Estimated memory usage: $(mem_usage.base / 1e6) MB (base), " *
              "$(mem_usage.qr_method / 1e6) MB (QR method), " *
              "$(mem_usage.normal_equations / 1e6) MB (Normal equations)"
    end
    
    try
        # Choose optimal method based on problem characteristics
        m, n = size(D_compute)
        _, p = size(Rt_compute)
        
        if gpu_active || use_backslash 
            # GPU: Use built-in backslash operator (most efficient)
            @info "Using backslash for least squares solve"
            O = D_compute \ Rt_compute
            return Array(O)
        elseif use_normal_equations && (m > 3n)  # Overdetermined system
            # Use normal equations for very overdetermined systems
            @info "Using normal equations method for overdetermined system"
            O = solve_normal_equations(D_compute, Rt_compute, solver)
        elseif p > chunk_size  # Many right-hand sides
            # Use QR with batching for multiple RHS
            @info "Using QR decomposition with batching for multiple RHS"
            O = solve_qr_batch(D_compute, Rt_compute, solver)
        else
            # Use direct method with LinearSolve.jl for small to medium problems
            @info "Using direct method with batching for least squares solve"
            O = solve_direct_batch(D_compute, Rt_compute, solver)
        end
        
        return O
        
    catch e
        if isa(e, OutOfMemoryError)
            @warn "Out of memory. Switching to normal equations method."
            # Fallback to most memory-efficient method
            if gpu_active
                D_compute, Rt_compute = Array(D_compute), Array(Rt_compute)
            end
            solver_fallback = LeastSquaresSolver{T}(; 
                use_normal_equations=true, chunk_size=min(100, chunk_size))
            return solve_normal_equations(D_compute, Rt_compute, solver_fallback)
        else
            rethrow(e)
        end
    end
end

# # Convenience method for different numeric types
# standard_least_squares(D::AbstractArray, Rt::AbstractArray; kwargs...) = 
#     standard_least_squares(convert_to_compatible_types(D, Rt)...; kwargs...)

"""
$(SIGNATURES)

Convenience method of `standard_least_squares` for inputs with different element types or
`Adjoint` wrappers: materializes adjoints, promotes `D` and `Rt` to a common element type
and calls the typed method with the same keyword arguments.

## Arguments
- `D::AbstractArray`: data matrix of size ``m \\times n``.
- `Rt::AbstractArray`: right-hand-side matrix of size ``m \\times p``.

## Keyword Arguments
- `kwargs...`: forwarded to the typed `standard_least_squares` method.

## Returns
- `O::AbstractMatrix`: solution of size ``n \\times p``.
"""
function standard_least_squares(D::AbstractArray, Rt::AbstractArray; kwargs...)
    # Convert Adjoint to Matrix to avoid promotion issues
    D_matrix = D isa Adjoint ? Matrix(D) : D
    Rt_matrix = Rt isa Adjoint ? Matrix(Rt) : Rt
    
    # Promote element types
    T = promote_type(eltype(D_matrix), eltype(Rt_matrix))
    D_promoted = convert(AbstractMatrix{T}, D_matrix)
    Rt_promoted = convert(AbstractMatrix{T}, Rt_matrix)
    
    return standard_least_squares(D_promoted, Rt_promoted; kwargs...)
end

"""
$(SIGNATURES)

Estimate the memory (in bytes) required by the least-squares problem
``\\mathbf{D}\\mathbf{O} \\approx \\mathbf{R}^\\top``. Internal helper.

## Arguments
- `D::AbstractArray{T}`: data matrix of size ``m \\times n``.
- `Rt::AbstractArray{T}`: right-hand-side matrix of size ``m \\times p``.

## Returns
- `mem::NamedTuple`: `(base, qr_method, normal_equations)` where `base = sizeof(T)*(mn + mp + np)`,
  `qr_method = base + sizeof(T)*mn` and `normal_equations = base + sizeof(T)*n^2`.
"""
function estimate_memory_usage(D::AbstractArray{T}, Rt::AbstractArray{T}) where T
    m, n = size(D)
    _, p = size(Rt)
    
    # Base memory for matrices
    base_memory = sizeof(T) * (m * n + m * p + n * p)
    
    # Additional memory for different methods
    qr_memory = sizeof(T) * m * n  # Q matrix storage
    normal_eq_memory = sizeof(T) * n * n  # D'*D matrix
    
    return (
        base = base_memory,
        qr_method = base_memory + qr_memory,
        normal_equations = base_memory + normal_eq_memory
    )
end





#==============================================================================#
#==================== Old code left for deubbging purposes ====================#
#==============================================================================#

# """
#     standard_least_squares(D::AbstractArray, Rt::AbstractArray; use_gpu::Bool=false, 
#                            use_backslash::Bool=false)

# Solve the standard least squares problem. Finds O such that D*O ≈ Rt. 

# ## Features
# - This function utilizes the LinearSolve.jl package to solve the least squares problem. 
# - If the direct solve fails due to memory issues, it switches to the iterative approach (Krylov GMRES).
# - The function also supports GPU acceleration using CUDA.jl (Windows/Linux) or Metal.jl (Apple M series).
#   To enable GPU acceleration, set `use_gpu=true`.
# - The function also supports using the backslash operator for the least-squares solve (default: true).
#   To enable this feature, set `use_backslash=true`.
# - Note that when using GPU acceleration, the function uses the backslash operator for the least-squares solve.
#   But the computation is done on the GPU. This is due to some implementation issues using LinearSolve.jl.

# ## Arguments
# - `D::AbstractArray`: data matrix
# - `Rt::AbstractArray`: derivative data matrix
# - `use_gpu::Bool`: use GPU for least-squares solve (default: false)
# - `use_backslash::Bool`: use backslash operator for least-squares solve (default: true)

# ## Returns
# - operator matrix solution `O`
# """
# function standard_least_squares(D::AbstractArray, Rt::AbstractArray; 
#                                 use_gpu::Bool=false, use_backslash::Bool=true)
#     if use_gpu
#         if Sys.isapple()
#             @info "GPU least squares requested on macOS. Using Metal.jl."
#             metal_devs = Metal.devices()
#             has_compatible = any(dev -> occursin("Apple M", string(dev)), metal_devs)
#             @assert has_compatible "Metal.jl is only available for Apple M series GPUs."
#             D = Metal.MetalArray(D)
#             Rt = Metal.MetalArray(Rt)
#         else
#             @info "GPU least squares requested. Using CUDA.jl."
#             if CUDA.has_cuda()
#                 D = CUDA.CuArray(D)
#                 Rt = CUDA.CuArray(Rt)
#             else
#                 @warn "CUDA GPU not available on this machine. Falling back to CPU"
#                 use_gpu = false
#             end
#         end
#     end

#     # Solve using the backslash operator
#     # Works for CUDA/Metal as well
#     if use_backslash || use_gpu
#         try 
#             if use_gpu
#                 O = D \ Rt  # Operator matrix solution
#                 return Array(O)
#             else
#                 return D \ Rt  # Operator matrix solution
#             end
#         catch e 
#             if isa(e, OutOfMemoryError)
#                 @warn "OutOfMemory with backslash least squares solve. Switching to LinearSolve.jl approach."
#                 @assert !use_gpu "Disable `use_gpu` to switch to LinearSolve.jl approach."
#             else
#                 rethrow(e)
#             end
#         end
#     end

#     # Solve using LinearSolve.jl 
#     O = similar(D, size(D, 2), size(Rt, 2))
#     try
#         # Try solving the least squares problem directly:
#         # Finds O such that D*O ≈ Rt.
#         # ATTENTION: LinearSolve.jl works for only vector right-hand side
#         # so we need to solve for each column of Rt separately in a loop.
#         ls = nothing
#         for i in axes(Rt, 2)  
#             if i == 1
#                 prob = LinearSolve.LinearProblem(D, view(Rt, :, i))
#                 ls = LinearSolve.init(prob)
#             else # reuse the linear problem
#                 ls.b .= view(Rt, :, i)
#             end
#             sol = LinearSolve.solve(ls)
#             O[:,i] .= sol.u
#         end
#     catch e
#         if isa(e, OutOfMemoryError) 
#             @warn "OutOfMemory in direct least squares solve. Switching to memory-efficient vector version."
#             # Solve normal equations: (D' * D) x = D' * Rt.
#             btilde = D' * Rt
#             n = size(D, 2)
#             op = let  # construct a linear operator to reduce memory usage
#                 f = (u,p,t) -> D' * (D * u)
#                 f = (du,u,p,t) -> (mul!(du,D,u); du .= D' * du)
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