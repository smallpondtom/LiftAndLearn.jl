"""
    LeastSquaresSolver

A high-performance solver for least squares problems D*O ≈ Rt with multiple 
optimization strategies.
"""
struct LeastSquaresSolver{T}
    use_gpu::Bool
    use_normal_equations::Bool
    chunk_size::Int
    tolerance::T
    algorithm::Union{Function,Nothing}
    
    function LeastSquaresSolver{T}(; 
        use_gpu::Bool=false, use_normal_equations::Bool=false, 
        chunk_size::Int=1000, tolerance::T=1e-12, 
        algorithm::Union{Function,Nothing}=nothing) where T

        new{T}(use_gpu, use_normal_equations, chunk_size, tolerance, algorithm)
    end
end

LeastSquaresSolver(; kwargs...) = LeastSquaresSolver{Float64}(; kwargs...)

"""
    setup_gpu_arrays(D, Rt, use_gpu)

Efficiently transfer arrays to GPU if requested and available.
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
    solve_direct_batch(D, Rt, solver)

Solve using direct method with batching for memory efficiency.
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
    solve_normal_equations(D, Rt, solver)

Solve using normal equations approach for better memory efficiency.
"""
function solve_normal_equations(D::AbstractArray{T}, Rt::AbstractArray{T}, 
                                solver::LeastSquaresSolver{T}) where T
    m, n = size(D)
    _, p = size(Rt)
    
    # Compute D'*Rt once (more efficient than computing D'*D)
    DtRt = D' * Rt
    
    # Pre-allocate output
    O = similar(D, n, p)
    
    # Create efficient operator for D'*D without storing the full matrix
    op = let D = D
        function matvec!(y, x, p, t)
            mul!(y, D, x)      # y = D*x
            mul!(y, D', y)     # y = D'*(D*x) = D'*D*x
        end
        LinearOperator{T}(matvec!, n, n; ismutating=true, issymmetric=true)
    end
    
    # Solve normal equations for each column
    ls = nothing
    for i in 1:p
        if i == 1
            prob = LinearProblem(op, view(DtRt, :, i))
            ls = init(prob, KrylovJL_CG())  # Use CG for symmetric positive definite
        else
            ls.b .= view(DtRt, :, i)
        end
        sol = solve(ls)
        O[:, i] .= sol.u
    end
    
    return O
end

"""
    solve_qr_batch(D, Rt, solver)

Solve using QR decomposition with efficient batching.
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
    
    # Batch solve triangular systems
    chunk_size = min(solver.chunk_size, p)
    
    for start_idx in 1:chunk_size:p
        end_idx = min(start_idx + chunk_size - 1, p)
        chunk_range = start_idx:end_idx
        
        # Solve triangular system for chunk
        O[:, chunk_range] .= R \ view(QtRt, :, chunk_range)
    end
    
    return O
end

"""
    standard_least_squares(D::AbstractArray, Rt::AbstractArray; kwargs...)

High-performance least squares solver with automatic method selection.

## Arguments
- `D::AbstractArray`: Data matrix (m × n)
- `Rt::AbstractArray`: Target matrix (m × p)
- `use_gpu::Bool`: Enable GPU acceleration (default: false)
- `use_normal_equations::Bool`: Force normal equations method (default: false)
- `chunk_size::Int`: Batch size for memory management (default: 1000)
- `tolerance::Real`: Numerical tolerance (default: 1e-12)
- `algorithm::Union{Function,Nothing}`: Custom solver method (default: nothing)

## Returns
- `O::AbstractArray`: Solution matrix (n × p) such that D*O ≈ Rt
"""
function standard_least_squares(D::AbstractArray{T}, Rt::AbstractArray{T}; 
                                use_gpu::Bool=false,
                                use_normal_equations::Bool=false,
                                chunk_size::Int=400,
                                tolerance::Real=1e-12,
                                use_backslash::Bool=false,
                                algorithm::Union{Function,Nothing}=nothing,
                                estimate_memory::Bool=false) where T
    
    # Input validation
    size(D, 1) == size(Rt, 1) || throw(DimensionMismatch(
        "D and Rt must have same number of rows"))
    
    # Create solver configuration
    solver = LeastSquaresSolver{T}(; 
        use_gpu, use_normal_equations, chunk_size, tolerance, algorithm)
    
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
        elseif use_normal_equations || (m > 3n)  # Overdetermined system
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

# Convenience method for different numeric types
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
    estimate_memory_usage(D, Rt)

Estimate memory usage for the least squares problem.
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