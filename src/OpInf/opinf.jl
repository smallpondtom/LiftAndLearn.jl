export opinf

# Import necessary utility functions
include("time_derivative_approx.jl")
include("get_data_matrix.jl")
include("unpack_operators.jl")
include("tikhonov.jl")
include("reproject.jl")

"""
    leastsquares_solve(D::AbstractArray, Rt::AbstractArray, Y::AbstractArray, Xhat_t::AbstractArray, 
             dims::AbstractArray, operator_symbols::AbstractArray, options::AbstractOption)

Solve the standard Operator Inference with/without regularization

## Arguments
- `D::AbstractArray`: data matrix
- `Rt::AbstractArray`: derivative data matrix (tall)
- `Yt::AbstractArray`: output data matrix (tall)
- `Xhat_t::AbstractArray`: projected data matrix (tall)
- `dims::AbstractArray`: dimensions of the operators
- `operator_symbols::AbstractArray`: symbols of the operators
- `options::AbstractOption`: options for the operator inference set by the user

## Returns
- `operators::Operators`: All learned operators
"""
function leastsquares_solve(D::AbstractArray, Rt::AbstractArray, Yt::AbstractArray, Xhat_t::AbstractArray, 
                            dims::AbstractArray, operator_symbols::AbstractArray, options::AbstractOption)
    # Preallocate the Tikhonov weight Matrix
    Γ = spzeros(sum(dims))

    # Construct the Tikhonov matrix
    tikhonov_matrix!(Γ, dims, operator_symbols, options.λ)
    Γ = spdiagm(0 => Γ)  # convert to sparse diagonal matrix

    # compute least squares (pseudo inverse)
    if options.with_reg 
        Ot = tikhonov(Rt, D, Γ, options.pinv_tol; tol_flag=options.with_tol, use_gpu=options.use_gpu, 
                      use_backslash=options.use_backslash)
    else
        # Ot = D \ Rt  # INFO: This is not optimal
        Ot = standard_least_squares(D, Rt; use_gpu=options.use_gpu, use_backslash=options.use_backslash)
    end

    # Extract the operators from the operator matrix O
    O = transpose(Ot)

    # Extract the operators
    operators = Operators(O=O)

    # Unpack the operators
    unpack_operators!(operators, O, Yt, Xhat_t, dims, operator_symbols, options)

    return operators
end


"""
    opinf(X::AbstractArray, Vn::AbstractArray, options::AbstractOption; 
        U::AbstractArray=zeros(1,1), Y::AbstractArray=zeros(1,1),
        Xdot::AbstractArray=[]) → op::Operators

Infer the operators with derivative data given. NOTE: Make sure the data is 
constructed such that the row is the state vector and the column is the time.

## Arguments
- `X::AbstractArray`: state data matrix
- `Vn::AbstractArray`: POD basis
- `options::AbstractOption`: options for the operator inference defined by the user
- `U::AbstractArray`: input data matrix
- `Y::AbstractArray`: output data matix
- `Xdot::AbstractArray`: derivative data matrix

## Returns
- `op::Operators`: inferred operators
"""
function opinf(X::AbstractArray, Vn::AbstractArray, options::AbstractOption; 
               U::AbstractArray=[0.0], Y::AbstractArray=[0.0],
               Xdot::AbstractArray=[])::Operators
    Ut = fat2tall(U)  # make sure that the U-matrix is tall
    Yt = fat2tall(Y)  # make sure that the Y-matrix is tall

    if isempty(Xdot)
        # Approximate the derivative data with finite difference
        Xdot, idx = time_derivative_approx(X, options)
        Xhat = Vn' * X[:, idx]  # fix the index of states
        Xhat_t = Xhat'
        Ut = iszero(Ut) ? [0.0] : Ut[idx, :]  # fix the index of inputs
        Yt = iszero(Yt) ? [0.0] : Yt[idx, :]  # fix the index of outputs
        Rt = Xdot' * Vn
    else
        Xhat = Vn' * X
        Xhat_t = Xhat'
        Rt = Xdot' * Vn  
    end

    D, dims, op_symbols = get_data_matrix(Xhat, Xhat_t, Ut, options; verbose=true)
    op = leastsquares_solve(D, Rt, Yt, Xhat_t, dims, op_symbols, options)
    return op
end


"""
    opinf(X::AbstractArray, Vn::AbstractArray, full_op::Operators, options::AbstractOption;
        U::AbstractArray=zeros(1,1), Y::AbstractArray=zeros(1,1)) → op::Operators

Infer the operators with reprojection method (dispatch). NOTE: Make sure the data is
constructed such that the row is the state vector and the column is the time.

## Arguments
- `X::AbstractArray`: state data matrix
- `Vn::AbstractArray`: POD basis
- `full_op::Operators`: full order model operators
- `options::AbstractOption`: options for the operator inference defined by the user
- `U::AbstractArray`: input data matrix
- `Y::AbstractArray`: output data matix
- `return_derivative::Bool=false`: return the derivative matrix (or residual matrix)

## Returns
- `op::Operators`: inferred operators
"""
function opinf(X::AbstractArray, Vn::AbstractArray, full_op::Operators, options::AbstractOption;
               U::AbstractArray=zeros(1,1), Y::AbstractArray=zeros(1,1), return_derivative::Bool=false)
    Ut = fat2tall(U)
    Yt = fat2tall(Y)

    Xhat = Vn' * X
    Xhat_t = transpose(Xhat)

    # Reproject
    Rt = reproject(Xhat, Vn, Ut, full_op, options)
    D, dims, op_symbols = get_data_matrix(Xhat, Xhat_t, Ut, options; verbose=true)
    op = leastsquares_solve(D, Rt, Yt, Xhat_t, dims, op_symbols, options)

    if return_derivative
        return op, Rt
    else
        return op
    end
end


"""
    standard_least_squares(D::AbstractArray, Rt::AbstractArray; use_gpu::Bool=false, 
                           use_backslash::Bool=false)

Solve the standard least squares problem. Finds O such that D*O ≈ Rt. 

## Features
- This function utilizes the LinearSolve.jl package to solve the least squares problem. 
- If the direct solve fails due to memory issues, it switches to the iterative approach (Krylov GMRES).
- The function also supports GPU acceleration using CUDA.jl (Windows/Linux) or Metal.jl (Apple M series).
  To enable GPU acceleration, set `use_gpu=true`.
- The function also supports using the backslash operator for the least-squares solve (default: true).
  To enable this feature, set `use_backslash=true`.
- Note that when using GPU acceleration, the function uses the backslash operator for the least-squares solve.
  But the computation is done on the GPU. This is due to some implementation issues using LinearSolve.jl.

## Arguments
- `D::AbstractArray`: data matrix
- `Rt::AbstractArray`: derivative data matrix
- `use_gpu::Bool`: use GPU for least-squares solve (default: false)
- `use_backslash::Bool`: use backslash operator for least-squares solve (default: true)

## Returns
- operator matrix solution `O`
"""
function standard_least_squares(D::AbstractArray, Rt::AbstractArray; 
                                use_gpu::Bool=false, use_backslash::Bool=true)
    if use_gpu
        if Sys.isapple()
            @info "GPU least squares requested on macOS. Using Metal.jl."
            metal_devs = Metal.devices()
            has_compatible = any(dev -> occursin("Apple M", string(dev)), metal_devs)
            @assert has_compatible "Metal.jl is only available for Apple M series GPUs."
            D = Metal.MetalArray(D)
            Rt = Metal.MetalArray(Rt)
        else
            @info "GPU least squares requested. Using CUDA.jl."
            if CUDA.has_cuda()
                D = CUDA.CuArray(D)
                Rt = CUDA.CuArray(Rt)
            else
                @warn "CUDA GPU not available on this machine. Falling back to CPU"
                use_gpu = false
            end
        end
    end

    # Solve using the backslash operator
    # Works for CUDA/Metal as well
    if use_backslash || use_gpu
        try 
            if use_gpu
                O = D \ Rt  # Operator matrix solution
                return Array(O)
            else
                return D \ Rt  # Operator matrix solution
            end
        catch e 
            if isa(e, OutOfMemoryError)
                @warn "OutOfMemory with backslash least squares solve. Switching to LinearSolve.jl approach."
                @assert !use_gpu "Disable `use_gpu` to switch to LinearSolve.jl approach."
            else
                rethrow(e)
            end
        end
    end

    # Solve using LinearSolve.jl 
    O = similar(D, size(D, 2), size(Rt, 2))
    try
        # Try solving the least squares problem directly:
        # Finds O such that D*O ≈ Rt.
        # ATTENTION: LinearSolve.jl works for only vector right-hand side
        # so we need to solve for each column of Rt separately in a loop.
        ls = nothing
        for i in axes(Rt, 2)  
            if i == 1
                prob = LinearSolve.LinearProblem(D, view(Rt, :, i))
                ls = LinearSolve.init(prob)
            else # reuse the linear problem
                ls.b .= view(Rt, :, i)
            end
            sol = LinearSolve.solve(ls)
            O[:,i] .= sol.u
        end
    catch e
        if isa(e, OutOfMemoryError) 
            @warn "OutOfMemory in direct least squares solve. Switching to memory-efficient vector version."
            # Solve normal equations: (D' * D) x = D' * Rt.
            btilde = D' * Rt
            n = size(D, 2)
            op = let  # construct a linear operator to reduce memory usage
                f = (u,p,t) -> D' * (D * u)
                f = (du,u,p,t) -> (mul!(du,D,u); du .= D' * du)
                SciMLOperators.FunctionOperator(f, spzeros(n), spzeros(n))
            end
            ls2 = nothing
            for i in axes(btilde, 2)
                if i == 1
                    prob2 = LinearSolve.LinearProblem(op, view(btilde, :, i))
                    ls2 = LinearSolve.init(prob2)
                else
                    ls2.b .= view(btilde, :, i)
                end
                sol2 = LinearSolve.solve(ls2, LinearSolve.KrylovJL_GMRES())
                O[:,i] .= sol2.u
            end
        else
            rethrow(e)
        end
    end
    return O  # Operator matrix solution
end