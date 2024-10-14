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
    tikhonovMatrix!(Γ, dims, operator_symbols, options.λ)
    Γ = spdiagm(0 => Γ)  # convert to sparse diagonal matrix

    # compute least squares (pseudo inverse)
    if options.with_reg 
        Ot = tikhonov(Rt, D, Γ, options.pinv_tol; flag=options.with_tol)
    else
        Ot = D \ Rt
    end

    # Extract the operators from the operator matrix O
    O = transpose(Ot)

    # Extract the operators
    operators = Operators()

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
               U::AbstractArray=zeros(1,1), Y::AbstractArray=zeros(1,1),
               Xdot::AbstractArray=[])::Operators
    Ut = fat2tall(U)  # make sure that the U-matrix is tall
    Yt = fat2tall(Y)  # make sure that the Y-matrix is tall

    if isempty(Xdot)
        # Approximate the derivative data with finite difference
        Xdot, idx = time_derivative_approx(X, options)
        Xhat = Vn' * X[:, idx]  # fix the index of states
        Xhat_t = Xhat'
        Ut = iszero(Ut) ? 0 : Ut[idx, :]  # fix the index of inputs
        Yt = iszero(Yt) ? 0 : Yt[idx, :]  # fix the index of outputs
        Rt = Xdot' * Vn
    else
        Xhat = Vn' * X
        Xhat_t = transpose(Xhat)
        Rt = Xdot' * Vn  
        # Rt = transpose(Vn' * Xdot)
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

