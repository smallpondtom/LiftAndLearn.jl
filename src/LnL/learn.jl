export opinf


"""
    opinf(W::AbstractArray, Vn::AbstractArray, lm::lifting, full_op::Operators,
            options::AbstractOption; U::AbstractArray=zeros(1,1), 
            Y::AbstractArray=zeros(1,1), IG::Operators=Operators()) → op::Operators

Infer the operators for Lift And Learn for reprojected data (dispatch). NOTE: make
sure that the data is constructed such that the row dimension is the state dimension
and the column dimension is the time dimension. 

## Arguments
- `W::AbstractArray`: state data matrix
- `Vn::AbstractArray`: POD basis
- `lm::lifting`: struct of the lift map
- `full_op::Operators`: full order model operators
- `options::AbstractOption`: options for the operator inference defined by the user
- `U::AbstractArray`: input data matrix
- `Y::AbstractArray`: output data matix

## Returns
- `op::Operators`: inferred operators

## Note
- You can opt to use the unlifted version of the `opinf` function instead of this dispatch
  if it is not necessary to reproject the data.
"""
function opinf(W::AbstractArray, Vn::AbstractArray, lm::lifting, full_op::Operators, 
               options::AbstractOption; U::AbstractArray=zeros(1,1), 
               Y::AbstractArray=zeros(1,1))::Operators
    Ut = fat2tall(U)
    Yt = fat2tall(Y)

    # Project
    What = Vn' * W
    What_t = W' * Vn

    # Generate R matrix from finite difference if not provided as a function argument
    if options.optim.reproject == true
        Rt = reproject(What, Vn, Ut, lm, full_op, options)
    else
        Wdot, idx = time_derivative_approx(W, options)
        What = Vn' * W[:, idx]  # fix the index of states
        What_t = What'
        Ut = iszero(Ut) ? 0 : Ut[idx, :]  # fix the index of inputs
        Yt = iszero(Yt) ? 0 : Yt[idx, :]  # fix the index of outputs
        Rt = Wdot' * Vn
    end

    D, dims, op_symbols = get_data_matrix(What, What_t, Ut, options, verbose=true)
    op = leastsquares_solve(D, Rt, Yt, What_t, dims, op_symbols, options)

    return op
end



"""
    reproject(Xhat::Matrix, V::Union{VecOrMat,BlockDiagonal}, U::VecOrMat,
        lm::lifting, op::Operators, options::AbstractOption) → Rhat::Matrix

Reprojecting the lifted data

## Arguments
- `Xhat::AbstractArray`: state data matrix projected onto the basis
- `V::AbstractArray`: POD basis
- `Ut::AbstractArray`: input data matrix (tall)
- `lm::lifting`: struct of the lift map
- `op::Operators`: full order model operators
- `options::AbstractOption`: options for the operator inference defined by the user

## Returns
- `Rhat::Matrix`: R matrix (transposed) for the regression problem
"""
function reproject(Xhat::AbstractArray, V::AbstractArray, Ut::AbstractArray,
    lm::lifting, op::Operators, options::AbstractOption)::Matrix

    tmp = size(V, 1)
    n = tmp / options.vars.N_lift
    dt = options.data.Δt
    r, K = size(Xhat)
    Rt = zeros(K, r)  # Left hand side of the regression problem

    # Assuming the user gave the nonlinear functional or the Operator structure predefined the nonlinear functional
    f = (x, u) -> op.A * x .+ op.B * u .+ op.K .+ op.f(x,u) 

    for i in 1:K  # loop thru all data
        x = Xhat[:, i]  # julia automatically makes into column vec after indexing (still xrow-tcol)
        xrec = V * x
        states = f(xrec[1:Int(options.vars.N * n), :], Ut[i, :])

        # Lifted variables
        next = [xrec[Int((j - 1) * n + 1):Int(j * n)] + dt * states[Int((j - 1) * n + 1):Int(j * n)] for j in 1:options.vars.N]
        z = lm.mapNL(next)  # map the nonlifted variables to the lifting functions
        dz = (z - xrec[(Int(options.vars.N * n)+1):Int(options.vars.N_lift * n)]) ./ options.data.Δt

        # Combine the original and lifted variables
        states = vcat(states, dz)

        # Project to obtain derivative data
        Rt[i, :] = V' * states
    end
    return Rt
end