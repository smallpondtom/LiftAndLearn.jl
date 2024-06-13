export opinf


"""
    opinf(W::AbstractArray, Vn::AbstractArray, lm::lifting, full_op::operators,
            options::AbstractOption; U::AbstractArray=zeros(1,1), 
            Y::AbstractArray=zeros(1,1), IG::operators=operators()) → op::operators

Infer the operators for Lift And Learn for reprojected data (dispatch). NOTE: make
sure that the data is constructed such that the row dimension is the state dimension
and the column dimension is the time dimension.

## Arguments
- `W::AbstractArray`: state data matrix
- `Vn::AbstractArray`: POD basis
- `lm::lifting`: struct of the lift map
- `full_op::operators`: full order model operators
- `options::AbstractOption`: options for the operator inference defined by the user
- `U::AbstractArray`: input data matrix
- `Y::AbstractArray`: output data matix
- `IG::operators`: initial guesses for optimization

## Returns
- `op::operators`: inferred operators
"""
function opinf(W::AbstractArray, Vn::AbstractArray, lm::lifting, full_op::operators, 
                 options::AbstractOption; U::AbstractArray=zeros(1,1), 
                 Y::AbstractArray=zeros(1,1), IG::operators=operators())::operators
    U = fat2tall(U)
    # Project
    What = Vn' * W
    What_t = W' * Vn

    define_dimensions!(What, U, Y, options)

    # Generate R matrix from finite difference if not provided as a function argument
    if options.optim.reproject == true
        Rt = reproject(What, Vn, U, lm, full_op, options)
    else
        Wdot, idx = dtApprox(W, options)
        W = W[:, idx]  # fix the index of states
        U = U[idx, :]  # fix the index of inputs
        Y = Y[:, idx]  # fix the index of outputs
        R = Vn'Wdot
        Rt = transpose(R)
        What = Vn' * W
        What_t = W' * Vn
    end

    D = getDataMat(What, What_t, U, options)
    op = run_optimizer(D, Rt, Y, What_t, options, IG)

    return op
end



"""
    reproject(Xhat::Matrix, V::Union{VecOrMat,BlockDiagonal}, U::VecOrMat,
        lm::lifting, op::operators, options::AbstractOption) → Rhat::Matrix

Reprojecting the lifted data

## Arguments
- `Xhat::Matrix`: state data matrix projected onto the basis
- `V::Union{VecOrMat,BlockDiagonal}`: POD basis
- `U::VecOrMat`: input data matrix
- `lm::lifting`: struct of the lift map
- `op::operators`: full order model operators
- `options::AbstractOption`: options for the operator inference defined by the user

## Returns
- `Rhat::Matrix`: R matrix (transposed) for the regression problem
"""
function reproject(Xhat::Matrix, V::Union{VecOrMat,BlockDiagonal}, U::VecOrMat,
    lm::lifting, op::operators, options::AbstractOption)::Matrix

    tmp = size(V, 1)
    n = tmp / options.vars.N_lift
    dt = options.data.Δt
    Rt = zeros(options.system.dims[:K], options.system.dims[:n])  # Left hand side of the regression problem

    if options.system.has_funcOp
        f = (x, u) -> op.A * x + op.f(x) + op.B * u + op.K
    else
        m = options.system.dims[:m]

        fA = (x) -> options.system.is_lin ? op.A * x : 0
        fB = (u) -> options.system.has_control ? op.B * u : 0
        fF = (x) -> options.system.is_quad && options.optim.which_quad_term == "F" ? op.F * (x ⊘ x) : 0
        fH = (x) -> options.system.is_quad && options.optim.which_quad_term == "H" ? op.H * (x ⊗ x) : 0
        fE = (x) -> options.system.is_cubic && options.optim.which_cubic_term == "E" ? op.E * ⊘(x,x,x) : 0
        fG = (x) -> options.system.is_cubic && options.optim.which_cubic_term == "G" ? op.G * (x ⊗ x ⊗ x) : 0
        fN = (x,u) -> options.system.is_bilin ? ( m==1 ? (op.N * x) * u : sum([(op.N[i] * x) * u[i] for i in 1:m]) ) : 0
        fK = options.system.has_const ? op.K : 0

        f = (x,u) -> fA(x) .+ fB(u) .+ fF(x) .+ fH(x) .+ fE(x) .+ fG(x) .+ fN(x,u) .+ fK
    end

    for i in 1:options.system.dims[:K]  # loop thru all data
        x = Xhat[:, i]  # julia automatically makes into column vec after indexing (still xrow-tcol)
        xrec = V * x
        states = f(xrec[1:Int(options.vars.N * n), :], U[i, :])

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