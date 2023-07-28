"""
Learning function file.
"""


"""
Approximating the derivative values of the data with different integration schemes

# Arguments
- `X`: data matrix
- `options`: operator inference options

# Return
- `dXdt`: derivative data
- `idx`: index for the specific integration scheme (important for later use)
"""
function dtApprox(X::VecOrMat, options::Abstract_Options)
    N = size(X, 2)
    choice = options.data.deriv_type

    if choice == "FE"  # Forward Euler
        dXdt = (X[:, 2:end] - X[:, 1:end-1]) / options.data.Δt
        idx = 1:N-1
    elseif choice == "BE"  # Backward Euler
        dXdt = (X[:, 2:end] - X[:, 1:end-1]) / options.data.Δt
        idx = 2:N
    elseif choice == "SI"  # Semi-implicit Euler
        dXdt = (X[:, 2:end] - X[:, 1:end-1]) / options.data.Δt
        idx = 2:N
    else
        error("Undefined choice of numerical integration. Choose only an accepted method.")
    end
    return dXdt, idx
end


"""
Choose reduced order (ro) that preserves an acceptable energy.

# Arguments
- `Σ`: Singular value vector from the SVD of some Hankel Matrix
- `enmin`: minimum size for energy preservation

# Returns
- vector of all reduced orders.
"""
function choose_ro(Σ::Vector, enmin::Real=-15.0)
    # Energy loss from truncation
    en = 1 .- sqrt.(cumsum(Σ .^ 2)) / norm(Σ)

    # loop through ROM sizes
    en_vals = map(x -> 10.0^x, -1.0:-1.0:enmin)
    r_all = Vector{Float64}()
    for rr = axes(en_vals, 1)
        # determine # basis functions to retain based on energy lost
        en_thresh = en_vals[rr]
        push!(r_all, findfirst(x -> x < en_thresh, en))
    end

    return Int.(r_all)
end


"""
Ridge regression (tikhonov)

# Arguments
- `b`: Ax = b right-hand side 
- `A`: Ax = b left-hand side matrix
- `k`: ridge regression parameter

# Return
- regression solution
"""
function tikhonov(b::AbstractArray, A::AbstractArray, k::Real, tol::Real)
    q = size(b, 2)
    p = size(A, 2)

    pseudo = sqrt(k) * 1.0I(p)
    Aplus = vcat(A, pseudo)
    bplus = vcat(b, zeros(p, q))
    
    Aplus_svd = svd(Aplus)
    sing_idx = findfirst(Aplus_svd.S .< tol)

    # If singular values are nearly singular, truncate at a certain threshold
    # and fill in the rest with zeros
    if sing_idx !== nothing
        @warn "Rank difficient, rank = $(sing_idx), tol = $(Aplus_svd.S[sing_idx]).\n"
        foo = [1 ./ Aplus_svd.S[1:sing_idx-1]; zeros(length(Aplus_svd.S[sing_idx:end]))]
        bar = Aplus_svd.Vt' * Diagonal(foo) * Aplus_svd.U'
        return bar * bplus
    else
        return Aplus \ bplus
    end
end


"""
Get the data matrix for the regression problem

# Arguments
- `Xhat`: projected data matrix (not transposed)
- `Xhat_t`: projected data matrix (transpoesd)
- `U`: input data matrix
- `dims`: dictionary including important dimensions
- `options`: options for the operator inference set by the user

# Return
- `D`: data matrix for the regression problem
"""
function getDataMat(Xhat::Matrix, Xhat_t::Union{Matrix,Transpose}, U::Matrix,
    dims::Dict, options::Abstract_Options)
    flag = false

    if options.system.is_lin
        D = Xhat_t
        flag = true
    end

    # Data matrix
    if options.system.has_control
        if flag
            D = hcat(D, U)
        else
            D = U
            flag = true
        end
    end

    # Compute the operator matrix by solving the least squares problem
    if options.system.is_quad  # Quadratic term
        if options.optim.which_quad_term == "F"
            # Assemble matrices Xhat^(1), ..., Xhat^(n) following (11) corresponding to F matrix
            Xsq_t = squareMatStates(Xhat)'
        else
            Xsq_t = kronMatStates(Xhat)'
        end
        # Assemble D matrix
        if flag
            D = hcat(D, Xsq_t)
        else
            D = Xsq_t
            flag = true
        end
    end

    if options.system.is_bilin  # Bilinear term
        XU = Xhat_t .* U[:, 1]
        for i in 2:dims[:p]
            XU = hcat(XU, Xhat_t .* U[:, i])
        end
        if flag
            D = hcat(D, XU)
        else
            D = XU
            flag = true
        end
    end

    if options.system.has_const  # constant term
        I = ones(dims[:m], 1)
        if flag
            D = hcat(D, I)
        else
            D = I
            flag = true
        end
    end

    return D
end


"""
Extracting the operators after solving the regression problem

# Arguments
- `D`: data matrix 
- `Rt`: derivative data which is the left-hand-side of the regression problem (transposed)
- `Y`: output data matrix
- `Xhat_t`: projected data matrix (transposed)
- `dims`: dictionary including important dimensions
- `options`: options for the operator inference set by the user

# Return
- All operators A, B, C, F, H, N, K
"""
function LS_solve(D::Matrix, Rt::Union{Matrix,Transpose}, Y::Matrix,
    Xhat_t::Union{Matrix,Transpose}, dims::Dict, options::Abstract_Options)
    # Some dimensions to unpack for convenience
    n = dims[:n]
    p = dims[:p]
    q = dims[:q]
    s = dims[:s]
    v = dims[:v]
    w = dims[:w]

    Ot = tikhonov(Rt, D, options.λ, options.pinv_tol) # compute least squares (pseudo inverse)

    # Extract the operators from the operator matrix O
    O = transpose(Ot)
    Ahat = options.system.is_lin ? O[:, 1:n] : 0
    Bhat = options.system.has_control ? O[:, n+1:n+p] : 0

    # Compute Chat by solving the least square problem for the output values 
    if options.system.has_output
        Chat_t = zeros(n, q)
        Yt = transpose(Y)
        Chat_t = Xhat_t \ Yt
        Chat = transpose(Chat_t)
    else
        Chat = 0
    end

    # Extract Quadratic terms if the system includes such terms
    sv = 0  # initialize this dummy variable just in case
    if options.system.is_quad
        if options.optim.which_quad_term == "F"
            Fhat = O[:, n+p+1:n+p+s]
            Hhat = F2Hs(Fhat)
            sv = s  # dummy dimension variable since we use F
        else
            Hhat = O[:, n+p+1:n+p+v]
            Fhat = H2F(Hhat)
            sv = v  # dummy dimension variable since we use H
        end
    else
        Fhat = 0
        Hhat = 0
    end

    # Extract Bilinear terms 
    # FIX: fix this so that you can extract for a general case where there
    # are more than 1 input
    if options.system.is_bilin
        if p == 1
            Nhat = O[:, n+p+sv+1:n+p+sv+w]
        else 
            Nhat = zeros(p,n,n)
            tmp = O[:, n+p+sv+1:n+p+sv+w]
            for i in 1:p
                Nhat[i,:,:] .= tmp[:, Int(n*(i-1)+1):Int(n*i)]
            end
        end
    else
        Nhat = (p == 0) || (p == 1) ? 0 : zeros(p,n,n)
    end

    # Constant term
    Khat = options.system.has_const ? Matrix(O[:, n+p+sv+w+1:end]) : 0

    return Ahat, Bhat, Chat, Fhat, Hhat, Nhat, Khat
end


"""
Run the optimizer of choice.

# Arguments
- `D`: data matrix 
- `Rt`: derivative data which is the left-hand-side of the regression problem (transposed)
- `Y`: output data matrix
- `Xhat_t`: projected data matrix (transposed)
- `dims`: dictionary including important dimensions
- `options`: options for the operator inference set by the user

# Return
- All operators A, B, C, F, H, N, K
"""
function run_optimizer(D::AbstractArray, Rt::AbstractArray, Y::AbstractArray,
    Xhat_t::AbstractArray, dims::Dict, options::Abstract_Options,
    IG::operators=operators())

    if options.method == "LS"
        Ahat, Bhat, Chat, Fhat, Hhat, Nhat, Khat = LS_solve(D, Rt, Y, Xhat_t, dims, options)
        Qhat = 0.0
    elseif options.method == "NC"  # Non-constrained
        Ahat, Bhat, Fhat, Hhat, Nhat, Khat = NC_Optimize(D, Rt, dims, options, IG)
        Chat = options.system.has_output ? NC_Optimize_output(Y, Xhat_t, dims, options) : 0
        Qhat = 0.0
    elseif options.method == "EPHEC" || options.method == "EPHC"
        Ahat, Bhat, Fhat, Hhat, Nhat, Khat = EPHEC_Optimize(D, Rt, dims, options, IG)
        Chat = options.system.has_output ? NC_Optimize_output(Y, Xhat_t, dims, options) : 0
        Qhat = H2Q(Hhat)
    elseif options.method == "EPSC" || options.method == "EPSIC"
        Ahat, Bhat, Fhat, Hhat, Nhat, Khat = EPSIC_Optimize(D, Rt, dims, options, IG)
        Chat = options.system.has_output ? NC_Optimize_output(Y, Xhat_t, dims, options) : 0
        Qhat = H2Q(Hhat)
    elseif options.method == "EPP"
        Ahat, Bhat, Fhat, Hhat, Nhat, Khat = EPP_Optimize(D, Rt, dims, options, IG)
        Chat = options.system.has_output ? NC_Optimize_output(Y, Xhat_t, dims, options) : 0
        Qhat = H2Q(Hhat)
    else
        error("Incorrect optimization options.")
    end
    
    # Make sure to reformat column and row vectors into matrices
    Bhat = Bhat == 0 ? 0 : Matrix(Bhat)
    Chat = Chat == 0 ? 0 : Matrix(Chat)
    Khat = Khat == 0 ? 0 : Matrix(Khat)

    op = operators(
        A=Ahat,
        B=Bhat,
        C=Chat,
        F=Fhat,
        H=Hhat,
        N=Nhat,
        K=Khat,
        Q=Qhat,
    )

    return op
end


"""
Infer the operators using the previously defined functions 

# Arguments
- `X`: state data matrix
- `U`: input data matrix
- `Y`: output data matix
- `Vn`: POD basis
- `R`: derivative state data (given in this function)
- `options`: options for the operator inference defined by the user
- `IG`: initial guesses for optimization

# Return
- inferred operators
"""
function inferOp(X::Matrix, U::Matrix, Y::VecOrMat, Vn::Matrix, R::Matrix,
    options::Abstract_Options, IG::operators=operators())::operators
    Rt = transpose(R)
    Xhat = Vn' * X
    Xhat_t = transpose(Xhat)

    # Important dimensions
    n, m = size(Xhat)
    p = options.system.has_control ? size(U, 2) : 0  # make sure that the U-matrix is tall
    q = options.system.has_output ? size(Y, 1) : 0
    s = options.system.is_quad ? Int(n * (n + 1) / 2) : 0
    v = options.system.is_quad ? Int(n * n) : 0
    w = options.system.is_bilin ? Int(n * p) : 0
    dims = Dict(:n => n, :m => m, :p => p, :q => q, :s => s, :v => v, :w => w)  # create a dict

    D = getDataMat(Xhat, Xhat_t, U, dims, options)
    op = run_optimizer(D, Rt, Y, Xhat_t, dims, options, IG)

    return op
end


"""
Infer the operators using the previously defined functions (dispatch)

# Arguments
- `X`: state data matrix
- `U`: input data matrix
- `Y`: output data matix
- `Vn`: POD basis
- `options`: options for the operator inference defined by the user
- `IG`: initial guesses for optimization

# Return
- inferred operators
"""
function inferOp(X::Matrix, U::Matrix, Y::VecOrMat, Vn::Matrix,
    options::Abstract_Options, IG::operators=operators())::operators

    # Approximate the derivative data with finite difference
    Xdot, idx = dtApprox(X, options)
    X = X[:, idx]  # fix the index of states
    U = size(U)==(1,1) && U[1]==0 ? 0 : U[idx, :]  # fix the index of inputs
    Y = size(Y)==(1,1) && Y[1]==0 ? 0 : Y[:, idx]  # fix the index of outputs
    R = Vn'Xdot
    Rt = transpose(R)

    Xhat = Vn' * X
    Xhat_t = transpose(Xhat)

    # Important dimensions
    n, m = size(Xhat)
    p = options.system.has_control ? size(U, 2) : 0  # make sure that the U-matrix is tall
    q = options.system.has_output ? size(Y, 1) : 0
    s = options.system.is_quad ? Int(n * (n + 1) / 2) : 0
    v = options.system.is_quad ? Int(n * n) : 0
    w = options.system.is_bilin ? Int(n * p) : 0
    dims = Dict(:n => n, :m => m, :p => p, :q => q, :s => s, :v => v, :w => w)  # create a dict

    D = getDataMat(Xhat, Xhat_t, U, dims, options)
    op = run_optimizer(D, Rt, Y, Xhat_t, dims, options, IG)

    return op
end


"""
Infer the operators with reprojection (dispatch)

# Arguments
- `X`: state data matrix
- `U`: input data matrix
- `Y`: output data matix
- `Vn`: POD basis
- `full_op`: full order model operators
- `options`: options for the operator inference defined by the user
- `IG`: initial guesses for optimization

# Return
- inferred operators
"""
function inferOp(X::Matrix, U::Matrix, Y::VecOrMat, Vn::Matrix,
    full_op::operators, options::Abstract_Options, IG::operators=operators())::operators
    Xhat = Vn' * X
    Xhat_t = transpose(Xhat)

    # Important dimensions
    n, m = size(Xhat)
    p = options.system.has_control ? size(U, 2) : 0  # make sure that the U-matrix is tall
    q = options.system.has_output ? size(Y, 1) : 0
    s = options.system.is_quad ? Int(n * (n + 1) / 2) : 0
    v = options.system.is_quad ? Int(n * n) : 0
    w = options.system.is_bilin ? Int(n * p) : 0
    dims = Dict(:n => n, :m => m, :p => p, :q => q, :s => s, :v => v, :w => w)  # create a dict

    # Reproject
    Rt = reproject(Xhat, Vn, U, dims, full_op, options)

    D = getDataMat(Xhat, Xhat_t, U, dims, options)
    op = run_optimizer(D, Rt, Y, Xhat_t, dims, options, IG)

    return op
end


# NOTE: Reprojection allows to reduce error from the missing terms of the basis
"""
Reprojecting the data to minimize the error affected by the missing orders of the POD basis

# Arguments
- `Xhat`: state data matrix projected onto the basis
- `V`: POD basis
- `U`: input data matrix
- `dims`: dictionary including important dimensions
- `op`: full order model operators
- `options`: options of the operator inference defined by the user

# Return
- `Rhat`: R matrix (transposed) for the regression problem
"""
function reproject(Xhat::Matrix, V::Union{VecOrMat,BlockDiagonal}, U::VecOrMat,
    dims::Dict, op::operators, options::Abstract_Options)::Matrix
    
    # Just a simple error detection
    if options.system.is_quad && options.optim.which_quad_term=="F"
        if op.F == 0
            error("Quadratic F matrix should not be 0 if it is selected.")
        end
    elseif options.system.is_quad && options.optim.which_quad_term=="H"
        if op.H == 0
            error("Quadratic H matrix should not be 0 if it is selected.")
        end
    end

    Rt = zeros(dims[:m], dims[:n])  # Left hand side of the regression problem
    if options.system.has_funcOp
        f = (x, u) -> op.A * x + op.f(x) + op.B * u + op.K
    else
        p = dims[:p]

        fA = (x) -> options.system.is_lin ? op.A * x : 0
        fB = (u) -> options.system.has_control ? op.B * u : 0
        fF = (x) -> options.system.is_quad && options.optim.which_quad_term == "F" ? op.F * vech(x * x') : 0
        fH = (x) -> options.system.is_quad && options.optim.which_quad_term == "H" ? op.H * vec(x * x') : 0
        fN = (x,u) -> options.system.is_bilin ? ( p==1 ? (op.N * x) * u : sum([(op.N[i] * x) * u[i] for i in 1:p]) ) : 0
        fK = options.system.has_const ? op.K : 0

        f = (x,u) -> fA(x) .+ fB(u) .+ fF(x) .+ fH(x) .+ fN(x,u) .+ fK
    end

    for i in 1:dims[:m]  # loop thru all data
        x = Xhat[:, i]  # julia automatically makes into column vec after indexing (still xrow-tcol)
        xrec = V * x
        states = f(xrec, U[i, :])
        Rt[i, :] = V' * states
    end
    return Rt
end


# NOTE: Reprojection allows to reduce error from the missing terms of the basis
"""
Reprojecting the data to minimize the error affected by the missing orders of the POD basis

# Arguments
- `Xhat`: state data matrix projected onto the basis
- `V`: POD basis
- `U`: input data matrix
- `dims`: dictionary including important dimensions
- `lm`: structure of the lift map
- `op`: full order model operators
- `options`: options of the operator inference defined by the user

# Return
- `Rhat`: R matrix (transposed) for the regression problem
"""
function reproject(Xhat::Matrix, V::Union{VecOrMat,BlockDiagonal}, U::VecOrMat,
    dims::Dict, lm::lifting, op::operators, options::Abstract_Options)::Matrix

    tmp = size(V, 1)
    n = tmp / options.vars.N_lift
    dt = options.data.Δt
    Rt = zeros(dims[:m], dims[:n])  # Left hand side of the regression problem

    if options.system.has_funcOp
        f = (x, u) -> op.A * x + op.f(x) + op.B * u + op.K
    else
        p = dims[:p]

        fA = (x) -> options.system.is_lin ? op.A * x : 0
        fB = (u) -> options.system.has_control ? op.B * u : 0
        fF = (x) -> options.system.is_quad && options.optim.which_quad_term == "F" ? op.F * vech(x * x') : 0
        fH = (x) -> options.system.is_quad && options.optim.which_quad_term == "H" ? op.H * vec(x * x') : 0
        fN = (x,u) -> options.system.is_bilin ? ( p==1 ? (op.N * x) * u : sum([(op.N[i] * x) * u[i] for i in 1:p]) ) : 0
        fK = options.system.has_const ? op.K : 0

        f = (x,u) -> fA(x) .+ fB(u) .+ fF(x) .+ fH(x) .+ fN(x,u) .+ fK
    end

    for i in 1:dims[:m]  # loop thru all data
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


"""
Infer the operators using the previously defined functions (dispatch function: 
this one includes the reprjection operation better results)

# Arguments
- `W`: lifted state data matrix
- `U`: input data matrix
- `Y`: output data matrix
- `Vn`: POD basis
- `lm`: struct of the lift map
- `full_op`: full order model operators
- `options`: options of the operator inference defined by the user
- `IG`: initial guesses for optimization

# Return
- inferred operators
"""
function inferOp(W::Matrix, U::Matrix, Y::VecOrMat, Vn::Union{Matrix,BlockDiagonal},
    lm::lifting, full_op::operators, options::Abstract_Options, IG::operators=operators())::operators

    # Project
    What = Vn' * W
    What_t = W' * Vn

    # Important dimensions
    n, m = size(What)
    p = options.system.has_control ? size(U, 2) : 0
    q = options.system.has_output ? size(Y, 1) : 0
    s = options.system.is_quad ? Int(n * (n + 1) / 2) : 0
    v = options.system.is_quad ? Int(n * n) : 0
    w = options.system.is_bilin ? Int(n * p) : 0
    dims = Dict(:n => n, :m => m, :p => p, :q => q, :s => s, :v => v, :w => w)  # create a dict

    # Generate R matrix from finite difference if not provided as a function argument
    if options.optim.reproject == true
        Rt = reproject(What, Vn, U, dims, lm, full_op, options)
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

    D = getDataMat(What, What_t, U, dims, options)
    op = run_optimizer(D, Rt, Y, What_t, dims, options, IG)

    return op
end


# TODO: Create another inferOp function with no full model operators provided.

