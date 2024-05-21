export opinf

include("dtApprox.jl")
include("choose_ro.jl")
include("define_dimensions.jl")
include("getDataMat.jl")
include("tikhonov.jl")


"""
    LS_solve(D::AbstractArray, Rt::AbstractArray, Y::AbstractArray,
        Xhat_t::AbstractArray, options::AbstractOption) → Ahat, Bhat, Chat, Fhat, Hhat, Ehat, Ghat, Nhat, Khat

Solve the standard Operator Inference with/without regularization

## Arguments
- `D::AbstractArray`: data matrix
- `Rt::AbstractArray`: derivative data matrix (transposed)
- `Y::AbstractArray`: output data matrix
- `Xhat_t::AbstractArray`: projected data matrix (transposed)
- `options::AbstractOption`: options for the operator inference set by the user

## Returns
- All learned operators A, B, C, F, H, N, K
"""
function LS_solve(D::AbstractArray, Rt::AbstractArray, Y::AbstractArray,
    Xhat_t::AbstractArray, options::AbstractOption)
    # Some dimensions to unpack for convenience
    n = options.system.dims[:n]
    p = options.system.dims[:p]
    q = options.system.dims[:q]
    s2 = options.system.dims[:s2]
    v2 = options.system.dims[:v2]
    s3 = options.system.dims[:s3]
    v3 = options.system.dims[:v3]
    w1 = options.system.dims[:w1]

    # Construct the Tikhonov matrix
    if options.optim.which_quad_term == "F"
        if options.optim.which_cubic_term == "E"
            Γ = spzeros(n+p+s2+s3+w1)
        else
            Γ = spzeros(n+p+s2+v3+w1)
        end
    else
        if options.optim.which_cubic_term == "E"
            Γ = spzeros(n+p+v2+s3+w1)
        else
            Γ = spzeros(n+p+v2+v3+w1)
        end
    end
    tikhonovMatrix!(Γ, options)
    Γ = spdiagm(0 => Γ)  # convert to sparse diagonal matrix

    # compute least squares (pseudo inverse)
    if options.with_reg 
        Ot = tikhonov(Rt, D, Γ, options.pinv_tol; flag=options.with_tol)
    else
        Ot = D \ Rt
    end

    # Extract the operators from the operator matrix O
    O = transpose(Ot)
    TD = 0  # initialize this dummy variable for total dimension (TD)
    if options.system.is_lin
        Ahat = O[:, TD+1:n]
        TD += n
    else
        Ahat = 0
    end
    if options.system.has_control
        Bhat = O[:, TD+1:TD+p]
        TD += p
    else
        Bhat = 0
    end

    # Compute Chat by solving the least square problem for the output values 
    if options.system.has_output
        Chat_t = zeros(n, q)
        Yt = transpose(Y)
        if options.with_reg && options.λ.output != 0
            Chat_t = (Xhat_t' * Xhat_t + options.λ.output * I) \ (Xhat_t' * Yt)
        else
            Chat_t = Xhat_t \ Yt
        end
        Chat = transpose(Chat_t)
    else
        Chat = 0
    end

    # Extract Quadratic terms if the system includes such terms
    sv = 0  # initialize this dummy variable just in case
    if options.system.is_quad
        if options.optim.which_quad_term == "F"
            Fhat = O[:, TD+1:TD+s2]
            Hhat = F2Hs(Fhat)
            TD += s2
        else
            Hhat = O[:, TD+1:TD+v2]
            Fhat = H2F(Hhat)
            TD += v2
        end
    else
        Fhat = 0
        Hhat = 0
    end

    # Extract Cubic terms if the system includes such terms
    sv3 = 0  # initialize this dummy variable just in case
    if options.system.is_cubic
        if options.optim.which_cubic_term == "E"
            Ehat = O[:, TD+1:TD+s3]
            TD += s3
        else
            Ghat = O[:, TD+1:TD+v3]
            Ehat = G2E(Ghat)
            TD += v3
        end
    else
        Ehat = 0
        Ghat = 0
    end

    # Extract Bilinear terms 
    # FIX: fix this so that you can extract for a general case where there
    # are more than 1 input
    if options.system.is_bilin
        if p == 1
            Nhat = O[:, TD+1:TD+w1]
        else 
            Nhat = zeros(p,n,n)
            tmp = O[:, TD+1:TD+w1]
            for i in 1:p
                Nhat[:,:,i] .= tmp[:, Int(n*(i-1)+1):Int(n*i)]
            end
        end
        TD += w1
    else
        Nhat = (p == 0) || (p == 1) ? 0 : zeros(n,n,p)
    end

    # Constant term
    Khat = options.system.has_const ? Matrix(O[:, TD+1:end]) : 0

    return Ahat, Bhat, Chat, Fhat, Hhat, Ehat, Ghat, Nhat, Khat
end


"""
    run_optimizer(D::AbstractArray, Rt::AbstractArray, Y::AbstractArray,
        Xhat_t::AbstractArray, dims::Dict, options::AbstractOption,
        IG::operators=operators()) → op::operators

Run the optimizer of choice.

## Arguments
- `D::AbstractArray`: data matrix
- `Rt::AbstractArray`: derivative data matrix (transposed)
- `Y::AbstractArray`: output data matrix
- `Xhat_t::AbstractArray`: projected data matrix (transposed)
- `options::AbstractOption`: options for the operator inference set by the user
- `IG::operators`: initial guesses for optimization

## Returns
- `op::operators`: All learned operators 
"""
function run_optimizer(D::AbstractArray, Rt::AbstractArray, Y::AbstractArray,
    Xhat_t::AbstractArray, options::AbstractOption,
    IG::operators=operators())

    if options.method == "LS"
        Ahat, Bhat, Chat, Fhat, Hhat, Ehat, Ghat, Nhat, Khat = LS_solve(D, Rt, Y, Xhat_t, options)
        Qhat = 0
    elseif options.method == "NC"  # Non-constrained
        Ahat, Bhat, Fhat, Hhat, Nhat, Khat = NC_Optimize(D, Rt, options, IG)
        Chat = options.system.has_output ? NC_Optimize_output(Y, Xhat_t, options) : 0
        Qhat = 0
        Ghat = 0
        Ehat = 0
    elseif options.method == "EPHEC" || options.method == "EPHC"
        Ahat, Bhat, Fhat, Hhat, Nhat, Khat = EPHEC_Optimize(D, Rt, options, IG)
        Chat = options.system.has_output ? NC_Optimize_output(Y, Xhat_t, options) : 0
        Qhat = H2Q(Hhat)
        Ghat = 0
        Ehat = 0
    elseif options.method == "EPSC" || options.method == "EPSIC"
        Ahat, Bhat, Fhat, Hhat, Nhat, Khat = EPSIC_Optimize(D, Rt, options, IG)
        Chat = options.system.has_output ? NC_Optimize_output(Y, Xhat_t, options) : 0
        Qhat = H2Q(Hhat)
        Ghat = 0
        Ehat = 0
    elseif options.method == "EPP"
        Ahat, Bhat, Fhat, Hhat, Nhat, Khat = EPP_Optimize(D, Rt, options, IG)
        Chat = options.system.has_output ? NC_Optimize_output(Y, Xhat_t, options) : 0
        Qhat = H2Q(Hhat)
        Ghat = 0
        Ehat = 0
    else
        error("Incorrect optimization options.")
    end
    
    op = operators(
        A=Ahat, B=Bhat, C=Chat, F=Fhat, H=Hhat, 
        E=Ehat, G=Ghat, N=Nhat, K=Khat, Q=Qhat,
    )
    return op
end


"""
    opinf(X::AbstractArray, Vn::AbstractArray, options::AbstractOption; 
        U::AbstractArray=zeros(1,1), Y::AbstractArray=zeros(1,1),
        Xdot::AbstractArray=[], IG::operators=operators()) → op::operators

Infer the operators with derivative data given

## Arguments
- `X::AbstractArray`: state data matrix
- `Vn::AbstractArray`: POD basis
- `options::AbstractOption`: options for the operator inference defined by the user
- `U::AbstractArray`: input data matrix
- `Y::AbstractArray`: output data matix
- `Xdot::AbstractArray`: derivative data matrix
- `IG::operators`: initial guesses for optimization

## Returns
- `op::operators`: inferred operators
"""
function opinf(X::AbstractArray, Vn::AbstractArray, options::AbstractOption; 
                 U::AbstractArray=zeros(1,1), Y::AbstractArray=zeros(1,1),
                 Xdot::AbstractArray=[], IG::operators=operators())::operators
    if isempty(Xdot)
        # Approximate the derivative data with finite difference
        Xdot, idx = dtApprox(X, options)
        Xhat = Vn' * X[:, idx]  # fix the index of states
        Xhat_t = transpose(Xhat)
        U = iszero(U) ? 0 : U[idx, :]  # fix the index of inputs
        Y = iszero(Y) ? 0 : Y[:, idx]  # fix the index of outputs
        R = Vn'Xdot
        Rt = transpose(R)
    else
        Xhat = Vn' * X
        Xhat_t = transpose(Xhat)
        Rt = transpose(Vn' * Xdot)
    end

    define_dimensions!(Xhat, U, Y, options)
    D = getDataMat(Xhat, Xhat_t, U, options)
    op = run_optimizer(D, Rt, Y, Xhat_t, options, IG)
    return op
end


"""
    opinf(X::AbstractArray, Vn::AbstractArray, full_op::operators, options::AbstractOption;
        U::AbstractArray=zeros(1,1), Y::AbstractArray=zeros(1,1), IG::operators=operators()) → op::operators

Infer the operators with reprojection method (dispatch)

## Arguments
- `X::AbstractArray`: state data matrix
- `Vn::AbstractArray`: POD basis
- `full_op::operators`: full order model operators
- `options::AbstractOption`: options for the operator inference defined by the user
- `U::AbstractArray`: input data matrix
- `Y::AbstractArray`: output data matix
- `IG::operators`: initial guesses for optimization

## Returns
- `op::operators`: inferred operators
"""
function opinf(X::AbstractArray, Vn::AbstractArray, full_op::operators, options::AbstractOption;
                 U::AbstractArray=zeros(1,1), Y::AbstractArray=zeros(1,1), IG::operators=operators())::operators
    Xhat = Vn' * X
    Xhat_t = transpose(Xhat)
    define_dimensions!(Xhat, U, Y, options)

    # Reproject
    Rt = reproject(Xhat, Vn, U, full_op, options)
    D = getDataMat(Xhat, Xhat_t, U, options)
    op = run_optimizer(D, Rt, Y, Xhat_t, options, IG)
    return op
end


"""
    reproject(Xhat::AbstractArray, V::AbstractArray, U::AbstractArray,
        op::operators, options::AbstractOption) → Rhat::AbstractArray

Reprojecting the data to minimize the error affected by the missing orders of the POD basis

## Arguments
- `Xhat::AbstractArray`: state data matrix projected onto the basis
- `V::AbstractArray`: POD basis
- `U::AbstractArray`: input data matrix
- `op::operators`: full order model operators
- `options::AbstractOption`: options for the operator inference defined by the user

## Return
- `Rhat::AbstractArray`: R matrix (transposed) for the regression problem
"""
function reproject(Xhat::AbstractArray, V::AbstractArray, U::AbstractArray,
    op::operators, options::AbstractOption)::AbstractArray
    
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

    Rt = zeros(options.system.dims[:m], options.system.dims[:n])  # Left hand side of the regression problem
    if options.system.has_funcOp
        f = (x, u) -> op.A * x + op.f(x) + op.B * u + op.K
    else
        p = options.system.dims[:p]

        fA = (x) -> options.system.is_lin ? op.A * x : 0
        fB = (u) -> options.system.has_control ? op.B * u : 0
        fF = (x) -> options.system.is_quad && options.optim.which_quad_term == "F" ? op.F * (x ⊘ x) : 0
        fH = (x) -> options.system.is_quad && options.optim.which_quad_term == "H" ? op.H * (x ⊗ x) : 0
        fE = (x) -> options.system.is_cubic && options.optim.which_cubic_term == "E" ? op.E * ⊘(x,x,x) : 0
        fG = (x) -> options.system.is_cubic && options.optim.which_cubic_term == "G" ? op.G * (x ⊗ x ⊗ x) : 0
        fN = (x,u) -> options.system.is_bilin ? ( p==1 ? (op.N * x) * u : sum([(op.N[i] * x) * u[i] for i in 1:p]) ) : 0
        fK = options.system.has_const ? op.K : 0

        f = (x,u) -> fA(x) .+ fB(u) .+ fF(x) .+ fH(x) .+ fE(x) .+ fG(x) .+ fN(x,u) .+ fK
    end

    for i in 1:options.system.dims[:m]  # loop thru all data
        x = Xhat[:, i]  # julia automatically makes into column vec after indexing (still xrow-tcol)
        xrec = V * x
        states = f(xrec, U[i, :])
        Rt[i, :] = V' * states
    end
    return Rt
end

