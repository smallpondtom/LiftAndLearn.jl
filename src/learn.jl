export inferOp, choose_ro

"""
    dtApprox(X::VecOrMat, options::AbstractOption) → dXdt, idx

Approximating the derivative values of the data with different integration schemes

## Arguments
- `X::VecOrMat`: data matrix
- `options::AbstractOption`: operator inference options

## Returns
- `dXdt`: derivative data
- `idx`: index for the specific integration scheme (important for later use)
"""
function dtApprox(X::VecOrMat, options::AbstractOption)
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
    choose_ro(Σ::Vector; en_low=-15) → r_all, en

Choose reduced order (ro) that preserves an acceptable energy.

## Arguments
- `Σ::Vector`: Singular value vector from the SVD of some Hankel Matrix
- `en_low`: minimum size for energy preservation

## Returns
- `r_all`: vector of reduced orders
- `en`: vector of energy values
"""
function choose_ro(Σ::Vector; en_low=-15)
    # Energy loss from truncation
    en = 1 .- sqrt.(cumsum(Σ .^ 2)) / norm(Σ)

    # loop through ROM sizes
    en_vals = map(x -> 10.0^x, -1.0:-1.0:en_low)
    r_all = Vector{Float64}()
    for rr = axes(en_vals, 1)
        # determine # basis functions to retain based on energy lost
        en_thresh = en_vals[rr]
        push!(r_all, findfirst(x -> x < en_thresh, en))
    end

    return Int.(r_all), en
end


"""
    getDataMat(Xhat::AbstractArray, Xhat_t::AbstractArray, U::Matrix,
        dims::Dict, options::AbstractOption) → D

Get the data matrix for the regression problem

## Arguments
- `Xhat::AbstractArray`: projected data matrix
- `Xhat_t::AbstractArray`: projected data matrix (transposed)
- `U::Matrix`: input data matrix
- `options::AbstractOption`: options for the operator inference set by the user

## Returns
- `D`: data matrix for the regression problem
"""
function getDataMat(Xhat::AbstractArray, Xhat_t::AbstractArray, U::Matrix, options::AbstractOption)
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

    if options.system.is_cubic  # cubic term
        if options.optim.which_cubic_term == "E"
            Xcu_t = cubicMatStates(Xhat)'
        else
            Xcu_t = kron3MatStates(Xhat)'
        end
        if flag
            D = hcat(D, Xcu_t)
        else
            D = Xcu_t
            flag = true
        end
    end

    if options.system.is_bilin  # Bilinear term
        XU = Xhat_t .* U[:, 1]
        for i in 2:options.system.dims[:p]
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
        I = ones(options.system.dims[:m], 1)
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
    getDataMat(Xhat::AbstractArray, U::Matrix,
        dims::Dict, options::AbstractOption) → D

Dispatch to avoid having tranpose data matrix in function argument.
"""
getDataMat(Xhat::AbstractArray, U::Matrix, options::AbstractOption) = getDataMat(Xhat, transpose(Xhat), U, options)


"""
    tikhonov(b::AbstractArray, A::AbstractArray, Γ::AbstractMatrix, tol::Real;
        flag::Bool=false) → x

Tikhonov regression

## Arguments
- `b::AbstractArray`: right hand side of the regression problem
- `A::AbstractArray`: left hand side of the regression problem
- `Γ::AbstractMatrix`: Tikhonov matrix
- `tol::Real`: tolerance for the singular values
- `flag::Bool`: flag for the tolerance

## Returns
- regression solution
"""
function tikhonov(b::AbstractArray, A::AbstractArray, Γ::AbstractMatrix, tol::Real; flag::Bool=false)
    if flag
        Ag = A' * A + Γ' * Γ
        Ag_svd = svd(Ag)
        sing_idx = findfirst(Ag_svd.S .< tol)

        # If singular values are nearly singular, truncate at a certain threshold
        # and fill in the rest with zeros
        if sing_idx !== nothing
            @warn "Rank difficient, rank = $(sing_idx), tol = $(Ag_svd.S[sing_idx]).\n"
            foo = [1 ./ Ag_svd.S[1:sing_idx-1]; zeros(length(Ag_svd.S[sing_idx:end]))]
            bar = Ag_svd.Vt' * Diagonal(foo) * Ag_svd.U'
            return bar * (A' * b)
        else
            return Ag \ (A' * b)
        end
    else
        return (A' * A + Γ' * Γ) \ (A' * b)
    end
end


"""
    tikhonovMatrix!(Γ::AbstractArray, dims::Dict, options::AbstractOption)

Construct the Tikhonov matrix

## Arguments
- `Γ::AbstractArray`: Tikhonov matrix (pass by reference)
- `options::AbstractOption`: options for the operator inference set by the user

## Returns
- `Γ`: Tikhonov matrix (pass by reference)
"""
function tikhonovMatrix!(Γ::AbstractArray, options::AbstractOption)
    n = options.system.dims[:n]
    p = options.system.dims[:p]
    s2 = options.system.dims[:s2]
    v2 = options.system.dims[:v2]
    s3 = options.system.dims[:s3]
    v3 = options.system.dims[:v3]
    w1 = options.system.dims[:w1]

    λ = options.λ
    si = 0  # start index
    if n != 0
        Γ[1:n] .= λ.lin
        si += n 
    end

    if p != 0
        Γ[si+1:si+p] .= λ.ctrl
        si += p
    end

    if options.optim.which_quad_term == "F"
        if s2 != 0
            Γ[si+1:si+s2] .= λ.quad
        end
        si += s2
    else
        if v2 != 0
            Γ[si+1:si+v2] .= λ.quad
        end
        si += v2
    end

    if options.system.is_cubic
        if options.optim.which_cubic_term == "E"
            if s3 != 0
                Γ[si+1:si+s3] .= λ.cubic
            end
            si += s3
        else
            if v3 != 0
                Γ[si+1:si+v3] .= λ.cubic
            end
            si += v3
        end
    end

    if w1 != 0
        Γ[si+1:si+w1] .= λ.bilin
    end
end


"""
    LS_solve(D::Matrix, Rt::Union{Matrix,Transpose}, Y::Matrix, Xhat_t::Union{Matrix,Transpose}, 
        dims::Dict, options::AbstractOption) → Ahat, Bhat, Chat, Fhat, Hhat, Nhat, Khat

Solve the standard Operator Inference with/without regularization

## Arguments
- `D::Matrix`: data matrix
- `Rt::Union{Matrix,Transpose}`: derivative data matrix (transposed)
- `Y::Matrix`: output data matrix
- `Xhat_t::Union{Matrix,Transpose}`: projected data matrix (transposed)
- `options::AbstractOption`: options for the operator inference set by the user

## Returns
- All learned operators A, B, C, F, H, N, K
"""
function LS_solve(D::Matrix, Rt::Union{Matrix,Transpose}, Y::Matrix,
    Xhat_t::Union{Matrix,Transpose}, options::AbstractOption)
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
    # Ahat = options.system.is_lin ? O[:, 1:n] : 0
    # Bhat = options.system.has_control ? O[:, n+1:n+p] : 0

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
            # Fhat = O[:, n+p+1:n+p+s]
            Fhat = O[:, TD+1:TD+s]
            Hhat = F2Hs(Fhat)
            TD += s
            # sv = s  # dummy dimension variable since we use F
        else
            # Hhat = O[:, n+p+1:n+p+v]
            Hhat = O[:, TD+1:TD+v]
            Fhat = H2F(Hhat)
            TD += v
            # sv = v  # dummy dimension variable since we use H
        end
    else
        Fhat = 0
        Hhat = 0
    end

    # Extract Cubic terms if the system includes such terms
    sv3 = 0  # initialize this dummy variable just in case
    if options.system.is_cubic
        if options.optim.which_cubic_term == "E"
            # Ehat = O[:, n+p+sv+1:n+p+sv+s3]
            Ehat = O[:, TD+1:TD+s3]
            # sv3 = s3  # dummy dimension variable since we use E
            TD += s3
        else
            # Hhat = O[:, n+p+sv+1:n+p+sv+v3]
            Ghat = O[:, TD+1:TD+v3]
            Ehat = G2E(Ghat)
            # sv3 = v3  # dummy dimension variable since we use H
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
            # Nhat = O[:, n+p+sv+1:n+p+sv+w]
            Nhat = O[:, TD+1:TD+w]
        else 
            Nhat = zeros(p,n,n)
            # tmp = O[:, n+p+sv+1:n+p+sv+w]
            tmp = O[:, TD+1:TD+w]
            for i in 1:p
                # Nhat[i,:,:] .= tmp[:, Int(n*(i-1)+1):Int(n*i)]
                Nhat[:,:,i] .= tmp[:, Int(n*(i-1)+1):Int(n*i)]
            end
        end
        TD += w
    else
        # Nhat = (p == 0) || (p == 1) ? 0 : zeros(p,n,n)
        Nhat = (p == 0) || (p == 1) ? 0 : zeros(n,n,p)
    end

    # Constant term
    # Khat = options.system.has_const ? Matrix(O[:, n+p+sv+w+1:end]) : 0
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
    
    # Make sure to reformat column and row vectors into matrices
    Bhat = Bhat == 0 ? 0 : Matrix(Bhat)
    Chat = Chat == 0 ? 0 : Matrix(Chat)
    Khat = Khat == 0 ? 0 : Matrix(Khat)

    op = operators(
        A=Ahat, B=Bhat, C=Chat, F=Fhat, H=Hhat, 
        E=Ehat, G=Ghat, N=Nhat, K=Khat, Q=Qhat,
    )

    return op
end


"""
    define_dimensions!(Xhat::Matrix, U::Matrix, Y::VecOrMat, options::AbstractOption)

Define the dimensions of the system

## Arguments
- `Xhat::Matrix`: projected data matrix
- `U::Matrix`: input data matrix
- `Y::VecOrMat`: output data matrix
- `options::AbstractOption`: options for the operator inference set by the user
"""
function define_dimensions!(Xhat::Matrix, U::Matrix, Y::VecOrMat, options::AbstractOption)
    # Important dimensions
    n, m = size(Xhat)
    p = options.system.has_control ? size(U, 2) : 0  # make sure that the U-matrix is tall
    q = options.system.has_output ? size(Y, 1) : 0
    v2 = options.system.is_quad ? Int(n * n) : 0
    s2 = options.system.is_quad ? Int(n * (n + 1) / 2) : 0
    v3 = options.system.is_cubic ? Int(n * n * n) : 0
    s3 = options.system.is_cubic ? Int(n * (n + 1) * (n + 2) / 6) : 0
    w1 = options.system.is_bilin ? Int(n * p) : 0
    options.system.dims = Dict(
        :n => n, :m => m, :p => p, :q => q, 
        :v2 => v2, :s2 => s2, :v3 => v3, :s3 => s3,
        :w1 => w1
    )  # create dimension dict
    return nothing
end


"""
    inferOp(X::Matrix, U::Matrix, Y::VecOrMat, Vn::Matrix, R::Matrix,
        options::AbstractOption, IG::operators=operators()) → op::operators

Infer the operators with derivative data given

## Arguments
- `X::Matrix`: state data matrix
- `U::Matrix`: input data matrix
- `Y::VecOrMat`: output data matix
- `Vn::Matrix`: POD basis
- `R::Matrix`: derivative data matrix
- `options::AbstractOption`: options for the operator inference defined by the user
- `IG::operators`: initial guesses for optimization

## Returns
- `op::operators`: inferred operators
"""
function inferOp(X::Matrix, U::Matrix, Y::VecOrMat, Vn::Matrix, R::Matrix,
    options::AbstractOption, IG::operators=operators())::operators
    Rt = transpose(Vn' * R)
    Xhat = Vn' * X
    Xhat_t = transpose(Xhat)

    define_dimensions!(Xhat, U, Y, options)
    D = getDataMat(Xhat, Xhat_t, U, options)
    op = run_optimizer(D, Rt, Y, Xhat_t, options, IG)

    return op
end


"""
    inferOp(X::Matrix, U::Matrix, Y::VecOrMat, Vn::Matrix,
        options::AbstractOption, IG::operators=operators()) → op::operators

Infer the operators without derivative data (dispatch)

## Arguments
- `X::Matrix`: state data matrix
- `U::Matrix`: input data matrix
- `Y::VecOrMat`: output data matix
- `Vn::Matrix`: POD basis
- `options::AbstractOption`: options for the operator inference defined by the user
- `IG::operators`: initial guesses for optimization

## Returns
- `op::operators`: inferred operators
"""
function inferOp(X::Matrix, U::Matrix, Y::VecOrMat, Vn::Matrix,
    options::AbstractOption, IG::operators=operators())::operators

    # Approximate the derivative data with finite difference
    Xdot, idx = dtApprox(X, options)
    X = X[:, idx]  # fix the index of states
    U = size(U)==(1,1) && U[1]==0 ? 0 : U[idx, :]  # fix the index of inputs
    Y = size(Y)==(1,1) && Y[1]==0 ? 0 : Y[:, idx]  # fix the index of outputs
    R = Vn'Xdot
    Rt = transpose(R)

    Xhat = Vn' * X
    Xhat_t = transpose(Xhat)

    define_dimensions!(Xhat, U, Y, options)
    D = getDataMat(Xhat, Xhat_t, U, options)
    op = run_optimizer(D, Rt, Y, Xhat_t, options, IG)

    return op
end


"""
    inferOp(X::Matrix, U::Matrix, Y::VecOrMat, Vn::Matrix,
        full_op::operators, options::AbstractOption, IG::operators=operators()) → op::operators

Infer the operators with reprojection method (dispatch)

## Arguments
- `X::Matrix`: state data matrix
- `U::Matrix`: input data matrix
- `Y::VecOrMat`: output data matix
- `Vn::Matrix`: POD basis
- `full_op::operators`: full order model operators
- `options::AbstractOption`: options for the operator inference defined by the user
- `IG::operators`: initial guesses for optimization

## Returns
- `op::operators`: inferred operators
"""
function inferOp(X::Matrix, U::Matrix, Y::VecOrMat, Vn::Matrix,
    full_op::operators, options::AbstractOption, IG::operators=operators())::operators
    Xhat = Vn' * X
    Xhat_t = transpose(Xhat)

    define_dimensions!(Xhat, U, Y, options)

    # Reproject
    Rt = reproject(Xhat, Vn, U, dims, full_op, options)

    D = getDataMat(Xhat, Xhat_t, U, options)
    op = run_optimizer(D, Rt, Y, Xhat_t, options, IG)

    return op
end


"""
    inferOp(W::Matrix, U::Matrix, Y::VecOrMat, Vn::Union{Matrix,BlockDiagonal},
        lm::lifting, full_op::operators, options::AbstractOption, 
        IG::operators=operators()) → op::operators

Infer the operators for Lift And Learn for reprojected data (dispatch)

## Arguments
- `W::Matrix`: state data matrix
- `U::Matrix`: input data matrix
- `Y::VecOrMat`: output data matix
- `Vn::Union{Matrix,BlockDiagonal}`: POD basis
- `lm::lifting`: struct of the lift map
- `full_op::operators`: full order model operators
- `options::AbstractOption`: options for the operator inference defined by the user
- `IG::operators`: initial guesses for optimization

## Returns
- `op::operators`: inferred operators
"""
function inferOp(W::Matrix, U::Matrix, Y::VecOrMat, Vn::Union{Matrix,BlockDiagonal},
    lm::lifting, full_op::operators, options::AbstractOption, IG::operators=operators())::operators

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
        op::operators, options::AbstractOption) → Rhat::Matrix

Reprojecting the data to minimize the error affected by the missing orders of the POD basis

## Arguments
- `Xhat::Matrix`: state data matrix projected onto the basis
- `V::Union{VecOrMat,BlockDiagonal}`: POD basis
- `U::VecOrMat`: input data matrix
- `op::operators`: full order model operators
- `options::AbstractOption`: options for the operator inference defined by the user

## Return
- `Rhat::Matrix`: R matrix (transposed) for the regression problem
"""
function reproject(Xhat::Matrix, V::Union{VecOrMat,BlockDiagonal}, U::VecOrMat,
    op::operators, options::AbstractOption)::Matrix
    
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
    Rt = zeros(options.system.dims[:m], options.system.dims[:n])  # Left hand side of the regression problem

    if options.system.has_funcOp
        f = (x, u) -> op.A * x + op.f(x) + op.B * u + op.K
    else
        p = options.systems.dims[:p]

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

    for i in 1:options.systems.dims[:m]  # loop thru all data
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
