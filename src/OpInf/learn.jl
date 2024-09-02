export opinf

include("dtApprox.jl")
include("choose_ro.jl")
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
function LS_solve(D::AbstractArray, Rt::AbstractArray, Y::AbstractArray, Xhat_t::AbstractArray, 
                  dims::AbstractArray, operator_symbols::AbstractArray, options::AbstractOption)
    # Some dimensions to unpack for convenience
    # n = options.system.dims[:n]
    # m = options.system.dims[:m]
    # l = options.system.dims[:l]
    # s2 = options.system.dims[:s2]
    # v2 = options.system.dims[:v2]
    # s3 = options.system.dims[:s3]
    # v3 = options.system.dims[:v3]
    # w1 = options.system.dims[:w1]

    # Preallocate the Tikhonov weight Matrix
    Γ = spzeros(sum(dims))

    # Construct the Tikhonov matrix
    tikhonovMatrix!(Γ, dims, operator_symbols, options.λ)
    Γ = spdiagm(0 => Γ)  # convert to sparse diagonal matrix

    # # Construct the Tikhonov matrix
    # if options.optim.which_quad_term == "F"
    #     if options.optim.which_cubic_term == "E"
    #         Γ = spzeros(n+m+s2+s3+w1)
    #     else
    #         Γ = spzeros(n+m+s2+v3+w1)
    #     end
    # else
    #     if options.optim.which_cubic_term == "E"
    #         Γ = spzeros(n+m+v2+s3+w1)
    #     else
    #         Γ = spzeros(n+m+v2+v3+w1)
    #     end
    # end
    # tikhonovMatrix!(Γ, options)
    # Γ = spdiagm(0 => Γ)  # convert to sparse diagonal matrix

    # compute least squares (pseudo inverse)
    if options.with_reg 
        Ot = tikhonov(Rt, D, Γ, options.pinv_tol; flag=options.with_tol)
    else
        Ot = D \ Rt
    end

    # Extract the operators from the operator matrix O
    O = transpose(Ot)

    # # Compute Chat by solving the least square problem for the output values 
    # if options.system.has_output
    #     Chat_t = zeros(n, l)
    #     Yt = transpose(Y)
    #     if options.with_reg && options.λ.output != 0
    #         Chat_t = (Xhat_t' * Xhat_t + options.λ.output * I) \ (Xhat_t' * Yt)
    #     else
    #         Chat_t = Xhat_t \ Yt
    #     end
    #     Chat = transpose(Chat_t)
    # else
    #     Chat = 0
    # end

    # Extract the operators
    operators = Operators()

    # # Compute Chat by solving the least square problem for the output values 
    # if !iszero(options.system.output)
    #     Yt = fat2tall(Y)
    #     l = size(Yt, 2)
    #     Chat_t = zeros(n, l)
    #     if options.with_reg && options.λ.output != 0
    #         Chat_t = (Xhat_t' * Xhat_t + options.λ.output * I) \ (Xhat_t' * Yt)
    #     else
    #         Chat_t = Xhat_t \ Yt
    #     end
    #     Chat = transpose(Chat_t)
    #     setproperty!(operators, :C, Chat)
    # end

    # TD = 1  # initialize this dummy variable for total dimension (TD)
    # for (i, symbol) in zip(dims, operator_symbols)
    #     if 'N' in string(symbol)  # only implemented for bilinear terms
    #         m = i ÷ n  # number of inputs
    #         if m == 1
    #             setproperty!(operators, symbol, O[:, TD:TD+i-1])
    #         else 
    #             Nhat = zeros(n,n,m)
    #             tmp = O[:, TD:TD+i-1]
    #             for i in 1:m
    #                 Nhat[:,:,i] = tmp[:, int(n*(i-1)+1):int(n*i)]
    #             end
    #             setproperty!(operators, symbol, Nhat)
    #         end
    #     else
    #         setproperty!(operators, symbol, O[:, TD:TD+i-1])
    #     end
    #     TD += i
    # end

    unpack_operators!(operators, O, Y, Xhat_t, dims, operator_symbols, options)
    
    # TD = 0  # initialize this dummy variable for total dimension (TD)
    # if options.system.is_lin
    #     Ahat = O[:, TD+1:n]
    #     TD += n
    # else
    #     Ahat = 0
    # end
    # if options.system.has_control
    #     Bhat = O[:, TD+1:TD+m]
    #     TD += m
    # else
    #     Bhat = 0
    # end

    # # Compute Chat by solving the least square problem for the output values 
    # if options.system.has_output
    #     Chat_t = zeros(n, l)
    #     Yt = transpose(Y)
    #     if options.with_reg && options.λ.output != 0
    #         Chat_t = (Xhat_t' * Xhat_t + options.λ.output * I) \ (Xhat_t' * Yt)
    #     else
    #         Chat_t = Xhat_t \ Yt
    #     end
    #     Chat = transpose(Chat_t)
    # else
    #     Chat = 0
    # end

    # # Extract Quadratic terms if the system includes such terms
    # # sv = 0  # initialize this dummy variable just in case
    # if options.system.is_quad
    #     if options.optim.which_quad_term == "F"
    #         Fhat = O[:, TD+1:TD+s2]
    #         Hhat = F2Hs(Fhat)
    #         TD += s2
    #     else
    #         Hhat = O[:, TD+1:TD+v2]
    #         Fhat = H2F(Hhat)
    #         TD += v2
    #     end
    # else
    #     Fhat = 0
    #     Hhat = 0
    # end

    # # Extract Cubic terms if the system includes such terms
    # # sv3 = 0  # initialize this dummy variable just in case
    # if options.system.is_cubic
    #     if options.optim.which_cubic_term == "E"
    #         Ehat = O[:, TD+1:TD+s3]
    #         Ghat = E2Gs(Ehat)
    #         TD += s3
    #     else
    #         Ghat = O[:, TD+1:TD+v3]
    #         Ehat = G2E(Ghat)
    #         TD += v3
    #     end
    # else
    #     Ehat = 0
    #     Ghat = 0
    # end

    # # Extract Bilinear terms 
    # # FIX: fix this so that you can extract for a general case where there
    # # are more than 1 input
    # if options.system.is_bilin
    #     if m == 1
    #         Nhat = O[:, TD+1:TD+w1]
    #     else 
    #         Nhat = zeros(m,n,n)
    #         tmp = O[:, TD+1:TD+w1]
    #         for i in 1:m
    #             Nhat[:,:,i] .= tmp[:, Int(n*(i-1)+1):Int(n*i)]
    #         end
    #     end
    #     TD += w1
    # else
    #     Nhat = (m == 0) || (m == 1) ? 0 : zeros(n,n,m)
    # end

    # # Constant term
    # Khat = options.system.has_const ? Matrix(O[:, TD+1:end]) : 0

    # return Ahat, Bhat, Chat, Fhat, Hhat, Ehat, Ghat, Nhat, Khat
    return operators
end


function unpack_operators!(operators::Operators, O::AbstractArray, Y::AbstractArray, Xhat_t::AbstractArray,
                           dims::AbstractArray, operator_symbols::AbstractArray, options::AbstractOption)
    n = size(O, 1)

    # Compute Chat by solving the least square problem for the output values 
    if !iszero(options.system.output)
        Yt = fat2tall(Y)
        l = size(Yt, 2)
        Chat_t = zeros(n, l)
        if options.with_reg && options.λ.C != 0
            Chat_t = (Xhat_t' * Xhat_t + options.λ.C * I) \ (Xhat_t' * Yt)
        else
            Chat_t = Xhat_t \ Yt
        end
        Chat = transpose(Chat_t)
        setproperty!(operators, :C, Chat)
    end

    TD = 1  # initialize this dummy variable for total dimension (TD)
    for (i, symbol) in zip(dims, operator_symbols)
        if 'N' in string(symbol)  # only implemented for bilinear terms
            m = i ÷ n  # number of inputs
            if m == 1
                setproperty!(operators, symbol, O[:, TD:TD+i-1])
            else 
                Nhat = zeros(n,n,m)
                tmp = O[:, TD:TD+i-1]
                for i in 1:m
                    Nhat[:,:,i] = tmp[:, int(n*(i-1)+1):int(n*i)]
                end
                setproperty!(operators, symbol, Nhat)
            end
        else
            setproperty!(operators, symbol, O[:, TD:TD+i-1])
        end
        TD += i
    end
end


"""
    run_optimizer(D::AbstractArray, Rt::AbstractArray, Y::AbstractArray,
        Xhat_t::AbstractArray, dims::Dict, options::AbstractOption,
        IG::Operators=Operators()) → op::Operators

Run the optimizer of choice.

## Arguments
- `D::AbstractArray`: data matrix
- `Rt::AbstractArray`: derivative data matrix (transposed)
- `Y::AbstractArray`: output data matrix
- `Xhat_t::AbstractArray`: projected data matrix (transposed)
- `options::AbstractOption`: options for the operator inference set by the user
- `IG::Operators`: initial guesses for optimization

## Returns
- `op::Operators`: All learned operators 
"""
function run_optimizer(D::AbstractArray, Rt::AbstractArray, Y::AbstractArray,
    Xhat_t::AbstractArray, dims::AbstractArray, operator_symbols::AbstractArray,
    options::AbstractOption, IG::Operators=Operators())

    if options.method == "LS"
        # Ahat, Bhat, Chat, Fhat, Hhat, Ehat, Ghat, Nhat, Khat = LS_solve(D, Rt, Y, Xhat_t, options)
        operators = LS_solve(D, Rt, Y, Xhat_t, dims, operator_symbols, options)
    # elseif options.method == "NC"  # Non-constrained
    #     # Ahat, Bhat, Fhat, Hhat, Nhat, Khat = NC_Optimize(D, Rt, options, IG)
    #     # Chat = options.system.has_output ? NC_Optimize_output(Y, Xhat_t, options) : 0
    #     operators = NC_Optimize(D, Rt, dims, operator_symbols, options, IG)
    #     operators.C = 1 in options.system.output ? NC_Optimize_output(Y, Xhat_t, options) : 0
    #     # Qhat = 0
    #     # Ghat = 0
    #     # Ehat = 0
    # elseif options.method == "EPHEC" || options.method == "EPHC"
    #     Ahat, Bhat, Fhat, Hhat, Nhat, Khat = EPHEC_Optimize(D, Rt, options, IG)
    #     Chat = options.system.has_output ? NC_Optimize_output(Y, Xhat_t, options) : 0
    #     Qhat = H2Q(Hhat)
    #     Ghat = 0
    #     Ehat = 0
    # elseif options.method == "EPSC" || options.method == "EPSIC"
    #     Ahat, Bhat, Fhat, Hhat, Nhat, Khat = EPSIC_Optimize(D, Rt, options, IG)
    #     Chat = options.system.has_output ? NC_Optimize_output(Y, Xhat_t, options) : 0
    #     Qhat = H2Q(Hhat)
    #     Ghat = 0
    #     Ehat = 0
    # elseif options.method == "EPP"
    #     Ahat, Bhat, Fhat, Hhat, Nhat, Khat = EPP_Optimize(D, Rt, options, IG)
    #     Chat = options.system.has_output ? NC_Optimize_output(Y, Xhat_t, options) : 0
    #     Qhat = H2Q(Hhat)
    #     Ghat = 0
    #     Ehat = 0
    else
        error("Incorrect optimization options.")
    end
    
    # op = Operators(
    #     A=Ahat, B=Bhat, C=Chat, F=Fhat, H=Hhat, 
    #     E=Ehat, G=Ghat, N=Nhat, K=Khat, Q=Qhat,
    # )
    # return op
    return operators
end


"""
    opinf(X::AbstractArray, Vn::AbstractArray, options::AbstractOption; 
        U::AbstractArray=zeros(1,1), Y::AbstractArray=zeros(1,1),
        Xdot::AbstractArray=[], IG::Operators=Operators()) → op::Operators

Infer the operators with derivative data given. NOTE: Make sure the data is 
constructed such that the row is the state vector and the column is the time.

## Arguments
- `X::AbstractArray`: state data matrix
- `Vn::AbstractArray`: POD basis
- `options::AbstractOption`: options for the operator inference defined by the user
- `U::AbstractArray`: input data matrix
- `Y::AbstractArray`: output data matix
- `Xdot::AbstractArray`: derivative data matrix
- `IG::Operators`: initial guesses for optimization

## Returns
- `op::Operators`: inferred operators
"""
function opinf(X::AbstractArray, Vn::AbstractArray, options::AbstractOption; 
               U::AbstractArray=zeros(1,1), Y::AbstractArray=zeros(1,1),
               Xdot::AbstractArray=[], IG::Operators=Operators())::Operators
    U = fat2tall(U)  # make sure that the U-matrix is tall
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

    # define_dimensions!(Xhat, U, Y, options)
    D, dims, op_symbols = getDataMat(Xhat, Xhat_t, U, options; verbose=true)
    op = run_optimizer(D, Rt, Y, Xhat_t, dims, op_symbols, options, IG)
    return op
end


"""
    opinf(X::AbstractArray, Vn::AbstractArray, full_op::Operators, options::AbstractOption;
        U::AbstractArray=zeros(1,1), Y::AbstractArray=zeros(1,1), IG::Operators=Operators()) → op::Operators

Infer the operators with reprojection method (dispatch). NOTE: Make sure the data is
constructed such that the row is the state vector and the column is the time.

## Arguments
- `X::AbstractArray`: state data matrix
- `Vn::AbstractArray`: POD basis
- `full_op::Operators`: full order model operators
- `options::AbstractOption`: options for the operator inference defined by the user
- `U::AbstractArray`: input data matrix
- `Y::AbstractArray`: output data matix
- `IG::Operators`: initial guesses for optimization

## Returns
- `op::Operators`: inferred operators
"""
function opinf(X::AbstractArray, Vn::AbstractArray, full_op::Operators, options::AbstractOption;
               U::AbstractArray=zeros(1,1), Y::AbstractArray=zeros(1,1), IG::Operators=Operators(), 
               return_derivative::Bool=false)
    U = fat2tall(U)
    Xhat = Vn' * X
    Xhat_t = transpose(Xhat)
    # define_dimensions!(Xhat, U, Y, options)

    # Reproject
    Rt = reproject(Xhat, Vn, U, full_op, options)
    D, dims, op_symbols = getDataMat(Xhat, Xhat_t, U, options; verbose=true)
    op = run_optimizer(D, Rt, Y, Xhat_t, dims, op_symbols, options, IG)

    if return_derivative
        return op, Rt
    else
        return op
    end
end


"""
    reproject(Xhat::AbstractArray, V::AbstractArray, U::AbstractArray,
        op::Operators, options::AbstractOption) → Rhat::AbstractArray

Reprojecting the data to minimize the error affected by the missing orders of the POD basis

## Arguments
- `Xhat::AbstractArray`: state data matrix projected onto the basis
- `V::AbstractArray`: POD basis
- `U::AbstractArray`: input data matrix
- `op::Operators`: full order model operators
- `options::AbstractOption`: options for the operator inference defined by the user

## Return
- `Rhat::AbstractArray`: R matrix (transposed) for the regression problem
"""
function reproject(Xhat::AbstractArray, V::AbstractArray, U::AbstractArray,
    op::Operators, options::AbstractOption)::AbstractArray
    
    # Just a simple error detection
    # if options.system.is_quad && options.optim.which_quad_term=="F"
    #     if op.F == 0
    #         error("Quadratic F matrix should not be 0 if it is selected.")
    #     end
    # elseif options.system.is_quad && options.optim.which_quad_term=="H"
    #     if op.H == 0
    #         error("Quadratic H matrix should not be 0 if it is selected.")
    #     end
    # end

    n, K = size(Xhat)
    # Rt = zeros(options.system.dims[:K], options.system.dims[:n])  # Left hand side of the regression problem
    Rt = zeros(K, n)  # Left hand side of the regression problem

    # if options.system.has_funcOp
    # if options.system.funtion_operator
    #     f = (x, u) -> op.A * x + op.f(x) + op.B * u + op.K
    # else
    #     # m = options.system.dims[:m]
    #     m = op.dims[:B]

    #     # fA = (x) -> options.system.is_lin ? op.A * x : 0
    #     # fB = (u) -> options.system.has_control ? op.B * u : 0
    #     # fF = (x) -> options.system.is_quad && options.optim.which_quad_term == "F" ? op.F * (x ⊘ x) : 0
    #     # fH = (x) -> options.system.is_quad && options.optim.which_quad_term == "H" ? op.H * (x ⊗ x) : 0
    #     # fE = (x) -> options.system.is_cubic && options.optim.which_cubic_term == "E" ? op.E * ⊘(x,x,x) : 0
    #     # fG = (x) -> options.system.is_cubic && options.optim.which_cubic_term == "G" ? op.G * (x ⊗ x ⊗ x) : 0
    #     # fN = (x,u) -> options.system.is_bilin ? ( m==1 ? (op.N * x) * u : sum([(op.N[i] * x) * u[i] for i in 1:m]) ) : 0
    #     # fK = options.system.has_const ? op.K : 0

    #     fA = (x) -> 1 in options.system.state ? op.A * x : 0
    #     fB = (u) -> 1 in options.system.control ? op.B * u : 0
    #     fF = (x) -> 2 in options.system.state && options.optim.nonredundant_operators ? op.A2u * (x ⊘ x) : 0
    #     fH = (x) -> 2 in options.system.state && !options.optim.nonredundant_operators ? op.A2 * (x ⊗ x) : 0
    #     fE = (x) -> 3 in options.system.state && options.optim.nonredundant_operators ? op.A3u * ⊘(x,x,x) : 0
    #     fG = (x) -> 3 in options.system.state && !options.optim.nonredundant_operators ? op.A3 * (x ⊗ x ⊗ x) : 0
    #     fN = (x,u) -> 1 in options.system.coupled_input ? ( m==1 ? (op.N * x) * u : sum([(op.N[i] * x) * u[i] for i in 1:m]) ) : 0
    #     fK = !iszero(options.system.constant) ? op.K : 0

    #     f = (x,u) -> fA(x) .+ fB(u) .+ fF(x) .+ fH(x) .+ fE(x) .+ fG(x) .+ fN(x,u) .+ fK
    # end

    # Assuming the user gave the nonlinear functional or the Operator structure predefined the nonlinear functional
    f = (x, u) -> op.A * x .+ op.B * u .+ op.K .+ op.f(x,u)

    # for i in 1:options.system.dims[:K]  # loop thru all data
    for i in 1:K  # loop thru all data
        x = Xhat[:, i]  # julia automatically makes into column vec after indexing (still xrow-tcol)
        xrec = V * x
        states = f(xrec, U[i, :])
        Rt[i, :] = V' * states
    end
    return Rt
end

