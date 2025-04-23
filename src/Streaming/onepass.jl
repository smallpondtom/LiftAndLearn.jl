export OnePassStreamingOpInf

mutable struct OnePassStreamingOpInf{T<:Number}
    # iSVD components
    V::AbstractArray{T}
    Σ::AbstractVector{T}
    W::AbstractArray{T}
    P::AbstractArray{T}
    S::AbstractVector{T}
    Q::AbstractArray{T}

    # Dimensions
    state_dim::Int
    input_dim::Int
    r_s::Int
    r_d::Int
    rmax::Int

    # Options
    options::LSOpInfOption
end

function OnePassStreamingOpInf(
    x_init::AbstractVector{T},          # Initial state data 
    xdot_init::AbstractVector{T};       # Initial derivative data
    options::LSOpInfOption,             # Standard (Least-Squares) Operator Inference options
    n::Int, m::Int=0,                   # state (n) and input (m) dimensions
    # isvd_algo::Symbol=:baker,           # algorithm type
    # reorth_algo::Symbol=:gramschmidt,   # reorthogonalization algorithm
    rank::Int=1                         # maximum rank for the iSVDs
    ) where {T<:Number}

    # Initialize the iSVDs 
    # isvd_algo = (Symbol ∘ lowercase ∘ string)(isvd_algo) # Ensure the algorithm is lowercase
    # reorth_algo = (Symbol ∘ lowercase ∘ string)(reorth_algo) # Ensure lowercase
    # if isvd_algo == :baker
    #     # Initialize the Baker algorithm for state and derivative data
    #     state_isvd = initialize_baker(x_init, max_rank=rank)
    #     deriv_isvd = initialize_baker(xdot_init, max_rank=rank)
    # elseif isvd_algo == :brand
    #     # Initialize the Brand algorithm for state and derivative data
    #     state_isvd = initialize_brand(x_init, max_rank=rank, reorth_method=reorth_algo)
    #     deriv_isvd = initialize_brand(xdot_init, max_rank=rank, reorth_method=reorth_algo)
    # else
    #     error("Invalid algorithm specified. Use :baker or :brand.")
    # end

    V = reshape(x_init / norm(x_init), :, 1)
    Σ = [norm(x_init)]
    W = reshape([T(1)], 1, 1)
    P = reshape(xdot_init / norm(xdot_init), :, 1)
    S = [norm(xdot_init)]
    Q = reshape([T(1)], 1, 1)

    OnePassStreamingOpInf(
        V, Σ, W, P, S, Q,
        n, m, 1, 1, rank, options
    )
end

function stream!(obj::OnePassStreamingOpInf, x::AbstractVector{T}, 
    xdot::AbstractVector{T}) where {T<:Real}

    # # Increment the state and derivative iSVDs with the new data
    # increment!(obj.state_isvd, x)
    # increment!(obj.deriv_isvd, xdot)

    # # Update the reduced dimension
    # obj.reduced_dim = size(obj.state_isvd.V, 2)

    # dimensions
    r1 = obj.r_s
    r2 = obj.r_d
    rmax = copy(obj.rmax)

    q1 = obj.V' * x
    xperp = x - obj.V * q1
    q2 = obj.V' * xperp
    xperp = xperp - obj.V * q2
    q = q1 + q2
    p = norm(xperp)

    p = [p]
    xperp = reshape(xperp, :, 1)
    qrf!(xperp, p)
    p = p[1]

    C = zeros(r1+1, r1+1)
    for j in 1:r1
        C[j,j] = obj.Σ[j]
        C[j,end] = q[j]
    end
    C[end,end] = p

    Vc, Σc, Wc = svd(C)
    obj.V = hcat(obj.V, xperp) * Vc
    obj.Σ = Σc
    obj.W = [obj.W zeros(size(obj.W,1), 1); zeros(1, r1) 1.0] * Wc
    obj.r_s += 1

    q1 = obj.P' * xdot
    xdotperp = xdot - obj.P * q1
    q2 = obj.P' * xdotperp
    xdotperp = xdotperp - obj.P * q2
    q = q1 + q2
    p = norm(xdotperp)

    p = [p]
    xdotperp = reshape(xdotperp, :, 1)
    qrf!(xdotperp, p)
    p = p[1]

    C = zeros(r2+1, r2+1)
    for j in 1:r2
        C[j,j] = obj.S[j]
        C[j,end] = q[j]
    end
    C[end,end] = p

    Pc, Sc, Qc = svd(C)
    obj.P = hcat(obj.P, xdotperp) * Pc
    obj.S = Sc
    obj.Q = [obj.Q zeros(size(obj.Q,1), 1); zeros(1, r2) 1.0] * Qc
    obj.r_d += 1

    if obj.r_s > rmax
        obj.V = obj.V[:,1:rmax]
        obj.Σ = obj.Σ[1:rmax]
        obj.W = obj.W[:,1:rmax]
        obj.r_s = rmax
    end
    if obj.r_d > rmax
        obj.P = obj.P[:,1:rmax]
        obj.S = obj.S[1:rmax]
        obj.Q = obj.Q[:,1:rmax]
        obj.r_d = rmax
    end

    return nothing
end

function compute_onepass_operators(obj::OnePassStreamingOpInf, 
    U::AbstractArray{T}) where {T<:Real}

    # Diagonalize the singular values
    Σ_diag = Diagonal(obj.Σ)
    S_diag = Diagonal(obj.S)

    # Assemble the low-rank approximation of the time derivative data
    Xdot_t = obj.Q  * S_diag * obj.P'

    # Construct the reduce data matrix
    r = obj.rmax
    m = obj.input_dim
    d = 0  # total dimension of the data matrix
    d += sum(i != 0 ? binomial(r+i-1, i) : 0 for i in obj.options.system.state)
    d += sum(i != 0 ? binomial(m+i-1, i) : 0 for i in obj.options.system.control)
    # d += sum(i != 0 ? binomial(r+i-1, i) * m : 0 for i in obj.options.system.coupled_input)
    d += iszero(obj.options.system.constant) ? 0 : 1
    K = size(Xdot_t, 1)  # number of snapshots
    D = zeros(T, K, d)  # data matrix
    tmp = 0

    dims = []
    operator_symbols = []

    for i in obj.options.system.state
        if i == 1
            D[:, 1:r] = obj.W * Σ_diag
            push!(dims, r)
            push!(operator_symbols, :A)
            tmp += r
        else
            if obj.options.optim.nonredundant_operators
                ri = binomial(r+i-1, i)
                D[:, tmp+1:tmp+ri] = ⧁(obj.W, i) * Diagonal(⊘(obj.Σ, i))
                push!(dims, ri)
                push!(operator_symbols, Symbol("A$(i)u"))
            else
                ri = Int(r^i)
                D[:, tmp+1:tmp+ri] = ⊖(obj.W, i) * Diagonal(⊗(obj.Σ, i))
                push!(operator_symbols, Symbol("A$(i)"))
            end
            tmp += ri
        end
    end

    # NOTE: Only works for linear inputs (for now)
    if !iszero(obj.options.system.control)
        D[:, tmp+1:tmp+m] = U'
        tmp += m
        push!(dims, m)
        push!(operator_symbols, :B)
    end
    # NOTE: Coupled inputs are not implemented yet

    if !iszero(obj.options.system.constant)
        D[:, tmp+1] = ones(K)  # constant term
        push!(dims, 1)
        push!(operator_symbols, :K)
    end

    # Construct the reduced right-hand side matrix
    R = Xdot_t * obj.V

    # compute least squares (pseudo inverse)
    if obj.options.with_reg 
        # Preallocate the Tikhonov weight Matrix
        Γ = spzeros(d)

        # Construct the Tikhonov matrix
        tikhonov_matrix!(Γ, dims, operator_symbols, obj.options.λ)
        Γ = spdiagm(0 => Γ)  # convert to sparse diagonal matrix

        Ot = tikhonov(
            R, D, Γ, obj.options.pinv_tol; tol_flag=obj.options.with_tol, 
            use_gpu=obj.options.use_gpu, use_backslash=obj.options.use_backslash
        )
    else
        Ot = standard_least_squares(
            D, R; use_gpu=obj.options.use_gpu, 
            use_backslash=obj.options.use_backslash
        )
    end

    # Extract the operators from the operator matrix O
    O = transpose(Ot)

    # Extract the operators
    operators = Operators(O=O)

    # Unpack the operators
    unpack_operators!(operators, O, dims, operator_symbols)

    ## DELETE THIS AFTER DEBUGGING
    # D = [W * Σ_diag    U']
    # R = Xdot_t * V

    # O = D \ R

    # rmax = obj.reduced_dim

    # Astream = O[1:rmax,1:rmax]'
    # Bstream = O[rmax+1:rmax+1,1:rmax]'

    # operators = Operators(A=Astream, B=Bstream)
    ##

    return operators
end