export StreamingOpInf

abstract type StreamingOpInf end

# Import the algorithms
include("algorithms/RLS/rls.jl")
include("algorithms/iQRRLS/iqrrls.jl")
include("algorithms/QRRLS/qrrls.jl")

# Import the streaming methods
include("algorithms/RLS/stream.jl")
# include("algorithms/iQRRLS/stream.jl")
# include("algorithms/QRRLS/stream.jl")

"""
$(TYPEDEF)

Streaming Operator Inference/Lift And Learn
"""
function StreamingOpInf(;
    options::LSOpInfOption,             # Standard (Least-Squares) Operator Inference options
    n::Int, m::Int, l::Int,             # state, input, and output dimensions
    algorithm::Symbol=:RLS,             # algorithm type
    γs=0.0, γy=0.0, λ=1.0,              # regularization terms and forgetting factor
    variable_regularize::Bool=false     # variable regularization flag
    )

    # Initialize the dimensions 
    # TODO: Currently only supports up to
    # - quartic operators
    # - bilinear (state-input coupling) operators
    # nml = Dict(
    #     :A => n,  # state dimension
    #     :B => m,  # input dimension
    #     :C => l,  # output dimension 
    #     :A2  => (2 ∈ options.system.state) && !options.optim.nonredundant_operators ? Int(n^2) : 0,                     # quadratic terms
    #     :A2u => (2 ∈ options.system.state) && options.optim.nonredundant_operators  ? Int(n*(n+1)/2) : 0,               # 'u'nique quadratic terms
    #     :A3  => (3 ∈ options.system.state) && !options.optim.nonredundant_operators ? Int(n^3) : 0,                     # cubic terms
    #     :A3u => (3 ∈ options.system.state) && options.optim.nonredundant_operators  ? Int(n*(n+1)*(n+2)/6) : 0,         # 'u'nique cubic terms
    #     :A4  => (4 ∈ options.system.state) && !options.optim.nonredundant_operators ? Int(n^4) : 0,                     # quartic terms
    #     :A4u => (4 ∈ options.system.state) && options.optim.nonredundant_operators  ? Int(n*(n+1)*(n+2)*(n+3)/24) : 0,  # 'u'nique quartic terms
    #     :N  => 1 ∈ options.system.coupled_input ? Int(n*m) : 0,  # bilinear term
    #     :K  => options.system.constant ? 1 : 0,                  # constant term
    # ) 
    dims = Dict(:n => n, :m => m, :l => l)
    d = 0  # total dimension of the data matrix
    for i in options.system.state
        if i == 1
            d += n
        else
            d += binomial(n+i-1, i)
        end
    end
    for i in options.system.control
        if i == 1
            d += m
        else
            d += binomial(m+i-1, i)
        end
    end
    for i in options.system.coupled_input
        if i == 1
            d += n*m
        else
            d += binomial(n+i-1, i) * m
        end
    end
    if options.system.constant
        d += 1
    end

    # Initialize variables based on algorithm
    if algorithm == :RLS
        O  = zeros(d,n)
        P  = iszero(γs) ? Matrix{<:Number}(undef,0,0) : 1.0I(d) / γs
        K  = Matrix{<:Number}(undef, 0, 0)
        Y  = zeros(n,l)
        Py = iszero(γy) ? Matrix{<:Number}(undef,0,0) : 1.0I(n) / γy
        Ky = Matrix{<:Number}(undef,0,0)
        e  = Matrix{<:Number}(undef,0,0)
        ξ  = Matrix{<:Number}(undef,0,0)
        C  = Matrix{<:Number}(undef,0,0)
        J  = Matrix{<:Number}(undef,0,0)

        # State regression
        state_cache = RLSCache(
            O, P, K, e, ξ, C, J, γs, λ
        )
        state_rls = RLSOpInf(state_cache, dims, Dict{Symbol,Any}(), options, variable_regularize, iszero(γs))
        if iszero(l)
            return state_rls
        end

        # Output regression
        output_cache = RLSCache(
            Y, Py, Ky, e, ξ, C, J, γy, λ
        )
        output_rls = RLSOpInf(output_cache, dims, Dict{Symbol,Any}(), options, variable_regularize, iszero(γy))
        return state_rls, output_rls
    elseif algorithm == :QRRLS
        O  = zeros(d,n)
        P  = Matrix{<:Number}(undef,0,0)
        K  = Matrix{<:Number}(undef,0,0)
        Y  = zeros(n,l)
        Py = Matrix{<:Number}(undef,0,0)
        Ky = Matrix{<:Number}(undef,0,0)
        Φ  = sqrt(γs) * 1.0I(d)
        q  = zeros(d,n)
        Φy = sqrt(γy) * 1.0I(n)
        qy = zeros(n,l)
        e  = Matrix{<:Number}(undef,0,0)
        ξ  = Matrix{<:Number}(undef,0,0)
        C  = Matrix{<:Number}(undef,0,0)
        J  = Matrix{<:Number}(undef,0,0)

        # State regression
        state_cache = QRRLSCache(
            O, P, K, Φ, q, e, ξ, C, J, γs, λ
        )
        state_qrrls = QRRLSOpInf(state_cache, dims, options)
        if iszero(l)
            return state_qrrls
        end

        # Output regression
        output_cache = QRRLSCache(
            Y, Py, Ky, Φy, qy, e, ξ, C, J, γy, λ
        )
        output_qrrls = QRRLSOpInf(output_cache, dims, options)
        return state_qrrls, output_qrrls
    elseif algorithm == :iQRRLS
        O    = zeros(d,n)
        Psq  = 1.0I(d) / sqrt(γs)
        K    = Matrix{<:Number}(undef,0,0)
        Y    = zeros(n,l)
        Psqy = 1.0I(n) / sqrt(γy)
        Ky   = Matrix{<:Number}(undef,0,0)
        Φ  = Matrix{<:Number}(undef,0,0)
        q  = Matrix{<:Number}(undef,0,0)
        Φy = Matrix{<:Number}(undef,0,0)
        qy = Matrix{<:Number}(undef,0,0)
        e  = Matrix{<:Number}(undef,0,0)
        ξ  = Matrix{<:Number}(undef,0,0)
        C  = Matrix{<:Number}(undef,0,0)
        J  = Matrix{<:Number}(undef,0,0)

        # State regression
        state_cache = iQRRLSCache(
            O, Psq, K, e, ξ, C, J, γs, λ
        )
        state_iqrrls = iQRRLSOpInf(state_cache, dims, options)
        if iszero(l)
            return state_iqrrls
        end

        # Output regression
        output_cache = iQRRLSCache(
            Y, Psqy, Ky, e, ξ, C, J, γy, λ
        )
        output_iqrrls = iQRRLSOpInf(output_cache, dims, options)
        return state_iqrrls, output_iqrrls
    else
        error("Available algorithms are RLS, QRRLS, and iQRRLS.")
    end
end


"""
$(SIGNATURES)

Update the streaming operator inference with new data by solving a recursive least-squares problem via 
the standard Recursive Least-Squares (RLS) algorithm with regularization.

# Note 
- For the RLS algorithm, the regularization term is updated if `variable_regularize` is enabled
- The RLS algorithm also allows for rank-k update if the data-stream `X` is rank higher than 1
- The RLS algorithm also permits noise in terms of a noise covariance matrix `Q`
"""
function stream!(obj::StreamingOpInf, X::AbstractArray{T}, R::AbstractArray{T}; U::AbstractArray{T}=T[], 
                 Q::Union{T,AbstractArray{<:Real}}=size(X,2)==1 ? 1.0 : 1.0I(size(X,2)),
                 γs::Real=0.0) where T<:Number

    tdim = size(X_k, 2)  # number of data points (time dimension)

    # Construct the data matrix while checking the dimension of the input matrix
    foo, bar = checksize(U) 
    if foo == obj.dims[:m] && bar == tdim
        if foo == bar && foo != 1
            @warn "Transposing while assuming the row dim is the input dim and the column dim is the number of data points."
        end
        D = getDataMat(X, U', obj.options; verbose=false)
    else
        D = getDataMat(X, U, obj.options; verbose=false)
    end

    # Reorganize the dimension of the derivative data matrix
    foo, bar = checksize(R_k)
    if foo == obj.dims[:n] && bar == tdim
        if foo == bar
            @warn "Transposing while assuming the row dim is the state dim and the column dim is the number of data points."
        end
        R = R'
    end


    if stream.algorithm == :RLS
        # Execute the update
        if stream.variable_regularize  # if variable regularization is enabled
            stream.O_k, stream.P_k, stream.K_k = RLS(D_k, R_k, stream.O_k, stream.P_k, Q_k, 
                                                    γs_k, stream.γs_k, stream.atol[1], stream.rtol[1])
            stream.γs_k = γs_k  # update the regularization term
        else
            if stream.zero_reg_start_state
                Q_k_inv = Q_k \ I
                if iszero(stream.atol[1]) 
                    stream.P_k = (D_k' * Q_k_inv * D_k) \ I
                else
                    stream.P_k = pinv(D_k' * Q_k_inv * D_k; atol=stream.atol[1], rtol=stream.rtol[1])
                end
                stream.K_k = stream.P_k * D_k' * Q_k_inv
                stream.O_k = stream.K_k * R_k
                stream.zero_reg_start_state = false  # disable initial zero regularization
            else
                stream.O_k, stream.P_k, stream.K_k = RLS(D_k, R_k, stream.O_k, stream.P_k, Q_k, stream.atol[1], stream.rtol[1])
            end
        end
    elseif stream.algorithm == :QRRLS
        @assert stream.dims[:K] == 1 "QRRLS is only for rank-1 update."
        stream.O_k, stream.Φ_k, stream.q_k, stream.P_k, stream.K_k = QRRLS(D_k, R_k, stream.Φ_k, stream.q_k, 
                                                                           stream.dims[:d], stream.dims[:n])
    elseif stream.algorithm == :iQRRLS
        @assert stream.dims[:K] == 1 "iQRRLS is only for rank-1 update."
        stream.O_k, stream.P_k, stream.K_k = iQRRLS(D_k, R_k, stream.O_k, stream.P_k, stream.dims[:d])
    else
        error("Available algorithms are RLS, QRRLS, and iQRRLS.")
    end
    return D_k
end








"""
$(SIGNATURES)

Single stream update for the output data.
"""
function stream_output!(stream::RLSOpInf, X::AbstractArray{T}, Y::AbstractArray{T}; γy::Real=0.0, 
                        Z::Union{T,AbstractArray{T}}=size(X_k,2)==1 ? 1.0 : 1.0I(size(X_k,2))) where T<:Number
    tdim = size(X_k, 2)  # number of data points (time dimension)
    foo, bar = checksize(Y_k)
    if foo == stream.dims[:l] && bar == tdim
        if foo == bar && foo != 1
            @warn "Transpose while assuming the row dim is the output dim and the column dim is the number of data points."
        end
        Y_k = Y_k'
        stream.dims[:l] = foo
    else
        stream.dims[:l] = bar
    end
    Xt = X

    if stream.algorithm == :RLS
        if stream.variable_regularize  # if variable regularization is enabled
            stream.C_k, stream.Py_k, stream.Ky_k = RLS(Xt_k, Y_k, stream.C_k, stream.Py_k, Z_k, 
                                                        γo_k, stream.γo_k, stream.atol[2], stream.rtol[2])
            stream.γo_k = γo_k  # update the regularization term
        else 
            if stream.zero_reg_start_output
                Z_k_inv = Z_k \ I
                Xt_k = transpose(X_k)
                if iszero(stream.atol[2])
                    stream.Py_k =  (Xt_k' * Z_k_inv * Xt_k) \ I
                else
                    stream.Py_k = pinv(Xt_k' * Z_k_inv * Xt_k; atol=stream.atol[2], rtol=stream.rtol[2])
                end
                stream.Ky_k = stream.Py_k * Xt_k' * Z_k_inv
                stream.C_k = stream.Ky_k * Y_k
                stream.zero_reg_start_output = false  # disable initial zero regularization
            else
                stream.C_k, stream.Py_k, stream.Ky_k = RLS(Xt_k, Y_k, stream.C_k, stream.Py_k, Z_k, stream.atol[2], stream.rtol[2])
            end
        end
    elseif stream.algorithm == :QRRLS
        @assert stream.dims[:K] == 1 "QRRLS is only for rank-1 update."
        stream.C_k, stream.Φy_k, stream.qy_k, stream.Py_k, stream.Ky_k = QRRLS(Xt_k, Y_k, stream.Φy_k, stream.qy_k, 
                                                                               stream.dims[:n], stream.dims[:l])
    elseif stream.algorithm == :iQRRLS
        @assert stream.dims[:K] == 1 "iQRRLS is only for rank-1 update."
        stream.C_k, stream.Py_k, stream.Ky_k = iQRRLS(Xt_k, Y_k, stream.C_k, stream.Py_k, stream.dims[:n])
    end
    return nothing
end


"""
$(SIGNATURES)

Update the streaming operator inference continuously with all the data streams.
"""
function stream_all!(stream::StreamingOpInf, X::AbstractArray{<:AbstractArray{T}}, R::AbstractArray{<:AbstractArray{T}}; 
                     U::AbstractArray{<:AbstractArray{T}}=Vector{T}[], γs::AbstractArray{<:Real}=zeros(length(X)),
                     Q::Union{AbstractArray{<:AbstractArray{T}},AbstractArray{T},Real}=0.0) where T<:Number
    N = length(X)
    D = nothing # initialize the data matrix
    flag = typeof(Q) <: AbstractArray{T}
    no_input = isempty(U)
    for i in 1:N
        if iszero(Q)
            D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i], γs=γs[i])
        else
            D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i], γs=γs[i], Q=flag ? Q : Q[i])
        end
    end
    return D
end


function stream_output_all!(stream::StreamingOpInf, X::AbstractArray{<:AbstractArray{T}}, 
                            Y::AbstractArray{<:AbstractArray{T}}; γy::AbstractArray{<:Real}=zeros(length(X)),
                            Z::Union{AbstractArray{<:AbstractArray{T}},AbstractArray{T},Real}=0.0) where T<:Number
    N = length(X)
    flag = typeof(Z) <: AbstractArray{T}
    for i in 1:N
        if iszero(Z)
            stream_output!(stream, X[i], Y[i]; γy=γy[i])
        else
            stream_output!(stream, X[i], Y[i]; γy=γy[i], Z=flag ? Z : Z[i])
        end
    end
    return nothing
end


function terminate_stream(obj::StreamingOpInf) where T<:Number
    # Extract the operators
    operators = Operators()
    unpack_operators!(
        operators, obj.cache.O, 
        obj.termination_settings[:dims], obj.termination_settings[:syms])
    return operators
end