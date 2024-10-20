export StreamingOpInf

abstract type StreamingOpInf end

# Import the algorithms
include("algorithms/RLS/rls.jl")
include("algorithms/iQRRLS/iqrrls.jl")
include("algorithms/QRRLS/qrrls.jl")


# Import the streaming methods
include("algorithms/RLS/stream.jl")
include("algorithms/iQRRLS/stream.jl")
include("algorithms/QRRLS/stream.jl")

# Each Algorithm solver structs
mutable struct RLSOpInf{T<:Number} <: StreamingOpInf
    cache::RLSCache{T}
    # Dimensions
    dims::Dict{Symbol,Int}
    # Termination settings
    termination_settings::Dict{Symbol,Any}
    # Options
    options::LSOpInfOption         # Standard (Least-Squares) Operator Inference options
    variable_regularization::Bool  # variable regularization flag
    initial_step::Bool             # Flag for initial step when γs is zero
end

mutable struct iQRRLSOpInf{T<:Number} <: StreamingOpInf
    cache::iQRRLSCache{T}
    # Dimensions
    dims::Dict{Symbol,Int}
    # Termination settings
    termination_settings::Dict{Symbol,Any}
    # Options
    options::LSOpInfOption  # Standard (Least-Squares) Operator Inference options
end

mutable struct QRRLSOpInf{T<:Number} <: StreamingOpInf
    cache::QRRLSCache{T}
    # Dimensions
    dims::Dict{Symbol,Int}
    # Termination settings
    termination_settings::Dict{Symbol,Any}
    # Options
    options::LSOpInfOption  # Standard (Least-Squares) Operator Inference options
end


"""
$(TYPEDEF)

Streaming Operator Inference/Lift And Learn
"""
function StreamingOpInf(;
    options::LSOpInfOption,             # Standard (Least-Squares) Operator Inference options
    n::Int, m::Int, l::Int,             # state, input, and output dimensions
    algorithm::Symbol=:RLS,             # algorithm type
    γs=0.0, γo=0.0, λ=1.0,              # regularization terms and forgetting factor
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
        d += binomial(n+i-1, i)
    end
    for i in options.system.control
        d += binomial(m+i-1, i)
    end
    for i in options.system.coupled_input
        d += binomial(n+i-1, i) * m
    end
    if options.system.constant
        d += 1
    end
    dims[:d] = d

    # Initialize variables based on algorithm
    if algorithm == :RLS
        Os    = zeros(d,n)
        Ps    = iszero(γs) ? Matrix{<:Number}(undef,0,0) : 1.0I(d) / γs
        Ks    = Matrix{<:Number}(undef, 0, 0)
        Oo    = zeros(n,l)
        Po    = iszero(γo) ? Matrix{<:Number}(undef,0,0) : 1.0I(n) / γo
        Ko    = Matrix{<:Number}(undef,0,0)
        ξpre  = Matrix{<:Number}(undef,0,0)
        ξpost = Matrix{<:Number}(undef,0,0)
        C     = Matrix{<:Number}(undef,0,0)
        J     = Matrix{<:Number}(undef,0,0)

        # State regression
        state_cache = RLSCache(
            Os, Ps, Ks, ξpre, ξpost, C, J, γs, λ,
            zeros(d,1), Matrix{<:Number}(undef,0,0),
            Matrix{<:Number}(undef,0,0),
            zeros(d,d), zeros(d,n),
        )
        state_rls = RLSOpInf(
            state_cache, dims, Dict{Symbol,Any}(), options, 
            variable_regularize, iszero(γs)
        )
        if iszero(l)
            return state_rls
        end

        # Output regression
        output_cache = RLSCache(
            Oo, Po, Ko, ξpre, ξpost, C, J, γo, λ,
            zeros(d,1), Matrix{<:Number}(undef,0,0),
            Matrix{<:Number}(undef,0,0),
            zeros(d,d), zeros(d,n),
        )
        output_rls = RLSOpInf(
            output_cache, dims, Dict{Symbol,Any}(), options, 
            variable_regularize, iszero(γy)
        )
        return state_rls, output_rls
    elseif algorithm == :QRRLS
        Os    = zeros(d,n)
        Ps    = Matrix{<:Number}(undef,0,0)
        Ks    = Matrix{<:Number}(undef,0,0)
        Oo    = zeros(n,l)
        Po    = Matrix{<:Number}(undef,0,0)
        Ko    = Matrix{<:Number}(undef,0,0)
        Φsqs  = sqrt(γs) * 1.0I(d)
        qs    = zeros(d,n)
        Φsqo  = sqrt(γo) * 1.0I(n)
        qo    = zeros(n,l)
        ξpre  = Matrix{<:Number}(undef,0,0)
        ξpost = Matrix{<:Number}(undef,0,0)
        C     = 0
        J     = 0

        # State regression
        state_cache = QRRLSCache(
            Os, Ps, Ks, Φsqs, qs, ξpre, ξpost, C, J, γs, λ,
            zeros(d+n+1,d+n+1), zeros(1,n), zeros(d,1)
        )
        state_qrrls = QRRLSOpInf(state_cache, dims, Dict{Symbol,Any}(), options)
        if iszero(l)
            return state_qrrls
        end

        # Output regression
        output_cache = QRRLSCache(
            Oo, Po, Ko, Φsqo, qo, ξpre, ξpost, C, J, γo, λ,
            zeros(d+n+1,d+n+1), zeros(1,n), zeros(d,1)
        )
        output_qrrls = QRRLSOpInf(output_cache, dims, Dict{Symbol,Any}(), options)
        return state_qrrls, output_qrrls
    elseif algorithm == :iQRRLS
        Os    = zeros(d,n)
        Psqs  = 1.0I(d) / sqrt(γs)
        Ks    = Matrix{<:Number}(undef,0,0)
        Oo    = zeros(n,l)
        Psqo  = 1.0I(n) / sqrt(γo)
        Ko    = Matrix{<:Number}(undef,0,0)
        Φs    = Matrix{<:Number}(undef,0,0)
        qs    = Matrix{<:Number}(undef,0,0)
        Φo    = Matrix{<:Number}(undef,0,0)
        qo    = Matrix{<:Number}(undef,0,0)
        ξpre  = Matrix{<:Number}(undef,0,0)
        ξpost = Matrix{<:Number}(undef,0,0)
        C     = 0
        J     = 0

        # State regression
        state_cache = iQRRLSCache(
            Os, Psqs, Ks, ξpre, ξpost, C, J, γs, λ,
            zeros(d+1,d+1), zeros(d), zeros(1,n), zeros(d,n)
        )
        state_iqrrls = iQRRLSOpInf(state_cache, dims, Dict{Symbol,Any}(), options)
        if iszero(l)
            return state_iqrrls
        end

        # Output regression
        output_cache = iQRRLSCache(
            Oo, Psqo, Ko, ξpre, ξpost, C, J, γo, λ,
            zeros(d+1,d+1), zeros(d), zeros(1,n), zeros(d,n)
        )
        output_iqrrls = iQRRLSOpInf(output_cache, dims, Dict{Symbol,Any}(), options)
        return state_iqrrls, output_iqrrls
    else
        error("Available algorithms are RLS, QRRLS, and iQRRLS.")
    end
end


"""
$(SIGNATURES)

Update the streaming operator inference continuously with all the data streams using
the Recursive Least-Squares (RLS) algorithm with regularization.
"""
function stream_all!(stream::RLSOpInf, X::AbstractArray{<:AbstractArray{T}}, R::AbstractArray{<:AbstractArray{T}}; 
                     U::AbstractArray{<:AbstractArray{T}}=Vector{T}[], γs::AbstractArray{<:Real}=zeros(length(X)),
                     Q::Union{AbstractArray{<:AbstractArray{T}},AbstractArray{T},Real}=0.0,verbose::Bool=false) where T<:Number
    N = length(X)
    D = nothing # initialize the data matrix
    flag = typeof(Q) <: AbstractArray{T}
    no_input = isempty(U)
    p = Progress(N; desc="Streaming data...")
    for i in 1:N
        if iszero(Q)
            if i == N
                D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i], γs=γs[i], final_step=true)
            else
                D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i], γs=γs[i])
            end
        else
            if i == N
                D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i], γs=γs[i], Q=flag ? Q : Q[i], final_step=true)
            else
                D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i], γs=γs[i], Q=flag ? Q : Q[i])
            end
        end
        next!(p)
    end
    return D
end


"""
$(SIGNATURES)

Update the streaming operator inference continuously with all the data streams using
the inverse and QR Decomposition Recursive Least-Squares (iQRRLS/QRRLS) algorithm.
"""
function stream_all!(stream::Union{iQRRLSOpInf,QRRLSOpInf}, X::AbstractArray{<:AbstractArray{T}}, 
                     R::AbstractArray{<:AbstractArray{T}}; U::AbstractArray{<:AbstractArray{T}}=Vector{T}[]) where T<:Number
    N = length(X)
    D = nothing # initialize the data matrix
    no_input = isempty(U)
    p = Progress(N; desc="Streaming data...")
    for i in 1:N
        if i == N
            D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i], final_step=true)
        else
            D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i])
        end
        next!(p)
    end
    return D
end


"""
$(SIGNATURES)

Streaming all the data for the output system using the Recursive Least-Squares (RLS) algorithm.
"""
function stream_output_all!(stream::RLSOpInf, X::AbstractArray{<:AbstractArray{T}}, 
                            Y::AbstractArray{<:AbstractArray{T}}; γo::AbstractArray{<:Real}=zeros(length(X)),
                            Z::Union{AbstractArray{<:AbstractArray{T}},AbstractArray{T},Real}=0.0) where T<:Number
    N = length(X)
    flag = typeof(Z) <: AbstractArray{T}
    p = Progress(N; desc="Streaming data...")
    for i in 1:N
        if iszero(Z)
            stream_output!(stream, X[i], Y[i]; γo=γo[i])
        else
            stream_output!(stream, X[i], Y[i]; γo=γo[i], Z=flag ? Z : Z[i])
        end
        next!(p)
    end
    return nothing
end


"""
$(SIGNATURES)

Streaming all the data for the output system using the inverse and 
QR Decomposition Recursive Least-Squares (iQRRLS/QRRLS) algorithm.
"""
function stream_output_all!(stream::Union{iQRRLSOpInf,QRRLSOpInf}, X::AbstractArray{<:AbstractArray{T}}, 
                            Y::AbstractArray{<:AbstractArray{T}}) where T<:Number
    N = length(X)
    p = Progress(N; desc="Streaming data...")
    for i in 1:N
        stream_output!(stream, X[i], Y[i])
        next!(p)
    end
    return nothing
end


"""
$(SIGNATURES)

Terminate the streaming operator inference and return the operators.
"""
function terminate_stream(obj::StreamingOpInf) where T<:Number
    # Extract the operators
    operators = Operators()
    unpack_operators!(
        operators, obj.cache.O, 
        obj.termination_settings[:dims], obj.termination_settings[:syms])
    return operators
end