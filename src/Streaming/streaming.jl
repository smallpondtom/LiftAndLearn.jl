export StreamingOpInf

abstract type StreamingOpInf end

# Import the algorithms
include("algorithms/RLS/rls.jl")
include("algorithms/iQRRLS/iqrrls.jl")
include("algorithms/QRRLS/qrrls.jl")

# Each Algorithm solver structs
mutable struct RLSOpInf{T<:Real} <: StreamingOpInf
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

mutable struct iQRRLSOpInf{T<:Real} <: StreamingOpInf
    cache::iQRRLSCache{T}
    # Dimensions
    dims::Dict{Symbol,Int}
    # Termination settings
    termination_settings::Dict{Symbol,Any}
    # Options
    options::LSOpInfOption  # Standard (Least-Squares) Operator Inference options
end

mutable struct QRRLSOpInf{T<:Real} <: StreamingOpInf
    cache::QRRLSCache{T}
    # Dimensions
    dims::Dict{Symbol,Int}
    # Termination settings
    termination_settings::Dict{Symbol,Any}
    # Options
    options::LSOpInfOption  # Standard (Least-Squares) Operator Inference options
end

# Import the streaming methods
include("algorithms/RLS/stream.jl")
include("algorithms/iQRRLS/stream.jl")
include("algorithms/QRRLS/stream.jl")


"""
$(TYPEDEF)

Streaming Operator Inference/Lift And Learn
"""
function StreamingOpInf(;
    options::LSOpInfOption,             # Standard (Least-Squares) Operator Inference options
    n::Int, m::Int, l::Int,             # state, input, and output dimensions
    algorithm::Symbol=:RLS,             # algorithm type
    γs::T=0.0, γo::T=0.0, λ::T=1.0,     # regularization terms and forgetting factor
    rank::Int=1,                        # rank of the update (default rank-1 update)
    variable_regularize::Bool=false     # variable regularization flag
    ) where {T<:Real}

    # Initialize the dimensions 
    dims = Dict(:n => n, :m => m, :l => l)
    d = 0  # total dimension of the data matrix
    d += sum(i != 0 ? binomial(n+i-1, i) : 0 for i in options.system.state)
    d += sum(i != 0 ? binomial(m+i-1, i) : 0 for i in options.system.control)
    d += sum(i != 0 ? binomial(n+i-1, i) * m : 0 for i in options.system.coupled_input)
    d += iszero(options.system.constant) ? 0 : 1
    dims[:d] = d

    # Initialize variables based on algorithm
    if algorithm == :RLS
        Ps    = iszero(γs) ? Matrix{T}(undef,0,0) : Matrix(1.0I(d) / γs)
        Po    = iszero(γo) ? Matrix{T}(undef,0,0) : Matrix(1.0I(n) / γo)

        # State regression
        state_cache = RLSCache{T}(N=d, M=rank, n=n, P=Ps, γ=γs, λ=λ)
        state_rls = RLSOpInf{T}(
            state_cache, dims, Dict{Symbol,Any}(), options, 
            variable_regularize, iszero(γs)
        )
        if iszero(l)
            return state_rls
        end

        # Output regression
        output_cache = RLSCache{T}(N=n, M=rank, n=l, P=Po, γ=γo, λ=λ)
        output_rls = RLSOpInf{T}(
            output_cache, dims, Dict{Symbol,Any}(), options, 
            variable_regularize, iszero(γo)
        )
        return state_rls, output_rls
    elseif algorithm == :QRRLS
        Os    = zeros(T,d,n)
        Ps    = Matrix{T}(undef,0,0)
        Ks    = Matrix{T}(undef,0,0)
        Oo    = zeros(T,n,l)
        Po    = Matrix{T}(undef,0,0)
        Ko    = Matrix{T}(undef,0,0)
        Φsqs  = sqrt(γs) * 1.0I(d)
        qs    = zeros(T,d,n)
        Φsqo  = sqrt(γo) * 1.0I(n)
        qo    = zeros(T,n,l)
        ξpre  = Matrix{T}(undef,0,0)
        ξpost = Matrix{T}(undef,0,0)
        C     = zero(T)
        J     = zero(T)

        # State regression
        state_cache = QRRLSCache{T}(
            Os, Ps, Ks, Φsqs, qs, ξpre, ξpost, C, J, γs, λ,
            zeros(T,d+n+1,d+n+1), zeros(T,1,n), zeros(T,d,1)
        )
        state_qrrls = QRRLSOpInf{T}(state_cache, dims, Dict{Symbol,Any}(), options)
        if iszero(l)
            return state_qrrls
        end

        # Output regression
        output_cache = QRRLSCache{T}(
            Oo, Po, Ko, Φsqo, qo, ξpre, ξpost, C, J, γo, λ,
            zeros(T,d+n+1,d+n+1), zeros(T,1,n), zeros(T,d,1)
        )
        output_qrrls = QRRLSOpInf{T}(output_cache, dims, Dict{Symbol,Any}(), options)
        return state_qrrls, output_qrrls
    elseif algorithm == :iQRRLS
        Os    = zeros(T,d,n)
        Psqs  = 1.0I(d) / sqrt(γs)
        Ks    = Matrix{T}(undef,0,0)
        Oo    = zeros(T,n,l)
        Psqo  = 1.0I(n) / sqrt(γo)
        Ko    = Matrix{T}(undef,0,0)
        Φs    = Matrix{T}(undef,0,0)
        qs    = Matrix{T}(undef,0,0)
        Φo    = Matrix{T}(undef,0,0)
        qo    = Matrix{T}(undef,0,0)
        ξpre  = Matrix{T}(undef,0,0)
        ξpost = Matrix{T}(undef,0,0)
        C     = zero(T)
        J     = zero(T)

        # State regression
        state_cache = iQRRLSCache{T}(
            Os, Psqs, Ks, ξpre, ξpost, C, J, γs, λ,
            zeros(T,d+1,d+1), zeros(T,d), zeros(T,1,n), zeros(T,d,n)
        )
        state_iqrrls = iQRRLSOpInf{T}(state_cache, dims, Dict{Symbol,Any}(), options)
        if iszero(l)
            return state_iqrrls
        end

        # Output regression
        output_cache = iQRRLSCache{T}(
            Oo, Psqo, Ko, ξpre, ξpost, C, J, γo, λ,
            zeros(T,d+1,d+1), zeros(T,d), zeros(T,1,n), zeros(T,d,n)
        )
        output_iqrrls = iQRRLSOpInf{T}(output_cache, dims, Dict{Symbol,Any}(), options)
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
                     Q::Union{AbstractArray{<:AbstractArray{T}},AbstractArray{T},Real}=0.0,verbose::Bool=false) where T<:Real
    N = length(X)
    D = nothing # initialize the data matrix
    flag = typeof(Q) <: AbstractArray{T}
    no_input = isempty(U)
    if verbose
        p = Progress(N; desc="Streaming data...")
    end
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
        if verbose
            next!(p)
        end
    end
    return D
end


"""
$(SIGNATURES)

Update the streaming operator inference continuously with all the data streams using
the inverse and QR Decomposition Recursive Least-Squares (iQRRLS/QRRLS) algorithm.
"""
function stream_all!(stream::Union{iQRRLSOpInf,QRRLSOpInf}, X::AbstractArray{<:AbstractArray{T}}, 
                     R::AbstractArray{<:AbstractArray{T}}; U::AbstractArray{<:AbstractArray{T}}=Vector{T}[],
                     verbose::Bool=false) where T<:Real
    N = length(X)
    D = nothing # initialize the data matrix
    no_input = isempty(U)
    if verbose
        p = Progress(N; desc="Streaming data...")
    end
    for i in 1:N
        if i == N
            D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i], final_step=true)
        else
            D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i])
        end
        if verbose
            next!(p)
        end
    end
    return D
end


"""
$(SIGNATURES)

Streaming all the data for the output system using the Recursive Least-Squares (RLS) algorithm.
"""
function stream_output_all!(stream::RLSOpInf, X::AbstractArray{<:AbstractArray{T}}, 
                            Y::AbstractArray{<:AbstractArray{T}}; γo::AbstractArray{<:Real}=zeros(length(X)),
                            Z::Union{AbstractArray{<:AbstractArray{T}},AbstractArray{T},Real}=0.0,
                            verbose::Bool=false) where T<:Real
    N = length(X)
    flag = typeof(Z) <: AbstractArray{T}
    if verbose
        p = Progress(N; desc="Streaming data...")
    end
    for i in 1:N
        if iszero(Z)
            stream_output!(stream, X[i], Y[i]; γo=γo[i])
        else
            stream_output!(stream, X[i], Y[i]; γo=γo[i], Z=flag ? Z : Z[i])
        end
        if verbose
            next!(p)
        end
    end
    return nothing
end


"""
$(SIGNATURES)

Streaming all the data for the output system using the inverse and 
QR Decomposition Recursive Least-Squares (iQRRLS/QRRLS) algorithm.
"""
function stream_output_all!(stream::Union{iQRRLSOpInf,QRRLSOpInf}, X::AbstractArray{<:AbstractArray{T}}, 
                            Y::AbstractArray{<:AbstractArray{T}}, verbose::Bool=false) where T<:Real
    N = length(X)
    if verbose
        p = Progress(N; desc="Streaming data...")
    end
    for i in 1:N
        stream_output!(stream, X[i], Y[i])
        if verbose
            next!(p)
        end
    end
    return nothing
end


"""
$(SIGNATURES)

Terminate the streaming operator inference and return the operators.
"""
function terminate_stream(obj::StreamingOpInf) 
    # Extract the operators
    operators = Operators()
    unpack_operators!(
        operators, obj.cache.O',  # remember to transpose the operator matrix
        obj.termination_settings[:dims], obj.termination_settings[:syms])
    return operators
end