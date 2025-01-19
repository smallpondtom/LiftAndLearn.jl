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
    n::Int, m::Int=0, l::Int=0,         # state (n), input (m), and output (l) dimensions
    algorithm::Symbol=:RLS,             # algorithm type
    Γs::Union{T,AbstractArray{T}}=0.0,  # regularization term for state regression (regularization ||Γ^(1/2) * O||_F^2)
    Γo::Union{T,AbstractArray{T}}=0.0,  # regularization term for output regression (regularization ||Γ^(1/2) * O||_F^2)
    λ::T=1.0,                           # forgetting factor
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

    if algorithm == :RLS  # Standard Recursive Least-Squares (RLS)
        # Initialize the inverse correlation matrices
        Ps = iszero(Γs) ? Matrix{T}(undef,d,d) : Matrix(Γs \ 1.0I(d))  # State
        Po = iszero(Γo) ? Matrix{T}(undef,n,n) : Matrix(Γo \ 1.0I(n))  # Output

        # State regression
        state_cache = RLSCache{T}(N=d, M=rank, n=n, P=Ps, Γ=Γs, λ=λ)
        state_rls = RLSOpInf{T}(
            state_cache, dims, Dict{Symbol,Any}(), options, 
            variable_regularize, iszero(Γs)
        )
        if iszero(l)
            return state_rls
        end

        # Output regression
        output_cache = RLSCache{T}(N=n, M=rank, n=l, P=Po, Γ=Γo, λ=λ)
        output_rls = RLSOpInf{T}(
            output_cache, dims, Dict{Symbol,Any}(), options, 
            variable_regularize, iszero(Γo)
        )
        return state_rls, output_rls
    elseif algorithm == :QRRLS  # QR Decomposition Recursive Least-Squares (QRRLS)
        # Initialize the inverse correlation matrices (P) and square-root correlation matrices (Φsq)
        Ps    = Matrix(Γs \ 1.0I(d))  # State
        Po    = Matrix(Γo \ 1.0I(n))  # Output
        Φsqs  = typeof(Γs)<:Real ? Matrix(sqrt(Γs) * 1.0I(d)) : sqrt(Γs) # State 
        Φsqo  = typeof(Γo)<:Real ? Matrix(sqrt(Γo) * 1.0I(n)) : sqrt(Γo) # Output

        # State regression
        state_cache = QRRLSCache{T}(N=d, n=n, P=Ps, Φsq=Φsqs, λ=λ)
        state_qrrls = QRRLSOpInf{T}(state_cache, dims, Dict{Symbol,Any}(), options)
        if iszero(l)
            return state_qrrls
        end

        # Output regression
        output_cache = QRRLSCache{T}(N=n, n=l, P=Po, Φsq=Φsqo, λ=λ)
        output_qrrls = QRRLSOpInf{T}(output_cache, dims, Dict{Symbol,Any}(), options)
        return state_qrrls, output_qrrls
    elseif algorithm == :iQRRLS  # Inverse QR Decomposition Recursive Least-Squares (iQRRLS)
        # Initialize the square-root inverse correlation matrices (Psq)
        Psqs = Matrix(sqrt(Γs) \ 1.0I(d))  # State
        Psqo = Matrix(sqrt(Γo) \ 1.0I(n))  # Output

        # State regression
        state_cache = iQRRLSCache{T}(N=d, n=n, Psq=Psqs, λ=λ)
        state_iqrrls = iQRRLSOpInf{T}(state_cache, dims, Dict{Symbol,Any}(), options)
        if iszero(l)
            return state_iqrrls
        end

        # Output regression
        output_cache = iQRRLSCache{T}(N=n, n=l, Psq=Psqo, λ=λ)
        output_iqrrls = iQRRLSOpInf{T}(output_cache, dims, Dict{Symbol,Any}(), options)
        return state_iqrrls, output_iqrrls
    else
        error("Available algorithms are :RLS, :QRRLS, and :iQRRLS.")
    end
end


"""
$(SIGNATURES)

Update the streaming operator inference continuously with all the data streams using
the Recursive Least-Squares (RLS) algorithm with regularization.
"""
function stream_all!(stream::RLSOpInf, X::AbstractArray{<:AbstractArray{T}}, R::AbstractArray{<:AbstractArray{T}}; 
                     U::AbstractArray{<:AbstractArray{T}}=Vector{T}[], 
                     Γs::Union{AbstractArray{T},AbstractArray{AbstractArray{T}}}=zeros(length(X)),
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
            # if i == N
            #     D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i], Γs=Γs[i], final_step=true)
            # else
            #     D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i], Γs=Γs[i])
            # end
            D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i], Γs=Γs[i])
        else
            # if i == N
            #     D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i], Γs=Γs[i], Q=flag ? Q : Q[i], final_step=true)
            # else
            #     D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i], Γs=Γs[i], Q=flag ? Q : Q[i])
            # end
            D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i], Γs=Γs[i], Q=flag ? Q : Q[i])
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
        # if i == N
        #     D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i], final_step=true)
        # else
        #     D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i])
        # end
        D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i])
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
                            Y::AbstractArray{<:AbstractArray{T}}; 
                            Γo::Union{AbstractArray{T},AbstractArray{AbstractArray{T}}}=zeros(length(X)),
                            Z::Union{AbstractArray{<:AbstractArray{T}},AbstractArray{T},Real}=0.0,
                            verbose::Bool=false) where T<:Real
    N = length(X)
    flag = typeof(Z) <: AbstractArray{T}
    if verbose
        p = Progress(N; desc="Streaming data...")
    end
    for i in 1:N
        if iszero(Z)
            stream_output!(stream, X[i], Y[i]; Γo=Γo[i])
        else
            stream_output!(stream, X[i], Y[i]; Γo=Γo[i], Z=flag ? Z : Z[i])
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
                            Y::AbstractArray{<:AbstractArray{T}}; verbose::Bool=false) where T<:Real
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
    operators = Operators(O=obj.cache.O)
    unpack_operators!(
        operators, obj.cache.O',  # remember to transpose the operator matrix
        obj.termination_settings[:dims], obj.termination_settings[:syms])
    return operators
end


"""
$(SIGNATURES)

Terminate the streaming operator inference and return the operators (dispatch)
"""
function terminate_stream(state_obj::StreamingOpInf, output_obj::StreamingOpInf) 
    # Extract the operators
    operators = Operators(O=obj.cache.O)
    unpack_operators!(
        operators, state_obj.cache.O',  # remember to transpose the operator matrix
        state_obj.termination_settings[:dims], state_obj.termination_settings[:syms])
    operators.C = output_obj.cache.O'
    return operators
end