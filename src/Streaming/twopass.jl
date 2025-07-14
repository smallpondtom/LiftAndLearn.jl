export TwoPassStreamingOpInf

abstract type TwoPassStreamingOpInf end

# Import the algorithms
include("rls_algorithms/RLS/rls.jl")
include("rls_algorithms/iQRRLS/iqrrls.jl")
include("rls_algorithms/QRRLS/qrrls.jl")

# Each Algorithm solver structs
mutable struct RLSOpInf{T<:Real} <: TwoPassStreamingOpInf
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

mutable struct iQRRLSOpInf{T<:Real} <: TwoPassStreamingOpInf
    cache::iQRRLSCache{T}
    # Dimensions
    dims::Dict{Symbol,Int}
    # Termination settings
    termination_settings::Dict{Symbol,Any}
    # Options
    options::LSOpInfOption  # Standard (Least-Squares) Operator Inference options
end

mutable struct QRRLSOpInf{T<:Real} <: TwoPassStreamingOpInf
    cache::QRRLSCache{T}
    # Dimensions
    dims::Dict{Symbol,Int}
    # Termination settings
    termination_settings::Dict{Symbol,Any}
    # Options
    options::LSOpInfOption  # Standard (Least-Squares) Operator Inference options
end

# Import the streaming methods
include("rls_algorithms/RLS/stream.jl")
include("rls_algorithms/iQRRLS/stream.jl")
include("rls_algorithms/QRRLS/stream.jl")


"""
$(TYPEDEF)

Streaming Two-Pass Operator Inference/Lift And Learn.

# Arguments:
- `options::LSOpInfOption`: Standard (Least-Squares) Operator Inference options.
- `n::Int`: State dimension.
- `m::Int=0`: Input dimension (default is 0).
- `l::Int=0`: Output dimension (default is 0).
- `algorithm::Symbol=:RLS`: Algorithm type, can be `:RLS`, `:QRRLS`, or `:iQRRLS`.
- `Γs::Union{T,AbstractArray{T}}=0.0`: Regularization term for state regression (default is 0).
- `Γo::Union{T,AbstractArray{T}}=0.0`: Regularization term for output regression (default is 0).
- `λ::T=1.0`: Forgetting factor (default is 1).
- `rank::Int=1`: Rank of the update (default is 1).
- `variable_regularize::Bool=false`: Variable regularization (only for `:RLS`) flag (default is false). This is experimental.
- `qr_method::Symbol=:givens`: QR method for `iQRRLS` and `QRRLS` (default is `:givens` or built-in `qr`).
"""
function TwoPassStreamingOpInf(;
    options::LSOpInfOption,             # Standard (Least-Squares) Operator Inference options
    n::Int, m::Int=0, l::Int=0,         # state (n), input (m), and output (l) dimensions
    algorithm::Symbol=:RLS,             # algorithm type
    Γs::Union{T,AbstractArray{T}}=0.0,  # regularization term for state regression  (regularization ||Γ^(1/2) * O||_F^2)
    Γo::Union{T,AbstractArray{T}}=0.0,  # regularization term for output regression (regularization ||Γ^(1/2) * O||_F^2)
    λ::T=1.0,                           # forgetting factor
    rank::Int=1,                        # rank of the update (default rank-1 update)
    variable_regularize::Bool=false,    # variable regularization flag
    qr_method::Symbol=:givens,          # QR method for iQRRLS (default is :givens)
    ) where {T<:Real}

    # Ensure BLAS multi-threading is enabled
    BLAS.set_num_threads(Sys.CPU_THREADS)

    # Initialize the dimensions 
    dims = Dict(:n => n, :m => m, :l => l)
    state_dims = isa(options.system.state, Real) ? [options.system.state] : 
                 options.system.state
    control_dims = isa(options.system.control, Real) ? [options.system.control] : 
                   options.system.control
    coupled_input_dims = isa(options.system.coupled_input, Real) ? [options.system.coupled_input] : 
                         options.system.coupled_input 

    # Build operator info (!!!! with B after A !!!!)
    operator_info = Vector{Tuple{Int, Symbol}}()

    # Add linear state operator A first
    linear_state = filter(i -> i == 1, state_dims)
    if !isempty(linear_state)
        push!(operator_info, (binomial(n, 1), :A))
    end

    # Add control operator B right after A
    for i in filter(!iszero, control_dims)
        push!(operator_info, (binomial(m+i-1, i), :B))
    end

    # Add higher-order state operators (A2, A3, etc.)
    higher_order_state = filter(i -> i > 1, state_dims)
    for i in higher_order_state
        symbol = options.optim.nonredundant_operators ? Symbol("A$(i)u") : Symbol("A$(i)")
        push!(operator_info, (binomial(n+i-1, i), symbol))
    end

    # Add coupled input operators
    for i in filter(!iszero, coupled_input_dims)
        symbol = i == 1 ? :N : Symbol("N$(i)")
        push!(operator_info, (binomial(n+i-1, i) * m, symbol))
    end

    # Add constant term
    if options.system.constant != 0
        push!(operator_info, (1, :K))
    end

    operator_dimensions, operator_symbols = zip(operator_info...) .|> collect
    d = sum(operator_dimensions)  # total dimension of the data matrix
    dims[:d] = d

    # Build Tikhonov matrix using built-in function
    if iszero(Γs) && options.with_reg
        # Construct the Tikhonov matrix
        Γs = spzeros(d)
        tikhonov_matrix!(Γs, operator_dimensions, operator_symbols, options.λ)
        Γs = diagm(0 => Γs)  # convert to sparse diagonal matrix
    elseif !isa(Γs, Real)
        @info "We recommend using a sparse Tikhonov matrix for large systems." *
              " Or use the built-in function to construct the Tikhonov matrix." *
              " If you want to use a dense Tikhonov matrix, set `with_reg=false`."
    end

    # Termination settings
    term_setting = Dict{Symbol,Any}(
        :dims => operator_dimensions, :syms => operator_symbols,
    )

    if algorithm == :RLS  # Standard Recursive Least-Squares (RLS)
        # Initialize the inverse correlation matrices (state)
        Ps = if iszero(Γs)
            Matrix{T}(undef, d, d)
        elseif isa(Γs, UniformScaling) || (isa(Γs, Real) && Γs > 0)
            # For scalar regularization: P = (1/γ) * I
            γ_val = isa(Γs, Real) ? Γs : Γs.λ
            Matrix{T}(I(d) / γ_val)
        else
            # Fallback to matrix solve for general case
            Matrix(Γs \ I(d))
        end

        # State regression
        state_cache = RLSCache{T}(N=d, M=rank, n=n, P=Ps, λ=λ)
        state_rls = RLSOpInf{T}(
            state_cache, dims, term_setting, options, 
            variable_regularize, iszero(Γs)
        )
        if iszero(l)  # If no output regression, return only state
            return state_rls
        end

        # Initialize the inverse correlation matrices (output)
        Po = if iszero(Γo)
            Matrix{T}(undef, n, n)
        elseif isa(Γo, UniformScaling) || (isa(Γo, Real) && Γo > 0)
            # For scalar regularization: P = (1/γ) * I
            γ_val = isa(Γo, Real) ? Γo : Γo.λ
            Matrix{T}(I(n) / γ_val)
        else
            # Fallback to matrix solve for general case
            Matrix(Γo \ I(n))
        end

        # Output regression
        output_cache = RLSCache{T}(N=n, M=rank, n=l, P=Po, λ=λ)
        output_rls = RLSOpInf{T}(
            output_cache, dims, Dict{Symbol,Any}(), options, 
            variable_regularize, iszero(Γo)
        )
        return state_rls, output_rls

    elseif algorithm == :QRRLS  # QR Decomposition Recursive Least-Squares (QRRLS)
        # Initialize the inverse correlation matrices (P) 
        # and square-root correlation matrices (Φsq)
        # for state
        Ps = if isa(Γs, UniformScaling) || (isa(Γs, Real) && Γs > 0)
            γ_val = isa(Γs, Real) ? Γs : Γs.λ
            Matrix{T}(I(d) / γ_val)
        else
            Matrix(Γs \ I(d))
        end

        # Square-root matrices
        Φsqs = if isa(Γs, UniformScaling) || (isa(Γs, Real) && Γs > 0)
            γ_val = isa(Γs, Real) ? Γs : Γs.λ
            Matrix{T}(I(d) * sqrt(γ_val))
        else
            Matrix(sqrt(Γs))
        end

        # State regression
        state_cache = QRRLSCache{T}(N=d, n=n, P=Ps, Φsq=Φsqs, λ=λ)
        state_qrrls = QRRLSOpInf{T}(state_cache, dims, term_setting, options)
        if iszero(l)
            return state_qrrls
        end

        # Initialize the inverse correlation matrices (P) 
        # and square-root correlation matrices (Φsq)
        # for output
        Po = if isa(Γo, UniformScaling) || (isa(Γo, Real) && Γo > 0)
            γ_val = isa(Γo, Real) ? Γo : Γo.λ
            Matrix{T}(I(n) / γ_val)
        else
            Matrix(Γo \ I(n))
        end

        Φsqo = if isa(Γo, UniformScaling) || (isa(Γo, Real) && Γo > 0)
            γ_val = isa(Γo, Real) ? Γo : Γo.λ
            Matrix{T}(I(n) * sqrt(γ_val))
        else
            Matrix(sqrt(Γo))
        end

        # Output regression
        output_cache = QRRLSCache{T}(N=n, n=l, P=Po, Φsq=Φsqo, λ=λ)
        output_qrrls = QRRLSOpInf{T}(output_cache, dims, Dict{Symbol,Any}(), options)
        return state_qrrls, output_qrrls

    elseif algorithm == :iQRRLS  # Inverse QR Decomposition Recursive Least-Squares (iQRRLS)
        # Initialize the square-root inverse correlation matrices (Psq) (state)
        Psqs = if isa(Γs, UniformScaling) || (isa(Γs, Real) && Γs > 0)
            γ_val = isa(Γs, Real) ? Γs : Γs.λ
            Matrix{T}(I(d) / sqrt(γ_val))
        else
            Matrix(sqrt(Γs) \ I(d))
        end

        # State regression
        state_cache = iQRRLSCache{T}(N=d, n=n, Psq=Psqs, λ=λ, method=qr_method)
        state_iqrrls = iQRRLSOpInf{T}(state_cache, dims, term_setting, options)
        if iszero(l)
            return state_iqrrls
        end

        # Initialize the square-root inverse correlation matrices (Psq) (output)
        Psqo = if isa(Γo, UniformScaling) || (isa(Γo, Real) && Γo > 0)
            γ_val = isa(Γo, Real) ? Γo : Γo.λ
            Matrix{T}(I(n) / sqrt(γ_val))
        else
            Matrix(sqrt(Γo) \ I(n))
        end

        # Output regression
        output_cache = iQRRLSCache{T}(N=n, n=l, Psq=Psqo, λ=λ, method=qr_method)
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
function stream_all!(stream::RLSOpInf, X::AbstractArray{<:AbstractArray{T}}, 
                     R::AbstractArray{<:AbstractArray{T}}; 
                     U::AbstractArray{<:AbstractArray{T}}=Vector{T}[], 
                     Γs::Union{AbstractArray{T},AbstractArray{AbstractArray{T}}}=zeros(length(X)),
                     Q::Union{AbstractArray{<:AbstractArray{T}},AbstractArray{T},Real}=0.0,verbose::Bool=false
                     ) where T<:Real
    N = length(X)
    D = nothing # initialize the data matrix
    flag = typeof(Q) <: AbstractArray{T}
    no_input = isempty(U)
    if verbose
        p = Progress(N; desc="Streaming data...")
    end
    for i in 1:N
        if iszero(Q)
            D = stream!(stream, X[i], R[i]; U=no_input ? T[] : U[i], Γs=Γs[i])
        else
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
function stream_all!(stream::Union{iQRRLSOpInf,QRRLSOpInf}, 
                     X::AbstractArray{<:AbstractArray{T}}, 
                     R::AbstractArray{<:AbstractArray{T}}; 
                     U::AbstractArray{<:AbstractArray{T}}=Vector{T}[],
                     verbose::Bool=false) where T<:Real
    N = length(X)
    D = nothing # initialize the data matrix
    no_input = isempty(U)
    if verbose
        p = Progress(N; desc="Streaming data...")
    end
    for i in 1:N
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
function stream_output_all!(stream::Union{iQRRLSOpInf,QRRLSOpInf}, 
                            X::AbstractArray{<:AbstractArray{T}}, 
                            Y::AbstractArray{<:AbstractArray{T}}; 
                            verbose::Bool=false) where T<:Real
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
function terminate_stream(obj::TwoPassStreamingOpInf) 
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
function terminate_stream(state_obj::TwoPassStreamingOpInf, 
                          output_obj::TwoPassStreamingOpInf) 
    # Extract the operators
    operators = Operators(O=obj.cache.O)
    unpack_operators!(
        operators, state_obj.cache.O',  # remember to transpose the operator matrix
        state_obj.termination_settings[:dims], state_obj.termination_settings[:syms])
    operators.C = output_obj.cache.O'
    return operators
end
