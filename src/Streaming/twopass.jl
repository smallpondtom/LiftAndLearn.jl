export TwoPassStreamingOpInf

"""
$(TYPEDEF)

Abstract supertype of the two-pass Streaming Operator Inference solvers.

In two-pass Streaming OpInf the reduced basis ``\\mathbf{V}_r`` is computed in a first pass
over the data (e.g., with an incremental SVD), and the reduced operators are learned in a
second pass by solving the (regularized) OpInf least-squares problem recursively, one data
stream at a time, with a Recursive Least-Squares (RLS) type algorithm.

Concrete subtypes:
- [`RLSOpInf`](@ref): standard (covariance-form) RLS.
- [`QRRLSOpInf`](@ref): QR-decomposition RLS (QRRLS).
- [`iQRRLSOpInf`](@ref): inverse QR-decomposition RLS (iQRRLS).

Instances are created with the keyword constructor
[`TwoPassStreamingOpInf`](@ref).
"""
abstract type TwoPassStreamingOpInf end

# Import the algorithms
include("rls_algorithms/RLS/rls.jl")
include("rls_algorithms/iQRRLS/iqrrls.jl")
include("rls_algorithms/QRRLS/qrrls.jl")

"""
$(TYPEDEF)

Two-pass Streaming OpInf solver based on the standard Recursive Least-Squares (RLS) algorithm.

Supports rank-1 and block (rank-``k``) updates, a forgetting factor, and a noise covariance
weighting. Construct it with [`TwoPassStreamingOpInf`](@ref)
using `algorithm=:RLS`.

## Fields
- `cache::RLSCache{T}`: RLS state (operator matrix, inverse correlation matrix, gains, errors).
- `dims::Dict{Symbol,Int}`: problem dimensions (`:n` state, `:m` input, `:l` output, `:d`
  number of columns of the data matrix).
- `termination_settings::Dict{Symbol,Any}`: operator dimensions (`:dims`) and symbols
  (`:syms`) used by [`terminate_stream`](@ref) to unpack the operator matrix.
- `options::LSOpInfOption`: standard (least-squares) OpInf options defining the model structure.
- `variable_regularization::Bool`: enable the (experimental) variable-regularization RLS.
- `initial_step::Bool`: `true` until the first update when no regularization is used, in which
  case the first stream initializes the inverse correlation matrix by a batch solve.
"""
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

"""
$(TYPEDEF)

Two-pass Streaming OpInf solver based on the inverse QR-decomposition Recursive
Least-Squares (iQRRLS) algorithm.

Propagates a square-root factor of the inverse correlation matrix, which is numerically more
robust than standard RLS. Only rank-1 updates (one snapshot per stream) are supported.
Construct it with [`TwoPassStreamingOpInf`](@ref) using
`algorithm=:iQRRLS`.

## Fields
- `cache::iQRRLSCache{T}`: iQRRLS state (operator matrix, square-root inverse correlation
  matrix, gains, errors).
- `dims::Dict{Symbol,Int}`: problem dimensions (`:n`, `:m`, `:l`, `:d`).
- `termination_settings::Dict{Symbol,Any}`: operator dimensions (`:dims`) and symbols (`:syms`).
- `options::LSOpInfOption`: standard (least-squares) OpInf options.
"""
mutable struct iQRRLSOpInf{T<:Real} <: TwoPassStreamingOpInf
    cache::iQRRLSCache{T}
    # Dimensions
    dims::Dict{Symbol,Int}
    # Termination settings
    termination_settings::Dict{Symbol,Any}
    # Options
    options::LSOpInfOption  # Standard (Least-Squares) Operator Inference options
end

"""
$(TYPEDEF)

Two-pass Streaming OpInf solver based on the QR-decomposition Recursive Least-Squares
(QRRLS) algorithm.

Propagates a square-root factor of the correlation matrix and recovers the operator matrix by
a triangular solve at every step. Only rank-1 updates (one snapshot per stream) are supported.
Construct it with [`TwoPassStreamingOpInf`](@ref) using
`algorithm=:QRRLS`.

## Fields
- `cache::QRRLSCache{T}`: QRRLS state (operator matrix, square-root correlation matrix, gains,
  errors).
- `dims::Dict{Symbol,Int}`: problem dimensions (`:n`, `:m`, `:l`, `:d`).
- `termination_settings::Dict{Symbol,Any}`: operator dimensions (`:dims`) and symbols (`:syms`).
- `options::LSOpInfOption`: standard (least-squares) OpInf options.
"""
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
$(SIGNATURES)

Construct a two-pass Streaming OpInf solver for the reduced state equation and, if `l > 0`,
for the output equation.

The solver learns the operator matrix ``\\mathbf{O}`` of the regularized OpInf problem

```math
\\min_{\\mathbf{O}} \\sum_{k} \\lambda^{K-k} \\| \\mathbf{d}_k \\mathbf{O} - \\mathbf{r}_k \\|_2^2
    + \\| \\boldsymbol{\\Gamma}^{1/2} \\mathbf{O} \\|_F^2
```

recursively, where ``\\mathbf{d}_k`` are rows of the data matrix built from the reduced
state/input data (see [`get_data_matrix`](@ref)) and ``\\mathbf{r}_k`` are the corresponding
reduced time-derivative data. Feed data with [`stream!`](@ref) / [`stream_all!`](@ref) and
recover the operators with [`terminate_stream`](@ref).

## Keyword Arguments
- `options::LSOpInfOption`: standard OpInf options; `options.system` defines the model
  structure (e.g., `state=[1,2]`, `control=1`, `constant=1`).
- `n::Int`: reduced state dimension ``r``.
- `m::Int=0`: input dimension.
- `l::Int=0`: output dimension. If `l > 0` a second solver for the output equation is
  returned as well.
- `algorithm::Symbol=:RLS`: one of `:RLS`, `:QRRLS`, or `:iQRRLS`.
- `Γs::Union{Real,AbstractArray}=0.0`: Tikhonov regularization for the state regression,
  either a scalar ``\\gamma`` (meaning ``\\gamma\\mathbf{I}``) or a ``d\\times d`` matrix.
  If `0` and `options.with_reg == true`, the Tikhonov matrix is built from `options.λ`
  (see [`TikhonovParameter`](@ref)).
- `Γo::Union{Real,AbstractArray}=0.0`: Tikhonov regularization for the output regression
  (scalar or ``n\\times n`` matrix).
- `λ::Real=1.0`: forgetting factor ``0 < \\lambda \\le 1`` (`1` means no forgetting).
- `rank::Int=1`: number of snapshots per stream (block size) for `:RLS`. Every call to
  `stream!` must then pass exactly `rank` snapshots. `:QRRLS`/`:iQRRLS` are rank-1 only.
- `variable_regularize::Bool=false`: enable the variable-regularization RLS (`:RLS` only).
  Experimental and not functional in this release.
- `qr_method::Symbol=:qr`: factorization used by `:QRRLS`/`:iQRRLS`, either `:qr`
  (LAPACK Householder QR) or `:givens` (Givens rotations).
- `use_gpu::Bool=false`: store the solver state on a CUDA GPU.

## Returns
- `state_solver::TwoPassStreamingOpInf`: the solver for the state equation if `l == 0`.
- `(state_solver, output_solver)`: a tuple of solvers if `l > 0`.

## Notes
- `:QRRLS` and `:iQRRLS` require a positive regularization (`Γs > 0`, `Γo > 0`, or
  `options.with_reg = true`) because they are initialized with a square-root factor of the
  regularization matrix.
- `:RLS` with `Γs = 0` performs an unregularized batch solve on the first stream, so the first
  stream must contain at least `d` linearly independent snapshots.
- The constructor sets the number of BLAS threads to `Sys.CPU_THREADS`.

## Examples
```julia
options = LSOpInfOption(system=SystemStructure(state=[1, 2], control=1))
solver = TwoPassStreamingOpInf(options=options, n=r, m=1, algorithm=:iQRRLS, Γs=1e-8)
LiftAndLearn.stream_all!(solver, LiftAndLearn.streamify(Xhat, 1),
                         LiftAndLearn.streamify(Xhatdot, 1); U=LiftAndLearn.streamify(U, 1))
op = LiftAndLearn.terminate_stream(solver)
```
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
    qr_method::Symbol=:qr,              # QR method for iQRRLS (default is :qr)
    use_gpu::Bool=false,                # use GPU for computations (default is false)
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
        @info "Constructing Tikhonov matrix using provided `options.λ`."
    elseif !isa(Γs, Real)
        @info "We recommend using a sparse Tikhonov matrix for large systems." *
              " Or use the built-in function to construct the Tikhonov matrix." *
              " If you want to use a dense Tikhonov matrix, set `with_reg=false`."
    end

    # The square-root (QR-based) algorithms are initialized from a factor of Γ, so they
    # cannot be used without regularization.
    if algorithm in (:QRRLS, :iQRRLS)
        iszero(Γs) && throw(ArgumentError(
            "`$(algorithm)` requires a positive state regularization: pass `Γs > 0` " *
            "or set `options.with_reg = true` with a nonzero `options.λ`."))
        (l > 0 && iszero(Γo)) && throw(ArgumentError(
            "`$(algorithm)` requires a positive output regularization `Γo > 0` when `l > 0`."))
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
        state_cache = RLSCache{T}(N=d, M=rank, n=n, P=Ps, λ=λ,
                                  use_gpu=use_gpu)
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
        output_cache = RLSCache{T}(N=n, M=rank, n=l, P=Po, λ=λ,
                                   use_gpu=use_gpu)
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
        state_cache = QRRLSCache{T}(N=d, n=n, P=Ps, Φsq=Φsqs, λ=λ,
                                    method=qr_method, use_gpu=use_gpu)
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
        output_cache = QRRLSCache{T}(N=n, n=l, P=Po, Φsq=Φsqo, λ=λ,
                                     method=qr_method, use_gpu=use_gpu)
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
        state_cache = iQRRLSCache{T}(N=d, n=n, Psq=Psqs, λ=λ,
                                     method=qr_method, use_gpu=use_gpu)
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
        output_cache = iQRRLSCache{T}(N=n, n=l, Psq=Psqo, λ=λ,
                                      method=qr_method, use_gpu=use_gpu)
        output_iqrrls = iQRRLSOpInf{T}(output_cache, dims, Dict{Symbol,Any}(), options)
        return state_iqrrls, output_iqrrls
    else
        error("Available algorithms are :RLS, :QRRLS, and :iQRRLS.")
    end
end


"""
$(SIGNATURES)

Stream every data block in `X`/`R` (and optionally `U`) through an [`RLSOpInf`](@ref) solver.

Equivalent to calling [`stream!`](@ref) on each `(X[i], R[i], U[i])` in order.

## Arguments
- `stream::RLSOpInf`: the solver to update (mutated).
- `X::AbstractArray{<:AbstractArray}`: reduced state streams, e.g. from [`streamify`](@ref);
  each element is ``r \\times k`` (``k`` snapshots per stream).
- `R::AbstractArray{<:AbstractArray}`: reduced time-derivative streams matching `X`.

## Keyword Arguments
- `U::AbstractArray{<:AbstractArray}=Vector{T}[]`: input streams matching `X` (omit if the
  model has no input).
- `Γs=zeros(length(X))`: per-stream regularization values used only by the experimental
  variable-regularization RLS.
- `Q=0.0`: noise covariance weighting; `0` uses the default of [`stream!`](@ref), a single
  matrix/scalar is used for every stream, otherwise `Q[i]` is used for stream `i`.
- `verbose::Bool=false`: show a progress bar.

## Returns
- `D::AbstractMatrix`: the data matrix of the last stream.
"""
function stream_all!(stream::RLSOpInf, X::AbstractArray{<:AbstractArray{T}},
                     R::AbstractArray{<:AbstractArray{T}};
                     U::AbstractArray{<:AbstractArray{T}}=Vector{T}[],
                     Γs::Union{AbstractArray{T},AbstractArray{AbstractArray{T}}}=zeros(length(X)),
                     Q::Union{AbstractArray{<:AbstractArray{T}},AbstractArray{T},Real}=0.0,verbose::Bool=false
                     ) where T<:Real
    N = length(X)
    D = nothing # initialize the data matrix
    flag = !(Q isa AbstractArray{<:AbstractArray})  # same Q for every stream
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

Stream every data block in `X`/`R` (and optionally `U`) through a [`QRRLSOpInf`](@ref) or
[`iQRRLSOpInf`](@ref) solver.

Equivalent to calling [`stream!`](@ref) on each `(X[i], R[i], U[i])` in order.

## Arguments
- `stream::Union{iQRRLSOpInf,QRRLSOpInf}`: the solver to update (mutated).
- `X::AbstractArray{<:AbstractArray}`: reduced state streams with one snapshot each
  (``r \\times 1``), e.g. `streamify(Xhat, 1)`.
- `R::AbstractArray{<:AbstractArray}`: reduced time-derivative streams matching `X`.

## Keyword Arguments
- `U::AbstractArray{<:AbstractArray}=Vector{T}[]`: input streams matching `X` (omit if the
  model has no input).
- `verbose::Bool=false`: show a progress bar.

## Returns
- `D::AbstractMatrix`: the data matrix of the last stream.
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

Stream every block of reduced state `X` and output `Y` through the output-equation
[`RLSOpInf`](@ref) solver, learning ``\\hat{\\mathbf{C}}`` in
``\\mathbf{y} = \\hat{\\mathbf{C}}\\hat{\\mathbf{x}}``.

## Arguments
- `stream::RLSOpInf`: the output solver returned by `TwoPassStreamingOpInf(...; l=l)` (mutated).
- `X::AbstractArray{<:AbstractArray}`: reduced state streams (``r \\times k`` each).
- `Y::AbstractArray{<:AbstractArray}`: output streams (``l \\times k`` each).

## Keyword Arguments
- `Γo=zeros(length(X))`: per-stream regularization values used only by the experimental
  variable-regularization RLS.
- `Z=0.0`: noise covariance weighting; `0` uses the default of [`stream_output!`](@ref).
- `verbose::Bool=false`: show a progress bar.

## Returns
- `nothing`; mutates `stream`.
"""
function stream_output_all!(stream::RLSOpInf, X::AbstractArray{<:AbstractArray{T}},
                            Y::AbstractArray{<:AbstractArray{T}};
                            Γo::Union{AbstractArray{T},AbstractArray{AbstractArray{T}}}=zeros(length(X)),
                            Z::Union{AbstractArray{<:AbstractArray{T}},AbstractArray{T},Real}=0.0,
                            verbose::Bool=false) where T<:Real
    N = length(X)
    flag = !(Z isa AbstractArray{<:AbstractArray})  # same Z for every stream
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

Stream every snapshot of reduced state `X` and output `Y` through the output-equation
[`QRRLSOpInf`](@ref)/[`iQRRLSOpInf`](@ref) solver.

## Arguments
- `stream::Union{iQRRLSOpInf,QRRLSOpInf}`: the output solver (mutated).
- `X::AbstractArray{<:AbstractArray}`: reduced state streams (``r \\times 1`` each).
- `Y::AbstractArray{<:AbstractArray}`: output streams (``l \\times 1`` each).

## Keyword Arguments
- `verbose::Bool=false`: show a progress bar.

## Returns
- `nothing`; mutates `stream`.
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

Finish streaming and return the learned reduced operators.

The current operator matrix of the solver is unpacked into the individual operators (`A`,
`B`, `A2u`, `K`, ...) according to `obj.options.system`.

## Arguments
- `obj::TwoPassStreamingOpInf`: the state-equation solver.

## Returns
- `operators::Operators`: the learned operators; `operators.O` holds the full operator
  matrix ``\\mathbf{O} \\in \\mathbb{R}^{r \\times d}`` (same orientation as [`opinf`](@ref)).
"""
function terminate_stream(obj::TwoPassStreamingOpInf)
    # Extract the operators (the cache stores the transposed operator matrix Oᵀ)
    O = Matrix(Array(obj.cache.O)')
    operators = Operators(O=O)
    unpack_operators!(
        operators, O, obj.termination_settings[:dims], obj.termination_settings[:syms])
    return operators
end


"""
$(SIGNATURES)

Finish streaming for both the state and output equations and return the learned operators.

## Arguments
- `state_obj::TwoPassStreamingOpInf`: the state-equation solver.
- `output_obj::TwoPassStreamingOpInf`: the output-equation solver.

## Returns
- `operators::Operators`: the learned state operators (see the single-argument method) with
  the output operator stored in `operators.C` (``l \\times r``).
"""
function terminate_stream(state_obj::TwoPassStreamingOpInf,
                          output_obj::TwoPassStreamingOpInf)
    operators = terminate_stream(state_obj)
    operators.C = Matrix(Array(output_obj.cache.O)')
    return operators
end
