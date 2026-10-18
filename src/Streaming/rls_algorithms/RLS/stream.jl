"""
$(SIGNATURES)

Update an [`RLSOpInf`](@ref) solver with one stream of reduced state/derivative data.

Builds the data matrix ``\\mathbf{D}_k`` of the stream from `X` (and `U`) according to
`obj.options` and performs one recursive least-squares update of the operator matrix with the
right-hand side `R`.

## Arguments
- `obj::RLSOpInf`: the solver (mutated).
- `X::AbstractArray{T}`: reduced state snapshots of this stream, ``r \\times k`` (columns are
  snapshots). ``k`` must be `1` or equal the block size `rank` set at construction.
- `R::AbstractArray{T}`: reduced time-derivative data, ``r \\times k`` (``k \\times r`` is also
  accepted).

## Keyword Arguments
- `U::AbstractArray{T}=T[]`: inputs of this stream, ``m \\times k`` (``k \\times m`` also
  accepted). Leave empty if the model has no input.
- `Q::Union{T,AbstractArray}`: noise covariance (weighting) of the stream; a scalar for
  ``k = 1`` or a ``k \\times k`` matrix. Defaults to the identity.
- `Γs::Union{Real,AbstractArray}=0.0`: new regularization value, only used by the experimental
  variable-regularization RLS.

## Returns
- `D::AbstractMatrix`: the data matrix ``\\mathbf{D}_k`` of this stream (``k \\times d``).

## Notes
- If the solver was built without regularization, the first call computes the inverse
  correlation matrix directly from ``\\mathbf{D}_k^\\top\\mathbf{Q}^{-1}\\mathbf{D}_k``, so the
  first stream must have at least `d` linearly independent snapshots.
- When `use_gpu=true`, `D` and `R` are moved to the GPU before the update.
"""
function stream!(
    obj::RLSOpInf, X::AbstractArray{T}, R::AbstractArray{T};
    U::AbstractArray{T}=T[],
    Q::Union{T,AbstractArray{<:Real}}=size(X,2)==1 ? 1.0 : 1.0I(size(X,2)),
    Γs::Union{Real,AbstractArray{<:Real}}=0.0) where T<:Number

    tdim = size(X, 2)  # number of data points (time dimension)

    # Construct the data matrix while checking the dimension of the input matrix
    foo, bar = checksize(U)
    if foo == obj.dims[:m] && bar == tdim
        if foo == bar && foo != 1
            @debug "Transposing while assuming the row dim is the input dim " *
              "and the column dim is the number of data points."
        end
        D = get_data_matrix(X, U', obj.options; verbose=false)
    else
        D = get_data_matrix(X, U, obj.options; verbose=false)
    end

    # Reorganize the dimension of the derivative data matrix
    foo, bar = checksize(R)
    if foo == obj.dims[:n] && bar == tdim
        if foo == bar
            @debug "Transposing while assuming the row dim is the state dim " *
                "and the column dim is the number of data points."
        end
        R = R'
    end

    # GPU support
    if obj.cache.use_gpu
        D = CUDA.CuArray(D)
        R = CUDA.CuArray(R)
    end

    # Execute the update
    if obj.variable_regularization  # if variable regularization is enabled
        vrrls!(obj.cache, D, R, Q, Γs, obj.cache.Γ)
    else
        if obj.initial_step
            # No regularization: initialize with the batch (weighted) least-squares solution
            # of the first stream.
            Q_inv = isa(Q, Number) ? 1 / Q : Q \ I
            P = Matrix((D' * Q_inv * D) \ I)
            Kinit = P * D' * Q_inv
            obj.cache.P = Symmetric(P, :U)
            obj.cache.O .= Kinit * R
            size(Kinit) == size(obj.cache.K) && copyto!(obj.cache.K, Kinit)
            obj.initial_step = false  # disable initial zero regularization
        else
            rls!(obj.cache, D, R, Q)
        end
    end

    return D
end


"""
$(SIGNATURES)

Update the output-equation [`RLSOpInf`](@ref) solver with one stream of reduced state `X` and
output `Y`, learning ``\\hat{\\mathbf{C}}`` in ``\\mathbf{y} = \\hat{\\mathbf{C}}\\hat{\\mathbf{x}}``.

## Arguments
- `obj::RLSOpInf`: the output solver (mutated).
- `X::AbstractArray{T}`: reduced state snapshots, ``r \\times k``.
- `Y::AbstractArray{T}`: output data, ``l \\times k`` (``k \\times l`` also accepted).

## Keyword Arguments
- `Γo::Union{Real,AbstractArray}=0.0`: new regularization value, only used by the experimental
  variable-regularization RLS.
- `Z::Union{T,AbstractArray{T}}`: noise covariance of the stream (scalar for ``k = 1``,
  ``k \\times k`` matrix otherwise). Defaults to the identity.

## Returns
- `nothing`; mutates `obj`.
"""
function stream_output!(obj::RLSOpInf, X::AbstractArray{T}, Y::AbstractArray{T}; Γo::Union{Real,AbstractArray{<:Real}}=0.0,
                        Z::Union{T,AbstractArray{T}}=size(X,2)==1 ? 1.0 : 1.0I(size(X,2))) where T<:Number
    tdim = size(X, 2)  # number of data points (time dimension)
    foo, bar = checksize(Y)
    if foo == obj.dims[:l] && bar == tdim
        if foo == bar && foo != 1
            @warn "Transpose while assuming the row dim is the output dim and the column dim is the number of data points."
        end
        Y = Y'
        obj.dims[:l] = foo
    else
        obj.dims[:l] = bar
    end
    Xt = X'

    # GPU support
    if obj.cache.use_gpu
        Xt = CUDA.CuArray(Xt)
        Y = CUDA.CuArray(Y)
    end

    if obj.variable_regularization  # if variable regularization is enabled
        vrrls!(obj.cache, Xt, Y, Z, Γo, obj.cache.Γ)
    else
        if obj.initial_step
            # No regularization: initialize with the batch (weighted) least-squares solution
            # of the first stream.
            Z_inv = isa(Z, Number) ? 1 / Z : Z \ I
            P = Matrix((X * Z_inv * Xt) \ I)
            Kinit = P * X * Z_inv
            obj.cache.P = Symmetric(P, :U)
            obj.cache.O .= Kinit * Y
            size(Kinit) == size(obj.cache.K) && copyto!(obj.cache.K, Kinit)
            obj.initial_step = false  # disable initial zero regularization
        else
            rls!(obj.cache, Xt, Y, Z)
        end
    end
    return nothing
end
