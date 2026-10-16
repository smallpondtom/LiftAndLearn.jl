"""
$(SIGNATURES)

Update a [`QRRLSOpInf`](@ref) solver with one snapshot of reduced state/derivative data.

Builds the data row ``\\mathbf{d}_k`` from `X` (and `U`) according to `obj.options` and
performs one rank-1 QRRLS update of the operator matrix with the right-hand side `R`.

## Arguments
- `obj::QRRLSOpInf`: the solver (mutated).
- `X::AbstractArray{T}`: one reduced state snapshot, ``r \\times 1``.
- `R::AbstractArray{T}`: the matching reduced time derivative, ``r \\times 1``
  (``1 \\times r`` also accepted).

## Keyword Arguments
- `U::AbstractArray{T}=T[]`: the input at this snapshot (length ``m``). Leave empty if the
  model has no input.

## Returns
- `D::AbstractMatrix`: the data row ``\\mathbf{d}_k`` (``1 \\times d``).

## Notes
- Only rank-1 updates are supported; stream the data one snapshot at a time
  (e.g., `streamify(Xhat, 1)`).
"""
function stream!(obj::QRRLSOpInf, X::AbstractArray{T}, R::AbstractArray{T}; 
                 U::AbstractArray{T}=T[]) where T<:Number

    tdim = size(X, 2)  # number of data points (time dimension)
    @assert tdim == 1 "QRRLS only supports rank-1 updates (one snapshot per stream)."

    # Construct the data matrix while checking the dimension of the input matrix
    foo, bar = checksize(U) 
    if foo == obj.dims[:m] && bar == tdim
        if foo == bar && foo != 1
            @debug "Transposing while assuming the row dim is the input dim and the column dim is the number of data points."
        end
        D = get_data_matrix(X, U', obj.options; verbose=false)
    else
        D = get_data_matrix(X, U, obj.options; verbose=false)
    end

    # Reorganize the dimension of the derivative data matrix
    foo, bar = checksize(R)
    if foo == obj.dims[:n] && bar == tdim
        if foo == bar
            @debug "Transposing while assuming the row dim is the state dim and the column dim is the number of data points."
        end
        R = R'
    end

    # GPU support
    if obj.cache.use_gpu
        D = CUDA.CuArray(D)
        R = CUDA.CuArray(R)
    end

    qrrls!(obj.cache, D, R)

    return D
end


"""
$(SIGNATURES)

Update the output-equation [`QRRLSOpInf`](@ref) solver with one snapshot of reduced state
`X` and output `Y`, learning ``\\hat{\\mathbf{C}}`` in
``\\mathbf{y} = \\hat{\\mathbf{C}}\\hat{\\mathbf{x}}``.

## Arguments
- `obj::QRRLSOpInf`: the output solver (mutated).
- `X::AbstractArray{T}`: one reduced state snapshot, ``r \\times 1``.
- `Y::AbstractArray{T}`: the matching output, ``l \\times 1`` (``1 \\times l`` also accepted).

## Returns
- `nothing`; mutates `obj`.
"""
function stream_output!(obj::QRRLSOpInf, X::AbstractArray{T}, Y::AbstractArray{T}) where T<:Number
    tdim = size(X, 2)  # number of data points (time dimension)
    @assert tdim == 1 "QRRLS only supports rank-1 updates (one snapshot per stream)."
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

    qrrls!(obj.cache, Xt, Y)

    return nothing
end
