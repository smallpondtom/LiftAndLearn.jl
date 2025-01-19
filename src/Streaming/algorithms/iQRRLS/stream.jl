"""
$(SIGNATURES)

Update the streaming operator inference with new data by solving a recursive least-squares problem via 
the inverse-QR Decomposition Recursive Least-Squares (iQRRLS) algorithm.
"""
function stream!(obj::iQRRLSOpInf, X::AbstractArray{T}, R::AbstractArray{T}; U::AbstractArray{T}=T[], 
                 final_step::Bool=false) where T<:Number

    tdim = size(X, 2)  # number of data points (time dimension)

    # Construct the data matrix while checking the dimension of the input matrix
    foo, bar = checksize(U) 
    if foo == obj.dims[:m] && bar == tdim
        if foo == bar && foo != 1
            @warn "Transposing while assuming the row dim is the input dim and the column dim is the number of data points."
        end

        if iszero(obj.cache.counter)
            D, operator_dims, operator_symbols = get_data_matrix(X, U', obj.options; verbose=true)
            obj.termination_settings[:dims] = operator_dims
            obj.termination_settings[:syms] = operator_symbols
        else
            D = get_data_matrix(X, U', obj.options; verbose=false)
        end

        # if final_step
        #     D, operator_dims, operator_symbols = get_data_matrix(X, U', obj.options; verbose=true)
        #     obj.termination_settings[:dims] = operator_dims
        #     obj.termination_settings[:syms] = operator_symbols
        # else
        #     D = get_data_matrix(X, U', obj.options; verbose=false)
        # end
    else

        if iszero(obj.cache.counter)
            D, operator_dims, operator_symbols = get_data_matrix(X, U, obj.options; verbose=true)
            obj.termination_settings[:dims] = operator_dims
            obj.termination_settings[:syms] = operator_symbols
        else
            D = get_data_matrix(X, U, obj.options; verbose=false)
        end

        # if final_step
        #     D, operator_dims, operator_symbols = get_data_matrix(X, U, obj.options; verbose=true)
        #     obj.termination_settings[:dims] = operator_dims
        #     obj.termination_settings[:syms] = operator_symbols
        # else
        #     D = get_data_matrix(X, U, obj.options; verbose=false)
        # end
    end

    # Reorganize the dimension of the derivative data matrix
    foo, bar = checksize(R)
    if foo == obj.dims[:n] && bar == tdim
        if foo == bar
            @warn "Transposing while assuming the row dim is the state dim and the column dim is the number of data points."
        end
        R = R'
    end

    @assert tdim == 1 "iQRRLS is only for rank-1 update."
    iqrrls!(obj.cache, D, R)

    # Update the counter
    obj.cache.counter += 1

    return D
end


"""
$(SIGNATURES)

Single stream update for the output data.
"""
function stream_output!(obj::iQRRLSOpInf, X::AbstractArray{T}, Y::AbstractArray{T}) where T<:Number
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

    @assert tdim == 1 "iQRRLS is only for rank-1 update."
    iqrrls!(obj.cache, Xt, Y)

    return nothing
end