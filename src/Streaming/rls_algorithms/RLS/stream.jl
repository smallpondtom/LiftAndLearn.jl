"""
$(SIGNATURES)

Update the streaming operator inference with new data by solving a recursive 
least-squares problem via the standard Recursive Least-Squares (RLS) algorithm 
with regularization.

# Note 
- For the RLS algorithm, the regularization term is updated if 
  `variable_regularization` is enabled
- The RLS algorithm also allows for rank-k update if the data-stream `X` is rank
  higher than 1
- The RLS algorithm also permits noise in terms of a noise covariance matrix `Q`
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
            @warn "Transposing while assuming the row dim is the input dim " *
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
            @warn "Transposing while assuming the row dim is the state dim " * 
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
        if obj.initial_step && iszero(obj.cache.Γ)
            Q_inv = isa(Q, Number) ? 1 / Q : Q \ I
            obj.cache.P = (D' * Q_inv * D) \ I
            obj.cache.K = obj.cache.P * D' * Q_inv
            obj.cache.O = obj.cache.K * R
            obj.initial_step = false  # disable initial zero regularization
        else
            rls!(obj.cache, D, R, Q)
        end
    end

    return D
end


"""
$(SIGNATURES)

Single stream update for the output data.
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

    if obj.variable_regularization  # if variable regularization is enabled
        vrrls!(obj.cache, Xt, Y, Z, Γo, obj.cache.Γ)
    else 
        if obj.initial_step && iszero(obj.cache.Γ)
            Z_inv = isa(Z, Number) ? 1 / Z : Z \ I
            obj.cache.P = (X * Z_inv * Xt) \ I
            obj.cache.K = obj.Py * X * Z_inv
            obj.cache.O = obj.Ky_k * Y
            obj.initial_step = false  # disable initial zero regularization
        else
            rls!(obj.cache, Xt, Y, Z)
        end
    end
    return nothing
end
