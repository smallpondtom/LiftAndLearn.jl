export StreamingOpInf


"""
$(TYPEDEF)

Streaming Operator Inference/Lift And Learn
"""
mutable struct StreamingOpInf
    # State and input
    O_k::AbstractArray   # operator matrix
    P_k::AbstractArray   # projection matrix
    K_k::AbstractArray   # gain matrix
    # Output
    C_k::AbstractArray   # output matrix
    Py_k::AbstractArray  # projection matrix
    Ky_k::AbstractArray  # gain matrix

    # Regularization terms (state and output)
    α_k::Union{Real}
    β_k::Union{Real}

    tol::Union{Real,Array{<:Real},Nothing}       # tolerance of the pseudo-inverse for state and output
    dims::Dict{Symbol,Int}                       # dimensions
    options::AbstractOption                      # options
    variable_regularize::Bool                    # variable regularization flag

    # Methods
    init!::Function
    stream!::Function
    stream_output!::Function
    unpack_operators::Function
end


function StreamingOpInf(options::AbstractOption; variable_regularize::Bool=false,
                           tol::Union{Real,Array{<:Real},Nothing}=nothing)
    if !isnothing(tol)
        @assert length(tol) <= 2 "The length of the tolerance should be at most 2."
    end

    dims = Dict(
        :n => 0, :K => 0, :m => 0, :l => 0, 
        :s2 => 0, :v2 => 0, :s3 => 0, :v3 => 0,
        :w1 => 0, :d => 0
    ) 

    O_k = []; P_k = []; K_k = []
    C_k = []; Py_k = []; Ky_k = []

    α_k = 0.0
    β_k = 0.0

    return StreamingOpInf(
        O_k, P_k, K_k, C_k, Py_k, Ky_k, α_k, β_k, tol, dims, options, variable_regularize,
        init!, stream!, stream_output!, unpack_operators
    )
end


function init!(stream::StreamingOpInf, X_k::AbstractArray{}, R_k::AbstractArray{T}; 
               U_k::AbstractArray{T}=[], Y_k::AbstractArray{T}=[], α_k::Union{Real,Nothing}=0.0, 
               β_k::Union{Real,Nothing}=0.0, Q_k::Union{AbstractArray{T}, T, Nothing}=nothing, 
               Z_k::Union{AbstractArray{T}, T, Nothing}=nothing) where T<:Real

    # Obtain the dimensions
    n, K = size(X_k)
    stream.dims[:n], stream.dims[:K] = n, K
    stream.dims[:m] = stream.options.system.has_control ? size(U_k, 2) : 0 
    stream.dims[:l] = stream.options.system.has_output ? size(Y_k, 1) : 0
    stream.dims[:s2] = stream.options.system.is_quad ? Int(n * (n + 1) / 2) : 0
    stream.dims[:v2] = stream.options.system.is_quad ? Int(n * n) : 0
    stream.dims[:s3] = stream.options.system.is_cubic ? Int(n * (n + 1) * (n + 2) / 6) : 0
    stream.dims[:v3] = stream.options.system.is_cubic ? Int(n * n * n) : 0
    stream.dims[:w1] = stream.options.system.is_bilin ? Int(n * p) : 0
    d = 0
    for (key, val) in stream.dims
        if key != :K && key != :l && key != :d
            d += val
        end
    end
    stream.dims[:d] = d

    ## System (input-state)
    Q_k = isnothing(Q_k) ? sparse(Matrix(1.0I, K, K)) : Q_k
    Q_k_inv = isnothing(Q_k) ? sparse(Matrix(1.0I, K, K)) : Q_k \ Matrix(1.0I, K, K)

    # Construct the data matrix
    D_k = getDataMat(X_k, U_k, stream.options)

    # Aggregated data matrix and Operator matrix
    if isnothing(stream.tol)
        stream.P_k = (D_k' * Q_k_inv * D_k + α_k * I) \ I
    else
        stream.P_k = pinv(D_k' * Q_k_inv * D_k + α_k * I; atol=stream.tol[1])
    end
    stream.O_k = stream.P_k * D_k' * Q_k_inv * R_k

    ## Output (state-output) 
    if !isempty(Y_k) && stream.options.system.has_output
        Z_k = isnothing(Z_k) ? sparse(Matrix(1.0I, K, K)) : Z_k
        Z_k_inv = isnothing(Z_k) ? sparse(Matrix(1.0I, K, K)) : Z_k \ Matrix(1.0I, K, K)

        # Aggregated data matrix and Output matrix
        Xt_k = transpose(X_k)
        if isnothing(stream.tol)
            stream.Py_k =  (Xt_k' * Z_k_inv * Xt_k + β_k * I) \ I
        else
            if length(stream.tol) == 1  # use same tolerance for both state and output
                stream.Py_k = pinv(Xt_k' * Z_k_inv * Xt_k + β_k * I; atol=stream.tol[1])
            else  # use different tolerance for state and output
                stream.Py_k = pinv(Xt_k' * Z_k_inv * Xt_k + β_k * I; atol=stream.tol[2])
            end
        end
        stream.C_k = stream.Py_k * Xt_k' * Z_k_inv * Y_k
    else
        error("Output option is not enabled. Check the system options.")
    end

    if stream.variable_regularize # if variable regularization is enabled
        stream.α_k = α_k
        stream.β_k = β_k
    end

    return D_k
end 


"""
$(SIGNATURES)

Update the streaming operator inference with new data. Including standard RLS, fixed regularization, 
and variable regularization.
"""
function stream!(stream::StreamingOpInf, X_kp1::AbstractArray{T}, R_kp1::AbstractArray{T}; 
                 U_kp1::AbstractArray{T}=[], α_kp1::Real=0.0,
                 Q_kp1::Union{AbstractArray{T}, T, Nothing}=nothing) where T<:Real

    K = size(X_kp1,2)
    if stream.dims[:K] != K
        stream.dims[:K] = K
    end

    # Construct the data matrix
    D_kp1 = getDataMat(X_kp1, U_kp1, stream.options)

    if stream.variable_regularize  # if variable regularization is enabled
        if !isnothing(Q_kp1)
            if K == 1  # rank-1 update
                u = stream.P_k * D_kp1'
                T_k = stream.P_k - u * u' / (Q_kp1 + dot(D_kp1, u))
                stream.P_k = (I - (α_kp1 - stream.α_k) * T_k) * T_k
            else
                Q_k_inv = Q_kp1 \ I
                if isnothing(stream.tol)
                    T_k = stream.P_k - stream.P_k * D_kp1' * ((Q_kp1 + D_kp1 * stream.P_k * D_kp1') \ D_kp1) * stream.P_k 
                    stream.P_k = (I - (α_kp1 - stream.α_k) * T_k) * T_k
                else
                    T_k = stream.P_k - stream.P_k * D_kp1' * (pinv(Q_kp1 + D_kp1 * stream.P_k * D_kp1'; atol=stream.tol[1]) * D_kp1) * stream.P_k
                    stream.P_k = (I - (α_kp1 - stream.α_k) * T_k) * T_k
                end
            end
            stream.K_k = stream.P_k * D_kp1' * Q_k_inv
        else
            if K == 1  # rank-1 update
                u = stream.P_k * D_kp1'
                T_k = stream.P_k - u * u' / (1.0 + dot(D_kp1, u))
                stream.P_k = (I - (α_kp1 - stream.α_k) * T_k) * T_k
            else
                if isnothing(stream.tol)
                    T_k = stream.P_k - stream.P_k * D_kp1' * ((1.0I(K) + D_kp1 * stream.P_k * D_kp1') \ D_kp1) * stream.P_k
                    stream.P_k = (I - (α_kp1 - stream.α_k) * T_k) * T_k
                else
                    T_k = stream.P_k - stream.P_k * D_kp1' * (pinv(1.0I(K) + D_kp1 * stream.P_k * D_kp1'; atol=stream.tol[1]) * D_kp1) * stream.P_k
                    stream.P_k = (I - (α_kp1 - stream.α_k) * T_k) * T_k
                end
            end
            stream.K_k = stream.P_k * D_kp1'
        end
        stream.O_k = (I - (α_kp1 - stream.α_k) * stream.P_k) * stream.O_k + stream.K_k * (R_kp1 - D_kp1 * stream.O_k)
        stream.α_k = α_kp1 # update the regularization term
    else
        if !isnothing(Q_kp1)
            if K == 1  # rank-1 update
                u = stream.P_k * D_kp1'
                stream.P_k -= u * u' / (Q_kp1 + dot(D_kp1, u))
            else
                Q_k_inv = Q_kp1 \ I
                if isnothing(stream.tol)
                    stream.P_k -= stream.P_k * D_kp1' * ((Q_kp1 + D_kp1 * stream.P_k * D_kp1') \ D_kp1) * stream.P_k
                else
                    stream.P_k -= stream.P_k * D_kp1' * (pinv(Q_kp1 + D_kp1 * stream.P_k * D_kp1'; atol=stream.tol[1]) * D_kp1) * stream.P_k
                end
            end
            stream.K_k = stream.P_k * D_kp1' * Q_k_inv
        else
            if K == 1  # rank-1 update
                u = stream.P_k * D_kp1'
                stream.P_k -= u * u' / (1.0 + dot(D_kp1, u))
            else
                if isnothing(stream.tol)
                    stream.P_k -= stream.P_k * D_kp1' * ((1.0I(K) + D_kp1 * stream.P_k * D_kp1') \ D_kp1) * stream.P_k
                else
                    stream.P_k -= stream.P_k * D_kp1' * (pinv(1.0I(K) + D_kp1 * stream.P_k * D_kp1'; atol=stream.tol[1]) * D_kp1) * stream.P_k
                end
            end
            stream.K_k = stream.P_k * D_kp1'
        end
        stream.O_k += stream.K_k * (R_kp1 - D_kp1 * stream.O_k)
    end

    return D_kp1
end


"""
$(SIGNATURES)

Update the streaming operator inference with new data for multiple batches of data matrices.
"""
function stream!(stream::StreamingOpInf, X_kp1::AbstractArray{<:AbstractArray{T}}, R_kp1::AbstractArray{<:AbstractArray{T}}; 
                 U_kp1::AbstractArray{<:AbstractArray{T}}=[[]], α_kp1::AbstractArray{T}=zeros(length(X_kp1)),
                 Q_kp1::Union{AbstractArray{<:AbstractArray{T}},AbstractArray{T},Nothing}=nothing) where T<:Real
    N = length(X_kp1)
    D_kp1 = nothing # initialize the data matrix
    for i in 1:N
        D_kp1 = stream!(stream, X_kp1[i], R_kp1[i]; U_kp1=U_kp1[i], α_kp1=α_kp1[i], 
                        Q_kp1=isnothing(Q_kp1) ? nothing : Q_kp1[i])
    end
    return D_kp1
end


function stream_output!(stream::StreamingOpInf, X_kp1::AbstractArray{T}, Y_kp1::AbstractArray{T}; 
                            β_kp1::Real=0.0,
                            Z_kp1::Union{AbstractArray{T}, T, Nothing}=nothing) where T<:Real
    K, l = size(Y_kp1)
    @assert K == size(X_kp1, 2) "The number of data points should be the same."
    Xt_kp1 = transpose(X_kp1)

    if stream.dims[:l] != l
        stream.dims[:l] = l
    end

    if stream.dims[:K] != K
        stream.dims[:K] = K
    end

    if stream.variable_regularize  # if variable regularization is enabled
        if !isnothing(Z_kp1)
            if K == 1  # rank-1 update
                u = stream.Py_k * Xt_kp1'
                T_k = stream.Py_k - u * u' / (Z_kp1 + dot(Xt_kp1, u))
                stream.Py_k = (I - (β_kp1 - stream.β_k) * T_k) * T_k
            else
                Z_k_inv = Z_kp1 \ I
                if isnothing(stream.tol)
                    T_k = stream.Py_k - stream.Py_k * Xt_kp1' * ((Z_kp1 + Xt_kp1 * stream.Py_k * Xt_kp1') \ Xt_kp1) * stream.Py_k
                    stream.Py_k = (I - (β_kp1 - stream.β_k) * T_k) * T_k
                else
                    if length(stream.tol) == 1
                        T_k = stream.Py_k - stream.Py_k * Xt_kp1' * (pinv(Z_kp1 + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[1]) * Xt_kp1) * stream.Py_k
                        stream.Py_k = (I - (β_kp1 - stream.β_k) * T_k) * T_k
                    else
                        T_k = stream.Py_k - stream.Py_k * Xt_kp1' * (pinv(Z_kp1 + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[2]) * Xt_kp1) * stream.Py_k
                        stream.Py_k = (I - (β_kp1 - stream.β_k) * T_k) * T_k
                    end
                end
            end
            stream.Ky_k = stream.Py_k * Xt_kp1' * Z_k_inv
        else
            if K == 1 # rank-1 update
                u = stream.Py_k * Xt_kp1'
                T_k = stream.Py_k - u * u' / (1.0 + dot(Xt_kp1, u))
                stream.Py_k = (I - (β_kp1 - stream.β_k) * T_k) * T_k
            else
                if isnothing(stream.tol)
                    T_k = stream.Py_k - stream.Py_k * Xt_kp1' * ((1.0I(K) + Xt_kp1 * stream.Py_k * Xt_kp1') \ Xt_kp1) * stream.Py_k
                    stream.Py_k = (I - (β_kp1 - stream.β_k) * T_k) * T_k
                else
                    if length(stream.tol) == 1
                        T_k = stream.Py_k - stream.Py_k * Xt_kp1' * (pinv(1.0I(K) + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[1]) * Xt_kp1) * stream.Py_k
                        stream.Py_k = (I - (β_kp1 - stream.β_k) * T_k) * T_k
                    else
                        T_k = stream.Py_k - stream.Py_k * Xt_kp1' * (pinv(1.0I(K) + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[2]) * Xt_kp1) * stream.Py_k
                        stream.Py_k = (I - (β_kp1 - stream.β_k) * T_k) * T_k
                    end
                end
            end
            stream.Ky_k = stream.Py_k * Xt_kp1'
        end
        stream.C_k = (I - (β_kp1 - stream.β_k) * stream.Py_k) * stream.C_k + stream.Ky_k * (Y_kp1 - Xt_kp1 * stream.C_k)
        stream.β_k = β_kp1 # update the regularization term
    else 
        if !isnothing(Z_kp1)
            if K == 1  # rank-1 update
                u = stream.Py_k * Xt_kp1'
                stream.Py_k -= u * u' / (Z_kp1 + dot(Xt_kp1, u))
            else
                Z_k_inv = Z_kp1 \ I
                if isnothing(stream.tol)
                    stream.Py_k -= stream.Py_k * Xt_kp1' * ((Z_kp1 + Xt_kp1 * stream.Py_k * Xt_kp1') \ Xt_kp1) * stream.Py_k
                else
                    if length(stream.tol) == 1
                        stream.Py_k -= stream.Py_k * Xt_kp1' * (pinv(Z_kp1 + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[1]) * Xt_kp1) * stream.Py_k
                    else
                        stream.Py_k -= stream.Py_k * Xt_kp1' * (pinv(Z_kp1 + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[2]) * Xt_kp1) * stream.Py_k
                    end
                end
            end
            stream.Ky_k = stream.Py_k * Xt_kp1' * Z_k_inv
        else
            if K == 1 # rank-1 update
                u = stream.Py_k * Xt_kp1'
                stream.Py_k -= u * u' / (1.0 + dot(Xt_kp1, u))
            else
                if isnothing(stream.tol)
                    stream.Py_k -= stream.Py_k * Xt_kp1' * ((1.0I(K) + Xt_kp1 * stream.Py_k * Xt_kp1') \ Xt_kp1) * stream.Py_k
                else
                    if length(stream.tol) == 1
                        stream.Py_k -= stream.Py_k * Xt_kp1' * (pinv(1.0I(K) + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[1]) * Xt_kp1) * stream.Py_k
                    else
                        stream.Py_k -= stream.Py_k * Xt_kp1' * (pinv(1.0I(K) + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[2]) * Xt_kp1) * stream.Py_k
                    end
                end
            end
            stream.Ky_k = stream.Py_k * Xt_kp1'
        end
        stream.C_k += stream.Ky_k * (Y_kp1 - Xt_kp1 * stream.C_k)
    end
end

function stream_output!(stream::StreamingOpInf, X_kp1::AbstractArray{<:AbstractArray{T}}, 
                        Y_kp1::AbstractArray{<:AbstractArray{T}}; β_kp1::AbstractArray{T}=zeros(length(X_kp1)),
                        Z_kp1::Union{AbstractArray{<:AbstractArray{T}},AbstractArray{T},Nothing}=nothing) where T<:Real
    N = length(X_kp1)
    for i in 1:N
        stream_output!(stream, X_kp1[i], Y_kp1[i]; β_kp1=β_kp1[i], 
                       Z_kp1=isnothing(Z_kp1) ? nothing : Z_kp1[i])
    end
end

function unpack_operators(stream::StreamingOpInf)
    # Extract the operators from the operator matrix O
    O = transpose(stream.O_k)
    options = stream.options

    # Dimensions
    n = stream.dims[:n]; m = stream.dims[:m]; l = stream.dims[:l]
    s2 = stream.dims[:s2]; v2 = stream.dims[:v2]; w1 = stream.dims[:w1]
    s3 = stream.dims[:s3]; v3 = stream.dims[:v3]

    TD = 0  # initialize this dummy variable for total dimension (TD)
    if options.system.is_lin
        Ahat = O[:, TD+1:n]
        TD += n
    else
        Ahat = 0
    end
    if options.system.has_control
        Bhat = O[:, TD+1:TD+m]
        TD += m
    else
        Bhat = 0
    end

    # Extract Quadratic terms if the system includes such terms
    sv = 0  # initialize this dummy variable just in case
    if options.system.is_quad
        if options.optim.which_quad_term == "F"
            Fhat = O[:, TD+1:TD+s2]
            Hhat = F2Hs(Fhat)
            TD += s2
        else
            Hhat = O[:, TD+1:TD+v2]
            Fhat = H2F(Hhat)
            TD += v2
        end
    else
        Fhat = 0
        Hhat = 0
    end

    # Extract Cubic terms if the system includes such terms
    sv3 = 0  # initialize this dummy variable just in case
    if options.system.is_cubic
        if options.optim.which_cubic_term == "E"
            Ehat = O[:, TD+1:TD+s3]
            TD += s3
        else
            Ghat = O[:, TD+1:TD+v3]
            Ehat = G2E(Ghat)
            TD += v3
        end
    else
        Ehat = 0
        Ghat = 0
    end

    # Extract Bilinear terms 
    if options.system.is_bilin
        if m == 1
            Nhat = O[:, TD+1:TD+w1]
        else 
            Nhat = zeros(m,n,n)
            tmp = O[:, TD+1:TD+w1]
            for i in 1:m
                Nhat[:,:,i] .= tmp[:, Int(n*(i-1)+1):Int(n*i)]
            end
        end
        TD += w1
    else
        Nhat = (m == 0) || (m == 1) ? 0 : zeros(n,n,m)
    end

    # Constant term
    Khat = options.system.has_const ? Matrix(O[:, TD+1:end]) : 0

    # Output matrix
    Chat = options.system.has_output ? Matrix(transpose(stream.C_k)) : 0

    return operators(
        A=Ahat, B=Bhat, C=Chat, F=Fhat, H=Hhat, E=Ehat, G=Ghat, N=Nhat, K=Khat
    )
end



# function init!(stream::StreamingOpInf, X_k::AbstractArray{T}, R_k::AbstractArray{T}; 
#                U_k::AbstractArray{T}=[], α_k::Union{Real,Nothing}=0.0, 
#                Q_k::Union{AbstractArray{T}, T, Nothing}=nothing) where T<:Real

#     # Obtain the dimensions
#     stream.dims[:n], stream.dims[:m] = size(X_k)
#     stream.dims[:p] = stream.options.system.has_control ? size(U_k, 2) : 0 
#     stream.dims[:s] = stream.options.system.is_quad ? Int(n * (n + 1) / 2) : 0
#     stream.dims[:v] = stream.options.system.is_quad ? Int(n * n) : 0
#     stream.dims[:s3] = stream.options.system.is_cubic ? Int(n * (n + 1) * (n + 2) / 6) : 0
#     stream.dims[:v3] = stream.options.system.is_cubic ? Int(n * n * n) : 0
#     stream.dims[:w] = stream.options.system.is_bilin ? Int(n * p) : 0

#     Q_k = isnothing(Q_k) ? sparse(Matrix(1.0I, K, K)) : Q_k
#     Q_k_inv = isnothing(Q_k) ? sparse(Matrix(1.0I, K, K)) : Q_k \ Matrix(1.0I, K, K)

#     # Construct the data matrix
#     D_k = getDataMat(X_k, U_k, stream.options)

#     # Aggregated data matrix and Operator matrix
#     tmp = D_k' * Q_k_inv * D_k + α_k * I
#     if isnothing(stream.tol)
#         stream.P_k = (tmp) \ I
#     else
#         stream.P_k = pinv(tmp; atol=stream.tol[1])
#     end
#     stream.O_k = P_k * D_k' * Q_k_inv * R_k

#     if variable_regularize # if variable regularization is enabled
#         stream.α_k = α_k
#     end

#     return D_k
# end 



# function stream_output!(stream::StreamingOpInf, X_kp1::AbstractArray{<:AbstractArray{T}}, 
#                         Y_kp1::AbstractArray{<:AbstractArray{T}}; β_kp1::Union{AbstractArray{T},Nothing}=nothing,
#                         Z_kp1::Union{AbstractArray{<:AbstractArray{T}}, T, Nothing}=nothing) where T<:Real
#     N = length(X_kp1)
#     if !isnothing(Z_kp1)
#         # if both is true so that we can choose to do fixed/variable regularization for the state and/or the output
#         if stream.variable_regularize && !isnothing(β_kp1) 
#             for i in 1:N
#                 reg_stream_output!(stream, X_kp1[i], Y_kp1[i]; β_kp1=β_kp1[i], Z_kp1=Z_kp1[i])
#             end
#         else
#         end
#             for i in 1:N
#                 stream_output!(stream, X_kp1[i], Y_kp1[i]; Z_kp1=Z_kp1[i])
#             end
#     else
#         if stream.variable_regularize && !isnothing(β_kp1)
#             for i in 1:N
#                 reg_stream_output!(stream, X_kp1[i], Y_kp1[i]; β_kp1=β_kp1[i])
#             end
#         else
#             for i in 1:N
#                 stream_output!(stream, X_kp1[i], Y_kp1[i])
#             end
#         end
#     end
# end


# """
# $(SIGNATURES)

# Update the streaming operator inference with new data.
# """
# function stream!(stream::StreamingOpInf, X_kp1::AbstractArray{T}, R_kp1::AbstractArray{T}; 
#                  U_kp1::AbstractArray{T}=[], Q_kp1::Union{AbstractArray{T}, T, Nothing}=nothing) where T<:Real

#     K = size(X_kp1,2)
#     if stream.dims[:m] != K
#         stream.dims[:m] = K
#     end

#     # Construct the data matrix
#     D_kp1 = getDataMat(X_kp1, U_kp1, stream.options)

#     if !isnothing(Q_kp1)
#         if K == 1  # rank-1 update
#             u = stream.P_k * D_kp1'
#             stream.P_k -= u * u' / (Q_kp1 + dot(D_kp1, u))
#         else
#             Q_k_inv = Q_kp1 \ I
#             if isnothing(stream.tol)
#                 stream.P_k -= stream.P_k * D_kp1' * ((Q_kp1 + D_kp1 * stream.P_k * D_kp1') \ D_kp1) * stream.P_k
#             else
#                 stream.P_k -= stream.P_k * D_kp1' * (pinv(Q_kp1 + D_kp1 * stream.P_k * D_kp1'; atol=stream.tol[1]) * D_kp1) * stream.P_k
#             end
#         end
#         stream.K_k = stream.P_k * D_kp1' * Q_k_inv
#     else
#         if K == 1  # rank-1 update
#             u = stream.P_k * D_kp1'
#             stream.P_k -= u * u' / (1.0 + dot(D_kp1, u))
#         else
#             if isnothing(stream.tol)
#                 stream.P_k -= stream.P_k * D_kp1' * ((1.0I(K) + D_kp1 * stream.P_k * D_kp1') \ D_kp1) * stream.P_k
#             else
#                 stream.P_k -= stream.P_k * D_kp1' * (pinv(1.0I(K) + D_kp1 * stream.P_k * D_kp1'; atol=stream.tol[1]) * D_kp1) * stream.P_k
#             end
#         end
#         stream.K_k = stream.P_k * D_kp1'
#     end
#     stream.O_k += stream.K_k * (R_kp1 - D_kp1 * stream.O_k)

#     return D_kp1
# end


# """
# $(SIGNATURES)

# Update the streaming operator inference with new data for multiple batches of data matrices.
# """
# function stream!(stream::StreamingOpInf, X_kp1::AbstractArray{<:AbstractArray{T}}, R_kp1::AbstractArray{<:AbstractArray{T}}; 
#                  U_kp1::AbstractArray{<:AbstractArray{T}}=[[]], α_kp1::Union{AbstractArray{T},Nothing}=nothing,
#                  Q_kp1::Union{AbstractArray{<:AbstractArray{T}},AbstractArray{T},Nothing}=nothing) where T<:Real
#     N = length(X_kp1)
#     D_kp1 = nothing
#     if !isnothing(Q_kp1)
#         # if both is true so that we can choose to do fixed/variable regularization for the state and/or the output
#         if stream.variable_regularize && !isnothing(α_kp1)  
#             for i in 1:N
#                 D_kp1 = reg_stream!(stream, X_kp1[i], R_kp1[i]; U_kp1=U_kp1[i], α_kp1=α_kp1[i], Q_kp1=Q_kp1[i])
#             end
#         else
#             for i in 1:N
#                 D_kp1 = stream!(stream, X_kp1[i], R_kp1[i]; U_kp1=U_kp1[i], Q_kp1=Q_kp1[i])
#             end
#         end
#     else
#         if stream.variable_regularize && !isnothing(α_kp1)
#             for i in 1:N
#                 D_kp1 = reg_stream!(stream, X_kp1[i], R_kp1[i]; U_kp1=U_kp1[i], α_kp1=α_kp1[i])
#             end
#         else
#             for i in 1:N
#                 D_kp1 = stream!(stream, X_kp1[i], R_kp1[i]; U_kp1=U_kp1[i])
#             end
#         end
#     end
#     return D_kp1
# end


# function stream_output!(stream::StreamingOpInf, X_kp1::AbstractArray{T}, Y_kp1::AbstractArray{T}; 
#                         Z_kp1::Union{AbstractArray{T}, T, Nothing}=nothing) where T<:Real
#     K, q = size(Y_kp1)
#     @assert K == size(X_kp1, 2) "The number of data points should be the same."
#     Xt_kp1 = transpose(X_kp1)

#     if stream.dims[:q] != q
#         stream.dims[:q] = q
#     end

#     if stream.dims[:m] != K
#         stream.dims[:m] = K
#     end

#     if !isnothing(Z_kp1)
#         if K == 1  # rank-1 update
#             u = stream.Py_k * Xt_kp1'
#             stream.Py_k -= u * u' / (Z_kp1 + dot(Xt_kp1, u))
#         else
#             Z_k_inv = Z_kp1 \ I
#             if isnothing(stream.tol)
#                 stream.Py_k -= stream.Py_k * Xt_kp1' * ((Z_kp1 + Xt_kp1 * stream.Py_k * Xt_kp1') \ Xt_kp1) * stream.Py_k
#             else
#                 if length(stream.tol) == 1
#                     stream.Py_k -= stream.Py_k * Xt_kp1' * (pinv(Z_kp1 + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[1]) * Xt_kp1) * stream.Py_k
#                 else
#                     stream.Py_k -= stream.Py_k * Xt_kp1' * (pinv(Z_kp1 + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[2]) * Xt_kp1) * stream.Py_k
#                 end
#             end
#         end
#         stream.Ky_k = stream.Py_k * Xt_kp1' * Z_k_inv
#     else
#         if K == 1 # rank-1 update
#             u = stream.Py_k * Xt_kp1'
#             stream.Py_k -= u * u' / (1.0 + dot(Xt_kp1, u))
#         else
#             if isnothing(stream.tol)
#                 stream.Py_k -= stream.Py_k * Xt_kp1' * ((1.0I(K) + Xt_kp1 * stream.Py_k * Xt_kp1') \ Xt_kp1) * stream.Py_k
#             else
#                 if length(stream.tol) == 1
#                     stream.Py_k -= stream.Py_k * Xt_kp1' * (pinv(1.0I(K) + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[1]) * Xt_kp1) * stream.Py_k
#                 else
#                     stream.Py_k -= stream.Py_k * Xt_kp1' * (pinv(1.0I(K) + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[2]) * Xt_kp1) * stream.Py_k
#                 end
#             end
#         end
#         stream.Ky_k = stream.Py_k * Xt_kp1'
#     end
#     stream.C_k += stream.Ky_k * (Y_kp1 - Xt_kp1 * stream.C_k)
# end