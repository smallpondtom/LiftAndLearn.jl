export Streaming_InferOp


"""
$(TYPEDEF)

Streaming Operator Inference/Lift And Learn
"""
mutable struct Streaming_InferOp
    # State and input
    O_k::Union{AbstractArray{Float64,2}, Nothing}   # operator matrix
    P_k::Union{AbstractArray{Float64,2}, Nothing}   # projection matrix
    K_k::Union{AbstractArray{Float64,2}, Nothing}   # gain matrix
    # Output
    C_k::Union{AbstractArray{Float64,2}, Nothing}  # output matrix
    Py_k::Union{AbstractArray{Float64,2}, Nothing}  # projection matrix
    Ky_k::Union{AbstractArray{Float64,2}, Nothing}  # gain matrix

    dims::Dict{Symbol,Int}                       # dimensions
    options::Abstract_Option                     # options
    init!::Function
    stream!::Function
    stream_output!::Function
    unpack_operators::Function
end


function Streaming_InferOp(options::Abstract_Option)
    dims = Dict(
        :n => 0, :m => 0, :p => 0, :q => 0, 
        :s => 0, :v => 0, :s3 => 0, :v3 => 0,
        :w => 0
    ) 

    O_k = nothing
    P_k = nothing
    K_k = nothing

    C_k = nothing
    Py_k = nothing
    Ky_k = nothing

    return Streaming_InferOp(O_k, P_k, K_k, C_k, Py_k, Ky_k, dims, options, 
                                init!, stream!, stream_output!, unpack_operators)
end


function init!(stream::Streaming_InferOp, X_k::AbstractArray{T}, U_k::AbstractArray{T},
                R_k::AbstractArray{T}, Q_k::Union{AbstractArray{T}, Nothing}=nothing) where T<:Real

    # Obtain the dimensions
    stream.dims[:n], stream.dims[:m] = size(X_k)
    stream.dims[:p] = stream.options.system.has_control ? size(U_k, 2) : 0 
    stream.dims[:s] = stream.options.system.is_quad ? Int(n * (n + 1) / 2) : 0
    stream.dims[:v] = stream.options.system.is_quad ? Int(n * n) : 0
    stream.dims[:s3] = stream.options.system.is_cubic ? Int(n * (n + 1) * (n + 2) / 6) : 0
    stream.dims[:v3] = stream.options.system.is_cubic ? Int(n * n * n) : 0
    stream.dims[:w] = stream.options.system.is_bilin ? Int(n * p) : 0

    Q_k = isnothing(Q_k) ? sparse(Matrix(1.0I, K, K)) : Q_k
    Q_k_inv = isnothing(Q_k) ? sparse(Matrix(1.0I, K, K)) : Q_k \ Matrix(1.0I, K, K)

    # Construct the data matrix
    D_k = getDataMat(X_k, transpose(X_k), U_k, stream.dims, stream.options)

    # Aggregated data matrix and Operator matrix
    stream.P_k = (D_k' * Q_k_inv * D_k) \ I
    stream.O_k = P_k * D_k' * Q_k_inv * R_k

    return D_k
end 


function init!(stream::Streaming_InferOp, X_k::AbstractArray{T}, U_k::AbstractArray{T}, Y_k::AbstractArray{T},
                R_k::AbstractArray{T}, Q_k::Union{AbstractArray{T}, Nothing}=nothing, 
                Z_k::Union{AbstractArray{T}, Nothing}=nothing) where T<:Real

    # Obtain the dimensions
    n, K = size(X_k)
    stream.dims[:n], stream.dims[:m] = n, K
    stream.dims[:p] = stream.options.system.has_control ? size(U_k, 2) : 0 
    stream.dims[:q] = stream.options.system.has_output ? size(Y_k, 1) : 0
    stream.dims[:s] = stream.options.system.is_quad ? Int(n * (n + 1) / 2) : 0
    stream.dims[:v] = stream.options.system.is_quad ? Int(n * n) : 0
    stream.dims[:s3] = stream.options.system.is_cubic ? Int(n * (n + 1) * (n + 2) / 6) : 0
    stream.dims[:v3] = stream.options.system.is_cubic ? Int(n * n * n) : 0
    stream.dims[:w] = stream.options.system.is_bilin ? Int(n * p) : 0

    ## System (input-state)
    Q_k = isnothing(Q_k) ? sparse(Matrix(1.0I, K, K)) : Q_k
    Q_k_inv = isnothing(Q_k) ? sparse(Matrix(1.0I, K, K)) : Q_k \ Matrix(1.0I, K, K)

    # Construct the data matrix
    D_k = getDataMat(X_k, transpose(X_k), U_k, stream.dims, stream.options)

    # Aggregated data matrix and Operator matrix
    stream.P_k = (D_k' * Q_k_inv * D_k) \ I
    stream.O_k = stream.P_k * D_k' * Q_k_inv * R_k

    ## Output (state-output)
    Z_k = isnothing(Z_k) ? sparse(Matrix(1.0I, K, K)) : Z_k
    Z_k_inv = isnothing(Z_k) ? sparse(Matrix(1.0I, K, K)) : Z_k \ Matrix(1.0I, K, K)

    # Aggregated data matrix and Output matrix
    Xt_k = transpose(X_k)
    stream.Py_k = (Xt_k' * Z_k_inv * Xt_k) \ I
    stream.C_k = stream.Py_k * Xt_k' * Z_k_inv * Y_k

    return D_k
end 



"""
$(SIGNATURES)

Update the streaming operator inference with new data.
"""
function stream!(stream::Streaming_InferOp, X_kp1::AbstractArray{T}, U_kp1::AbstractArray{T},
                        R_kp1::AbstractArray{T}, Q_kp1::Union{AbstractArray{T}, Nothing}=nothing) where T<:Real

    K = size(X_kp1,2)
    if stream.dims[:m] != K
        stream.dims[:m] = K
    end

    # Construct the data matrix
    D_kp1 = getDataMat(X_kp1, transpose(X_kp1), U_kp1, stream.dims, stream.options)

    if !isnothing(Q_kp1)
        Q_k_inv = Q_kp1 \ I
        stream.P_k -= stream.P_k * D_kp1' * ((Q_kp1 + D_kp1 * stream.P_k * D_kp1') \ D_kp1) * stream.P_k
        stream.K_k = stream.P_k * D_kp1' * Q_k_inv
        stream.O_k += stream.K_k * (R_kp1 - D_kp1 * stream.O_k)
    else
        stream.P_k -= stream.P_k * D_kp1' * ((1.0I(K) + D_kp1 * stream.P_k * D_kp1') \ D_kp1) * stream.P_k
        stream.K_k = stream.P_k * D_kp1'
        stream.O_k += stream.K_k * (R_kp1 - D_kp1 * stream.O_k)
    end
end


"""
$(SIGNATURES)

Update the streaming operator inference with new data for multiple batches of data matrices.
"""
function stream!(stream::Streaming_InferOp, X_kp1::AbstractArray{<:AbstractArray{T}}, U_kp1::AbstractArray{<:AbstractArray{T}},
                    R_kp1::AbstractArray{<:AbstractArray{T}}, Q_kp1::Union{AbstractArray{<:AbstractArray{T}}, Nothing}=nothing) where T<:Real
    N = length(X_kp1)
    if !isnothing(Q_kp1)
        for i in 1:N
            stream!(stream, X_kp1[i], U_kp1[i], R_kp1[i], Q_kp1[i])
        end
    else
        for i in 1:N
            stream!(stream, X_kp1[i], U_kp1[i], R_kp1[i])
        end
    end
end


function stream_output!(stream::Streaming_InferOp, X_kp1::AbstractArray{T}, Y_kp1::AbstractArray{T}, 
                            Z_kp1::Union{AbstractArray{T}, Nothing}=nothing) where T<:Real
    K, q = size(Y_kp1)
    @assert K == size(X_kp1, 2) "The number of data points should be the same."
    Xt_kp1 = transpose(X_kp1)

    if stream.dims[:q] != q
        stream.dims[:q] = q
    end

    if stream.dims[:m] != K
        stream.dims[:m] = K
    end

    if !isnothing(Z_kp1)
        Z_k_inv = Z_kp1 \ I
        stream.Py_k -= stream.Py_k * Xt_kp1' * ((Z_kp1 + Xt_kp1 * stream.Py_k * Xt_kp1') \ Xt_kp1) * stream.Py_k
        stream.Ky_k = stream.Py_k * Xt_kp1' * Z_k_inv
        stream.C_k += stream.Ky_k * (Y_kp1 - Xt_kp1 * stream.C_k)
    else
        stream.Py_k -= stream.Py_k * Xt_kp1' * ((1.0I(K) + Xt_kp1 * stream.Py_k * Xt_kp1') \ Xt_kp1) * stream.Py_k
        stream.Ky_k = stream.Py_k * Xt_kp1'
        stream.C_k += stream.Ky_k * (Y_kp1 - Xt_kp1 * stream.C_k)
    end
end


function stream_output!(stream::Streaming_InferOp, X_kp1::AbstractArray{<:AbstractArray{T}}, Y_kp1::AbstractArray{<:AbstractArray{T}},
                            Z_kp1::Union{AbstractArray{<:AbstractArray{T}}, Nothing}=nothing) where T<:Real
    N = length(X_kp1)
    if !isnothing(Z_kp1)
        for i in 1:N
            stream_output!(stream, X_kp1[i], Y_kp1[i], Z_kp1[i])
        end
    else
        for i in 1:N
            stream_output!(stream, X_kp1[i], Y_kp1[i])
        end
    end
end


function unpack_operators(stream::Streaming_InferOp)
    # Extract the operators from the operator matrix O
    O = transpose(stream.O_k)
    options = stream.options

    # Dimensions
    n = stream.dims[:n]; p = stream.dims[:p]; q = stream.dims[:q]
    s = stream.dims[:s]; v = stream.dims[:v]; w = stream.dims[:w]
    s3 = stream.dims[:s3]; v3 = stream.dims[:v3]

    TD = 0  # initialize this dummy variable for total dimension (TD)
    if options.system.is_lin
        Ahat = O[:, TD+1:n]
        TD += n
    else
        Ahat = 0
    end
    if options.system.has_control
        Bhat = O[:, TD+1:TD+p]
        TD += p
    else
        Bhat = 0
    end

    # Extract Quadratic terms if the system includes such terms
    sv = 0  # initialize this dummy variable just in case
    if options.system.is_quad
        if options.optim.which_quad_term == "F"
            Fhat = O[:, TD+1:TD+s]
            Hhat = F2Hs(Fhat)
            TD += s
        else
            Hhat = O[:, TD+1:TD+v]
            Fhat = H2F(Hhat)
            TD += v
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
        if p == 1
            Nhat = O[:, TD+1:TD+w]
        else 
            Nhat = zeros(p,n,n)
            tmp = O[:, TD+1:TD+w]
            for i in 1:p
                Nhat[:,:,i] .= tmp[:, Int(n*(i-1)+1):Int(n*i)]
            end
        end
        TD += w
    else
        Nhat = (p == 0) || (p == 1) ? 0 : zeros(n,n,p)
    end

    # Constant term
    Khat = options.system.has_const ? Matrix(O[:, TD+1:end]) : 0

    # Output matrix
    Chat = options.system.has_output ? Matrix(transpose(stream.C_k)) : 0

    return operators(
        A=Ahat, B=Bhat, C=Chat, F=Fhat, H=Hhat, E=Ehat, G=Ghat, N=Nhat, K=Khat
    )
end