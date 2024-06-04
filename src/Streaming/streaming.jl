export StreamingOpInf


"""
$(TYPEDEF)

Streaming Operator Inference/Lift And Learn
"""
mutable struct StreamingOpInf <: AbstractOption
    # State and input
    O_k::AbstractArray   # operator matrix
    P_k::AbstractArray   # inverse correlation matrix (state)
    K_k::AbstractArray   # Kalman gain matrix (state)
    Φ_k::AbstractArray   # correlation matrix (QRRLS)
    q_k::AbstractArray   # auxiliary matrix (QRRLS)

    # Output
    C_k::AbstractArray   # output matrix
    Py_k::AbstractArray  # inverse correlation matrix (output)
    Ky_k::AbstractArray  # Kalman gain matrix (output)
    Φy_k::AbstractArray  # correlation matrix (QRRLS)
    qy_k::AbstractArray  # auxiliary matrix (QRRLS)

    # Regularization terms (state and output)
    γs_k::Union{Real}
    γo_k::Union{Real}

    # Tolerance of the pseudo-inverse for state and output
    atol::Array{<:Real,1}  # absolute tolerance (state, output)
    rtol::Array{<:Real,1}  # relative tolerance (state, output)

    dims::Dict{Symbol,Int}       # dimensions
    options::LSOpInfOption       # least-squares operator inference options
    variable_regularize::Bool    # variable regularization flag
    zero_reg_start_state::Bool   # zero regularization at the start (state)
    zero_reg_start_output::Bool  # zero regularization at the start (output)

    # Algorithms
    algorithm::Symbol  # algorithm to use

    # Methods
    stream!::Function
    stream_output!::Function
    unpack_operators::Function
end


function StreamingOpInf(options::AbstractOption, n::Int, m::Int=0, l::Int=0; 
                        variable_regularize::Bool=false, algorithm::Symbol=:RLS,
                        atol=[0.0,0.0], rtol=[0.0,0.0], γs_k=0.0, γo_k=0.0)
    @assert length(atol) <= 2 "The length of the absolute tolerance should be at most 2."

    # Initialize the dimensions
    dims = Dict(
        :n => n, :K => 0, 
        :m => m, :l => 0, 
        :s2 => options.system.is_quad ? Int(n * (n + 1) / 2) : 0, 
        :v2 => options.system.is_quad ? Int(n * n) : 0, 
        :s3 => options.system.is_cubic ? Int(n * (n + 1) * (n + 2) / 6) : 0,
        :v3 => options.system.is_cubic ? Int(n * n * n) : 0,
        :w1 => options.system.is_bilin ? Int(n * p) : 0, 
        :d => 0
    ) 
    d = 0
    for (key, val) in dims
        if key != :K && key != :l && key != :d
            if key == :s2
                d += (options.optim.which_quad_term == "F") * val
            elseif key == :v2
                d += (options.optim.which_quad_term == "H") * val
            elseif key == :s3
                d += (options.optim.which_cubic_term == "E") * val
            elseif key == :v3
                d += (options.optim.which_cubic_term == "G") * val
            else
                d += val 
            end
        end
    end
    d += (options.system.has_const) * 1  # if has constant term
    dims[:d] = d

    if algorithm == :RLS
        # Initialize the operators
        O_k = zeros(d,n)
        P_k = iszero(γs_k) ? [] : 1.0I(d) / γs_k
        K_k = []
        C_k = zeros(n,l)
        Py_k = iszero(γo_k) ? [] : 1.0I(n) / γo_k
        Ky_k = []; Φ_k = []; q_k = []; Φy_k = []; qy_k = []
    elseif algorithm == :QRRLS
        # Initialize the operators
        O_k = zeros(d,n)
        P_k = []; K_k = []
        C_k = zeros(n,l)
        Py_k = []; Ky_k = []
        Φ_k = sqrt(γs_k) * 1.0I(d)
        q_k = zeros(d,n)
        Φy_k = sqrt(γo_k) * 1.0I(n)
        qy_k = zeros(n,l)
    elseif algorithm == :iQRRLS
        # Initialize the operators
        O_k = zeros(d,n)
        P_k = 1.0I(d) / sqrt(γs_k)
        K_k = []
        C_k = zeros(n,l)
        Py_k = 1.0I(n) / sqrt(γo_k)
        Ky_k = []; Φ_k = []; q_k = []
        Φy_k = []; qy_k = []
    else
        error("Available algorithms are RLS, QRRLS, and iQRRLS.")
    end

    # Check if the initial regularizations are zero
    zero_reg_start_state = iszero(γs_k) ? true : false
    zero_reg_start_output = iszero(γo_k) ? true : false

    # Initialize the relative tolerance
    if all(rtol .== 0.0) # if relative tolerance is not provided
        rtol[1] = atol[1] > 0.0 ? 0.0 : d*eps()
        rtol[2] = atol[2] > 0.0 ? 0.0 : d*eps()
    end

    return StreamingOpInf(
        O_k, P_k, K_k, Φ_k, q_k,
        C_k, Py_k, Ky_k, Φy_k, qy_k,
        γs_k, γo_k, atol, rtol, 
        dims, options, variable_regularize, 
        zero_reg_start_state, zero_reg_start_output,
        algorithm, stream!, stream_output!, unpack_operators
    )
end


"""
```math
\\Vert \\mathbf{R}_k - \\mathbf{D}_k\\mathbf{O}_k \\Vert_F^2
```
"""
function RLS(D_k::AbstractArray{T}, R_k::AbstractArray{T}, O_km1::AbstractArray{T},
              P_km1::AbstractArray{T}, Q_k::Union{Real,AbstractArray{T}}, 
              atol::Real, rtol::Real) where T<:Real
    M = size(D_k, 1)
    if M == 1  # rank-1 update
        u = P_km1 * D_k'
        P_km1 -= u * u' / (Q_k + dot(D_k, u))
    else  # block (rank-M) update
        if iszero(atol)  # if absolute tolerance is not provided (use backslash)
            P_km1 -= P_km1 * D_k' * ((Q_k + D_k * P_km1 * D_k') \ D_k) * P_km1
        else  # use pinv with tolerance
            P_km1 -= P_km1 * D_k' * (pinv(Q_k + D_k * P_km1 * D_k'; atol=atol, rtol=rtol) * D_k) * P_km1
        end
    end
    K_k = P_km1 * D_k' * (Q_k \ I)
    O_km1 += K_k * (R_k - D_k * O_km1)
    return O_km1, P_km1, K_k
end


"""
RLS dispatch with variable-regularization.
"""
function RLS(D_k::AbstractArray{T}, R_k::AbstractArray{T}, O_km1::AbstractArray{T},
              P_km1::AbstractArray{T}, Q_k::Union{Real,AbstractArray{T}}, 
              γ_k::Real, γ_km1::Real, atol::Real, rtol::Real) where T<:Real
    M = size(D_k, 1)
    if M == 1  # rank-1 update
        u = P_km1 * D_k'
        T_k = P_km1 - u * u' / (Q_k + dot(D_k, u))
    else  # block (rank-M) update
        if iszero(atol)  # if absolute tolerance is not provided (use backslash)
            T_k = P_km1 - P_km1 * D_k' * ((Q_k + D_k * P_km1 * D_k') \ D_k) * P_km1
        else  # use pinv with tolerance
            T_k = P_km1 - P_km1 * D_k' * (pinv(Q_k + D_k * P_km1 * D_k'; atol=atol, rtol=rtol) * D_k) * P_km1
        end
    end
    P_km1 = (I - (γ_k - γ_km1) * T_k) * T_k  # inverse covariance matrix
    K_k = P_km1 * D_k' * (Q_k \ I)  # Kalman gain matrix
    O_km1 += K_k * (R_k - D_k * O_km1)  # update operator matrix
    return O_km1, P_km1, K_k
end


"""
QRRLS
"""
function QRRLS(d_k::AbstractArray{T}, r_k::AbstractArray{T}, Φ_km1::AbstractArray{T}, 
               q_km1::AbstractArray{T}, d::Int, r::Int) where T<:Real
    # Prearray
    A_k = [Φ_km1' q_km1; d_k r_k]  # note: it's actually the transpose

    # Compute postarray using QR factorization
    qr!(A_k)  # in-place QR factorization (B_k = A_k)

    # Extract the inverse covariance matrix and auxiliary matrix
    Φ_km1 = A_k[1:d, 1:d]  # keep it upper triangular here
    q_km1 = A_k[1:d, d+1:d+r] 

    # Compute the next operator matrix with inverse of upper triangular matrix
    O_k = Φ_km1 \ q_km1   # (backslash inverse) automatically does backward substitution
    # O_k = copy(q_km1)
    # backsub!(Φ_km1', O_k)  # (backward subtitution) transpose to make upper triangular

    # Compute the inverse covariance matrix and Kalman gain matrix
    P_k = (Φ_km1'*Φ_km1) \ I   # Φ_km1 is still upper triangular
    K_k = P_k * d_k'
    return O_k, Φ_km1', q_km1, P_k, K_k
end


function backsub!(U::Matrix{T}, x::Vector{T}) where T<:Real
    n = length(x)
    # Backward substitution for U*x = y
    @inbounds for i = n:-1:1
        x[i] /= U[i, i]
        for j = 1:i-1
            x[j] -= A[j, i] * x[i]
        end
    end
end


function backsub!(U::Matrix{T}, X::Matrix{T}) where T<:Real
    n = size(X,1)
    
    # Ensure the dimensions match
    if size(U, 1) != n || size(U, 2) != n
        error("Dimensions of U and X do not match")
    end
    
    # vectorized backward substitution for U*X = Y
    @inbounds for i in n:-1:1
        X[i, :] ./= U[i, i]
        X[1:i-1, :] .-= U[1:i-1, i] .* X[i, :]'
    end
end


"""
iQRRLS

P2_km1: is actually the square-root of the inverse of the correlation matrix
"""
function iQRRLS(d_k::AbstractArray{T}, r_k::AbstractArray{T}, O_km1::AbstractArray{T},
                P2_km1::AbstractArray{T}, d::Int) where T<:Real
    # Prearray
    A_k = [1 zeros(1,d); P2_km1'*d_k' P2_km1']  # note: it's actually the transpose

    # Compute postarray using QR factorization
    _, B_k = qr(A_k)  

    # Extract the square-root of the conversion factor and 
    # the Kalman gain matrix multiplied by square-root of the conversion factor
    α2_k_inv = B_k[1,1]
    gα2_k_inv = B_k[1,2:end]  # becomes a column vector after slicing
    P2_k = B_k[2:end, 2:end]'  # make sure it's lower triangular

    # Compute the next operator matrix and Kalman gain matrix
    K_k = gα2_k_inv * (α2_k_inv)^(-1)
    O_k = O_km1 + K_k * (r_k - d_k * O_km1)
    return O_k, P2_k, K_k
end


"""
$(SIGNATURES)

Update the streaming operator inference with new data. Including standard RLS, fixed regularization, 
and variable regularization.
"""
function stream!(stream::StreamingOpInf, X_k::AbstractArray{T}, R_k::AbstractArray{T}; U_k::AbstractArray{T}=[], 
                 Q_k::Union{T,AbstractArray{T}}=size(X_k,2)==1 ? 1.0 : 1.0I(size(X_k,2)),
                 γs_k::T=0.0) where T<:Real
    stream.dims[:K] = size(X_k, 2)
    stream.dims[:m] = size(U_k, 2)

    # Construct the data matrix
    D_k = getDataMat(X_k, U_k, stream.options)

    if stream.algorithm == :RLS
        # Execute the update
        if stream.variable_regularize  # if variable regularization is enabled
            stream.O_k, stream.P_k, stream.K_k = RLS(D_k, R_k, stream.O_k, stream.P_k, Q_k, 
                                                    γs_k, stream.γs_k, stream.atol[1], stream.rtol[1])
            stream.γs_k = γs_k  # update the regularization term
        else
            if stream.zero_reg_start_state
                Q_k_inv = Q_k \ I
                if iszero(stream.atol[1]) 
                    stream.P_k = (D_k' * Q_k_inv * D_k) \ I
                else
                    stream.P_k = pinv(D_k' * Q_k_inv * D_k; atol=stream.atol[1], rtol=stream.rtol[1])
                end
                stream.K_k = stream.P_k * D_k' * Q_k_inv
                stream.O_k = stream.K_k * R_k
                stream.zero_reg_start_state = false  # disable initial zero regularization
            else
                stream.O_k, stream.P_k, stream.K_k = RLS(D_k, R_k, stream.O_k, stream.P_k, Q_k, stream.atol[1], stream.rtol[1])
            end
        end
    elseif stream.algorithm == :QRRLS
        @assert stream.dims[:K] == 1 "QRRLS is only for rank-1 update."
        stream.O_k, stream.Φ_k, stream.q_k, stream.P_k, stream.K_k = QRRLS(D_k, R_k, stream.Φ_k, stream.q_k, 
                                                                           stream.dims[:d], stream.dims[:n])
    elseif stream.algorithm == :iQRRLS
        @assert stream.dims[:K] == 1 "iQRRLS is only for rank-1 update."
        stream.O_k, stream.P_k, stream.K_k = iQRRLS(D_k, R_k, stream.O_k, stream.P_k, stream.dims[:d])
    else
        error("Available algorithms are RLS, QRRLS, and iQRRLS.")
    end
    return D_k
end


"""
$(SIGNATURES)

Update the streaming operator inference with new data for multiple batches of data matrices.
"""
function stream!(stream::StreamingOpInf, X_k::AbstractArray{<:AbstractArray{T}}, R_k::AbstractArray{<:AbstractArray{T}}; 
                 U_k::AbstractArray{<:AbstractArray{T}}=[[]], γs_k::AbstractArray{T}=zeros(length(X_k)),
                 Q_k::Union{AbstractArray{<:AbstractArray{T}},AbstractArray{T},Real}=0.0) where T<:Real
    N = length(X_k)
    D_k = nothing # initialize the data matrix
    flag = typeof(Q_k) <: AbstractArray{T} 
    for i in 1:N
        if iszero(Q_k)
            D_k = stream!(stream, X_k[i], R_k[i]; U_k=U_k[i], γs_k=γs_k[i])
        else
            D_k = stream!(stream, X_k[i], R_k[i]; U_k=U_k[i], γs_k=γs_k[i], Q_k=flag ? Q_k : Q_k[i])
        end
    end
    return D_k
end


function stream_output!(stream::StreamingOpInf, X_k::AbstractArray{T}, Y_k::AbstractArray{T}; γo_k::Real=0.0, 
                        Z_k::Union{T,AbstractArray{T}}=size(X_k,2)==1 ? 1.0 : 1.0I(size(X_k,2))) where T<:Real
    K, l = size(Y_k)
    @assert K == size(X_k, 2) "The number of data points should be the same."
    Xt_k = transpose(X_k)
    stream.dims[:l] = l
    stream.dims[:K] = K

    if stream.algorithm == :RLS
        if stream.variable_regularize  # if variable regularization is enabled
            stream.C_k, stream.Py_k, stream.Ky_k = RLS(Xt_k, Y_k, stream.C_k, stream.Py_k, Z_k, 
                                                        γo_k, stream.γo_k, stream.atol[2], stream.rtol[2])
            stream.γo_k = γo_k  # update the regularization term
        else 
            if stream.zero_reg_start_output
                Z_k_inv = Z_k \ I
                Xt_k = transpose(X_k)
                if iszero(stream.atol[2])
                    stream.Py_k =  (Xt_k' * Z_k_inv * Xt_k) \ I
                else
                    stream.Py_k = pinv(Xt_k' * Z_k_inv * Xt_k; atol=stream.atol[2], rtol=stream.rtol[2])
                end
                stream.Ky_k = stream.Py_k * Xt_k' * Z_k_inv
                stream.C_k = stream.Ky_k * Y_k
                stream.zero_reg_start_output = false  # disable initial zero regularization
            else
                stream.C_k, stream.Py_k, stream.Ky_k = RLS(Xt_k, Y_k, stream.C_k, stream.Py_k, Z_k, stream.atol[2], stream.rtol[2])
            end
        end
    elseif stream.algorithm == :QRRLS
        @assert stream.dims[:K] == 1 "QRRLS is only for rank-1 update."
        stream.C_k, stream.Φy_k, stream.qy_k, stream.Py_k, stream.Ky_k = QRRLS(Xt_k, Y_k, stream.Φy_k, stream.qy_k, 
                                                                               stream.dims[:n], stream.dims[:l])
    elseif stream.algorithm == :iQRRLS
        @assert stream.dims[:K] == 1 "iQRRLS is only for rank-1 update."
        stream.C_k, stream.Py_k, stream.Ky_k = iQRRLS(Xt_k, Y_k, stream.C_k, stream.Py_k, stream.dims[:n])
    end
    return nothing
end


function stream_output!(stream::StreamingOpInf, X_k::AbstractArray{<:AbstractArray{T}}, 
                        Y_k::AbstractArray{<:AbstractArray{T}}; γo_k::AbstractArray{T}=zeros(length(X_k)),
                        Z_k::Union{AbstractArray{<:AbstractArray{T}},AbstractArray{T},Real}=0.0) where T<:Real
    N = length(X_k)
    flag = typeof(Z_k) <: AbstractArray{T}
    for i in 1:N
        if iszero(Z_k)
            stream_output!(stream, X_k[i], Y_k[i]; γo_k=γo_k[i])
        else
            stream_output!(stream, X_k[i], Y_k[i]; γo_k=γo_k[i], Z_k=flag ? Z_k : Z_k[i])
        end
    end
    return nothing
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


## __OLD__ ##
# function stream_output!(stream::StreamingOpInf, X_kp1::AbstractArray{T}, Y_kp1::AbstractArray{T}; 
#                             β_kp1::Real=0.0,
#                             Z_kp1::Union{AbstractArray{T}, T, Nothing}=nothing) where T<:Real
#     K, l = size(Y_kp1)
#     @assert K == size(X_kp1, 2) "The number of data points should be the same."
#     Xt_kp1 = transpose(X_kp1)

#     if stream.dims[:l] != l
#         stream.dims[:l] = l
#     end

#     if stream.dims[:K] != K
#         stream.dims[:K] = K
#     end

#     if stream.variable_regularize  # if variable regularization is enabled
#         if !isnothing(Z_kp1)
#             if K == 1  # rank-1 update
#                 u = stream.Py_k * Xt_kp1'
#                 T_k = stream.Py_k - u * u' / (Z_kp1 + dot(Xt_kp1, u))
#                 stream.Py_k = (I - (β_kp1 - stream.β_k) * T_k) * T_k
#             else
#                 Z_k_inv = Z_kp1 \ I
#                 if isnothing(stream.tol)
#                     T_k = stream.Py_k - stream.Py_k * Xt_kp1' * ((Z_kp1 + Xt_kp1 * stream.Py_k * Xt_kp1') \ Xt_kp1) * stream.Py_k
#                     stream.Py_k = (I - (β_kp1 - stream.β_k) * T_k) * T_k
#                 else
#                     if length(stream.tol) == 1
#                         T_k = stream.Py_k - stream.Py_k * Xt_kp1' * (pinv(Z_kp1 + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[1]) * Xt_kp1) * stream.Py_k
#                         stream.Py_k = (I - (β_kp1 - stream.β_k) * T_k) * T_k
#                     else
#                         T_k = stream.Py_k - stream.Py_k * Xt_kp1' * (pinv(Z_kp1 + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[2]) * Xt_kp1) * stream.Py_k
#                         stream.Py_k = (I - (β_kp1 - stream.β_k) * T_k) * T_k
#                     end
#                 end
#             end
#             stream.Ky_k = stream.Py_k * Xt_kp1' * Z_k_inv
#         else
#             if K == 1 # rank-1 update
#                 u = stream.Py_k * Xt_kp1'
#                 T_k = stream.Py_k - u * u' / (1.0 + dot(Xt_kp1, u))
#                 stream.Py_k = (I - (β_kp1 - stream.β_k) * T_k) * T_k
#             else
#                 if isnothing(stream.tol)
#                     T_k = stream.Py_k - stream.Py_k * Xt_kp1' * ((1.0I(K) + Xt_kp1 * stream.Py_k * Xt_kp1') \ Xt_kp1) * stream.Py_k
#                     stream.Py_k = (I - (β_kp1 - stream.β_k) * T_k) * T_k
#                 else
#                     if length(stream.tol) == 1
#                         T_k = stream.Py_k - stream.Py_k * Xt_kp1' * (pinv(1.0I(K) + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[1]) * Xt_kp1) * stream.Py_k
#                         stream.Py_k = (I - (β_kp1 - stream.β_k) * T_k) * T_k
#                     else
#                         T_k = stream.Py_k - stream.Py_k * Xt_kp1' * (pinv(1.0I(K) + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[2]) * Xt_kp1) * stream.Py_k
#                         stream.Py_k = (I - (β_kp1 - stream.β_k) * T_k) * T_k
#                     end
#                 end
#             end
#             stream.Ky_k = stream.Py_k * Xt_kp1'
#         end
#         stream.C_k = (I - (β_kp1 - stream.β_k) * stream.Py_k) * stream.C_k + stream.Ky_k * (Y_kp1 - Xt_kp1 * stream.C_k)
#         stream.β_k = β_kp1 # update the regularization term
#     else 
#         if !isnothing(Z_kp1)
#             if K == 1  # rank-1 update
#                 u = stream.Py_k * Xt_kp1'
#                 stream.Py_k -= u * u' / (Z_kp1 + dot(Xt_kp1, u))
#             else
#                 Z_k_inv = Z_kp1 \ I
#                 if isnothing(stream.tol)
#                     stream.Py_k -= stream.Py_k * Xt_kp1' * ((Z_kp1 + Xt_kp1 * stream.Py_k * Xt_kp1') \ Xt_kp1) * stream.Py_k
#                 else
#                     if length(stream.tol) == 1
#                         stream.Py_k -= stream.Py_k * Xt_kp1' * (pinv(Z_kp1 + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[1]) * Xt_kp1) * stream.Py_k
#                     else
#                         stream.Py_k -= stream.Py_k * Xt_kp1' * (pinv(Z_kp1 + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[2]) * Xt_kp1) * stream.Py_k
#                     end
#                 end
#             end
#             stream.Ky_k = stream.Py_k * Xt_kp1' * Z_k_inv
#         else
#             if K == 1 # rank-1 update
#                 u = stream.Py_k * Xt_kp1'
#                 stream.Py_k -= u * u' / (1.0 + dot(Xt_kp1, u))
#             else
#                 if isnothing(stream.tol)
#                     stream.Py_k -= stream.Py_k * Xt_kp1' * ((1.0I(K) + Xt_kp1 * stream.Py_k * Xt_kp1') \ Xt_kp1) * stream.Py_k
#                 else
#                     if length(stream.tol) == 1
#                         stream.Py_k -= stream.Py_k * Xt_kp1' * (pinv(1.0I(K) + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[1]) * Xt_kp1) * stream.Py_k
#                     else
#                         stream.Py_k -= stream.Py_k * Xt_kp1' * (pinv(1.0I(K) + Xt_kp1 * stream.Py_k * Xt_kp1'; atol=stream.tol[2]) * Xt_kp1) * stream.Py_k
#                     end
#                 end
#             end
#             stream.Ky_k = stream.Py_k * Xt_kp1'
#         end
#         stream.C_k += stream.Ky_k * (Y_kp1 - Xt_kp1 * stream.C_k)
#     end
# end



# function init!(stream::StreamingOpInf, X_k::AbstractArray{}, R_k::AbstractArray{T}; 
#                U_k::AbstractArray{T}=[], Y_k::AbstractArray{T}=[], α_k::Union{Real,Nothing}=0.0, 
#                β_k::Union{Real,Nothing}=0.0, Q_k::Union{AbstractArray{T}, T, Nothing}=nothing, 
#                Z_k::Union{AbstractArray{T}, T, Nothing}=nothing) where T<:Real

#     # Obtain the dimensions
#     n, K = size(X_k)
#     stream.dims[:n], stream.dims[:K] = n, K
#     stream.dims[:m] = stream.options.system.has_control ? size(U_k, 2) : 0 
#     stream.dims[:l] = stream.options.system.has_output ? size(Y_k, 1) : 0
#     stream.dims[:s2] = stream.options.system.is_quad ? Int(n * (n + 1) / 2) : 0
#     stream.dims[:v2] = stream.options.system.is_quad ? Int(n * n) : 0
#     stream.dims[:s3] = stream.options.system.is_cubic ? Int(n * (n + 1) * (n + 2) / 6) : 0
#     stream.dims[:v3] = stream.options.system.is_cubic ? Int(n * n * n) : 0
#     stream.dims[:w1] = stream.options.system.is_bilin ? Int(n * p) : 0
#     d = 0
#     for (key, val) in stream.dims
#         if key != :K && key != :l && key != :d
#             if key == :s2
#                 d += (stream.options.optim.which_quad_term == "F") * val
#             elseif key == :v2
#                 d += (stream.options.optim.which_quad_term == "R") * val
#             elseif key == :s3
#                 d += (stream.options.optim.which_cubic_term == "E") * val
#             elseif key == :v3
#                 d += (stream.options.optim.which_cubic_term == "G") * val
#             else
#                 d += val 
#             end
#         end
#     end
#     d += (stream.options.system.has_const) * 1  # if has constant term
#     stream.dims[:d] = d

#     ## System (input-state)
#     Q_k = isnothing(Q_k) ? sparse(Matrix(1.0I, K, K)) : Q_k
#     Q_k_inv = isnothing(Q_k) ? sparse(Matrix(1.0I, K, K)) : Q_k \ Matrix(1.0I, K, K)

#     # Construct the data matrix
#     D_k = getDataMat(X_k, U_k, stream.options)

#     # Aggregated data matrix and Operator matrix
#     if isnothing(stream.tol)
#         stream.P_k = (D_k' * Q_k_inv * D_k + α_k * I) \ I
#     else
#         stream.P_k = pinv(D_k' * Q_k_inv * D_k + α_k * I; atol=stream.tol[1])
#     end
#     stream.O_k = stream.P_k * D_k' * Q_k_inv * R_k

#     ## Output (state-output) 
#     if !isempty(Y_k) && stream.options.system.has_output
#         Z_k = isnothing(Z_k) ? sparse(Matrix(1.0I, K, K)) : Z_k
#         Z_k_inv = isnothing(Z_k) ? sparse(Matrix(1.0I, K, K)) : Z_k \ Matrix(1.0I, K, K)

#         # Aggregated data matrix and Output matrix
#         Xt_k = transpose(X_k)
#         if isnothing(stream.tol)
#             stream.Py_k =  (Xt_k' * Z_k_inv * Xt_k + β_k * I) \ I
#         else
#             if length(stream.tol) == 1  # use same tolerance for both state and output
#                 stream.Py_k = pinv(Xt_k' * Z_k_inv * Xt_k + β_k * I; atol=stream.tol[1])
#             else  # use different tolerance for state and output
#                 stream.Py_k = pinv(Xt_k' * Z_k_inv * Xt_k + β_k * I; atol=stream.tol[2])
#             end
#         end
#         stream.C_k = stream.Py_k * Xt_k' * Z_k_inv * Y_k
#     else
#         error("Output option is not enabled. Check the system options.")
#     end

#     if stream.variable_regularize # if variable regularization is enabled
#         stream.α_k = α_k
#         stream.β_k = β_k
#     end

#     return D_k
# end 


# """
# $(SIGNATURES)

# Update the streaming operator inference with new data. Including standard RLS, fixed regularization, 
# and variable regularization.
# """
# function stream!(stream::StreamingOpInf, X_kp1::AbstractArray{T}, R_kp1::AbstractArray{T}; 
#                  U_kp1::AbstractArray{T}=[], α_kp1::Real=0.0,
#                  Q_kp1::Union{AbstractArray{T}, T, Nothing}=nothing) where T<:Real

#     K = size(X_kp1,2)
#     if stream.dims[:K] != K
#         stream.dims[:K] = K
#     end

#     # Construct the data matrix
#     D_kp1 = getDataMat(X_kp1, U_kp1, stream.options)

#     if stream.variable_regularize  # if variable regularization is enabled
#         if !isnothing(Q_kp1)
#             if K == 1  # rank-1 update
#                 u = stream.P_k * D_kp1'
#                 T_k = stream.P_k - u * u' / (Q_kp1 + dot(D_kp1, u))
#                 stream.P_k = (I - (α_kp1 - stream.α_k) * T_k) * T_k
#             else
#                 Q_k_inv = Q_kp1 \ I
#                 if isnothing(stream.tol)
#                     T_k = stream.P_k - stream.P_k * D_kp1' * ((Q_kp1 + D_kp1 * stream.P_k * D_kp1') \ D_kp1) * stream.P_k 
#                     stream.P_k = (I - (α_kp1 - stream.α_k) * T_k) * T_k
#                 else
#                     T_k = stream.P_k - stream.P_k * D_kp1' * (pinv(Q_kp1 + D_kp1 * stream.P_k * D_kp1'; atol=stream.tol[1]) * D_kp1) * stream.P_k
#                     stream.P_k = (I - (α_kp1 - stream.α_k) * T_k) * T_k
#                 end
#             end
#             stream.K_k = stream.P_k * D_kp1' * Q_k_inv
#         else
#             if K == 1  # rank-1 update
#                 u = stream.P_k * D_kp1'
#                 T_k = stream.P_k - u * u' / (1.0 + dot(D_kp1, u))
#                 stream.P_k = (I - (α_kp1 - stream.α_k) * T_k) * T_k
#             else
#                 if isnothing(stream.tol)
#                     T_k = stream.P_k - stream.P_k * D_kp1' * ((1.0I(K) + D_kp1 * stream.P_k * D_kp1') \ D_kp1) * stream.P_k
#                     stream.P_k = (I - (α_kp1 - stream.α_k) * T_k) * T_k
#                 else
#                     T_k = stream.P_k - stream.P_k * D_kp1' * (pinv(1.0I(K) + D_kp1 * stream.P_k * D_kp1'; atol=stream.tol[1]) * D_kp1) * stream.P_k
#                     stream.P_k = (I - (α_kp1 - stream.α_k) * T_k) * T_k
#                 end
#             end
#             stream.K_k = stream.P_k * D_kp1'
#         end
#         stream.O_k = (I - (α_kp1 - stream.α_k) * stream.P_k) * stream.O_k + stream.K_k * (R_kp1 - D_kp1 * stream.O_k)
#         stream.α_k = α_kp1 # update the regularization term
#     else
#         if !isnothing(Q_kp1)
#             if K == 1  # rank-1 update
#                 u = stream.P_k * D_kp1'
#                 stream.P_k -= u * u' / (Q_kp1 + dot(D_kp1, u))
#             else
#                 Q_k_inv = Q_kp1 \ I
#                 if isnothing(stream.tol)
#                     stream.P_k -= stream.P_k * D_kp1' * ((Q_kp1 + D_kp1 * stream.P_k * D_kp1') \ D_kp1) * stream.P_k
#                 else
#                     stream.P_k -= stream.P_k * D_kp1' * (pinv(Q_kp1 + D_kp1 * stream.P_k * D_kp1'; atol=stream.tol[1]) * D_kp1) * stream.P_k
#                 end
#             end
#             stream.K_k = stream.P_k * D_kp1' * Q_k_inv
#         else
#             if K == 1  # rank-1 update
#                 u = stream.P_k * D_kp1'
#                 stream.P_k -= u * u' / (1.0 + dot(D_kp1, u))
#             else
#                 if isnothing(stream.tol)
#                     stream.P_k -= stream.P_k * D_kp1' * ((1.0I(K) + D_kp1 * stream.P_k * D_kp1') \ D_kp1) * stream.P_k
#                 else
#                     stream.P_k -= stream.P_k * D_kp1' * (pinv(1.0I(K) + D_kp1 * stream.P_k * D_kp1'; atol=stream.tol[1]) * D_kp1) * stream.P_k
#                 end
#             end
#             stream.K_k = stream.P_k * D_kp1'
#         end
#         stream.O_k += stream.K_k * (R_kp1 - D_kp1 * stream.O_k)
#     end

#     return D_kp1
# end



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