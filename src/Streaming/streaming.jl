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
        :m => m, :l => l, 
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
function stream!(stream::StreamingOpInf, X_k::AbstractArray{T}, R_k::AbstractArray{T}; U_k::AbstractArray{T}=T[], 
                 Q_k::Union{T,AbstractArray{T}}=size(X_k,2)==1 ? 1.0 : 1.0I(size(X_k,2)),
                 γs_k::T=0.0) where T<:Real
    stream.dims[:K] = size(X_k, 2)
    # tmp, stream.dims[:m] = size(U_k)

    # Construct the data matrix
    # reorganize the dimension of the input matrix
    foo, bar = checksize(U_k)
    if foo == stream.dims[:m] && bar == stream.dims[:K]
        if foo == bar && foo != 1
            @warn "Assuming the row dim is the input dim and the column dim is the number of data points."
        end
        D_k = getDataMat(X_k, U_k', stream.options)
    else
        D_k = getDataMat(X_k, U_k, stream.options)
    end

    # Reorganize the dimension of the derivative data matrix
    foo, bar = checksize(R_k)
    if foo == stream.dims[:n] && bar == stream.dims[:K]
        if foo == bar
            @warn "Assuming the row dim is the state dim and the column dim is the number of data points."
        end
        R_k = R_k'
    end

    # # Construct the data matrix
    # if tmp == 1 && stream.dims[:m] != 1
    #     D_k = getDataMat(X_k, U_k, stream.options)
    # else
    #     D_k = getDataMat(X_k, U_k', stream.options)
    # end

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
                 U_k::AbstractArray{<:AbstractArray{T}}=Vector{T}[], γs_k::AbstractArray{T}=zeros(length(X_k)),
                 Q_k::Union{AbstractArray{<:AbstractArray{T}},AbstractArray{T},Real}=0.0) where T<:Real
    N = length(X_k)
    D_k = nothing # initialize the data matrix
    flag = typeof(Q_k) <: AbstractArray{T} 
    no_input = isempty(U_k)
    for i in 1:N
        if iszero(Q_k)
            D_k = stream!(stream, X_k[i], R_k[i]; U_k=no_input ? T[] : U_k[i], γs_k=γs_k[i])
        else
            D_k = stream!(stream, X_k[i], R_k[i]; U_k=no_input ? T[] : U_k[i], γs_k=γs_k[i], Q_k=flag ? Q_k : Q_k[i])
        end
    end
    return D_k
end


function stream_output!(stream::StreamingOpInf, X_k::AbstractArray{T}, Y_k::AbstractArray{T}; γo_k::Real=0.0, 
                        Z_k::Union{T,AbstractArray{T}}=size(X_k,2)==1 ? 1.0 : 1.0I(size(X_k,2))) where T<:Real
    foo, bar = checksize(Y_k)
    if foo == stream.dims[:l] && bar == stream.dims[:K]
        if foo == bar && foo != 1
            @warn "Assuming the row dim is the output dim and the column dim is the number of data points."
        end
        Y_k = Y_k'
        stream.dims[:l] = foo
        stream.dims[:K] = bar
    else
        stream.dims[:l] = bar
        stream.dims[:K] = foo
    end
    # @assert K == size(X_k, 2) "The number of data points should be the same."
    Xt_k = X_k'

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
            Ghat = E2Gs(Ehat)
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

    return Operators(
        A=Ahat, B=Bhat, C=Chat, F=Fhat, H=Hhat, E=Ehat, G=Ghat, N=Nhat, K=Khat
    )
end

