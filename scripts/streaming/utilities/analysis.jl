"""
    compute_rse(op_int::LnL.operators, op_inf::LnL.operators, op_stream::LnL.operators, 
        model::LnL.Abstract_Model, Vr::Matrix, Xf::Matrix, Uf::Matrix, Yf::Matrix)

Computes the relative state and output errors of the intrusive, inferred, and streaming inferred models.

# Arguments
- `op_int::LnL.operators`: The intrusive model operators.
- `op_inf::LnL.operators`: The inferred model operators.
- `op_stream::LnL.operators`: The streaming inferred model operators.
- `model::LnL.Abstract_Model`: The model to simulate.
- `Vr::Matrix`: The reduced basis.
- `Xf::Matrix`: The full state matrix.
- `Uf::Matrix`: The input matrix.
- `Yf::Matrix`: The full output matrix.

# Returns
A tuple containing the relative state errors and the relative output errors.
"""
function compute_rse(op_int::LnL.operators, op_inf::LnL.operators, op_stream::LnL.operators, 
        model::LnL.Abstract_Model, Vr::Matrix, Xf::Matrix, Uf::Matrix, Yf::Matrix;
        op_inf_reg::Union{LnL.operators,Nothing}=nothing)
    r = size(Vr, 2)
    if isnothing(op_inf_reg)
        rel_state_err = zeros(r, 3)
        rel_output_err = zeros(r, 3)
    else
        rel_state_err = zeros(r, 4)
        rel_output_err = zeros(r, 4)
    end

    for i = 1:r
        Vrr = Vr[:, 1:i]

        # Integrate the intrusive model
        Xint = LnL.backwardEuler(op_int.A[1:i, 1:i], op_int.B[1:i, :], Uf, model.t, Vrr' * model.IC)
        Yint = op_int.C[1:1, 1:i] * Xint

        # Integrate the inferred model
        Xinf = LnL.backwardEuler(op_inf.A[1:i, 1:i], op_inf.B[1:i, :], Uf, model.t, Vrr' * model.IC)
        Yinf = op_inf.C[1:1, 1:i] * Xinf

        # Integrate the inferred regularized model
        if !isnothing(op_inf_reg)
            Xinf_reg = LnL.backwardEuler(op_inf_reg.A[1:i, 1:i], op_inf_reg.B[1:i, :], Uf, model.t, Vrr' * model.IC)
            Yinf_reg = op_inf_reg.C[1:1, 1:i] * Xinf_reg
            rel_state_err[i, end-1] = LnL.compStateError(Xf, Xinf_reg, Vrr)
            rel_output_err[i, end-1] = LnL.compOutputError(Yf, Yinf_reg)
        end

        # Integrate the streaming inferred model
        Xstr = LnL.backwardEuler(op_stream.A[1:i, 1:i], op_stream.B[1:i, :], Uf, model.t, Vrr' * model.IC)
        Ystr = op_stream.C[1:1, 1:i] * Xstr

        # Compute relative state errors
        rel_state_err[i, 1] = LnL.compStateError(Xf, Xint, Vrr)
        rel_state_err[i, 2] = LnL.compStateError(Xf, Xinf, Vrr)
        rel_state_err[i, end] = LnL.compStateError(Xf, Xstr, Vrr)

        # Compute relative output errors
        rel_output_err[i, 1] = LnL.compOutputError(Yf, Yint)
        rel_output_err[i, 2] = LnL.compOutputError(Yf, Yinf)
        rel_output_err[i, end] = LnL.compOutputError(Yf, Ystr)
    end
    return rel_state_err, rel_output_err
end


"""
    compute_rse_per_stream(Xhat_batch::Array{<:Array}, U_batch::Array{<:Array}, Y_batch::Array{<:Array},
        R_batch::Array{<:Array}, batchsize::Union{Int,Array{<:Int}}, r_select::Vector{<:Int}, options::LnL.Abstract_Option; CONST_BATCH::Bool=true)

Computes the relative state and output errors of the intrusive, inferred, and streaming inferred models per stream update.

# Arguments
- `Xhat_batch::Array{<:Array}`: The state data.
- `U_batch::Array{<:Array}`: The input data.
- `Y_batch::Array{<:Array}`: The output data.
- `R_batch::Array{<:Array}`: The residual data.
- `batchsize::Union{Int,Array{<:Int}}`: The batch size.
- `r_select::Vector{<:Int}`: The reduced basis sizes.
- `options::LnL.Abstract_Option`: The options for the streaming inference.

# Returns
A tuple containing the relative state errors, the relative output errors, the state error accuracy, the output error accuracy, and error condition numbers.
"""
function compute_rse_per_stream(Xhat_batch::Array{<:Array}, U_batch::Array{<:Array}, Y_batch::Array{<:Array},
         R_batch::Array{<:Array}, batchsize::Union{Int,Array{<:Int}}, r_select::Vector{<:Int}, 
         options::LnL.Abstract_Option; tol::Union{Real,Array{<:Real},Nothing}=nothing, CONST_BATCH::Bool=true,
         VR::Bool=false, α::Union{Real,Array{<:Real}}=0.0, β::Union{Real,Array{<:Real}}=0.0)

    # Initialize the stream
    M = CONST_BATCH ? size(X,2)÷batchsize : length(batchsize)
    rel_state_err_stream = zeros(M, length(r_select))
    rel_output_err_stream = zeros(M, length(r_select))
    err_state_acc_stream = zeros(M,2)
    err_output_acc_stream = zeros(M,2)
    err_state_cond = zeros(M)
    err_output_cond = zeros(M)

    trace_cov = zeros(M)

    err_state_acc_stream[1,:] .= 1.0  # Iteration 0 is 1
    err_output_acc_stream[1,:] .= 1.0  # Iteration 0 is 1
    err_state_cond[1] = 1.0  # Iteration 0 is 1
    err_output_cond[1] = 1.0  # Iteration 0 is 1

    stream = isnothing(tol) ? LnL.Streaming_InferOp(options; variable_regularize=VR) : LnL.Streaming_InferOp(options; tol=tol, variable_regularize=VR)
    for (j,ri) in enumerate(r_select)
        for i in 1:M
            if i == 1
                _ = stream.init!(stream, Xhat_batch[i], U_batch[i], Y_batch[i], R_batch[i], α[i], β[i])
            else 
                if VR
                    D_k = stream.stream!(stream, Xhat_batch[i], U_batch[i], R_batch[i], α[i], β[i])
                    stream.stream_output!(stream, Xhat_batch[i], Y_batch[i])
                else
                    D_k = stream.stream!(stream, Xhat_batch[i], U_batch[i], R_batch[i])
                    stream.stream_output!(stream, Xhat_batch[i], Y_batch[i])
                end

                if j == length(r_select)
                    # Store the error factor ||I - K_k * D_k||_2
                    err_state_acc_stream[i,1] = opnorm(I - stream.K_k * D_k, 2)
                    err_output_acc_stream[i,1] = opnorm(I - stream.Ky_k * Xhat_batch[i]', 2)

                    err_state_acc_stream[i,2] = minimum(svd(I - stream.K_k * D_k).S)
                    err_output_acc_stream[i,2] = minimum(svd(I - stream.Ky_k * Xhat_batch[i]').S)

                    # condition numbers
                    err_state_cond[i] = cond(I - stream.K_k * D_k)
                    err_output_cond[i] = cond(I - stream.Ky_k * Xhat_batch[i]')

                    # trace of covariance
                    d = size(stream.O_k,1)
                    cov = stream.O_k * stream.O_k' / (d - 1)
                    trace_cov[i] = sum(diag(cov))
                end
            end
            op_stream = stream.unpack_operators(stream)

            # Integrate the streaming inferred model
            Xstr = LnL.backwardEuler(op_stream.A[1:ri, 1:ri], op_stream.B[1:ri, :], Uf, heat1d.t, Vr[:,1:ri]' * heat1d.IC)
            Ystr = op_stream.C[1:1, 1:ri] * Xstr

            # Compute relative state errors
            rel_state_err_stream[i,j] = LnL.compStateError(Xf, Xstr, Vr[:,1:ri])

            # Compute relative output errors
            rel_output_err_stream[i,j] = LnL.compOutputError(Yf, Ystr)
        end
    end
    err_state_acc_stream[err_output_acc_stream .>= 1e20] .= NaN
    err_output_acc_stream[err_output_acc_stream .>= 1e20] .= NaN
    rel_state_err_stream[rel_state_err_stream .>= 1e20] .= NaN
    rel_output_err_stream[rel_output_err_stream .>= 1e20] .= NaN
    err_state_cond[err_state_cond .>= 1e20] .= NaN
    err_output_cond[err_output_cond .>= 1e20] .= NaN
    trace_cov[trace_cov .>= 1e20] .= NaN

    replace!(err_state_acc_stream, Inf=>NaN)
    replace!(err_output_acc_stream, Inf=>NaN)
    replace!(rel_output_err_stream, Inf=>NaN)
    replace!(rel_state_err_stream, Inf=>NaN)
    replace!(err_state_cond, Inf=>NaN)
    replace!(err_output_cond, Inf=>NaN)
    replace!(trace_cov, Inf=>NaN)
    return rel_state_err_stream, rel_output_err_stream, err_state_acc_stream, err_output_acc_stream, err_state_cond, err_output_cond, trace_cov
end



"""
    compute_inital_stream_error(batchsizes::Union{AbstractArray{<:Int},Int}, Vr::Matrix, X::Matrix, U::Matrix, 
        Y::Matrix, R::Matrix, op_inf::LnL.operators, options::LnL.Abstract_Option)

Computes the initial state and output errors of the streaming inferred model over different batch sizes.

# Arguments
- `batchsizes::Union{AbstractArray{<:Int},Int}`: The batch sizes to consider.
- `Vr::Matrix`: The reduced basis.
- `X::Matrix`: The state matrix.
- `U::Matrix`: The input matrix.
- `Y::Matrix`: The output matrix.
- `R::Matrix`: The residual matrix.
- `op_inf::LnL.operators`: The inferred model operators.
- `options::LnL.Abstract_Option`: The options for the streaming inference.

# Returns
A tuple containing the initial state errors and the initial output errors.
"""
function compute_inital_stream_error(batchsizes::Union{AbstractArray{<:Int},Int}, Vr::Matrix, X::Matrix, U::Matrix, 
        Y::Matrix, R::Matrix, op_inf::LnL.operators, options::LnL.Abstract_Option, tol::Union{Real,Array{<:Real},Nothing}=nothing;
        α::Union{Real,Array{<:Real}}=0.0, β::Union{Real,Array{<:Real}}=0.0)

    initial_errs = zeros(length(batchsizes))
    initial_output_errs = zeros(length(batchsizes))

    O_star = [op_inf.A'; op_inf.B']
    for (i, batchsize) in enumerate(batchsizes)
        # Xhat_batch = LnL.batchify(Vr' * X, batchsize)
        # U_batch = LnL.batchify(U, batchsize)
        # Y_batch = LnL.batchify(Y', batchsize)
        # R_batch = LnL.batchify(R', batchsize)
        Xhat_batch = (Vr' * X)[:, 1:batchsize]
        U_batch = U[1:batchsize, :]
        Y_batch = Y[:, 1:batchsize]'
        R_batch = R[:, 1:batchsize]'

        # Initialize the stream
        # INFO: Remember to make data matrices a tall matrix except X matrix
        stream = isnothing(tol) ? LnL.Streaming_InferOp(options) : LnL.Streaming_InferOp(options; tol=tol)

        # _ = stream.init!(stream, Xhat_batch[1], U_batch[1], Y_batch[1], R_batch[1], α[i], β[i])
        _ = stream.init!(stream, Xhat_batch, U_batch, Y_batch, R_batch, α, β)

        # Stream all at once
        # if variable_regularize
        #     _ = stream.stream!(stream, Xhat_batch[2:end], U_batch[2:end], R_batch[2:end], α[i], β[i])
        #     stream.stream_output!(stream, Xhat_batch[2:end], Y_batch[2:end])
        # else
        #     _ = stream.stream!(stream, Xhat_batch[2:end], U_batch[2:end], R_batch[2:end])
        #     stream.stream_output!(stream, Xhat_batch[2:end], Y_batch[2:end])
        # end

        # Compute state error
        initial_errs[i] = norm(O_star - stream.O_k, 2) / norm(O_star, 2)

        # Compute output error
        initial_output_errs[i] = norm(op_inf.C' - stream.C_k, 2) / norm(op_inf.C', 2)
    end

    return initial_errs, initial_output_errs
end