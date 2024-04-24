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
        model::LnL.Abstract_Model, Vr::Matrix, Xf::Matrix, Uf::Matrix, Yf::Matrix)
    r = size(Vr, 2)
    rel_state_err = zeros(r, 3)
    rel_output_err = zeros(r, 3)

    for i = 1:r
        Vrr = Vr[:, 1:i]

        # Integrate the intrusive model
        Xint = LnL.backwardEuler(op_int.A[1:i, 1:i], op_int.B[1:i, :], Uf, model.t, Vrr' * model.IC)
        Yint = op_int.C[1:1, 1:i] * Xint

        # Integrate the inferred model
        Xinf = LnL.backwardEuler(op_inf.A[1:i, 1:i], op_inf.B[1:i, :], Uf, model.t, Vrr' * model.IC)
        Yinf = op_inf.C[1:1, 1:i] * Xinf

        # Integrate the streaming inferred model
        Xstr = LnL.backwardEuler(op_stream.A[1:i, 1:i], op_stream.B[1:i, :], Uf, model.t, Vrr' * model.IC)
        Ystr = op_stream.C[1:1, 1:i] * Xstr

        # Compute relative state errors
        rel_state_err[i, 1] = LnL.compStateError(Xf, Xint, Vrr)
        rel_state_err[i, 2] = LnL.compStateError(Xf, Xinf, Vrr)
        rel_state_err[i, 3] = LnL.compStateError(Xf, Xstr, Vrr)

        # Compute relative output errors
        rel_output_err[i, 1] = LnL.compOutputError(Yf, Yint)
        rel_output_err[i, 2] = LnL.compOutputError(Yf, Yinf)
        rel_output_err[i, 3] = LnL.compOutputError(Yf, Ystr)
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
         R_batch::Array{<:Array}, batchsize::Union{Int,Array{<:Int}}, r_select::Vector{<:Int}, options::LnL.Abstract_Option; CONST_BATCH::Bool=true)

    # Initialize the stream
    M = CONST_BATCH ? size(X,2)÷batchsize : length(batchsize)
    rel_state_err_stream = zeros(M, length(r_select))
    rel_output_err_stream = zeros(M, length(r_select))
    err_state_acc_stream = zeros(M,2)
    err_output_acc_stream = zeros(M,2)
    err_state_cond = zeros(M)
    err_output_cond = zeros(M)

    err_state_acc_stream[1,:] .= 1.0  # Iteration 0 is 1
    err_output_acc_stream[1,:] .= 1.0  # Iteration 0 is 1
    err_state_cond[1] = 1.0  # Iteration 0 is 1
    err_output_cond[1] = 1.0  # Iteration 0 is 1

    stream = LnL.Streaming_InferOp(options)
    for (j,ri) in enumerate(r_select)
        for i in 1:M
            if i == 1
                _ = stream.init!(stream, Xhat_batch[i], U_batch[i], Y_batch[i], R_batch[i])
            else 
                D_k = stream.stream!(stream, Xhat_batch[i], U_batch[i], R_batch[i])
                stream.stream_output!(stream, Xhat_batch[i], Y_batch[i])

                if j == length(r_select)
                    # Store the error factor ||I - K_k * D_k||_2
                    err_state_acc_stream[i,1] = opnorm(I - stream.K_k * D_k, 2)
                    err_output_acc_stream[i,1] = opnorm(I - stream.Ky_k * Xhat_batch[i]', 2)

                    err_state_acc_stream[i,2] = minimum(svd(I - stream.K_k * D_k).S)
                    err_output_acc_stream[i,2] = minimum(svd(I - stream.Ky_k * Xhat_batch[i]').S)

                    # condition numbers
                    err_state_cond[i] = cond(I - stream.K_k * D_k)
                    err_output_cond[i] = cond(I - stream.Ky_k * Xhat_batch[i]')
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
    return rel_state_err_stream, rel_output_err_stream, err_state_acc_stream, err_output_acc_stream, err_state_cond, err_output_cond
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
        Y::Matrix, R::Matrix, op_inf::LnL.operators, options::LnL.Abstract_Option)

    initial_errs = zeros(length(batchsizes))
    initial_output_errs = zeros(length(batchsizes))

    O_star = [op_inf.A'; op_inf.B']
    for (i, batchsize) in enumerate(batchsizes)
        Xhat_batch = LnL.batchify(Vr' * X, batchsize)
        U_batch = LnL.batchify(U, batchsize)
        Y_batch = LnL.batchify(Y', batchsize)
        R_batch = LnL.batchify(R', batchsize)

        # Initialize the stream
        # INFO: Remember to make data matrices a tall matrix except X matrix
        stream = LnL.Streaming_InferOp(options)
        _ = stream.init!(stream, Xhat_batch[1], U_batch[1], Y_batch[1], R_batch[1])

        # Stream all at once
        stream.stream!(stream, Xhat_batch[2:end], U_batch[2:end], R_batch[2:end])
        stream.stream_output!(stream, Xhat_batch[2:end], Y_batch[2:end])

        # Compute state error
        initial_errs[i] = norm(O_star - stream.O_k, 2) / norm(O_star, 2)

        # Compute output error
        initial_output_errs[i] = norm(op_inf.C' - stream.C_k, 2) / norm(op_inf.C', 2)
    end

    return initial_errs, initial_output_errs
end