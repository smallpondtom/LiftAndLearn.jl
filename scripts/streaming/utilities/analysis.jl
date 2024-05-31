function quad_indices(N, r)
    xsq_idx = [1 + (N + 1) * (n - 1) - n * (n - 1) / 2 for n in 1:N]
    extract_idx = [collect(x:x+(r-j)) for (j, x) in enumerate(xsq_idx[1:r])]
    return Int.(reduce(vcat, extract_idx))
end

function cube_indices(N,r)
    ct = 0
    tmp = []
    for i in 1:N, j in i:N, k in j:N
        ct += 1
        if (i <= r) && (j <= r) && (k <= r)
            push!(tmp, ct)
        end
    end
    return tmp
end

function extract_indices(stream, r)
    # Start with linear term
    extract_idx = collect(1:r)

    # Quadratic 
    if !iszero(stream.dims[:s2])
        tmp = quad_indices(stream.dims[:n], r)
        extract_idx = vcat(extract_idx, tmp .+ stream.dims[:n])
    end

    # Cubic
    if !iszero(stream.dims[:s3])
        tmp = cube_indices(stream.dims[:n], r)
        extract_idx = vcat(extract_idx, tmp .+ (stream.dims[:n] + stream.dims[:s2]))
    end

    # control
    if !iszero(stream.dims[:m])
        extract_idx = vcat(
            extract_idx, 
            collect(1:stream.dims[:m]) .+ (stream.dims[:n] + stream.dims[:s2] + stream.dims[:s3])
        )
    end

    return extract_idx
end


function construct_Ostar(stream, op, r)
    O_star = op.A[1:r, 1:r]'
    if !iszero(stream.dims[:s2])
        idx = quad_indices(stream.dims[:n], r)
        O_star = [O_star; op.F[1:r, idx]']
    end
    if !iszero(stream.dims[:s3])
        idx = cube_indices(stream.dims[:n], r)
        O_star = [O_star; op.E[1:r, idx]']
    end
    if !iszero(stream.dims[:m])
        O_star = [O_star; op.B[1:r, :]']
    end
    return O_star
end

function get_operators!(tmp, op, r, i, required_operators)
    for symb in required_operators
        if symb == :A
            push!(tmp, op.A[1:i, 1:i])
        elseif symb == :B
            push!(tmp, op.B[1:i, :])
        elseif symb == :F 
            idx = quad_indices(r, i)
            push!(tmp, op.F[1:i, idx])
        elseif symb == :E 
            idx = cube_indices(r, i)
            push!(tmp, op.E[1:i, idx])
        end
    end
end

function compute_rse(op, Xfull, Ufull, Vr, tspan, IC, solver)
    X = solver(op..., Ufull, tspan, Vr' * IC)
    return LnL.compStateError(Xfull, X, Vr), X
end


function analysis_1(ops, model, V, Xfull, Ufull, Yfull, required_operators, solver)
    r = size(V,2)
    rel_state_err = Dict{String, Vector{Float64}}()
    rel_output_err = Dict{String, Vector{Float64}}()
    for (key, op) in ops
        rel_state_err[key] = Vector{Float64}[]
        rel_output_err[key] = Vector{Float64}[]
        for i = 1:r 
            Vr = V[:, 1:i]
            tmp = []
            get_operators!(tmp, op, r, i, required_operators)
            foo, X = compute_rse(tmp, Xfull, Ufull, Vr, model.t, model.IC, solver)
            Y = op.C[1:1, 1:i] * X
            bar = LnL.compOutputError(Yfull, Y)
            push!(rel_state_err[key], foo)
            push!(rel_output_err[key], bar)
        end
    end
    return rel_state_err, rel_output_err
end


function analysis_2(Xhat_stream, U_stream, Y_stream, R_stream, num_of_streams, 
                        op_inf, Xfull, Vr, Ufull, Yfull, model, r_select, options, 
                        required_operators, solver; atol=[0.0,0.0], rtol=[0.0,0.0], 
                        VR=false, α=0.0, β=0.0)
    results = Dict(
        "rse_stream" => Dict(r => zeros(num_of_streams) for r in r_select),
        "roe_stream" => Dict(r => zeros(num_of_streams) for r in r_select),
        "cond_state_EF" => Dict(r => zeros(num_of_streams) for r in r_select),
        "cond_output_EF" => Dict(r => zeros(num_of_streams) for r in r_select),
        "streaming_error" => Dict(r => zeros(num_of_streams) for r in r_select),
        "true_streaming_error" => Dict(r => zeros(num_of_streams) for r in r_select),
        "streaming_error_output" => Dict(r => zeros(num_of_streams) for r in r_select),
        "true_streaming_error_output" => Dict(r => zeros(num_of_streams) for r in r_select)
    )

    # Compute the quantities of interest
    for (i,ri) in collect(enumerate(r_select))
        # Initialize the Streaming-OpInf
        if iszero(atol[1]) && iszero(atol[2])
            stream = LnL.StreamingOpInf(options, size(Vr,2), size(Ufull,2), size(Yfull,1); 
                                        variable_regularize=VR, γs_k=α, γo_k=β)
        else
            stream = LnL.StreamingOpInf(options, size(Vr,2), size(Ufull,2), size(Yfull,1); 
                                        atol=atol, rtol=rtol, variable_regularize=VR, γs_k=α, γo_k=β)
        end

        # _ = stream.init!(stream, Xhat_stream[1], R_stream[1]; 
        #                     U_k=U_stream[1], Y_k=Y_stream[1], α_k=α[1], β_k=β[1])

        # Compute the indices extracted from nonredundant quadratic matrix
        # for a smaller dimension r
        if ri == stream.dims[:n]
            extract_idx = collect(1:stream.dims[:d])
        else
            extract_idx = extract_indices(stream, ri)
        end

        # Compute the initial errors
        O_star = construct_Ostar(stream, op_inf, ri)
        O_star_full = construct_Ostar(stream, op_inf, stream.dims[:n])
        E_k_full = nothing
        # E_k_full = O_star_full - stream.O_k
        # results["streaming_error"][ri][1] = norm(E_k_full[extract_idx, 1:ri], 2) / norm(O_star, 2)
        # results["true_streaming_error"][ri][1] = norm(E_k_full[extract_idx, 1:ri], 2) / norm(O_star, 2)

        # Compute the initial output errors
        C_star = op_inf.C[:, 1:ri]'
        E_ko_full = nothing
        # E_ko_full = op_inf.C' - stream.C_k
        # results["streaming_error_output"][ri][1] = norm(E_ko_full[1:ri, :], 2) / norm(C_star, 2)
        # results["true_streaming_error_output"][ri][1] = norm(E_ko_full[1:ri, :], 2) / norm(C_star, 2)

        # # Unpack the operators from Streaming-OpInf
        # op_stream = stream.unpack_operators(stream)

        # # Integrate the streaming inferred model
        # tmp = []
        # get_operators!(tmp, op_stream, stream.dims[:n], ri, required_operators)
        # Xstream = solver(tmp..., Ufull, model.t, Vr[:,1:ri]' * model.IC)
        # Ystream = op_stream.C[1:1, 1:ri] * Xstream

        # # Compute relative state and output errors
        # results["rse_stream"][ri][1] = LnL.compStateError(Xfull, Xstream, Vr[:,1:ri])
        # results["roe_stream"][ri][1] = LnL.compOutputError(Yfull, Ystream)

        # D_k = nothing  # Initialize D_k
        prog = Progress(num_of_streams, desc="Stream $(ri)-th order")
        for k in 1:num_of_streams
            if VR  # Variable regularization
                D_k = stream.stream!(stream, Xhat_stream[k], R_stream[k]; U_k=U_stream[k], γs_k=α[k])
                stream.stream_output!(stream, Xhat_stream[k], Y_stream[k]; γo_k=β[k])
            else  # Fixed regularization or no regularization
                D_k = stream.stream!(stream, Xhat_stream[k], R_stream[k]; U_k=U_stream[k])
                stream.stream_output!(stream, Xhat_stream[k], Y_stream[k])
            end

            # Compute the error factors
            error_factor = I - stream.K_k * D_k
            error_factor_output = I - stream.Ky_k * Xhat_stream[k]'

            # Compute the condition number of the error factor
            results["cond_state_EF"][ri][k] = cond(error_factor[extract_idx, extract_idx])
            results["cond_output_EF"][ri][k] = cond(error_factor_output[1:ri, 1:ri])

            # Compute the streaming error
            E_k_full = k>1 ? error_factor * E_k_full : O_star_full - stream.O_k
            results["streaming_error"][ri][k] = norm(E_k_full[extract_idx, 1:ri], 2) / norm(O_star, 2)
            results["true_streaming_error"][ri][k] = norm(O_star - stream.O_k[extract_idx, 1:ri], 2) / norm(O_star, 2)

            # Compute the streaming output error
            E_ko_full = k>1 ? error_factor_output * E_ko_full : op_inf.C' - stream.C_k
            results["streaming_error_output"][ri][k] = norm(E_ko_full[1:ri, :], 2) / norm(C_star, 2)
            results["true_streaming_error_output"][ri][k] = norm(C_star - stream.C_k[1:ri, :], 2) / norm(C_star, 2)

            # Unpack the operators from Streaming-OpInf
            op_stream = stream.unpack_operators(stream)

            # Integrate the streaming inferred model
            tmp = []
            get_operators!(tmp, op_stream, stream.dims[:n], ri, required_operators)
            Xstream = solver(tmp..., Ufull, model.t, Vr[:,1:ri]' * model.IC)
            Ystream = op_stream.C[1:1, 1:ri] * Xstream

            # Compute relative state and output errors
            results["rse_stream"][ri][k] = LnL.compStateError(Xfull, Xstream, Vr[:,1:ri])
            results["roe_stream"][ri][k] = LnL.compOutputError(Yfull, Ystream)

            next!(prog)
        end

        # Upper bound values
        results["rse_stream"][ri][results["rse_stream"][ri] .>= 1e20] .= NaN
        results["roe_stream"][ri][results["roe_stream"][ri] .>= 1e20] .= NaN
        results["streaming_error"][ri][results["streaming_error"][ri] .>= 1e20] .= NaN
        results["true_streaming_error"][ri][results["true_streaming_error"][ri] .>= 1e20] .= NaN
        results["streaming_error_output"][ri][results["streaming_error_output"][ri] .>= 1e20] .= NaN
        results["true_streaming_error_output"][ri][results["true_streaming_error_output"][ri] .>= 1e20] .= NaN

        # Replace Inf with NaN
        replace!(results["rse_stream"][ri], Inf=>NaN)
        replace!(results["roe_stream"][ri], Inf=>NaN)
        replace!(results["streaming_error"][ri], Inf=>NaN)
        replace!(results["true_streaming_error"][ri], Inf=>NaN)
        replace!(results["streaming_error_output"][ri], Inf=>NaN)
        replace!(results["true_streaming_error_output"][ri], Inf=>NaN)
    end
    return results
end


function analysis_3(streamsizes, Vr, X, U, Y, R, op_inf, r_select, options; 
                    tol=nothing, α=0.0, β=0.0)

    initial_errs = zeros(length(streamsizes), length(r_select))
    initial_output_errs = zeros(length(streamsizes), length(r_select))

    n = size(Vr, 2)
    m = size(U, 2)
    l = size(Y, 1)

    prog = Progress(length(streamsizes), desc="Initial error over streamsize") 
    for (i, streamsize) in enumerate(streamsizes)
        Xhat_i = (Vr' * X)[:, 1:streamsize]
        U_i = U[1:streamsize, :]
        Y_i = Y[:, 1:streamsize]'
        R_i = R[:, 1:streamsize]'

        # Initialize the stream
        # INFO: Remember to make data matrices a tall matrix except X matrix
        if isnothing(tol)
            stream = LnL.StreamingOpInf(options,n,m,l;γs_k=α, γo_k=β)
        else
            stream = LnL.StreamingOpInf(options,n,m,l; tol=tol, γs_k=α, γo_k=β)
        end
        # _ = stream.init!(stream, Xhat_i,R_i; U_k=U_i, Y_k=Y_i, α_k=α, β_k=β)
        _ = stream.stream!(stream, Xhat_i, R_i; U_k=U_i)
        stream.stream_output!(stream, Xhat_i, Y_i)
        ops = stream.unpack_operators(stream)

        for (j, r) in enumerate(r_select)
            O_star_r = construct_Ostar(stream, op_inf, r)
            O_tmp = construct_Ostar(stream, ops, r)

            # Compute state error
            initial_errs[i,j] = norm(O_star_r - O_tmp, 2) / norm(O_star_r, 2)

            # Compute output error
            C_tmp = ops.C[:, 1:r]'
            initial_output_errs[i,j] = norm(op_inf.C[:,1:r]' - C_tmp, 2) / norm(op_inf.C[:,1:r]', 2)
        end
        next!(prog)
    end
    return initial_errs, initial_output_errs
end
