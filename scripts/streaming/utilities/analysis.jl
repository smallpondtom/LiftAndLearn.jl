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
        O_star = [O_star; op.G[1:r, idx]']
    end
    if !iszero(stream.dims[:m])
        O_star = [O_star; op.B[1:r, :]']
    end
    return O_star
end


function compute_rse(op, Xfull, Ufull, Vr, tspan, IC, solver)
    X = solver(op..., Ufull, tspan, Vr' * IC)
    return LnL.compStateError(Xfull, X, Vr), X
end


function analyze_heat_1(ops, model, V, Xfull, Ufull, Yfull)
    r = size(V,2)
    rel_state_err = Dict{String, Vector{Float64}}()
    rel_output_err = Dict{String, Vector{Float64}}()
    for (key, op) in ops
        rel_state_err[key] = Vector{Float64}[]
        rel_output_err[key] = Vector{Float64}[]
        for i = 1:r 
            Vr = V[:, 1:i]
            tmp = [op.A[1:i, 1:i], op.B[1:i, :]]
            foo, X = compute_rse(tmp, Xfull, Ufull, Vr, model.t, model.IC, LnL.backwardEuler)
            Y = op.C[1:1, 1:i] * X
            bar = LnL.compOutputError(Yfull, Y)
            push!(rel_state_err[key], foo)
            push!(rel_output_err[key], bar)
        end
    end
    return rel_state_err, rel_output_err
end


function analyze_heat_2(Xhat_stream, U_stream, Y_stream, R_stream, num_of_streams, 
                        op_inf, Xfull, Vr, Ufull, Yfull, model, r_select, options; 
                        tol=0.0, VR=false, α=0.0, β=0.0)
    results = Dict(
        "rse_stream" => Dict(r => Float64[] for r in r_select),
        "roe_stream" => Dict(r => Float64[] for r in r_select),
        "cond_state_EF" => Dict(r => Float64[] for r in r_select),
        "cond_output_EF" => Dict(r => Float64[] for r in r_select),
        "streaming_error" => Dict(r => Float64[] for r in r_select),
        "true_streaming_error" => Dict(r => Float64[] for r in r_select),
        "streaming_error_output" => Dict(r => Float64[] for r in r_select),
        "true_streaming_error_output" => Dict(r => Float64[] for r in r_select)
    )

    # Initialize the Streaming-OpInf
    if iszero(tol)
        stream = LnL.StreamingOpInf(options; variable_regularize=VR)
    else
        stream = LnL.StreamingOpInf(options; tol=tol, variable_regularize=VR)
    end

    # Compute the quantities of interest
    for (i,ri) in enumerate(r_select)
        extract_idx = nothing
        O_star = nothing
        C_star = nothing
        E_k = nothing
        E_k_output = nothing
        for k in 1:num_of_streams
            if k == 1  # First iteration
                if VR  # Variable regularization
                    _ = stream.init!(stream, Xhat_stream[i], R_stream[i]; 
                                     U_k=U_stream[i], Y_k=Y_stream[i], α_k=α[i], β_k=β[i])
                else  # Fixed regularization or no regularization
                    _ = stream.init!(stream, Xhat_stream[i], R_stream[i]; 
                                    U_k=U_stream[i], Y_k=Y_stream[i], α_k=α, β_k=β)
                end

                # Compute the indices extracted from nonredundant quadratic matrix
                # for a smaller dimension r
                extract_idx = extract_indices(stream, ri)

                # Compute the initial errors
                O_star = construct_Ostar(stream, op_inf, ri)
                E_k = O_star - stream.O_k[extract_idx, 1:ri]
                push!(results["streaming_error"][ri], norm(E_k, 2) / norm(O_star, 2))
                push!(results["true_streaming_error"][ri], norm(E_k, 2) / norm(O_star, 2))

                # Compute the initial output errors
                C_star = op_inf.C[:, 1:ri]'
                E_k_output = C_star - stream.C_k[1:ri, :]
                push!(results["streaming_error_output"][ri], norm(E_k_output, 2) / norm(C_star, 2))
                push!(results["true_streaming_error_output"][ri], norm(E_k_output, 2) / norm(C_star, 2))                
            else  
                if VR  # Variable regularization
                    D_k = stream.stream!(stream, Xhat_stream[i], R_stream[i]; U_kp1=U_stream[i], α_kp1=α[i])
                    stream.stream_output!(stream, Xhat_stream[i], Y_stream[i]; β_kp1=β[i])
                else  # Fixed regularization or no regularization
                    D_k = stream.stream!(stream, Xhat_stream[i], R_stream[i]; U_kp1=U_stream[i])
                    stream.stream_output!(stream, Xhat_stream[i], Y_stream[i])
                end

                # Extract the submatrices from K_k and D_k according to ri
                K_ri = stream.K_k[extract_idx, :]
                D_ri = D_k[:, extract_idx]

                # Compute the error factors
                error_factor = I - K_ri * D_ri
                error_factor_output = I - stream.Ky_k[1:ri, :] * Xhat_stream[k][1:ri, :]'

                # Compute the condition number of the error factor
                push!(results["cond_state_EF"][ri], cond(error_factor))
                push!(results["cond_output_EF"][ri], cond(error_factor_output))

                # Compute the streaming error
                E_k = error_factor * E_k
                push!(results["streaming_error"][ri], norm(E_k, 2) / norm(O_star, 2))
                push!(results["true_streaming_error"][ri], norm(O_star - stream.O_k[extract_idx, 1:ri], 2) / norm(O_star, 2))

                # Compute the streaming output error
                E_k_output = error_factor_output * E_k_output
                push!(results["streaming_error_output"][ri], norm(E_k_output, 2) / norm(C_star, 2))
                push!(results["true_streaming_error_output"][ri], norm(C_star - stream.C_k[1:ri, :], 2) / norm(C_star, 2))
            end

            # Unpack the operators from Streaming-OpInf
            op_stream = stream.unpack_operators(stream)

            # Integrate the streaming inferred model
            Xstream = LnL.backwardEuler(op_stream.A[1:ri, 1:ri], op_stream.B[1:ri, :], Ufull, model.t, Vr[:,1:ri]' * model.IC)
            Ystream = op_stream.C[1:1, 1:ri] * Xstream

            # Compute relative state errors
            push!(results["rse_stream"][ri], LnL.compStateError(Xfull, Xstream, Vr[:,1:ri]))

            # Compute relative output errors
            push!(results["roe_stream"][ri], LnL.compOutputError(Yfull, Ystream))
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


function compute_inital_stream_error(batchsizes::Union{AbstractArray{<:Int},Int}, Vr::Matrix, X::Matrix, U::Matrix, 
        Y::Matrix, R::Matrix, op_inf::LnL.operators, options::LnL.AbstractOption, tol::Union{Real,Array{<:Real},Nothing}=nothing;
        α::Union{Real,Array{<:Real}}=0.0, β::Union{Real,Array{<:Real}}=0.0, orders::Union{Int,AbstractArray{<:Int}}=size(Vr,2))

    initial_errs = zeros(length(batchsizes), length(orders))
    initial_output_errs = zeros(length(batchsizes), length(orders))

    O_star = [op_inf.A'; op_inf.B']
    for (i, batchsize) in enumerate(batchsizes)
        Xhat_i = (Vr' * X)[:, 1:batchsize]
        U_i = U[1:batchsize, :]
        Y_i = Y[:, 1:batchsize]'
        R_i = R[:, 1:batchsize]'

        # Initialize the stream
        # INFO: Remember to make data matrices a tall matrix except X matrix
        stream = isnothing(tol) ? LnL.StreamingOpInf(options) : LnL.StreamingOpInf(options; tol=tol)

        _ = stream.init!(stream, Xhat_i,R_i; U_k=U_i, Y_k=Y_i, α_k=α, β_k=β)

        if length(orders) == 1
            # Compute state error
            initial_errs[i] = norm(O_star[] - stream.O_k, 2) / norm(O_star, 2)

            # Compute output error
            initial_output_errs[i] = norm(op_inf.C' - stream.C_k, 2) / norm(op_inf.C', 2)
        else
            ops = stream.unpack_operators(stream)
            for (j, r) in enumerate(orders)
                O_star_r = [op_inf.A[1:r, 1:r]'; op_inf.B[1:r, :]']
                O_tmp = [ops.A[1:r, 1:r]'; ops.B[1:r, :]']

                # Compute state error
                initial_errs[i,j] = norm(O_star_r - O_tmp, 2) / norm(O_star_r, 2)

                # Compute output error
                C_tmp = ops.C[:, 1:r]'
                initial_output_errs[i,j] = norm(op_inf.C[:,1:r]' - C_tmp, 2) / norm(op_inf.C[:,1:r]', 2)
            end
        end
    end

    return initial_errs, initial_output_errs
end

function compute_operator_error()
end