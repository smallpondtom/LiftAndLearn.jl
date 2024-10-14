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
    if isempty(Ufull)
        X = solver(op..., tspan, Vr' * IC)
    else
        X = solver(op..., Ufull, tspan, Vr' * IC)
    end
    return LnL.compStateError(Xfull, X, Vr), X
end


function analysis_1(ops, model, V, Xfull, Ufull, Yfull, required_operators, solver; r_select=nothing)
    r = size(V,2)
    rel_state_err = Dict{String, Vector{Float64}}()
    rel_output_err = Dict{String, Vector{Float64}}()
    for (key, op) in ops
        rel_state_err[key] = Vector{Float64}[]
        rel_output_err[key] = Vector{Float64}[]
        for i = (isnothing(r_select) ? (1:r) : r_select)
            Vr = V[:, 1:i]
            tmp = []
            get_operators!(tmp, op, r, i, required_operators)
            if hasfield(typeof(model), :t)
                foo, X = compute_rse(tmp, Xfull, Ufull, Vr, model.t, model.IC, solver)
            else
                foo, X = compute_rse(tmp, Xfull, Ufull, Vr, model.tspan, model.IC, solver)
            end
            Y = op.C[1:end, 1:i] * X
            bar = LnL.compOutputError(Yfull, Y)
            push!(rel_state_err[key], foo)
            push!(rel_output_err[key], bar)
            @info "($key) r = $i, State Error = $foo, Output Error = $bar"
        end
    end
    return rel_state_err, rel_output_err
end