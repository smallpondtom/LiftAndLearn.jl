"""
    getDataMat(Xhat::AbstractArray, Xhat_t::AbstractArray, U::AbstractArray,
        options::AbstractOption) → D

Get the data matrix for the regression problem

## Arguments
- `Xhat::AbstractArray`: projected data matrix
- `Xhat_t::AbstractArray`: projected data matrix (transposed)
- `U::AbstractArray`: input data matrix
- `options::AbstractOption`: options for the operator inference set by the user
- `verbose::Bool=false`: verbose mode returning the dimension breakdown and operator symbols

## Returns
- `D`: data matrix for the regression problem
- `dims`: dimension breakdown of the data matrix
- `operator_symbols`: operator symbols corresponding to `dims` for the regression problem
"""
function getDataMat(Xhat::AbstractArray, Xhat_t::AbstractArray, U::AbstractArray, options::AbstractOption;
                    verbose::Bool=false)
    dims = []
    operator_symbols = []
    K, m = size(U)
    state_struct = copy(options.system.state)
    flag = false

    # State matrix
    if 1 in options.system.state  # State matrix
        D = Xhat_t
        if state_struct == 1
            state_struct = []
        else
            deleteat!(state_struct, findfirst(isequal(1), state_struct))
        end
        push!(dims, size(Xhat_t, 2))
        push!(operator_symbols, :A)
        flag = true
    end

    # if options.system.is_lin
    #     D = Xhat_t
    #     flag = true
    # end

    # Control matrix
    if 1 in options.system.control  # Control matrix
        if flag
            D = hcat(D, U)
        else
            D = U
            flag = true
        end
        push!(dims, size(U, 2))
        push!(operator_symbols, :B)
    end

    # Data matrix
    # if options.system.has_control
    #     if flag
    #         D = hcat(D, U)
    #     else
    #         D = U
    #         flag = true
    #     end
    # end

    # Polynomial state operators
    for i in state_struct
        if options.optim.nonredundant_operators
            Xk_t = unique_kron_snapshot_matrix(Xhat, i)'
            push!(operator_symbols, Symbol("A$(i)u"))
        else
            Xk_t = kron_snapshot_matrix(Xhat, i)'
            push!(operator_symbols, Symbol("A$(i)"))
        end
        if flag
            D = hcat(D, Xk_t)
        else
            D = Xk_t
            flag = true
        end
        push!(dims, size(Xk_t, 2))
    end

    # if options.system.is_quad  # Quadratic term
    #     if options.optim.which_quad_term == "F"
    #         # Assemble matrices Xhat^(1), ..., Xhat^(n) following (11) corresponding to F matrix
    #         Xsq_t = squareMatStates(Xhat)'
    #     else
    #         Xsq_t = kronMatStates(Xhat)'
    #     end
    #     # Assemble D matrix
    #     if flag
    #         D = hcat(D, Xsq_t)
    #     else
    #         D = Xsq_t
    #         flag = true
    #     end
    # end

    # if options.system.is_cubic  # cubic term
    #     if options.optim.which_cubic_term == "E"
    #         Xcu_t = cubeMatStates(Xhat)'
    #     else
    #         Xcu_t = kron3MatStates(Xhat)'
    #     end
    #     if flag
    #         D = hcat(D, Xcu_t)
    #     else
    #         D = Xcu_t
    #         flag = true
    #     end
    # end

    # State and input coupled operators
    if !iszero(options.system.coupled_input)
        for i in options.system.coupled_input
            if i == 1
                XU = Xhat_t .* U[:, 1]
                for j in 2:m
                    XU = hcat(XU, Xhat_t .* U[:, j])
                end
                push!(operator_symbols, :N)
            else
                if options.optim.nonredundant_operators
                    XU = unique_kron_snapshot_matrix(Xhat, i)' .* U[:, 1]
                    for j in 2:m
                        XU = hcat(XU, unique_kron_snapshot_matrix(Xhat, i)' .* U[:, j])
                    end
                else
                    XU = kron_snapshot_matrix(Xhat, i)' .* U[:, 1]
                    for j in 2:m
                        XU = hcat(XU, kron_snapshot_matrix(Xhat, i)' .* U[:, j])
                    end
                end
                push!(operator_symbols, Symbol("N$(i)"))
            end
            if flag
                D = hcat(D, XU)
            else
                D = XU
                flag = true
            end
            push!(dims, size(XU, 2))
        end
    end

    # if options.system.is_bilin  # Bilinear term
    #     XU = Xhat_t .* U[:, 1]
    #     for i in 2:options.system.dims[:m]
    #         XU = hcat(XU, Xhat_t .* U[:, i])
    #     end
    #     if flag
    #         D = hcat(D, XU)
    #     else
    #         D = XU
    #         flag = true
    #     end
    # end

    # Constant term
    if !iszero(options.system.constant)
        I = ones(K,1)
        if flag
            D = hcat(D, I)
        else
            D = I
            flag = true
        end
        push!(dims, 1)
        push!(operator_symbols, :K)
    end

    if verbose
        return D, dims, operator_symbols
    else
        return D
    end
end


"""
$(SIGNATURES)
"""
getDataMat(Xhat::AbstractArray, U::AbstractArray, options::AbstractOption) = getDataMat(Xhat, transpose(Xhat), U, options)