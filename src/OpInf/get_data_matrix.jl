"""
$(SIGNATURES)

Get the data matrix for the regression problem

## Arguments
- `Xhat::AbstractArray`: projected data matrix
- `Xhat_t::AbstractArray`: projected data matrix (transposed)
- `Ut::AbstractArray`: input data matrix (transposed)
- `options::AbstractOption`: options for the operator inference set by the user
- `verbose::Bool=false`: verbose mode returning the dimension breakdown and operator symbols

## Returns
- `D`: data matrix for the regression problem
- `dims`: dimension breakdown of the data matrix
- `operator_symbols`: operator symbols corresponding to `dims` for the regression problem
"""
function get_data_matrix(Xhat::AbstractArray, Xhat_t::AbstractArray, Ut::AbstractArray, options::AbstractOption;
                         verbose::Bool=true)
    dims = []
    operator_symbols = []
    K, m = size(Ut)
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

    # Control matrix
    if 1 in options.system.control  # Control matrix
        if flag
            D = hcat(D, Ut)
        else
            D = Ut
            flag = true
        end
        push!(dims, size(Ut, 2))
        push!(operator_symbols, :B)
    end

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

    # State and input coupled operators
    if !iszero(options.system.coupled_input)
        for i in options.system.coupled_input
            if i == 1
                XU = Xhat_t .* Ut[:, 1]
                for j in 2:m
                    XU = hcat(XU, Xhat_t .* Ut[:, j])
                end
                push!(operator_symbols, :N)
            else
                if options.optim.nonredundant_operators
                    XU = unique_kron_snapshot_matrix(Xhat, i)' .* Ut[:, 1]
                    for j in 2:m
                        XU = hcat(XU, unique_kron_snapshot_matrix(Xhat, i)' .* Ut[:, j])
                    end
                else
                    XU = kron_snapshot_matrix(Xhat, i)' .* Ut[:, 1]
                    for j in 2:m
                        XU = hcat(XU, kron_snapshot_matrix(Xhat, i)' .* Ut[:, j])
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
get_data_matrix(Xhat::AbstractArray, Ut::AbstractArray, options::AbstractOption) = get_data_matrix(Xhat, Xhat', Ut, options)