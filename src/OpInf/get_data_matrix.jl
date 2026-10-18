"""
$(SIGNATURES)

Assemble the data matrix ``\\mathbf{D}`` of the Operator Inference (OpInf) regression problem.

The column blocks of ``\\mathbf{D}`` are stacked in the following order, according to
`options.system`:
1. linear state ``\\hat{\\mathbf{X}}^\\top`` (symbol `:A`),
2. input ``\\mathbf{U}^\\top`` (symbol `:B`),
3. polynomial state terms of order ``i \\geq 2`` (symbols `:Aiu` with the non-redundant
   Kronecker product if `options.optim.nonredundant_operators`, otherwise `:Ai`),
4. state-input coupling terms (symbol `:N` for bilinear, `:Ni` for higher orders),
5. a column of ones for the constant term (symbol `:K`).

## Arguments
- `Xhat::AbstractArray`: reduced state snapshot matrix of size ``r \\times K`` (columns = snapshots).
- `Xhat_t::AbstractArray`: transpose of `Xhat` (size ``K \\times r``).
- `Ut::AbstractArray`: tall input data matrix of size ``K \\times m``.
- `options::AbstractOption`: OpInf options defining the system structure.

## Keyword Arguments
- `verbose::Bool=true`: if `true`, also return the dimension breakdown and the operator symbols.

## Returns
- `D::AbstractMatrix`: data matrix of size ``K \\times d``.
- `dims::Vector`: number of columns of each block of `D` (only if `verbose=true`).
- `operator_symbols::Vector{Symbol}`: operator symbol of each block, in the same order as `dims` (only if `verbose=true`).
"""
function get_data_matrix(Xhat::AbstractArray, Xhat_t::AbstractArray, Ut::AbstractArray, options::AbstractOption;
                         verbose::Bool=true)
    dims = []
    operator_symbols = []
    _, K = checksize(Xhat)
    _, m = checksize(Ut)
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
        if flag
            D = hcat(D, ones(K,1))
        else
            D = ones(K,1)
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

Assemble the data matrix ``\\mathbf{D}`` from `Xhat` and the tall input data `Ut`; same as
the four-argument method with `Xhat_t = Xhat'`.

## Arguments
- `Xhat::AbstractArray`: reduced state snapshot matrix of size ``r \\times K`` (columns = snapshots).
- `Ut::AbstractArray`: tall input data matrix of size ``K \\times m``.
- `options::AbstractOption`: OpInf options defining the system structure.

## Keyword Arguments
- `verbose::Bool`: (required) if `true`, also return the dimension breakdown and the operator symbols.

## Returns
- `D::AbstractMatrix`: data matrix of size ``K \\times d``.
- `dims::Vector`: number of columns of each block of `D` (only if `verbose=true`).
- `operator_symbols::Vector{Symbol}`: operator symbol of each block (only if `verbose=true`).
"""
get_data_matrix(Xhat::AbstractArray, Ut::AbstractArray, options::AbstractOption; verbose::Bool) = get_data_matrix(Xhat, Xhat', Ut, options; verbose)