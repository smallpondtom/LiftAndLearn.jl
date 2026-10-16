"""
$(SIGNATURES)

Unpack the operator matrix ``\\mathbf{O}`` into the fields of `operators` and, if the
system has an output, fit the output operator ``\\mathbf{C}``.

The columns of ``\\mathbf{O}`` are split into consecutive blocks of widths `dims` and
assigned to the fields named by `operator_symbols`. Bilinear blocks (symbols containing
`N`) with ``m > 1`` inputs are reshaped into an ``r \\times r \\times m`` array. If
`options.system.output` is nonzero, ``\\mathbf{C}`` is obtained from the least-squares
problem ``\\hat{\\mathbf{X}}^\\top \\mathbf{C}^\\top \\approx \\mathbf{Y}^\\top``, with Tikhonov
regularization `options.λ.C` if `options.with_reg` is `true` and `options.λ.C != 0`.

## Arguments
- `operators::Operators`: operator structure to be filled (mutated).
- `O::AbstractArray`: operator matrix of size ``r \\times d``.
- `Yt::AbstractArray`: tall output data of size ``K \\times l``.
- `Xhat_t::AbstractArray`: tall reduced state data of size ``K \\times r``.
- `dims::AbstractArray`: number of columns of each operator block of `O`.
- `operator_symbols::AbstractArray`: operator symbols in the same order as `dims`.
- `options::AbstractOption`: OpInf options (needs `system.output`, `with_reg` and `λ`).

## Returns
- `nothing`; mutates `operators` in place.

## Notes
- A warning is emitted if `:B` is not placed right after `:A` in `operator_symbols`.
"""
function unpack_operators!(operators::Operators, O::AbstractArray, Yt::AbstractArray, Xhat_t::AbstractArray,
                           dims::AbstractArray, operator_symbols::AbstractArray, options::AbstractOption)
    n = size(O, 1)

    # Compute Chat by solving the least square problem for the output values 
    if !iszero(options.system.output)
        l = size(Yt, 2)
        Chat_t = zeros(n, l)
        if options.with_reg && options.λ.C != 0
            Chat_t = (Xhat_t' * Xhat_t + options.λ.C * I) \ (Xhat_t' * Yt)
        else
            Chat_t = Xhat_t \ Yt
        end
        Chat = transpose(Chat_t)
        setproperty!(operators, :C, Chat)
    end

    B_index = findfirst(==(:B), operator_symbols)
    if !isnothing(B_index)
        B_index > 2 && @warn ":B (control) operator should come right after :A (linear) \
             operator to be compatible with the `get_data_matrix` function."
    end

    TD = 1  # initialize this dummy variable for total dimension (TD)
    for (i, symbol) in zip(dims, operator_symbols)
        if 'N' in string(symbol)  # only implemented for bilinear terms
            m = i ÷ n  # number of inputs
            if m == 1
                setproperty!(operators, symbol, O[:, TD:TD+i-1])
            else 
                Nhat = zeros(n,n,m)
                tmp = O[:, TD:TD+i-1]
                for i in 1:m
                    Nhat[:,:,i] = tmp[:, n*(i-1)+1:n*i]
                end
                setproperty!(operators, symbol, Nhat)
            end
        else
            setproperty!(operators, symbol, O[:, TD:TD+i-1])
        end
        TD += i
    end
end


"""
$(SIGNATURES)

Unpack the operator matrix ``\\mathbf{O}`` into the fields of `operators`.

The columns of ``\\mathbf{O}`` are split into consecutive blocks of widths `dims` and
assigned to the fields named by `operator_symbols`. Bilinear blocks (symbols containing
`N`) with ``m > 1`` inputs are reshaped into an ``r \\times r \\times m`` array.

## Arguments
- `operators::Operators`: operator structure to be filled (mutated).
- `O::AbstractArray`: operator matrix of size ``r \\times d``.
- `dims::AbstractArray`: number of columns of each operator block of `O`.
- `operator_symbols::AbstractArray`: operator symbols in the same order as `dims`.

## Returns
- `nothing`; mutates `operators` in place.

## Notes
- A warning is emitted if `:B` is not placed right after `:A` in `operator_symbols`.
"""
function unpack_operators!(operators::Operators, O::AbstractArray, dims::AbstractArray, operator_symbols::AbstractArray)
    n = size(O, 1)

    B_index = findfirst(==(:B), operator_symbols)
    if !isnothing(B_index)
        # Check if :B operator is right after :A operator
        B_index > 2 && @warn ":B (control) operator should come right after :A (linear) \
            operator to be compatible with the `get_data_matrix` function."
    end

    TD = 1  # initialize this dummy variable for total dimension (TD)
    for (i, symbol) in zip(dims, operator_symbols)
        if 'N' in string(symbol)  # only implemented for bilinear terms
            m = i ÷ n  # number of inputs
            if m == 1
                setproperty!(operators, symbol, O[:, TD:TD+i-1])
            else 
                Nhat = zeros(n,n,m)
                tmp = O[:, TD:TD+i-1]
                for i in 1:m
                    Nhat[:,:,i] = tmp[:, n*(i-1)+1:n*i]
                end
                setproperty!(operators, symbol, Nhat)
            end
        else
            setproperty!(operators, symbol, O[:, TD:TD+i-1])
        end
        TD += i
    end
end