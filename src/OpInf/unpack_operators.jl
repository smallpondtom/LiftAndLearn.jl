"""
$(SIGNATURES)

Unpack the operators from the operator matrix O including the output.
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
                    Nhat[:,:,i] = tmp[:, int(n*(i-1)+1):int(n*i)]
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

Unpack the operators from the operator matrix O.
"""
function unpack_operators!(operators::Operators, O::AbstractArray, dims::AbstractArray, operator_symbols::AbstractArray)
    n = size(O, 1)

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
                    Nhat[:,:,i] = tmp[:, int(n*(i-1)+1):int(n*i)]
                end
                setproperty!(operators, symbol, Nhat)
            end
        else
            setproperty!(operators, symbol, O[:, TD:TD+i-1])
        end
        TD += i
    end
end


