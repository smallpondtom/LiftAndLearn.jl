"""
    getDataMat(Xhat::AbstractArray, Xhat_t::AbstractArray, U::AbstractArray,
        options::AbstractOption) → D

Get the data matrix for the regression problem

## Arguments
- `Xhat::AbstractArray`: projected data matrix
- `Xhat_t::AbstractArray`: projected data matrix (transposed)
- `U::AbstractArray`: input data matrix
- `options::AbstractOption`: options for the operator inference set by the user

## Returns
- `D`: data matrix for the regression problem
"""
function getDataMat(Xhat::AbstractArray, Xhat_t::AbstractArray, U::AbstractArray, options::AbstractOption)
    flag = false

    if options.system.is_lin
        D = Xhat_t
        flag = true
    end

    # Data matrix
    if options.system.has_control
        if flag
            D = hcat(D, U)
        else
            D = U
            flag = true
        end
    end

    if options.system.is_quad  # Quadratic term
        if options.optim.which_quad_term == "F"
            # Assemble matrices Xhat^(1), ..., Xhat^(n) following (11) corresponding to F matrix
            Xsq_t = squareMatStates(Xhat)'
        else
            Xsq_t = kronMatStates(Xhat)'
        end
        # Assemble D matrix
        if flag
            D = hcat(D, Xsq_t)
        else
            D = Xsq_t
            flag = true
        end
    end

    if options.system.is_cubic  # cubic term
        if options.optim.which_cubic_term == "E"
            Xcu_t = cubicMatStates(Xhat)'
        else
            Xcu_t = kron3MatStates(Xhat)'
        end
        if flag
            D = hcat(D, Xcu_t)
        else
            D = Xcu_t
            flag = true
        end
    end

    if options.system.is_bilin  # Bilinear term
        XU = Xhat_t .* U[:, 1]
        for i in 2:options.system.dims[:p]
            XU = hcat(XU, Xhat_t .* U[:, i])
        end
        if flag
            D = hcat(D, XU)
        else
            D = XU
            flag = true
        end
    end

    if options.system.has_const  # constant term
        I = ones(options.system.dims[:m], 1)
        if flag
            D = hcat(D, I)
        else
            D = I
            flag = true
        end
    end

    return D
end


"""
$(SIGNATURES)
"""
getDataMat(Xhat::AbstractArray, U::AbstractArray, options::AbstractOption) = getDataMat(Xhat, transpose(Xhat), U, options)