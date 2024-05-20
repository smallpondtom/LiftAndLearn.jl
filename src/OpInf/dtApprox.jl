"""
    dtApprox(X::VecOrMat, options::AbstractOption) → dXdt, idx

Approximating the derivative values of the data with different integration schemes

## Arguments
- `X::VecOrMat`: data matrix
- `options::AbstractOption`: operator inference options

## Returns
- `dXdt`: derivative data
- `idx`: index for the specific integration scheme (important for later use)
"""
function dtApprox(X::VecOrMat, options::AbstractOption)
    N = size(X, 2)
    choice = options.data.deriv_type

    if choice == "FE"  # Forward Euler
        dXdt = (X[:, 2:end] - X[:, 1:end-1]) / options.data.Δt
        idx = 1:N-1
    elseif choice == "BE"  # Backward Euler
        dXdt = (X[:, 2:end] - X[:, 1:end-1]) / options.data.Δt
        idx = 2:N
    elseif choice == "SI"  # Semi-implicit Euler
        dXdt = (X[:, 2:end] - X[:, 1:end-1]) / options.data.Δt
        idx = 2:N
    else
        error("Undefined choice of numerical integration. Choose only an accepted method.")
    end
    return dXdt, idx
end


