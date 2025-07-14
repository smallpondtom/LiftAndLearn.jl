"""
$(SIGNATURES)

Approximating the derivative values of the data with different integration schemes

## Arguments
- `X::VecOrMat`: data matrix
- `options::AbstractOption`: operator inference options

## Returns
- `dXdt`: derivative data
- `idx`: index for the specific integration scheme (important for later use)
"""
function time_derivative_approx(X::VecOrMat, options::AbstractOption)
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
    elseif choice == "FE4"  # 4th order Forward Euler
        dXdt = (- 25 * X[:,1:end-4] 
                + 48 * X[:,2:end-3] 
                - 36 * X[:,3:end-2] 
                + 16 * X[:,4:end-1] 
                -  3 * X[:,5:end]) / (12*options.data.Δt)
        idx = 1:N-4
    elseif choice == "BE4"  # 4th order Backward Euler
        dXdt = (   3 * X[:,1:end-4] 
                - 16 * X[:,2:end-3] 
                + 36 * X[:,3:end-2] 
                - 48 * X[:,4:end-1] 
                + 25 * X[:,5:end]) / (12*options.data.Δt)
        idx = 5:N
    elseif choice == "CTD4"  # 4th order Central
        dXdt = (      X[:,1:end-4] 
                - 8 * X[:,2:end-3] 
                + 8 * X[:,4:end-1] 
                -     X[:,5:end]) / (12*options.data.Δt)
        idx = 3:N-2
    elseif choice == "FBCT4"  # 4th order Forward Backward Central
        dXdt = hcat(
            fwd4(X[:, 1:5], options.data.Δt, true),
            fwd4(X[:, 1:5], options.data.Δt, false),
            (      X[:,3:end-5] 
             - 8 * X[:,4:end-4] 
             + 8 * X[:,5:end-3] 
             -     X[:,6:end-2]) / (12*options.data.Δt),
            bwd4(X[:, end-4:end], options.data.Δt, false),
            bwd4(X[:, end-4:end], options.data.Δt, true)
        )
        idx = 1:N
    else
        error("Undefined choice of numerical integration. Choose only an accepted method from: FE (Forward Euler), BE (Backward Euler), SI (Semi-implicit Euler)")
    end
    return dXdt, idx
end


"""
fwd4(xs, dt)

Forward 4th order finite difference time derivative estimation.
"""
function fwd4(xs::Matrix, dt::Float64, is_init::Bool)
    if size(xs,2) < 5
        error("Need at least 5 time points for 4th order finite differences.")
    end
    if is_init
        # Initial point
        dxdt = (-25 * xs[:,1] + 48 * xs[:,2] - 36 * xs[:,3] + 16 * xs[:,4] - 3 * xs[:,5]) / (12*dt)
    else
        # Second point due to using the initial point to the left making it an
        # asymmetric stencil
        dxdt = (-3 * xs[:,1] - 10 * xs[:,2] + 18 * xs[:,3] - 6 * xs[:,4] + xs[:,5]) / (12*dt)
    end
    return dxdt
end
# Dispatch
function fwd4(xs::Array{<:Vector}, dt::Float64, is_init::Bool)
    if length(xs) < 5
        error("Need at least 5 time points for 4th order finite differences.")
    end
    if is_init
        # Initial point
        dxdt = (-25 * xs[1] + 48 * xs[2] - 36 * xs[3] + 16 * xs[4] - 3 * xs[5]) / (12*dt)
    else
        # Second point due to using the initial point to the left making it an
        # asymmetric stencil
        dxdt = (-3 * xs[1] - 10 * xs[2] + 18 * xs[3] - 6 * xs[4] + xs[5]) / (12*dt)
    end
    return dxdt
end


"""
    bwd4(xs, dt)

Backwards 4th order finite difference time derivative estimation.
"""
function bwd4(xs::Matrix, dt::Float64, is_final::Bool)
    if size(xs,2) < 5
        error("Need at least 5 time points for 4th order finite differences.")
    end
    if is_final
        # Final point
        dxdt = (3 * xs[:,end-4] - 16 * xs[:,end-3] + 36 * xs[:,end-2] - 48 * xs[:,end-1] + 25 * xs[:,end]) / (12*dt)
    else
        # Second to last point due to using the final point to the right making it an
        # asymmetric stencil
        dxdt = (-xs[:,end-4] + 6 * xs[:,end-3] - 18 * xs[:,end-2] + 10 * xs[:,end-1] + 3 * xs[:,end]) / (12*dt)
    end
    return dxdt
end
# Dispatch
function bwd4(xs::Array{<:Vector}, dt::Float64, is_final::Bool)
    if length(xs) < 5
        error("Need at least 5 time points for 4th order finite differences.")
    end
    if is_final
        # Final point
        dxdt = (3 * xs[end-4] - 16 * xs[end-3] + 36 * xs[end-2] - 48 * xs[end-1] + 25 * xs[end]) / (12*dt)
    else
        # Second to last point due to using the final point to the right making it an
        # asymmetric stencil
        dxdt = (-xs[end-4] + 6 * xs[end-3] - 18 * xs[end-2] + 10 * xs[end-1] + 3 * xs[end]) / (12*dt)
    end
    return dxdt
end


"""
    ctd4(xs, dt)

Central 4th order finite difference time derivative estimation.
"""
function ctd4(xs::Matrix, dt::Float64)
    if size(xs,2) < 5
        error("Need at least 5 time points for 4th order finite differences.")
    end
    dxdt = (xs[:,1] - 8 * xs[:,2] + 8 * xs[:,4] - xs[:,5]) / (12*dt)
    return dxdt
end
# Dispatch
function ctd4(xs::Array{<:Vector}, dt::Float64)
    if length(xs) < 5
        error("Need at least 5 time points for 4th order finite differences.")
    end
    dxdt = (xs[1] - 8 * xs[2] + 8 * xs[4] - xs[5]) / (12*dt)
    return dxdt
end
