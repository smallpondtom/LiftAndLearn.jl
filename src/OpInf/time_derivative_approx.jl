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
            (      X[:,1:end-4] 
             - 8 * X[:,2:end-3] 
             + 8 * X[:,4:end-1] 
             -     X[:,5:end]) / (12*options.data.Δt),
            bwd4(X[:, end-4:end], options.data.Δt, false),
            bwd4(X[:, end-4:end], options.data.Δt, true)
        )
        idx = 1:N
    else
        error("Undefined choice of numerical integration. Choose only an " * 
              "accepted method from: FE (Forward Euler), BE (Backward Euler)," *
              "SI (Semi-implicit Euler)")
    end
    return dXdt, idx
end


function finite_diff_matrix(fd_method::String, n::Int, dt::Float64)
    if fd_method == "FE"
        if n < 2
            error("Need at least 2 time points for 1st order finite differences.")
        end
        D = spzeros(n, n-1)
        for i in 1:n-1
            D[i, i]   = -1 / dt
            D[i+1, i] =  1 / dt
        end
        idx = 1:n-1
    elseif fd_method == "BE"
        if n < 2
            error("Need at least 2 time points for 1st order finite differences.")
        end
        D = spzeros(n, n-1)
        for i in 2:n
            D[i-1, i-1] = -1 / dt
            D[i, i-1]   =  1 / dt
        end
        idx = 2:n
    elseif fd_method == "SI"
        if n < 2
            error("Need at least 2 time points for 1st order finite differences.")
        end
        D = spzeros(n, n-1)
        for i in 2:n
            D[i-1, i-1] = -1 / dt
            D[i, i-1]   =  1 / dt
        end
        idx = 2:n
    elseif fd_method == "FE4"
        if n < 5
            error("Need at least 5 time points for 4th order finite differences.")
        end
        D = spzeros(n, n-4)
        for i in 1:n-4
            D[i, i]   = -25 / (12*dt)
            D[i+1, i] =  48 / (12*dt)
            D[i+2, i] = -36 / (12*dt)
            D[i+3, i] =  16 / (12*dt)
            D[i+4, i] =  -3 / (12*dt)
        end
        idx = 1:n-4
    elseif fd_method == "BE4"
        if n < 5
            error("Need at least 5 time points for 4th order finite differences.")
        end
        D = spzeros(n, n-4)
        for i in 1:n-4
            D[i, i]   =   3 / (12*dt)
            D[i+1, i] = -16 / (12*dt)
            D[i+2, i] =  36 / (12*dt)
            D[i+3, i] = -48 / (12*dt)
            D[i+4, i] =  25 / (12*dt)
        end
        idx = 5:n
    elseif fd_method == "CTD4"
        if n < 5
            error("Need at least 5 time points for 4th order finite differences.")
        end
        D = spzeros(n, n-4)
        for i in 3:n-2
            D[i-2, i-2] =  1 / (12*dt)
            D[i-1, i-2] = -8 / (12*dt)
            D[i+1, i-2] =  8 / (12*dt)
            D[i+2, i-2] = -1 / (12*dt)
        end
        idx = 3:n-2
    elseif fd_method == "FBCT4"
        if n < 5
            error("Need at least 5 time points for 4th order finite differences.")
        end
        D = spzeros(n, n)
        # Forward difference for first two points
        D[1, 1] = -25 / (12*dt)
        D[2, 1] =  48 / (12*dt)
        D[3, 1] = -36 / (12*dt)
        D[4, 1] =  16 / (12*dt)
        D[5, 1] =  -3 / (12*dt)

        D[1, 2] =  -3 / (12*dt)
        D[2, 2] = -10 / (12*dt)
        D[3, 2] =  18 / (12*dt)
        D[4, 2] =  -6 / (12*dt)
        D[5, 2] =   1 / (12*dt)

        # central difference for middle points
        for i in 3:n-2
            D[i-2, i] =  1 / (12*dt)
            D[i-1, i] = -8 / (12*dt)
            D[i+1, i] =  8 / (12*dt)
            D[i+2, i] = -1 / (12*dt)
        end

        # Backward difference for last two points
        D[n-4, n-1] =  -1 / (12*dt)
        D[n-3, n-1] =   6 / (12*dt)
        D[n-2, n-1] = -18 / (12*dt)
        D[n-1, n-1] =  10 / (12*dt)
        D[n, n-1]   =   3 / (12*dt)

        D[n-4, n] =   3 / (12*dt)
        D[n-3, n] = -16 / (12*dt)
        D[n-2, n] =  36 / (12*dt)
        D[n-1, n] = -48 / (12*dt)
        D[n, n]   =  25 / (12*dt)

        idx = 1:n
    else
        error("Undefined choice of numerical integration. Choose only an " * 
              "accepted method from: FE (Forward Euler), BE (Backward Euler)," *
              "SI (Semi-implicit Euler), FE4 (4th order Forward Euler), " *
              "BE4 (4th order Backward Euler), CTD4 (4th order Central), " *
              "FBCT4 (4th order Forward Backward Central)")
    end
    return D, idx
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
