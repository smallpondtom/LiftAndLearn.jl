"""
    f(x, u, A, A2u, B)

ODE function.
"""
function f(x, u, A, A2u, B)
    return A * x + A2u * (x ⊘ x) + B * u
end

"""
    rk4_step(x, u, dt, A, A2u, B)

Perform a single RK4 step. 
"""
function rk4_step(x, u, dt, A, A2u, B)
    k1 = f(x, u, A, A2u, B)
    k2 = f(x + (dt/2) * k1, u, A, A2u, B)
    k3 = f(x + (dt/2) * k2, u, A, A2u, B)
    k4 = f(x + dt * k3, u, A, A2u, B)
    return x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
end

"""
    rk4_integrate(x0, Uinput, tspan, A, A2u, B)

Integrate the ODE using the RK4 method over a specified time span.
"""
function rk4_integrate(x0, Uinput, tspan, A, A2u, B)
    N = length(tspan)  # number of time points
    n = length(x0)     # dimension of the state
    xs = zeros(n, N)
    xs[:, 1] = x0
    for i in 1:(N-1)
        x = xs[:,i]
        u = Uinput[i]
        dt = tspan[i+1] - tspan[i]
        k1 = f(x, u, A, A2u, B)
        k2 = f(x + (dt/2)*k1, u, A, A2u, B)
        k3 = f(x + (dt/2)*k2, u, A, A2u, B)
        k4 = f(x + dt*k3, u, A, A2u, B)
        xs[:, i+1] .= x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    end
    return xs
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
function fwd4(xs::Array{Vector}, dt::Float64, is_init::Bool)
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
function bwd4(xs::Array{Vector}, dt::Float64, is_final::Bool)
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
function ctd4(xs::Array{Vector}, dt::Float64)
    if length(xs) < 5
        error("Need at least 5 time points for 4th order finite differences.")
    end
    dxdt = (xs[1] - 8 * xs[2] + 8 * xs[4] - xs[5]) / (12*dt)
    return dxdt
end