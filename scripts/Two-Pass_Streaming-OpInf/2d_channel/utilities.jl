# Define the ODE right-hand side function
function f(x, A, A2u)
    return A * x + A2u * (x ⊘ x)
end

# RK4 integrator function: returns the state at each time step
function rk4_integrate(x0, tspan, A, A2u)
    N = length(tspan)         # number of time points
    n = length(x0)            # dimension of the state
    xs = zeros(n, N)
    xs[:, 1] = x0
    @inbounds for i in 1:(N-1)
        x = view(xs, :, i)
        dt = tspan[i+1] - tspan[i]
        k1 = f(x, A, A2u)
        k2 = f(x + (dt/2)*k1, A, A2u)
        k3 = f(x + (dt/2)*k2, A, A2u)
        k4 = f(x + dt*k3, A, A2u)
        xs[:, i+1] .= x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    end
    return xs
end

# RK4 derivative approximation function: returns a matrix of approximated time derivatives.
function rk4_time_derivatives(x0, tspan, A, A2u)
    N = length(tspan)         # number of time points
    n = length(x0)            # dimension of the state
    dX = zeros(n, N)          # to store the derivative approximations
    x = x0
    # Optionally, store the derivative at the initial condition.
    dX[:, 1] = f(x, A, A2u)
    for i in 1:(N-1)
        dt = tspan[i+1] - tspan[i]
        k1 = f(x, A, A2u)
        k2 = f(x + (dt/2)*k1, A, A2u)
        k3 = f(x + (dt/2)*k2, A, A2u)
        k4 = f(x + dt*k3, A, A2u)
        # RK4 derivative estimate (weighted average of slopes)
        d_est = (k1 + 2*k2 + 2*k3 + k4) / 6
        dX[:, i] = d_est
        # Update state with the RK4 step
        x = x + dt * d_est
    end
    # Compute derivative at the final state
    dX[:, N] = f(x, A, A2u)
    return dX
end

# Fourth-order finite difference derivative estimation from state data.
# xs is assumed to be a matrix where each column is the state at a time point.
# dt is the uniform time step.

function finite_difference_derivative(xs, tspan)
    n, N = size(xs)
    dX = similar(xs)
    
    # Check if we have enough points for a 4th order scheme
    if N < 5
        error("Need at least 5 time points for 4th order finite differences.")
    end

    # Forward difference (4th order) for the first two time points:
    dt = tspan[2] - tspan[1]
    dX[:, 1] .= (-25 * xs[:,1] + 48 * xs[:,2] - 36 * xs[:,3] + 16 * xs[:,4] - 3 * xs[:,5]) / (12*dt)
    dt = tspan[3] - tspan[2]
    dX[:, 2] .= (-3 * xs[:,1] - 10 * xs[:,2] + 18 * xs[:,3] - 6 * xs[:,4] + xs[:,5]) / (12*dt)
    
    # Central difference (4th order) for interior points:
    
    @inbounds for i in 3:(N-2)
        dt = tspan[i+1] - tspan[i]
        dX[:, i] .= ( xs[:, i-2] - 8*xs[:, i-1] + 8*xs[:, i+1] - xs[:, i+2] ) / (12*dt)
    end

    # Backward difference (4th order) for the last two time points:
    dt = tspan[N] - tspan[N-1]
    dX[:, N-1] .= (-xs[:, N-4] + 6*xs[:, N-3] - 18*xs[:, N-2] + 10*xs[:, N-1] + 3*xs[:, N]) / (12*dt)
    dX[:, N]   .= ( 3*xs[:, N-4] - 16*xs[:, N-3] + 36*xs[:, N-2] - 48*xs[:, N-1] + 25*xs[:, N]) / (12*dt)
    
    return dX
end