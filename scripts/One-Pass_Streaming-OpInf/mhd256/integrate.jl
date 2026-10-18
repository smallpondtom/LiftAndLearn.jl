using UniqueKronecker

"""
    reduced_model(x, u, A, A2u, A3u, K)

ODE function.
"""
function reduced_model(x, A, A2u, A3u, K)
    return A * x + A2u * (x ⊘ x) + A3u * ⊘(x,3) + K
end

"""
    rk4_step(x, u, dt, A, A2u, A3u, B)

Perform a single RK4 step. 
"""
function rk4_step(x, dt, A, A2u, A3u, K)
    k1 = reduced_model(x, A, A2u, A3u, K)
    k2 = reduced_model(x + (dt/2) * k1, A, A2u, A3u, K)
    k3 = reduced_model(x + (dt/2) * k2, A, A2u, A3u, K)
    k4 = reduced_model(x + dt * k3, A, A2u, A3u, K)
    return x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
end

"""
    rk4_integrate(x0, Uinput, tspan, A, A2u, A3u, B)

Integrate the ODE using the RK4 method over a specified time span.
"""
function rk4_integrate(x0, tspan, A, A2u, A3u, K)
    N = length(tspan)  # number of time points
    n = length(x0)     # dimension of the state
    xs = zeros(n, N)
    xs[:, 1] = x0
    fidx = 0
    @inbounds for i in 1:(N-1)
        x = xs[:,i]
        dt = tspan[i+1] - tspan[i]
        xs[:, i+1] .= rk4_step(x, dt, A, A2u, A3u, K)
        fidx = i+1
        if any(isnan.(xs[:, i+1]))
            @info "NaN encountered in RK4 integration at time step $i"
            break
        end
    end
    return xs, fidx
end


"""
    reduced_model(x, u, A, A2u, K)

ODE function.
"""
function reduced_model(x, A, A2u, K)
    return A * x + A2u * (x ⊘ x) + K
end

"""
    rk4_step(x, u, dt, A, A2u, B)

Perform a single RK4 step. 
"""
function rk4_step(x, dt, A, A2u, K)
    k1 = reduced_model(x, A, A2u, K)
    k2 = reduced_model(x + (dt/2) * k1, A, A2u, K)
    k3 = reduced_model(x + (dt/2) * k2, A, A2u, K)
    k4 = reduced_model(x + dt * k3, A, A2u, K)
    return x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
end

"""
    rk4_integrate(x0, Uinput, tspan, A, A2u, B)

Integrate the ODE using the RK4 method over a specified time span.
"""
function rk4_integrate(x0, tspan, A, A2u, K)
    N = length(tspan)  # number of time points
    n = length(x0)     # dimension of the state
    xs = zeros(n, N)
    xs[:, 1] = x0
    fidx = 0
    @inbounds for i in 1:(N-1)
        x = xs[:,i]
        dt = tspan[i+1] - tspan[i]
        xs[:, i+1] .= rk4_step(x, dt, A, A2u, K)
        fidx = i+1
        if any(isnan.(xs[:, i+1]))
            @info "NaN encountered in RK4 integration at time step $i"
            break
        end
    end
    return xs, fidx
end

