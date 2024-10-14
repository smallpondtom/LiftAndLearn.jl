function rk4(f, tspan, y0, h; u=nothing)
    t0, tf = tspan
    N = Int(ceil((tf - t0) / h))
    h = (tf - t0) / N  # Adjust step size to fit N steps exactly
    t = t0:h:tf
    if isa(y0, Number)
        y = zeros(N + 1)
        dy = zeros(N + 1)
        y[1] = y0
        # Initialize derivative at the first time point
        un = u === nothing ? 0 : isa(u, Function) ? u(t[1]) : u[1]
        dy[1] = f(t[1], y[1], un)
        for n in 1:N
            tn = t[n]
            yn = y[n]
            # Determine the input u at the required times
            if u === nothing
                un = 0
                un_half = 0
                un_next = 0
            elseif isa(u, Number)
                un = u
                un_half = u
                un_next = u
            elseif typeof(u) <: AbstractArray
                un = u[n]
                un_half = (u[n] + u[n+1])/2
                un_next = u[n+1]
            elseif isa(u, Function)
                un = u(tn)
                un_half = u(tn + h / 2)
                un_next = u(tn + h)
            else
                error("Unsupported type for input u")
            end

            k1 = f(tn, yn, un)
            k2 = f(tn + h / 2, yn + h * k1 / 2, un_half)
            k3 = f(tn + h / 2, yn + h * k2 / 2, un_half)
            k4 = f(tn + h, yn + h * k3, un_next)
            y[n + 1] = yn + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            dy[n] = k1  # Store derivative at time tn
            if n == N
                dy[n + 1] = f(t[n + 1], y[n + 1], un_next)
            end
        end
    else
        m = length(y0)
        y = zeros(m, N + 1)
        dy = zeros(m, N + 1)
        y[:, 1] = y0
        # Initialize derivative at the first time point
        un = u === nothing ? zeros(size(y0)) : isa(u, Function) ? u(t[1]) : u[:, 1]
        dy[:, 1] = f(t[1], y[:, 1], un)
        for n in 1:N
            tn = t[n]
            yn = y[:, n]
            # Determine the input u at the required times
            if u === nothing
                un = zeros(size(y0))
                un_half = zeros(size(y0))
                un_next = zeros(size(y0))
            elseif isa(u, Number) || (isa(u, AbstractArray) && length(u) == 1)
                un = u
                un_half = u
                un_next = u
            elseif typeof(u) <: AbstractArray
                un = u[:, n]
                un_half = (u[:, n] + u[:, n + 1]) / 2
                un_next = u[:, n + 1]
            elseif isa(u, Function)
                un = u(tn)
                un_half = u(tn + h / 2)
                un_next = u(tn + h)
            else
                error("Unsupported type for input u")
            end

            k1 = f(tn, yn, un)
            k2 = f(tn + h / 2, yn + h * k1 / 2, un_half)
            k3 = f(tn + h / 2, yn + h * k2 / 2, un_half)
            k4 = f(tn + h, yn + h * k3, un_next)
            y[:, n + 1] = yn + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            dy[:, n] = k1  # Store derivative at time tn
            if n == N
                dy[:, n + 1] = f(t[n + 1], y[:, n + 1], un_next)
            end
        end
    end
    return t, y, dy
end
