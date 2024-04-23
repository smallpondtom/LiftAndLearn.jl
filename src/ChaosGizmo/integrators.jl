"""
    RK4(J, Q, dt) → Qnew

4th order Runge-Kutta method for the perturbation integrator.

## Arguments
- `J::AbstractMatrix`: Jacobian of the system
- `Q::AbstractArray`: perturbation state
- `dt::Real`: timestep

## Returns
- `Qnew::AbstractArray`: updated perturbation state
"""
function RK4(J, Q, dt)
    # RK4 steps
    k1 = J * Q
    k2 = J * (Q + 0.5*dt*k1)
    k3 = J * (Q + 0.5*dt*k2)
    k4 = J * (Q + dt*k3)
    # Update the perturbation state
    Qnew = Q + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return Qnew
end


"""
    RK2(J, Q, dt) → Qnew

2nd order Runge-Kutta method for the perturbation integrator.

## Arguments
- `J::AbstractMatrix`: Jacobian of the system
- `Q::AbstractArray`: perturbation state
- `dt::Real`: timestep

## Returns
- `Qnew::AbstractArray`: updated perturbation state
"""
function RK2(J, Q, dt)
    # RK2 (Midpoint method) steps
    k1 = J * Q
    k2 = J * (Q + 0.5*dt*k1)
    # Update the perturbation state
    Qnew = Q + dt * k2
    return Qnew
end


"""
    SSPRK3(J, Q, dt) → Qnew

3rd order Strong Stability Preserving Runge-Kutta method for the perturbation integrator.

## Arguments
- `J::AbstractMatrix`: Jacobian of the system
- `Q::AbstractArray`: perturbation state
- `dt::Real`: timestep

## Returns
- `Qnew::AbstractArray`: updated perturbation state
"""
function SSPRK3(J, Q, dt)
    # First step
    k1 = J * Q
    Q1 = Q + dt * k1

    # Second step
    k2 = J * Q1
    Q2 = (3/4) * Q + (1/4) * Q1 + (1/4) * dt * k2

    # Third step
    k3 = J * Q2
    Qnext = (1/3) * Q + (2/3) * Q2 + (2/3) * dt * k3

    return Qnext
end


"""
    RALSTON4(J, Q, dt) → Qnew

Ralston's fourth-order method for the perturbation integrator.

## Arguments
- `J::AbstractMatrix`: Jacobian of the system
- `Q::AbstractArray`: perturbation state
- `dt::Real`: timestep

## Returns
- `Qnew::AbstractArray`: updated perturbation state
"""
function RALSTON4(J, Q, dt)
    # Coefficients for Ralston's fourth-order method
    b21 = 0.4
    b31 = 0.29697761
    b32 = 0.15875964
    b41 = 0.21810040
    b42 = -3.05096516
    b43 = 3.83286476
    c1 = 0.17476028
    c2 = -0.55148066
    c3 = 1.20553560
    c4 = 0.17118478

    # Ralston's fourth-order steps
    k1 = J * Q
    k2 = J * (Q + b21 * dt * k1)
    k3 = J * (Q + dt * (b31 * k1 + b32 * k2))
    k4 = J * (Q + dt * (b41 * k1 + b42 * k2 + b43 * k3))

    # Update the perturbation state
    Qnew = Q + dt * (c1 * k1 + c2 * k2 + c3 * k3 + c4 * k4)
    
    return Qnew
end


"""
    EULER(J, Q, dt) → Qnew

Euler method for the perturbation integrator.

## Arguments
- `J::AbstractMatrix`: Jacobian of the system
- `Q::AbstractArray`: perturbation state
- `dt::Real`: timestep

## Returns
- `Qnew::AbstractArray`: updated perturbation state
"""
function EULER(J, Q, dt)
    return Q + dt * (J * Q)
end
