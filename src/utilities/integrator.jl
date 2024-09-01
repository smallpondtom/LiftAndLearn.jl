"""
$(SIGNATURES)

Forward euler scheme integration.

## Arguments
- `A`: linear state operator
- `B`: linear input operator
- `U`: input vector/matrix
- `tdata`: time data
- `IC`: initial conditions

## Return
- `states`: integrated states
"""
function forwardEuler(A::Array, B::Array, U::Array, tdata::VecOrMat, IC::VecOrMat)::Array
    Xdim = length(IC)
    Tdim = length(tdata)
    states = zeros(Xdim, Tdim)
    states[:, 1] = IC

    # If row dim corresponds to # of time steps, transpose the input data
    if size(U, 1) == Tdim
        U = U'
    end

    for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        states[:, j] = (I(Xdim) + Δt * A) * states[:, j-1] + B * U[:,j-1] * Δt
    end
    return states
end


"""
$(SIGNATURES)

Forward euler scheme [dispatch for f(x,u) and u = g(u,t)]

## Arguments
- `f`: Xdot = f(x,g(u,t)) right-hand-side of the dynamics
- `g`: Xdot = f(x,g(u,t)) input function g(u,t)
- `tdata`: time data
- `IC`: initial conditions

## Return
- `states`: integrated states
"""
function forwardEuler(f::Function, g::Function, tdata::VecOrMat, IC::VecOrMat)::Matrix
    Xdim = length(IC)
    Tdim = length(tdata)
    states = zeros(Xdim, Tdim)
    states[:, 1] = IC
    for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        states[:, j] = states[:, j-1] + Δt * f(states[:, j-1], g(tdata[j]))
    end
    return states
end


"""
$(SIGNATURES)

Forward euler scheme [dispatch for f(x,u) and u-input as U-matrix]

## Arguments
- `f`: Xdot = f(x,U) right-hand-side of the dynamics
- `U`: Xdot = f(x,U) input data U
- `tdata`: time data
- `IC`: initial conditions

## Return
- `states`: integrated states
"""
function forwardEuler(f::Function, U::Array, tdata::VecOrMat, IC::VecOrMat)::Matrix
    Xdim = length(IC)
    Tdim = length(tdata)
    states = zeros(Xdim, Tdim)
    states[:, 1] = IC

    # If row dim corresponds to # of time steps, transpose the input data
    if size(U, 1) == Tdim
        U = U'
    end

    @inbounds @views for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        states[:, j] = states[:, j-1] + Δt * f(states[:, j-1], U[:,j])
    end
    return states
end


"""
$(SIGNATURES)

Backward Euler scheme integration.

## Arguments
- `A`: linear state operator
- `B`: linear input operator
- `U`: input data
- `tdata`: time data
- `IC`: initial condtions

## Return
- `states`: integrated states
"""
function backwardEuler(A, B, U, tdata, IC)
    Xdim = length(IC)
    Tdim = length(tdata)
    state = zeros(Xdim, Tdim)
    state[:, 1] = IC

    # If row dim corresponds to # of time steps, transpose the input data
    if size(U, 1) == Tdim
        U = U'
    end

    @inbounds @views for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        state[:, j] = (I(Xdim) - Δt * A) \ (state[:, j-1] + B * U[:,j-1] * Δt)
    end
    return state
end


"""
$(SIGNATURES)

Crank-Nicolson scheme

## Arguments
- `A`: linear state operator
- `B`: linear input operator
- `U`: input data
- `tdata`: time data
- `IC`: initial condtions


## Return
- `states`: integrated states
"""
function crankNicolson(A, B, U, tdata, IC)
    Xdim = length(IC)
    Tdim = length(tdata)
    states = zeros(Xdim, Tdim)
    states[:, 1] = IC

    # If row dim corresponds to # of time steps, transpose the input data
    if size(U, 1) == Tdim
        U = U'
    end

    @inbounds @views for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        states[:, j] = (I(Xdim) - 0.5 * Δt * A) \ ((I(Xdim) + 0.5 * Δt * A) * states[:, j-1] + B * U[:,j-1] * Δt)
    end
    return states
end


"""
$(SIGNATURES)

Semi-Implicit Euler scheme

## Arguments
- `A`: linear state operator
- `B`: linear input operator
- `F`: quadratic state operator
- `U`: input data
- `tdata`: time data
- `IC`: initial condtions

## Return
- `states`: integrated states
"""
function semiImplicitEuler(A, B, F, U, tdata, IC)
    Xdim = length(IC)
    Tdim = length(tdata)
    state = zeros(Xdim, Tdim)
    state[:, 1] = IC

    # If row dim corresponds to # of time steps, transpose the input data
    if size(U, 1) == Tdim
        U = U'
    end

    @inbounds @views for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        # state2 = vech(state[:, j-1] * state[:, j-1]')
        state2 = state[:, j-1] ⊘ state[:, j-1]
        state[:, j] = (I(Xdim) - Δt * A) \ (state[:, j-1] + F * state2 * Δt + B * U[:,j-1] * Δt)
    end
    return state
end


"""
$(SIGNATURES)

Semi-Implicit Euler scheme (dispatch)

## Arguments
- `A`: linear state operator
- `B`: linear input operator
- `F_or_H`: quadratic state operator (F or H)
- `U`: input data
- `tdata`: time data
- `IC`: initial condtions

## Return
- `states`: integrated states
"""
function semiImplicitEuler(A, B, F_or_H, U, tdata, IC, options)
    Xdim = length(IC)
    Tdim = length(tdata)
    state = zeros(Xdim, Tdim)
    state[:, 1] = IC

    # If row dim corresponds to # of time steps, transpose the input data
    if size(U, 1) == Tdim
        U = U'
    end
    
    if options.which_quad_term == "F"
        @inbounds @views for j in 2:Tdim
            Δt = tdata[j] - tdata[j-1]
            # state2 = vech(state[:, j-1] * state[:, j-1]')
            state2 = state[:, j-1] ⊘ state[:, j-1]
            state[:, j] = (I(Xdim) - Δt * A) \ (state[:, j-1] + F_or_H * state2 * Δt + B * U[:,j-1] * Δt)
        end
    elseif options.which_quad_term == "H"
        @inbounds @views for j in 2:Tdim
            Δt = tdata[j] - tdata[j-1]
            state2 = vec(state[:, j-1] * state[:, j-1]')
            state[:, j] = (I(Xdim) - Δt * A) \ (state[:, j-1] + F_or_H * state2 * Δt + B * U[:,j-1] * Δt)
        end
    end
    return state
end