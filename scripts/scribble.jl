using LiftAndLearn
using LinearAlgebra
const LnL = LiftAndLearn

function quad2d(t, x, u)
    return [-2*x[1] + x[1]*x[2]; -x[2] + x[1]*x[2]] + [0; -0.01] * u[1]
end

# function quad2d(t, x, u)
#     dx1 = -2 * x[1] + x[1] * x[2] + 0.2 * u[1]
#     dx2 = -x[2] + x[1] * x[2] + 0.1 * u[2]
#     return [dx1; dx2]
# end

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

function nonlinear_pendulum(t, x, u)
    return [x[2]; -sin(x[1])] + [0.01; -0.02] * u[1]
end



# options = LnL.LSOpInfOption(
#     system=LnL.SystemStructure(
#         state=[1,2],
#         control=1,
#         output=1,
#     ),
#     vars=LnL.VariableStructure(
#         N=1,
#     ),
#     data=LnL.DataStructure(
#         Δt=0.01,
#         deriv_type="BE"
#     ),
#     optim=LnL.OptimizationSetting(
#         verbose=true,
#     ),
# )

# X = Array[]
# U = Array[]
# Xdot = Array[]
# tspan = (0.0, 10.0)
# h = 0.01
# tdim = Int(ceil((tspan[2] - tspan[1]) / h))
# for _ in 1:20
#     u = zeros(1, tdim+1)
#     u[rand(1:tdim, 5)] .= 0.2
#     _, foo, bar = rk4(quad2d, tspan, rand(2), h, u=u)
#     push!(X, foo)
#     push!(U, u)
#     push!(Xdot, bar)
# end
# X = reduce(hcat, X)  
# U = reduce(hcat, U)
# Xdot = reduce(hcat, Xdot)
# Y = 0.5 * X[2, :]

# op1 = LnL.opinf(X, 1.0I(2), options; U=U, Y=Y, Xdot=Xdot)
# println(op1)

# full_op = LnL.Operators(
#     A=[-2 0; 0 -1], A2u=[0 1 0; 0 1 0], B=reshape([0; 0.01], 2,1), C=[0, 0.5]
# )
# op2 = LnL.opinf(X, 1.0I(2), full_op, options; U=U, Y=Y)

# ##
# op3 = LnL.opinf(X, 1.0I(2), options; U=U, Y=Y)

# options = LnL.LSOpInfOption(
#     system=LnL.SystemStructure(
#         state=[1, 2],
#         control=1,
#         output=1,
#     ),
#     vars=LnL.VariableStructure(
#         N=2,
#         N_lift=4,
#     ),
#     data=LnL.DataStructure(
#         Δt=1e-4,
#         DS=100,
#     ),
#     optim=LnL.OptimizationSetting(
#         verbose=true,
#         nonredundant_operators=true,
#         reproject=true,
#     ),
# )

# X = Array[]
# U = Array[]
# Xdot = Array[]
# tspan = (0.0, 10.0)
# h = 0.01
# tdim = Int(ceil((tspan[2] - tspan[1]) / h))
# for _ in 1:20
#     u = zeros(1, tdim+1)
#     u[rand(1:tdim, 5)] .= 0.1
#     _, foo, bar = rk4(nonlinear_pendulum, tspan, rand(2), h, u=u)
#     push!(X, foo)
#     push!(U, u)
#     push!(Xdot, bar)
# end
# X = reduce(hcat, X)  
# U = reduce(hcat, U)
# Xdot = reduce(hcat, Xdot)
# Y = -0.1 * X[1, :] + 0.5 * X[2, :]

# Xsep = [X[1:1, :], X[2:2, :]]
# lifter = LnL.lifting(2, 4, [x -> sin.(x[1]), x -> cos.(x[1])])
# Xlift = lifter.map(Xsep)

# full_op = begin
#     A = zeros(2,2)
#     A[1,2] = 1.0

#     f = (x,u) -> [0; -sin(x[1])]

#     LnL.Operators(
#         A=A, f=f, B=reshape([0.01; -0.02],2,1), C=reshape([-0.1, 0.5],1,2)
#     )
# end

# op1 = LnL.opinf(Xlift, 1.0I(4), lifter, full_op, options; U=U, Y=Y)

# ##
# options.optim.reproject = false
# op3 = LnL.opinf(Xlift, 1.0I(4), lifter, full_op, options; U=U, Y=Y)

system_ = LnL.SystemStructure(
    state=[1,2],
)
vars_ = LnL.VariableStructure(
    N=1,
)
data_ = LnL.DataStructure(
    Δt=1e-4,
    DS=100,
)
optim_ = LnL.OptimizationSetting(
    verbose=true,
    initial_guess=false,
    max_iter=1000,
    reproject=false,
    SIGE=false,
    nonredundant_operators=true,
)

X = Array[]
U = Array[]
Xdot = Array[]
tspan = (0.0, 10.0)
h = 0.01
tdim = Int(ceil((tspan[2] - tspan[1]) / h))
for _ in 1:20
    u = zeros(1, tdim+1)
    _, foo, bar = rk4(quad2d, tspan, rand(2), h, u=u)
    push!(X, foo)
    push!(U, u)
    push!(Xdot, bar)
end
X = reduce(hcat, X)  
U = reduce(hcat, U)
Xdot = reduce(hcat, Xdot)

full_op = LnL.Operators(
    A=[-2 0; 0 -1], A2u=[0 1 0; 0 1 0]
)

options = LnL.EPHECOpInfOption(
    system=system_,
    vars=vars_,
    data=data_,
    optim=optim_,
)
op1 = LnL.epopinf(X, 1.0I(2), options; Xdot=Xdot)