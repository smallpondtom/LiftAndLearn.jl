"""
One-Pass Streaming-OpInf prototype for 1D heat equation
"""

#=================#
## Load packages
#=================#
using LinearAlgebra
using BlockDiagonals
using CairoMakie
using ProgressMeter
using Random
import PolynomialModelReductionDataset: Heat1DModel
import LiftAndLearn as LnL

#=================#
## Generate data
#=================#
Ω = (0.0, 1.0)
Nx = 2^7; dt = 1e-3
heat1d = Heat1DModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), 
    Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt, 
    diffusion_coeffs=0.3,
)

# Some options for operator inference
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=1,
        control=1,
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=dt,
        deriv_type="BE"
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)

# Input from the boundary condition
Ubc = ones(heat1d.time_dim)

μ = heat1d.diffusion_coeffs[1]
A, B = heat1d.finite_diff_model(heat1d, μ)
C = ones(1, heat1d.spatial_dim) / heat1d.spatial_dim
op_heat = LnL.Operators(A=A, B=B, C=C)

# Only one initial condition and input
heat1d.IC = cos.(2π * heat1d.xspan)

# Compute the states with backward Euler
state = heat1d.integrate_model(heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
                            system_input=true, integrator_type=:BackwardEuler)
Xref = copy(state)
Uref = Ubc
Xdot = (state[:, 2:end] - state[:, 1:end-1]) / dt

ICref = heat1d.IC
X = state[:, 2:end]
U = Ubc[2:end]'

# Different initial conditions and inputs
# rng = MersenneTwister(1234)
# for i in 1:9
#     heat1d.IC = randn(rng) * cos.(2π * heat1d.xspan) + randn(rng) * sin.(2π * heat1d.xspan) * 0.01
#     # heat1d.IC[2:end-1] += randn(heat1d.spatial_dim-2) * 0.1
#     Ubc = ones(heat1d.time_dim) * (rand(rng) * 2 - 1)

#     state = heat1d.integrate_model(heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
#                             system_input=true, integrator_type=:BackwardEuler)
#     X = hcat(X, state[:, 2:end])
#     Xdot = hcat(Xdot, (state[:, 2:end] - state[:, 1:end-1]) / dt)
#     U = hcat(U, Ubc[2:end]')
# end

rmax = 10
tmp = svd(X)
Vrmax = tmp.U[:, 1:rmax]
Σrmax = tmp.S[1:rmax]

#====================================#
## One-Pass Streaming-OpInf function
#====================================#
function reorthogonalize!(V::AbstractMatrix{T}, tol::Real) where {T<:Number}
    # Dimension
    r = size(V, 2)
    R = zeros(T, r, r)
    if abs(dot(V[:, end], V[:, 1])) > tol
        @views for k in 1:r
            for _ = 1:2  # do this twice (from p307 algo 6.11 in [GanderGK2014])
                for i = 1:k-1
                    E = dot(V[:, i], V[:, k])
                    V[:, k] .-= E * V[:, i]
                    R[i, k] += E
                end
            end
            R[k, k] = sqrt(dot(V[:, k], V[:, k]))
            V[:, k] ./= R[k, k]
        end
    end
end

function past_algorithm(X::AbstractMatrix, d::Int, β::Real;
                        W_init=nothing, P_init=nothing)
    n, T = size(X)

    # If no initial W or P is given, initialize them
    if W_init === nothing
        W = 1.0I(n)[:, 1:d]  # identity initialization
    else
        W = copy(W_init)
    end

    if P_init === nothing
        P = 1.0I(d)  # small diagonal initialization
    else
        P = copy(P_init)
    end

    for t in 1:T
        x = X[:, t]            # current sample
        y = W' * x             # y(t) = W^H(t-1)*x(t)  (here W' is Hermitian transpose)
        h = P * y              # h(t) = P(t-1)*y(t)
        denom = β + y' * h
        g = h / denom          # g(t) = h(t)/(β + y^H(t)*h(t))

        # P(t) = (1/β)[ P(t-1) - g(t)*h(t)^H ]
        P .= (1/β) .* (P .- g * h')

        # e(t) = x(t) - W(t-1)*y(t)
        e = x .- W * y

        # W(t) = W(t-1) + e(t)*g(t)^H
        W .+= e * g'
    end

    return W, P
end

function rls!(d::Array{T}, r::Array{T}, P::Array{T}, O::Array{T}) where {T<:Number}
    d = reshape(d, 1, :)
    N = length(d)
    r = reshape(r, 1, :)

    u = (P * d')[:]
    denom = 1 + dot(d, u)
    c = 1 / denom 
    K = c * u
    BLAS.syr!('U', -1.0 / denom, u, P)
    @inbounds for i in 1:N, j in i+1:N
        P[j, i] = P[i, j]
    end

    ξpre = r
    mul!(ξpre, d, O, -1.0, 1.0)
    mul!(O, K, ξpre, 1.0, 1.0)
end

# function OnePassStreamingOpInf(X, Xdot, U, rmax, ϵ, γ, β=1.0)
#     n, num_of_snapshots = size(X)
#     m = size(U, 1)
#     dmax = rmax + m

#     # Initialization for RLS
#     O = zeros(dmax, rmax)
#     P = Matrix(1.0I(dmax) / γ)

#     # Initialization for PAST
#     P_past = Matrix(1.0I(rmax)) / 1e-14
#     V = Matrix(1.0I(n)[:, 1:rmax])

#     for i in 1:num_of_snapshots
#         x = X[:,i] # n x 1
#         xdot = Xdot[:,i] # n x 1
#         u = U[:,i] # m x 1

#         # --- Run the PAST algorithm to compute the subspace/POD basis ---
#         y = V' * x             # y(t) = W^H(t-1)*x(t)  (here W' is Hermitian transpose)
#         h = P_past * y         # h(t) = P(t-1)*y(t)
#         denom = β + y' * h
#         g = h / denom          # g(t) = h(t)/(β + y^H(t)*h(t))
#         P_past .= (1/β) .* triu(P_past .- g * h')
#         @inbounds for j in 1:rmax, k in j+1:rmax
#             P_past[k, j] = P_past[j, k]
#         end
#         e = x .- V * y
#         V .+= e * g'

#         @views reorthogonalize!(V, ϵ)

#         # --- Run the RLS algorithm to compute the operator ---
#         xhat = V' * x
#         rvec = V' * xdot
#         dvec = vcat(xhat, u)
#         rls!(dvec, rvec, P, O)
#     end

#     return O, V
# end

function OnePassStreamingOpInf(X, Xdot, U, rmax, ϵ, γ, μbar=1e-4)
    n, num_of_snapshots = size(X)
    m = size(U, 1)
    dmax = rmax + m

    # Initialization for RLS
    O = zeros(dmax, rmax)
    P = Matrix(1.0I(dmax) / γ)

    # Initialization for PAST
    V = Matrix(1.0I(n)[:, 1:rmax])

    for i in 1:num_of_snapshots
        x = X[:,i] # n x 1
        xdot = Xdot[:,i] # n x 1
        u = U[:,i] # m x 1

        # --- Run the FDPM algorithm to compute the subspace/POD basis ---
        μ = μbar / norm(x)
        r = V' * x 
        T = V + μ * x * r'
        e1 = zeros(rmax)
        e1[1] = 1.0
        a = r - norm(r) * e1
        V = T - 2 * (T * a) * a' / dot(a, a)
        foreach(normalize!, eachcol(V))
        @views reorthogonalize!(V, ϵ)

        # --- Run the RLS algorithm to compute the operator ---
        xhat = V' * x
        rvec = V' * xdot
        dvec = vcat(xhat, u)
        rls!(dvec, rvec, P, O)
    end

    return O, V
end

#====================#
## Generate operators
#====================#
# Intrusive
op_heat = LnL.Operators(A=A, B=B)
op_heat_new = LnL.pod(op_heat, Vrmax, options.system)
Aint = op_heat_new.A
Bint = op_heat_new.B

## OpInf
op_infer = LnL.opinf(X, Vrmax, options; U=U, Xdot=Xdot)
Ainf = op_infer.A
Binf = op_infer.B

## One-Pass Streaming-OpInf
Ostream, Vstream = OnePassStreamingOpInf(X, Xdot, U, rmax, 1e-12, 1e-9)
Astream = Ostream[1:rmax,1:rmax]'
Bstream = Ostream[rmax+1:rmax+1,1:rmax]'

# # Check RLS is correct
# Xhat = Vrmax' * X
# Xhatdot = Vrmax' * Xdot
# dmax = rmax+1
# Ostream = zeros(dmax, rmax)
# P = Matrix(1.0I(dmax) / 1e-9) 
# for i in axes(Xhat, 2)
#     xhat = Xhat[:,i] # rmax x 1
#     rvec = Xhatdot[:,i] # rmax x 1
#     u = U[:,i] # m x 1
#     dvec = vcat(xhat, u)
#     rls!(dvec, rvec, P, Ostream)
# end
# Astream = Ostream[1:rmax,1:rmax]'
# Bstream = Ostream[rmax+1:rmax+1,1:rmax]'

#=========#
## Analyze
#=========#
@info "Compute errors"

# Error analysis 
intru_state_err = zeros(rmax)
opinf_state_err = zeros(rmax)
stream_state_err = zeros(rmax)
proj_err = zeros(rmax)
proj_err_stream = zeros(rmax)

@showprogress for i = 1:rmax
    Vr = Vrmax[:,1:i]
    Vr_stream = Vstream[:,1:i]
    # Vr_stream = Vrmax[:,1:i]

    # Integrate the intrusive model
    Xint = heat1d.integrate_model(
        heat1d.tspan, Vr' * ICref, Uref,
        linear_matrix=Aint[1:i, 1:i], control_matrix=Bint[1:i,:],
        system_input=true, integrator_type=:BackwardEuler
    )

    # Integrate the inferred model
    Xinf = heat1d.integrate_model(
        heat1d.tspan, Vr' * ICref, Uref,
        linear_matrix=Ainf[1:i, 1:i], control_matrix=Binf[1:i,:],
        system_input=true, integrator_type=:BackwardEuler
    )

    # Integrate the streaming model
    Xstream = heat1d.integrate_model(
        heat1d.tspan, Vr_stream' * ICref, Uref,
        linear_matrix=Astream[1:i, 1:i], control_matrix=Bstream[1:i,:],
        system_input=true, integrator_type=:BackwardEuler
    )

    # Compute errors
    PE = LnL.proj_error(Xref, Vr)
    PE_stream = LnL.proj_error(Xref, Vr_stream)

    # Relative state errors
    SE_int = LnL.rel_state_error(Xref, Xint, Vr)
    SE_inf = LnL.rel_state_error(Xref, Xinf, Vr)
    SE_stream = LnL.rel_state_error(Xref, Xstream, Vr_stream)

    # Sum of error values
    proj_err[i] = PE / heat1d.param_dim
    proj_err_stream[i] = PE_stream / heat1d.param_dim
    intru_state_err[i] = SE_int / heat1d.param_dim
    opinf_state_err[i] = SE_inf / heat1d.param_dim
    stream_state_err[i] = SE_stream / heat1d.param_dim
end

#=================#
## Plot the errors
#=================#
with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 600))
    ax = Axis(
        fig[1, 1], xlabel = "Reduced dimension", ylabel = "mean relative projection error",
        yscale=log10, xticks=1:rmax, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    scatterlines!(ax, 1:rmax, proj_err, label="batch", linewidth=8, markersize=30)
    scatterlines!(ax, 1:rmax, proj_err_stream, label="stream", linewidth=5, linestyle=:dash, markersize=20)
    axislegend(ax, position = :lb, labelsize=30)
    display(fig)
end

with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 600))
    ax = Axis(
        fig[1, 1], xlabel = "Reduced dimension", ylabel = "mean relative state error",
        yscale=log10, xticks=1:rmax, titlesize=30,
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        limits=(nothing, nothing, 1e-6, 1e+6),
    )
    scatterlines!(ax, 1:rmax, intru_state_err, label = "intrusive", linewidth=8, markersize=30)
    scatterlines!(ax, 1:rmax, opinf_state_err, label = "opinf", linewidth=5, markersize=20, linestyle=:dash)
    scatterlines!(ax, 1:rmax, stream_state_err, label = "stream", linewidth=3, markersize=15, linestyle=:dashdot)
    axislegend(ax, position = :lb, labelsize=30)
    display(fig)
end