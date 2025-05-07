"""
One-Pass Streaming-OpInf experiment for 1D heat equation
"""

#=================#
## Load packages
#=================#
using Revise
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
X = heat1d.integrate_model(heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
                            system_input=true, integrator_type=:BackwardEuler)
Xref = copy(X)
Uref = Ubc

# Finite difference matrix
E = zeros(heat1d.time_dim, heat1d.time_dim-1)
for i in 1:heat1d.time_dim, j in 1:heat1d.time_dim-1
    if i == j 
        E[i, j] = -1.0 / dt
    elseif i == (j + 1)
        E[i, j] = 1.0 / dt
    end
end

# Compute the time derivative data
Xdot = X * E

# Reference initial condition
ICref = heat1d.IC

# Training data for OpInf (aligned with time derivative data)
X_opinf = X[:, 2:end]
U_opinf = Ubc[2:end]'

# Flag to use finite difference matrix or not
use_finite_diff_matrix = true

# Compute the SVD of the data
rmax = 10
tmp = use_finite_diff_matrix ? svd(X) : svd(X_opinf)
Vrmax = tmp.U[:, 1:rmax]
Σrmax = tmp.S[1:rmax]

#====================#
## Generate operators
#====================#
# Compute the values for the intrusive model
op_heat = LnL.Operators(A=A, B=B)
op_heat_new = LnL.pod(op_heat, Vrmax, options.system)
Aint = op_heat_new.A
Bint = op_heat_new.B

## Compute OpInf
op_infer = LnL.opinf(X_opinf, Vrmax, options; U=U_opinf, Xdot=Xdot)
Ainf = op_infer.A
Binf = op_infer.B

## Compute One-Pass Streaming-OpInf
if use_finite_diff_matrix  # With finite difference matrix
    options.with_reg = true
    options.λ = LnL.TikhonovParameter(
        A = 1e-9,
    )
    stream = LnL.OnePassStreamingOpInf(
        X[:,1];
        options=options, n=size(X,1), m=size(Ubc',1), rank=rmax, finite_diff=true
    )
    for xi in eachcol(X[:,2:end])
        LnL.stream!(stream, xi)
    end
    op_stream = LnL.compute_onepass_operators(stream, Ubc, E, (2,heat1d.time_dim))
else  # Using the time derivative data
    stream = LnL.OnePassStreamingOpInf(
        X_opinf[:,1], Xdot[:,1];
        options=options, n=size(X,1), m=size(U_opinf,1), rank=rmax, finite_diff=false
    )
    for (xi, xdoti) in zip(eachcol(X_opinf[:,2:end]), eachcol(Xdot[:,2:end]))
        LnL.stream!(stream, xi, xdoti)
    end
    op_stream = LnL.compute_onepass_operators(stream, U_opinf)
end

# Extract operators
Astream = op_stream.A
Bstream = op_stream.B
Vstream = stream.V
Σ = stream.Σ

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
        fig[1, 1], xlabel = "Reduced dimension", ylabel = "Singular Values",
        yscale=log10, xticks=1:rmax, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    scatterlines!(ax, 1:rmax, Σrmax, label="batch", linewidth=8, markersize=30)
    scatterlines!(ax, 1:rmax, Σ[1:rmax], label="stream", linewidth=5, linestyle=:dash, markersize=20)
    axislegend(ax, position = :lb, labelsize=30)
    display(fig)
end

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
        limits=(nothing, nothing, 1e-7, 1e+1),
    )
    scatterlines!(ax, 1:rmax, intru_state_err, label = "intrusive", linewidth=8, markersize=30)
    scatterlines!(ax, 1:rmax, opinf_state_err, label = "opinf", linewidth=5, markersize=20, linestyle=:dash)
    scatterlines!(ax, 1:rmax, stream_state_err, label = "stream", linewidth=3, markersize=15, linestyle=:dashdot)
    axislegend(ax, position = :lb, labelsize=30)
    display(fig)
end
