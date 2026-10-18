"""
One-Pass Streaming-OpInf experiment for 1D heat equation
"""

#=================#
## Load packages ##
#=================#
using Revise
using LinearAlgebra
using BlockDiagonals
using CairoMakie
using ProgressMeter
using Random
import PolynomialModelReductionDataset: Heat1DModel
import LiftAndLearn as LnL

#=======================================#
## Some options for operator inference ##
#=======================================#
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=1,
        control=1,
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        deriv_type="BE"
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)

#=================#
## Generate data ##
#=================#
# System settings
Ω = (0.0, 1.0); Nx = 2^7; dt = 1e-3
heat1d = Heat1DModel(
    spatial_domain=Ω, 
    time_domain=(0.0, 1.0), 
    Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, 
    Δt=dt, 
    diffusion_coeffs=0.3,
)

# Input from the boundary condition
Ubc = ones(heat1d.time_dim)

# Generate the full-model operators
A, B = heat1d.finite_diff_model(heat1d, heat1d.diffusion_coeffs[1])

# Generate the initial condition
heat1d.IC = cos.(2π * heat1d.xspan)

# Compute the states with backward Euler
X = heat1d.integrate_model(
    heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
    system_input=true, integrator_type=:BackwardEuler
)

# Save the reference data for later
Xref = copy(X)
Uref = Ubc

# Generate the finite difference matrix and corresponding indices
E, Δidx = LnL.finite_diff_matrix(options.data.deriv_type, heat1d.time_dim, dt)

# Obtain the data used for training (for batch OpInf)
Xtrain = X[:, Δidx]
Utrain = Ubc[Δidx]
Xdot_train = X * E

# Compute the SVD of the data
rmax = 10
F = svd(Xtrain)
Vrmax = F.U[:, 1:rmax]
Σrmax = F.S[1:rmax]

#=================================#
## Compute the Reduced Operators ##
#=================================#
# Compute the operators for Intrusive-POD
op_pod = LnL.pod(LnL.Operators(A=A, B=B), Vrmax, options.system)
Apod = op_pod.A
Bpod = op_pod.B

## Compute OpInf
op_inf = LnL.opinf(Xtrain, Vrmax, options; U=Utrain, Xdot=Xdot_train)
Ainf = op_inf.A
Binf = op_inf.B

## Compute One-Pass Streaming-OpInf
options.with_reg = true
options.λ = LnL.TikhonovParameter(A=1e-9, B=1e-9)
stream = LnL.OnePassStreamingOpInf(
    X[:,1]; options=options, n=size(X,1), m=1, rank=rmax, finite_diff=true
)
for xi in eachcol(X[:,2:end])
    LnL.stream!(stream, xi)
end
op_stream = LnL.compute_stream_operators(stream, E, (Δidx[1],Δidx[end]), U=Ubc)
Astream = op_stream.A
Bstream = op_stream.B
Vstream = stream.V
Σ = stream.Σ

#===========#
## Analyze ##
#===========#
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
    Xpod = heat1d.integrate_model(
        heat1d.tspan, Vr' * heat1d.IC, Uref,
        linear_matrix=Apod[1:i, 1:i], control_matrix=Bpod[1:i,:],
        system_input=true, integrator_type=:BackwardEuler
    )

    # Integrate the inferred model
    Xinf = heat1d.integrate_model(
        heat1d.tspan, Vr' * heat1d.IC, Uref,
        linear_matrix=Ainf[1:i, 1:i], control_matrix=Binf[1:i,:],
        system_input=true, integrator_type=:BackwardEuler
    )

    # Integrate the streaming model
    Xstream = heat1d.integrate_model(
        heat1d.tspan, Vr_stream' * heat1d.IC, Uref,
        linear_matrix=Astream[1:i, 1:i], control_matrix=Bstream[1:i,:],
        system_input=true, integrator_type=:BackwardEuler
    )

    # Compute errors
    PE = LnL.proj_error(Xref, Vr)
    PE_stream = LnL.proj_error(Xref, Vr_stream)

    # Relative state errors
    SE_int = LnL.rel_state_error(Xref, Xpod, Vr)
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
    scatterlines!(ax, 1:rmax, Σ[1:rmax], label="stream", linewidth=5, 
                  linestyle=:dash, markersize=20)
    axislegend(ax, position = :lb, labelsize=30, patchsize=(80,20))
    display(fig)
end

with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 600))
    ax = Axis(
        fig[1, 1], xlabel = "Reduced dimension", 
        ylabel = "mean relative projection error",
        yscale=log10, xticks=1:rmax, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    scatterlines!(ax, 1:rmax, proj_err, label="batch", linewidth=8, 
                  markersize=30)
    scatterlines!(ax, 1:rmax, proj_err_stream, label="stream", linewidth=5, 
                  linestyle=:dash, markersize=20)
    axislegend(ax, position = :lb, labelsize=30, patchsize=(80,20))
    display(fig)
end

with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 600))
    ax = Axis(
        fig[1, 1], xlabel = "Reduced dimension", 
        ylabel = "mean relative state error",
        yscale=log10, xticks=1:rmax, titlesize=30,
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        limits=(nothing, nothing, 1e-7, 1),
    )
    scatterlines!(ax, 1:rmax, intru_state_err, label = "Intrusive-POD", 
                  linewidth=8, markersize=30)
    scatterlines!(ax, 1:rmax, opinf_state_err, label = "OpInf", 
                  linewidth=5, markersize=20, linestyle=:dash)
    scatterlines!(ax, 1:rmax, stream_state_err, label = "Streaming-OpInf", 
                  linewidth=3, markersize=15, linestyle=:dashdot)
    axislegend(ax, position = :lb, labelsize=30, patchsize=(80,20))
    display(fig)
end
