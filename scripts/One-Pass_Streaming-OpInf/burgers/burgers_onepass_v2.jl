"""
One-Pass Streaming-OpInf experiment for the Viscous Burgers Equation
"""

#=================#
## Load packages
#=================#
using LinearAlgebra
using BlockDiagonals
using CairoMakie
using ProgressMeter
using Random
using Revise
using SparseArrays: spzeros
import PolynomialModelReductionDataset: BurgersModel
import LiftAndLearn as LnL
using Kronecker
using UniqueKronecker

#=================#
## Generate data
#=================#
Ω = (0.0, 1.0)
Nx = 2^7; dt = 1e-4
burgers = BurgersModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    diffusion_coeffs=0.5, BC=:dirichlet,
)
burgers.IC = 0.1*cos.(π*burgers.xspan)

# WARNING: If you're using more than 1 input, you need to be very careful with
# how you form the difference matrix and the indices to align the snapshot data with
# the time derivative data.
num_inputs = 1  # number of random inputs for training data

options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=[1,2],
        control=1,
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=dt,
        deriv_type="SI",
        DS=1,  # downsampling factor
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)

seed = 1234
rgen = Random.MersenneTwister(seed)

Urand = randn(rgen, burgers.time_dim, num_inputs) # Random input/boundary condition for training data

μ = burgers.diffusion_coeffs[1]
A, F, B = burgers.finite_diff_model(burgers, μ)
op_burgers = LnL.Operators(A=A, B=B, A2u=F)

# Compute the reference data with the reference input
Uref = ones(burgers.time_dim, 1);  # Reference input/boundary condition for OpInf testing 
Xref = burgers.integrate_model(
    burgers.tspan, burgers.IC, Uref; linear_matrix=A,
    control_matrix=B, quadratic_matrix=F, system_input=true
)

# Finite difference matrix
E = spzeros(burgers.time_dim, burgers.time_dim-1)
for i in 1:burgers.time_dim, j in 1:burgers.time_dim-1
    if i == j 
        E[i, j] = -1.0 / dt
    elseif i == (j + 1)
        E[i, j] = 1.0 / dt
    end
end

# Compute the training with random input 
Xall = Vector{Matrix{Float64}}(undef, num_inputs)
Xdotall = Vector{Matrix{Float64}}(undef, num_inputs)
X_opinf_all = Vector{Matrix{Float64}}(undef, num_inputs)
for j in 1:num_inputs
    states = burgers.integrate_model(
        burgers.tspan, burgers.IC, Urand[:, j], linear_matrix=A,
        control_matrix=B, quadratic_matrix=F, system_input=true
    ) 
    Xall[j] = states
    X_opinf_all[j] = states[:, 2:end]
    Xdotall[j] = states * E
end
X = reduce(hcat, Xall)
X_opinf = reduce(hcat, X_opinf_all)
Xdot = reduce(hcat, Xdotall)
U = reshape(Urand, burgers.time_dim * num_inputs, 1)
U_opinf = reshape(Urand[2:end,:], (burgers.time_dim - 1) * num_inputs, 1)

# Down sample the training data
X = X[:, 1:options.data.DS:end]
X_opinf = X_opinf[:, 1:options.data.DS:end]
Xdot = Xdot[:, 1:options.data.DS:end]
U = reshape(U[1:options.data.DS:end], 1, :)
U_opinf = reshape(U_opinf[1:options.data.DS:end], 1, :)

## Flag to use finite difference matrix or not
use_finite_diff_matrix = false

# Compute the SVD
rmax = 15
tmp = use_finite_diff_matrix ? svd(X) : svd(X_opinf)
Vrmax = tmp.U[:, 1:rmax]
Σrmax = tmp.S[1:rmax]

#=================#
## Plot the data
#=================#
with_theme(theme_latexfonts()) do
    fig = Figure(size=(800, 600))
    ax = Axis3(
        fig[1, 1], xlabel=L"\omega", ylabel=L"t", zlabel=L"x(\omega,t)",
        titlesize=30, xlabelsize=30, ylabelsize=30, zlabelsize=30,
        xticklabelsize=25, yticklabelsize=25, zticklabelsize=25,
    )
    surface!(ax, burgers.xspan, burgers.tspan, Xref, colormap=:plasma)
    display(fig)
end

#====================#
## Generate operators
#====================#
# Compute the values for the intrusive model
op_heat = LnL.Operators(A=A, B=B, A2u=F)
op_heat_new = LnL.pod(op_heat, Vrmax, options.system)
Aint = op_heat_new.A
Bint = op_heat_new.B 
Fint = op_heat_new.A2u

## Compute OpInf
op_infer = LnL.opinf(X_opinf, Vrmax, options; U=U_opinf, Xdot=Xdot)
Ainf = op_infer.A
Binf = op_infer.B 
Finf = op_infer.A2u

## Compute One-Pass Streaming-OpInf
if use_finite_diff_matrix
    options.with_reg = true
    options.λ = LnL.TikhonovParameter(
        A = 1e-9,
        A2 = 1e-9,
        B = 1e-4,
    )
    stream = LnL.OnePassStreamingOpInf(
        X[:,1];
        options=options, n=size(X,1), m=size(U,1), rank=rmax, finite_diff=true
    )
    for xi in eachcol(X[:,2:end])
        LnL.stream!(stream, xi)
    end
    op_stream = LnL.compute_onepass_operators(stream, U, E, (2,burgers.time_dim))
else
    stream = LnL.OnePassStreamingOpInf(
        X_opinf[:,1], Xdot[:,1];
        options=options, n=size(X_opinf,1), m=size(U_opinf,1), rank=rmax, 
        finite_diff=false
    )
    for (xi, xdoti) in zip(eachcol(X_opinf[:,2:end]), eachcol(Xdot[:,2:end]))
        LnL.stream!(stream, xi, xdoti)
    end
    op_stream = LnL.compute_onepass_operators(stream, U_opinf)
end
Astream = op_stream.A
Fstream = op_stream.A2u
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
    Xint = burgers.integrate_model(
        burgers.tspan, Vr' * burgers.IC, Uref,
        linear_matrix=Aint[1:i, 1:i], control_matrix=Bint[1:i,:], 
        quadratic_matrix=UniqueKronecker.extractF(Fint, i), 
        system_input=true,
    )

    # Integrate the inferred model
    Xinf = burgers.integrate_model(
        burgers.tspan, Vr' * burgers.IC, Uref,
        linear_matrix=Ainf[1:i, 1:i], control_matrix=Binf[1:i,:],
        quadratic_matrix=UniqueKronecker.extractF(Finf, i), 
        system_input=true, 
    )

    # Integrate the streaming model
    Xstream = burgers.integrate_model(
        burgers.tspan, Vr_stream' * burgers.IC, Uref,
        linear_matrix=Astream[1:i, 1:i], control_matrix=Bstream[1:i,:],
        quadratic_matrix=UniqueKronecker.extractF(Fstream, i),
        system_input=true, 
    )

    # Compute errors
    PE = LnL.proj_error(Xref, Vr)
    PE_stream = LnL.proj_error(Xref, Vr_stream)

    # Relative state errors
    SE_int = LnL.rel_state_error(Xref, Xint, Vr)
    SE_inf = LnL.rel_state_error(Xref, Xinf, Vr)
    SE_stream = LnL.rel_state_error(Xref, Xstream, Vr_stream)

    # Sum of error values
    proj_err[i] = PE / burgers.param_dim
    proj_err_stream[i] = PE_stream / burgers.param_dim
    intru_state_err[i] = SE_int / burgers.param_dim
    opinf_state_err[i] = SE_inf / burgers.param_dim
    stream_state_err[i] = SE_stream / burgers.param_dim
end

#=================#
## Plot the errors
#=================#
# with_theme(theme_latexfonts()) do
#     fig = Figure(size = (800, 600))
#     ax = Axis(
#         fig[1, 1], xlabel = "Reduced dimension", ylabel = "Singular Values",
#         yscale=log10, xticks=1:rmax, titlesize=30, 
#         xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
#     )
#     scatterlines!(ax, 1:rmax, Σrmax, label="batch", linewidth=8, markersize=30)
#     scatterlines!(ax, 1:rmax, Σ, label="stream", linewidth=5, linestyle=:dash, markersize=20)
#     axislegend(ax, position = :lb, labelsize=30)
#     display(fig)
# end

# with_theme(theme_latexfonts()) do
#     fig = Figure(size = (800, 600))
#     ax = Axis(
#         fig[1, 1], xlabel = "Reduced dimension", ylabel = "mean relative projection error",
#         yscale=log10, xticks=1:rmax, titlesize=30, 
#         xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
#     )
#     scatterlines!(ax, 1:rmax, proj_err, label="batch", linewidth=8, markersize=30)
#     scatterlines!(ax, 1:rmax, proj_err_stream, label="stream", linewidth=5, linestyle=:dash, markersize=20)
#     axislegend(ax, position = :lb, labelsize=30)
#     display(fig)
# end

with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 600))
    ax = Axis(
        fig[1, 1], xlabel = "Reduced dimension", ylabel = "mean relative state error",
        yscale=log10, xticks=1:rmax, titlesize=30,
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    scatterlines!(ax, 1:rmax, intru_state_err, label = "intrusive", linewidth=8, markersize=30)
    scatterlines!(ax, 1:rmax, opinf_state_err, label = "opinf", linewidth=5, markersize=20, linestyle=:dash)
    scatterlines!(ax, 1:rmax, stream_state_err, label = "stream", linewidth=3, markersize=15, linestyle=:dashdot)
    axislegend(ax, position = :lb, labelsize=30)
    display(fig)
end