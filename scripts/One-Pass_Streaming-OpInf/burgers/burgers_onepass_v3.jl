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

#==================#
## Burgers' Setup ##
#==================#
Ω = (0.0, 1.0)
Nx = 2^8; dt = 1e-4
burgers = BurgersModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    diffusion_coeffs=0.5, BC=:dirichlet,
)
burgers.IC = 0.1*cos.(π*burgers.xspan)
num_inputs = 10  # number of random inputs for training data

#===============#
## OpInf Setup ##
#===============#
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
        DS=20,  # downsampling factor
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)

#=================#
## Generate data ##
#=================#
seed = 1234
rgen = Random.MersenneTwister(seed)

# Random input/boundary condition for training data
U = randn(rgen, burgers.time_dim, num_inputs) 

μ = burgers.diffusion_coeffs[1]
A, F, B = burgers.finite_diff_model(burgers, μ)
op_burgers = LnL.Operators(A=A, B=B, A2u=F)

# Compute the reference data with the reference input
# Reference input/boundary condition for OpInf testing 
Uref = ones(burgers.time_dim, 1)
Xref = burgers.integrate_model(
    burgers.tspan, burgers.IC, Uref; linear_matrix=A,
    control_matrix=B, quadratic_matrix=F, system_input=true
)

# Compute the training with random input 
X = Array{Float64,3}(undef, size(Xref,1), size(Xref,2)-1, num_inputs)
Xdot = Array{Float64,3}(undef, size(Xref,1), size(Xref,2)-1, num_inputs)
for j in 1:num_inputs
    states = burgers.integrate_model(
        burgers.tspan, burgers.IC, U[:, j], linear_matrix=A,
        control_matrix=B, quadratic_matrix=F, system_input=true
    ) 
    X[:,:,j] = states[:,2:end]
    Xdot[:,:,j] = (states[:,2:end] - states[:,1:end-1]) / dt
end

# Down sample the training data
Xtrain = X[:, 1:options.data.DS:end, :]
Utrain = U[2:end, :][1:options.data.DS:end, :]
Xdot = Xdot[:, 1:options.data.DS:end, :]

# Flattened training data 
Xtrain = reshape(Xtrain, burgers.spatial_dim, :)
Utrain = reshape(Utrain, :, 1)
Xdottrain = reshape(Xdot, burgers.spatial_dim, :)

# Compute the SVD
rmax = 14
tmp = svd(Xtrain)
Vrmax = tmp.U[:, 1:rmax]
Σrmax = tmp.S[1:rmax]

#====================#
## Generate operators
#====================#
# Compute the values for the intrusive model
op_burgers = LnL.Operators(A=A, B=B, A2u=F)
op_pod = LnL.pod(op_burgers, Vrmax, options.system)
Apod = op_pod.A
Bpod = op_pod.B 
Fpod = op_pod.A2u

## Compute OpInf
op_infer = LnL.opinf(
    Vrmax' * Xtrain,
    options; 
    U=Utrain,
    Xhatdot=Vrmax' * Xdottrain,
)
Ainf = op_infer.A
Binf = op_infer.B 
Finf = op_infer.A2u

## Compute One-Pass Streaming-OpInf
options.with_reg = true
options.λ = LnL.TikhonovParameter(A = 1e-9, A2 = 1e-9, B = 1e-9,)
stream = LnL.OnePassStreamingOpInf(
    Xtrain[:,1], Xdottrain[:,1];
    options=options, n=size(Xtrain,1), m=1,
    rank=rmax, finite_diff=false
)
@showprogress for (xi, xdi) in zip(eachcol(Xtrain[:,2:end]), eachcol(Xdottrain[:,2:end]))
    LnL.stream!(stream, xi, xdi, tol=1e-8)
end

##
op_stream = LnL.compute_stream_operators(stream; U=Utrain)

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
        linear_matrix=Apod[1:i, 1:i], control_matrix=Bpod[1:i,:], 
        quadratic_matrix=UniqueKronecker.extractF(Fpod, i), 
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
with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 600))
    ax = Axis(
        fig[1, 1], xlabel = "Reduced dimension", ylabel = "Singular Values",
        yscale=log10, xticks=1:rmax, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    scatterlines!(ax, 1:rmax, Σrmax, label="batch", linewidth=8, markersize=30)
    scatterlines!(ax, 1:rmax, Σ, label="stream", linewidth=5, linestyle=:dash, markersize=20)
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
        fig[1, 1], xlabel = "Reduced dimension", 
        ylabel = "mean relative state error",
        yscale=log10, xticks=1:rmax, titlesize=30,
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    scatterlines!(ax, 1:rmax, intru_state_err, label = "intrusive", 
                  linewidth=8, markersize=30)
    scatterlines!(ax, 1:rmax, opinf_state_err, label = "opinf", 
                  linewidth=5, markersize=20, linestyle=:dash)
    scatterlines!(ax, 1:rmax, stream_state_err, label = "stream", 
                  linewidth=3, markersize=15, linestyle=:dashdot)
    axislegend(ax, position = :lb, labelsize=30, patchsize=(80,30))
    display(fig)
end
