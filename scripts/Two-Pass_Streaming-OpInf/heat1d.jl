"""
1D heat equation iOpInf example
"""

#===============#
## Load Packages
#===============#
using CairoMakie
using IncrementalSVD
using LinearAlgebra
using LiftAndLearn
using PolynomialModelReductionDataset: Heat1DModel
const LnL = LiftAndLearn

#=================#
## Global Settings
#=================#
TOL = 1e-10
SAVEFIG = true

#=============================#
## Include functions and files
#=============================#
include("utilities/plot_theme.jl")
include("utilities/analysis.jl")
include("utilities/plotting.jl")

#========================#
## 1D Heat equation setup
#========================#
Nx = 2^7; dt = 1e-3
heat1d = Heat1DModel(  # define the model
    spatial_domain=(0.0, 1.0), time_domain=(0.0, 2.0), diffusion_coeffs=0.1,
    Δx=1/Nx, Δt=1e-3, BC=:dirichlet
)
foo = zeros(heat1d.spatial_dim)
foo[(Nx÷2+1):end] .= 1
heat1d.IC = foo .* (0.5 * sin.(2π * heat1d.xspan))  # change IC
U = ones(heat1d.time_dim)'  # boundary condition → control input (make sure to let column dim be # of time steps)

# OpInf options
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=1,
        control=1,
        output=1,
    ),
    vars=LnL.VariableStructure(
        N=1,  # number of state variables
    ),
    data=LnL.DataStructure(
        Δt=dt, # time step
        deriv_type="BE"  # backward Euler
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,  # show the optimization process
    ),
)

#===============#
## Generate Data
#===============#
# Construct full model
μ = heat1d.diffusion_coeffs
A, B = heat1d.finite_diff_model(heat1d, μ)
C = ones(1, heat1d.spatial_dim) / heat1d.spatial_dim
op_heat = LnL.Operators(A=A, B=B, C=C)

# Compute the state snapshot data with backward Euler
X = heat1d.integrate_model(
    heat1d.tspan, heat1d.IC, U; linear_matrix=A, control_matrix=B, 
    system_input=true, integrator_type=:BackwardEuler
)

# Compute the output of the system
Y = C * X

#==================================#
## Compute the SVD for the POD basis
#==================================#
r = 12  # order of the reduced form
V, Σ, _ = svd(X)
Vr = V[:, 1:r]
Σr = Σ[1:r]

#=======================#
## Compute the iPOD basis
#=======================#
isvd = iSVD(x1=X[:,1], algo=:baker, max_rank=r)
full_increment!(isvd, X[:,2:end], tol=1e-10, verbose=true)
iVr = isvd.Q[:,1:r]
iΣ = isvd.Σ
iΣr = sort(iΣ, rev=true)[1:r]

#======================#
## Plot Singular Values
#======================#
fig0 = Figure()
ax = Axis(fig0[1,1], title="Singular Values", xlabel="Index", ylabel="Value", yscale=log10)
scatterlines!(ax, 1:r, Σr, color=:black, linewidth=3, label="SVD")
scatterlines!(ax, 1:r, iΣr, color=:red, linewidth=2, linestyle=:dash, label="iSVD")
axislegend(ax, labelsize=20, position=:rt)
display(fig0)

#==============#
## POD-Galerkin
#==============#
op = LnL.pod(op_heat, Vr, options.system)

#=======#
## OpInf
#=======#
# Obtain derivative data
op_inf = LnL.opinf(X, Vr, options; U=U, Y=Y)

#============================#
## Tikhonov Regularized OpInf
#============================#
options.with_reg = true
options.λ = LnL.TikhonovParameter(
    A = 1e-13,
    B = 1e-13,
    C = 1e-10
)
op_inf_reg = LnL.opinf(X, Vr, options; U=U, Y=Y)

#==================#
## Streaming-OpInf
#==================#
# Save data 
Xfull = copy(X)
Yfull = copy(Y)
Ufull = copy(U)

# Obtain derivative data and adjust data
Xdot = (X[:, 2:end] - X[:, 1:end-1]) / heat1d.Δt
idx = 2:heat1d.time_dim
X = X[:, idx]  
U = U[:, idx]
Y = Y[:, idx] 

# Streamify the data based on the selected streamsizes
streamsize = 1
Xhat_stream = LnL.streamify(iVr' * X, streamsize)
U_stream = LnL.streamify(U, streamsize)
Y_stream = LnL.streamify(Y, streamsize)
Xdot_stream = LnL.streamify(iVr' * Xdot, streamsize)
num_of_streams = length(Xhat_stream)

## RLS-Streaming-OpInf
Γs = 1e-10
Γo = 7.6e-9
state_stream, output_stream = LnL.StreamingOpInf(options=options, n=r, m=size(U,1), l=size(Y,1), Γs=Γs, Γo=Γo, algorithm=:RLS)
LnL.stream_all!(state_stream, Xhat_stream, Xdot_stream; U=U_stream, verbose=true)
LnL.stream_output_all!(output_stream, Xhat_stream, Y_stream, verbose=true)
op_rls_stream = LnL.terminate_stream(state_stream, output_stream)

## iQR-Streaming-OpInf
Γs = 1e-13
Γo = 1e-10
state_stream, output_stream = LnL.StreamingOpInf(options=options, n=r, m=size(U,1), l=size(Y,1); Γs=Γs, Γo=Γo, algorithm=:iQRRLS)
LnL.stream_all!(state_stream, Xhat_stream, Xdot_stream; U=U_stream, verbose=true)
LnL.stream_output_all!(output_stream, Xhat_stream, Y_stream, verbose=true)
op_iqrrls_stream = LnL.terminate_stream(state_stream, output_stream)

## QR-Streaming-OpInf
Γs = 1e-13
Γo = 1e-10
state_stream, output_stream = LnL.StreamingOpInf(options=options, n=r, m=size(U,1), l=size(Y,1); Γs=Γs, Γo=Γo, algorithm=:QRRLS)
LnL.stream_all!(state_stream, Xhat_stream, Xdot_stream; U=U_stream, verbose=true)
LnL.stream_output_all!(output_stream, Xhat_stream, Y_stream, verbose=true)
op_qrrls_stream = LnL.terminate_stream(state_stream, output_stream)

#=============================#
## (Analysis 1) Relative Error 
#=============================#
# Collect all operators into a dictionary
op_dict = Dict(
    "POD" => op,
    "OpInf" => op_inf,
    "TR-OpInf" => op_inf_reg,
    "RLS-Streaming-OpInf" => op_rls_stream,
    "iQRRLS-Streaming-OpInf" => op_iqrrls_stream,
    "QRRLS-Streaming-OpInf" => op_qrrls_stream
)

r = size(Vr,2)
rse = Dict{String, Vector{Float64}}()
roe = Dict{String, Vector{Float64}}()
for (key, op) in op_dict
    rse[key] = Vector{Float64}[]
    roe[key] = Vector{Float64}[]
    for i = 1:r
        if occursin(r"Streaming-OpInf", key)
            Vri = iVr[:, 1:i]
        else
            Vri = Vr[:, 1:i]
        end
        # Integrate the system for reconstruction
        Xtmp = heat1d.integrate_model(
            heat1d.tspan, Vri' * heat1d.IC, Ufull; linear_matrix=op.A[1:i,1:i], control_matrix=op.B[1:i,:], 
            system_input=true, integrator_type=:BackwardEuler
        )

        foo = LnL.rel_state_error(Xfull, Xtmp, Vri)
        Ytmp = op.C[1:end, 1:i] * Xtmp
        bar = LnL.rel_output_error(Yfull, Ytmp)
        push!(rse[key], foo)
        push!(roe[key], bar)
        @info "($key) r = $i, State Error = $(round(foo,sigdigits=4)), Output Error = $(round(bar,sigdigits=4))"
    end
end

## Plot
provided_keys = ["POD", "OpInf", "TR-OpInf", "RLS-Streaming-OpInf", "iQRRLS-Streaming-OpInf", "QRRLS-Streaming-OpInf"]
marker_styles = [:circle, :diamond, :cross, :rect, :star5, :hexagon]
line_styles = [:solid, :dash, :dot, :dashdot, :dashdotdot, :dash]
with_theme(theme_latexfonts()) do
    fig1 = Figure(fontsize=20, size=(900,700))
    # Relative State Error
    lines = []
    labels = []
    ax1 = Axis(fig1[1, 1], 
        xlabel=L"reduced dimension, $r$",
        ylabel="Relative State Error", 
        # title="Relative State Error", 
        yscale=log10,
        xlabelsize=30,
        ylabelsize=30,
        xticklabelsize=25,
        yticklabelsize=25,
        xticks=1:r
    )
    for (i,key) in enumerate(provided_keys)
        l = scatterlines!(
            ax1, 1:r, rse[key],
            marker=marker_styles[i], markersize=(35-(i-1)*2),
            linestyle=line_styles[i], linewidth=7,
        )
        push!(lines, l)
        push!(labels, key)
    end
    Legend(fig1[2, 1], 
        lines, labels,
        orientation=:horizontal, 
        halign=:center, 
        tellwidth=false, 
        tellheight=true,
        labelsize=28,
        nbanks=2
    )
    display(fig1)
end