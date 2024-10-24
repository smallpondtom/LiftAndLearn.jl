"""
    Streaming-OpInf example of the 2D heat equation.
"""

#===========#
## Packages
#===========#
using CairoMakie
using LinearAlgebra
using ProgressMeter
using PolynomialModelReductionDataset: Heat2DModel
using UniqueKronecker: invec

#=============#
## My modules
#=============#
using LiftAndLearn
const LnL = LiftAndLearn

#=================#
## Global Settings
#=================#
SAVEFIG = true

#=============================#
## Include functions and files
#=============================#
include("utilities/plot_theme.jl")
include("utilities/analysis.jl")
include("utilities/plotting.jl")

#========================#
## 2D Heat equation setup
#========================#
Ω = ((0.0, 1.0), (0.0, 1.0))
Nx = 2^6
Ny = 2^6
heat2d = Heat2DModel(
    spatial_domain=Ω, time_domain=(0,2), 
    Δx=(Ω[1][2] + 1/Nx)/Nx, Δy=(Ω[2][2] + 1/Ny)/Ny, Δt=1e-3,
    diffusion_coeffs=0.1, BC=(:dirichlet, :dirichlet)
)
xgrid0 = heat2d.xspan' .* ones(heat2d.spatial_dim[1])
ygrid0 = ones(heat2d.spatial_dim[2])' .* heat2d.yspan
ux0 = sin.(2π * xgrid0) .* cos.(2π * ygrid0)
heat2d.IC = vec(ux0)  # initial condition

# OpInf options
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=1,
        control=1,
        output=1,
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=1e-3,
        deriv_type="BE"
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)

#===============#
## Generate Data
#===============#
# Construct full model
μ = heat2d.diffusion_coeffs
A, B = heat2d.finite_diff_model(heat2d, μ)
C = ones(1, (Int ∘ prod)(heat2d.spatial_dim)) / heat2d.spatial_dim[1] / heat2d.spatial_dim[2]
op_heat = LnL.Operators(A=A, B=B, C=C)

# Generate the input data
U = [1.0, 1.0, -1.0, -1.0]
U = repeat(U, 1, heat2d.time_dim)

# Compute the state snapshot data with backward Euler
X = heat2d.integrate_model(A, B, U, heat2d.tspan, heat2d.IC)

# Compute the SVD for the POD basis
r = 12  # order of the reduced form
Vr = svd(X).U[:, 1:r]

# Compute the output of the system
Y = C * X

# Copy the data for later analysis
Xfull = copy(X)
Yfull = copy(Y)
Ufull = copy(U)

#=====================#
## Plot Data to Check
#=====================#
Xflat = invec.(eachcol(X), heat2d.spatial_dim...)
with_theme(theme_latexfonts()) do
    fig0 = Figure(fontsize=20, size=(1200,1050))
    ax1 = Axis3(fig0[1, 1], xlabel=L"x", ylabel=L"y", zlabel=L"u(x,y,t)",
                xticks=heat2d.spatial_domain[1][1]:0.2:heat2d.spatial_domain[1][2],
                yticks=heat2d.spatial_domain[2][1]:0.2:heat2d.spatial_domain[2][2],
                xlabelsize=35, ylabelsize=35, zlabelsize=35,
                xticklabelsize=22, yticklabelsize=22, zticklabelsize=22)
    ax2 = Axis(fig0[1, 2], xlabel=L"x", ylabel=L"y", aspect=DataAspect(),
               xticks=heat2d.spatial_domain[1][1]:0.2:heat2d.spatial_domain[1][2],
               yticks=heat2d.spatial_domain[2][1]:0.2:heat2d.spatial_domain[2][2],
               xlabelsize=35, ylabelsize=35, xticklabelsize=22, yticklabelsize=22)
    ax3 = Axis3(fig0[2, 1], xlabel=L"x", ylabel=L"y", zlabel=L"u(x,y,t)",
                xticks=heat2d.spatial_domain[1][1]:0.2:heat2d.spatial_domain[1][2],
                yticks=heat2d.spatial_domain[2][1]:0.2:heat2d.spatial_domain[2][2],
                xlabelsize=35, ylabelsize=35, zlabelsize=35,
                xticklabelsize=22, yticklabelsize=22, zticklabelsize=22)
    ax4 = Axis(fig0[2, 2], xlabel=L"x", ylabel=L"y", aspect=DataAspect(),
               xticks=heat2d.spatial_domain[1][1]:0.2:heat2d.spatial_domain[1][2],
               yticks=heat2d.spatial_domain[2][1]:0.2:heat2d.spatial_domain[2][2],
               xlabelsize=35, ylabelsize=35, xticklabelsize=22, yticklabelsize=22)
    Label(fig0[0, :], "2D Heat Equation at initial (top) and final time (bottom)", fontsize=35)
    colsize!(fig0.layout, 2, Aspect(1, 0.8))
    sf1 = surface!(ax1, heat2d.xspan, heat2d.yspan, Xflat[1])
    hm1 = heatmap!(ax2, heat2d.xspan, heat2d.yspan, Xflat[1])
    sf2 = surface!(ax3, heat2d.xspan, heat2d.yspan, Xflat[end])
    hm2 = heatmap!(ax4, heat2d.xspan, heat2d.yspan, Xflat[end])
    Colorbar(fig0[1, 3], hm1) 
    Colorbar(fig0[2, 3], hm2)
    display(fig0)
    save(joinpath(FILEPATH, "plots/heat2d/heat2d_initial_final.png"), fig0)
end

#========================#
## Animate Data to Check
#========================#
X2d = LnL.invec.(eachcol(X), heat2d.spatial_dim...)
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(1300,500))
    ax1 = Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="u(x,y,t)")
    ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", aspect=DataAspect())
    colgap!(fig.layout, 0)
    sf = surface!(ax1, heat2d.xspan, heat2d.yspan, X2d[1])
    hm = heatmap!(ax2, heat2d.xspan, heat2d.yspan, X2d[1])
    Colorbar(fig[1, 3], hm)
    record(fig, "scripts/streaming/plots/heat2d/temperature.mp4", 1:heat2d.time_dim) do i
        sf[3] = X2d[i]
        hm[3] = X2d[i]
        autolimits!(ax1) # update limits
        autolimits!(ax2) # update limits
    end
end
##
X2d = nothing
GC.gc()

#===========#
## Intrusive
#===========#
op_int = LnL.pod(op_heat, Vr, options.system)

#====================#
## Operator Inference
#====================#
# Obtain derivative data
Xdot = (X[:, 2:end] - X[:, 1:end-1]) / heat2d.Δt
idx = 2:heat2d.time_dim
X = X[:, idx]  
U = U[:, idx]
Y = Y[:, idx] 
op_inf = LnL.opinf(X, Vr, options; U=U, Y=Y, Xdot=Xdot)

#============================#
## Tikhonov Regularized OpInf
#============================#
options.with_reg = true
options.λ = LnL.TikhonovParameter(
    A = 1e-6,
    B = 1e-6,
    C = 1e-3
)
op_inf_reg = LnL.opinf(X, Vr, options; U=U, Y=Y, Xdot=Xdot)

#=================#
## Streaming-OpInf
#=================#
# Streamify the data based on the selected streamsizes
streamsize = 1
X_stream = LnL.streamify(Vr' * X, streamsize)
U_stream = LnL.streamify(U, streamsize)
Y_stream = LnL.streamify(Y, streamsize)
R_stream = LnL.streamify(Vr' * Xdot, streamsize)
num_of_streams = length(X_stream)

# Initialize the stream
# γs = 0.0
# γo = 0.0
γs = 1e-9
γo = 1e-8
algo = :iQRRLS
state_stream, output_stream = LnL.StreamingOpInf(options=options, n=r, m=size(U,1), l=size(Y,1); γs=γs, γo=γo, algorithm=algo)

# Stream all at once
LnL.stream_all!(state_stream, X_stream, R_stream; U=U_stream)
LnL.stream_output_all!(output_stream, X_stream, Y_stream)

# Unpack solution operators
op_stream = LnL.terminate_stream(state_stream, output_stream)


###############################
## (Analysis 1) Relative Error 
###############################
# # Collect all operators into a dictionary
# op_dict = Dict(
#     "POD" => op_int,
#     "OpInf" => op_inf,
#     "TR-OpInf" => op_inf_reg,
#     "iQR-Streaming-OpInf" => op_stream
#     # "Streaming-OpInf" => op_stream
# )
# rse, roe = analysis_1(op_dict, heat2d, Vr, Xfull, Ufull, Yfull, [:A, :B], heat2d.integrate_model)

# ## Plot
# fig1 = plot_rse(rse, roe, r, ace_light; provided_keys=["POD", "OpInf", "TR-OpInf", "iQR-Streaming-OpInf"])
# display(fig1)

# Collect all operators into a dictionary
op_dict = Dict(
    "POD" => op_int,
    "OpInf" => op_inf,
    "TR-OpInf" => op_inf_reg,
    "Streaming-OpInf" => op_stream
)

r = size(Vr,2)
rse = Dict{String, Vector{Float64}}()
roe = Dict{String, Vector{Float64}}()
for (key, op) in op_dict
    rse[key] = Vector{Float64}[]
    roe[key] = Vector{Float64}[]
    for i = 1:r
        Vri = Vr[:, 1:i]

        # Integrate the system for reconstruction
        Xtmp = heat2d.integrate_model(
            heat2d.tspan, Vri' * heat2d.IC, Ufull; 
            operators=[A[1:i,1:i],B[1:i,:]], system_input=true, integrator_type=:BackwardEuler
        )

        foo = LnL.rel_state_error(Xfull, Xtmp, Vri)
        Y = op.C[1:end, 1:i] * Xtmp
        bar = LnL.rel_output_error(Yfull, Y)
        push!(rse[key], foo)
        push!(roe[key], bar)
        @info "($key) r = $i, State Error = $foo, Output Error = $bar"
    end
end

## Plot
provided_keys = ["POD", "OpInf", "TR-OpInf", "Streaming-OpInf"]
with_theme(theme_latexfonts()) do
    fig1 = Figure(fontsize=20, size=(1200,600))
    # Relative State Error
    ax1 = Axis(fig1[1, 1], 
        xlabel=L"reduced dimension, $r$",
        ylabel="Relative State Error", 
        title="Relative State Error", 
        yscale=log10,
        xlabelsize=30,
        ylabelsize=30,
        xticklabelsize=25,
        yticklabelsize=25
    )
    for key in provided_keys
        scatterlines!(ax1, 1:r, rse[key])
    end
    # Relative Output Error
    lines = []
    labels = []
    ax2 = Axis(fig1[1, 2], 
        xlabel=L"reduced dimensions, $r$", 
        ylabel="Relative Output Error", 
        title="Relative Output Error", 
        yscale=log10,
        xlabelsize=30,
        ylabelsize=30,
        xticklabelsize=25,
        yticklabelsize=25
    )
    for key in provided_keys
        l = scatterlines!(ax2, 1:r, roe[key], label=key)
        push!(lines, l)
        push!(labels, key)
    end
    Legend(fig1[2, 1:2], 
        lines, labels,
        orientation=:horizontal, 
        halign=:center, 
        tellwidth=false, 
        tellheight=true,
        labelsize=28
    )
    display(fig1)
end


##################################################
## (Analysis 2) Per stream quantities of interest
##################################################
r_select = 1:r
analysis_results = analysis_2(
    Xhat_stream, U_stream, Y_stream, R_stream, num_of_streams, 
    op_inf_reg, Xfull, Vr, Ufull, Yfull, heat2d, r_select, options, 
    [:A, :B], LnL.backwardEuler; VR=false, α=γs, β=γo, algo=algo
)

## Plot
fig2 = plot_rse_per_stream(analysis_results["rse_stream"], analysis_results["roe_stream"], 
                           analysis_results["streaming_error"], analysis_results["streaming_error_output"], 
                           [5,10,15], num_of_streams; ylimits=([1e-7,1.3e1], [1e-9,1e1]))
display(fig2)
##
fig3 = plot_errorfactor_condition(analysis_results["cond_state_EF"], analysis_results["cond_output_EF"], 
                                  r_select, num_of_streams, ace_light)
display(fig3)
##
fig4 = plot_streaming_error(analysis_results["streaming_error"], analysis_results["streaming_error_output"], 
                            analysis_results["true_streaming_error"], analysis_results["true_streaming_error_output"],
                            r_select, num_of_streams, ace_light)
display(fig4)


##############################################
## (Analysis 3) Initial error over streamsize
##############################################
streamsizes = 1:num_of_streams
init_rse, init_roe = analysis_3(streamsizes, Vr, X, U, Y, Vr' * Xdot, op_inf_reg, 1:15, options; 
                                tol=nothing, α=γs, β=γo, algo=algo)

## Plot
fig5 = plot_initial_error(streamsizes, init_rse, init_roe, ace_light, 1:15)
display(fig5)

