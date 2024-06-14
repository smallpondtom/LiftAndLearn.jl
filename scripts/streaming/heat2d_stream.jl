"""
    Streaming-OpInf example of the 2D heat equation.
"""

#############
## Packages
#############
using CairoMakie
using LinearAlgebra
using ProgressMeter


###############
## My modules
###############
using LiftAndLearn
const LnL = LiftAndLearn


###################
## Global Settings
##################
CONST_STREAM = true
SAVEFIG = true


###############################
## Include functions and files
###############################
include("utilities/plot_theme.jl")
include("utilities/analysis.jl")
include("utilities/plotting.jl")


##########################
## 1D Heat equation setup
##########################
Ω = ((0.0, 1.0), (0.0, 1.0))
Nx = 2^6
Ny = 2^6
heat2d = LnL.Heat2DModel(
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
        is_lin=true,
        has_control=true,
        has_output=true,
    ),
    vars=LnL.VariableStructure(
        N=1,  # number of state variables
    ),
    data=LnL.DataStructure(
        Δt=1e-3, # time step
        deriv_type="BE"  # backward Euler
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,  # show the optimization process
    ),
)


#################
## Generate Data
#################
# Construct full model
μ = heat2d.diffusion_coeffs
A, B = heat2d.finite_diff_model(heat2d, μ)
C = ones(1, (Int ∘ prod)(heat2d.spatial_dim)) / heat2d.spatial_dim[1] / heat2d.spatial_dim[2]
op_heat = LnL.operators(A=A, B=B, C=C)

# Generate the input data
U = [1.0, 1.0, -1.0, -1.0]
U = repeat(U, 1, heat2d.time_dim)

# Compute the state snapshot data with backward Euler
X = heat2d.integrate_model(A, B, U, heat2d.tspan, heat2d.IC)

# Compute the SVD for the POD basis
r = 15  # order of the reduced form
Vr = svd(X).U[:, 1:r]

# Compute the output of the system
Y = C * X

# Copy the data for later analysis
Xfull = copy(X)
Yfull = copy(Y)
Ufull = copy(U)


######################
## Plot Data to Check
######################
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


#############
## Intrusive
#############
op_int = LnL.pod(op_heat, Vr, options)


######################
## Operator Inference
######################
# Obtain derivative data
Xdot = (X[:, 2:end] - X[:, 1:end-1]) / heat2d.Δt
idx = 2:heat2d.time_dim
X = X[:, idx]  
U = U[:, idx]
Y = Y[:, idx] 
op_inf = LnL.opinf(X, Vr, options; U=U, Y=Y, Xdot=Xdot)


##############################
## Tikhonov Regularized OpInf
##############################
options.with_reg = true
options.λ = LnL.TikhonovParameter(
    lin = 1e-6,
    ctrl = 1e-6,
    output = 1e-3
)
op_inf_reg = LnL.opinf(X, Vr, options; U=U, Y=Y, Xdot=Xdot)


###################
## Streaming-OpInf
###################
# Construct batches of the training data
if CONST_STREAM  # using a single constant batchsize
    global streamsize = 1
else  # initial batch updated with smaller batches
    init_streamsize = 1
    update_size = 1
    global streamsize = vcat([init_streamsize], [update_size for _ in 1:((size(X,2)-init_streamsize)÷update_size)])
end

# Streamify the data based on the selected streamsizes
# INFO: Remember to make data matrices a tall matrix except X matrix
Xhat_stream = LnL.streamify(Vr' * X, streamsize)
U_stream = LnL.streamify(U', streamsize)
Y_stream = LnL.streamify(Y', streamsize)
R_stream = LnL.streamify((Vr' * Xdot)', streamsize)
num_of_streams = length(Xhat_stream)

# Initialize the stream
# γs = 0.0
# γo = 0.0
γs = 1e-9
γo = 1e-8
algo = :iQRRLS
stream = LnL.StreamingOpInf(options, r, size(U,1), size(Y,1); γs_k=γs, γo_k=γo, algorithm=algo)

# Stream all at once
stream.stream!(stream, Xhat_stream, R_stream; U_k=U_stream)
stream.stream_output!(stream, Xhat_stream, Y_stream)

# Unpack solution operators
op_stream = stream.unpack_operators(stream)


###############################
## (Analysis 1) Relative Error 
###############################
# Collect all operators into a dictionary
op_dict = Dict(
    "POD" => op_int,
    "OpInf" => op_inf,
    "TR-OpInf" => op_inf_reg,
    "iQR-Streaming-OpInf" => op_stream
    # "Streaming-OpInf" => op_stream
)
rse, roe = analysis_1(op_dict, heat2d, Vr, Xfull, Ufull, Yfull, [:A, :B], heat2d.integrate_model)

## Plot
fig1 = plot_rse(rse, roe, r, ace_light; provided_keys=["POD", "OpInf", "TR-OpInf", "iQR-Streaming-OpInf"])
display(fig1)


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

