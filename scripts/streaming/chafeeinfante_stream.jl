"""
    Streaming-OpInf example of the Chafee-Infante equation.
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
###################
SAVEFIG = true


###############################
## Include functions and files
###############################
include("utilities/plot_theme.jl")
include("utilities/analysis.jl")
include("utilities/plotting.jl")



#############################
## Fisher-KPP equation setup
#############################
Ω = (0.0, 1.0)
Nx = 2^6
dt = 1e-3
chafeeinfante = LnL.ChafeeInfanteModel(
    spatial_domain=Ω, time_domain=(0,2), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt, 
    diffusion_coeffs=1.0, nonlin_coeffs=1.0, BC=:mixed
)
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        is_lin=true,
        is_cubic=true,
        has_control=true,
        has_output=true,
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=dt,
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
        which_cubic_term="E",
    ),
)
num_of_inputs = 5
rmax = 12
DS = 1 # downsampling factor


##########################
## Generate training data
##########################
A, B, E = chafeeinfante.finite_diff_model(chafeeinfante, chafeeinfante.diffusion_coeffs, chafeeinfante.nonlin_coeffs)
C = ones(1, chafeeinfante.spatial_dim) / chafeeinfante.spatial_dim
op_chafeeinfante = LnL.Operators(A=A, B=B, C=C, E=E)
# chafeeinfante.IC = 1 .+ 5*sin.((2*chafeeinfante.xspan.+1)*π).^2
chafeeinfante.IC = zeros(length(chafeeinfante.xspan))  # change IC

# Reference solution
g = (t) -> 10*(sin.(π * t) .+ 1)
Uref = g(chafeeinfante.tspan)  # boundary condition → control input
Uref = hcat(Uref, zeros(chafeeinfante.time_dim))'
Xref = chafeeinfante.integrate_model(A, B, E, Uref, chafeeinfante.tspan, chafeeinfante.IC; const_stepsize=true)
Yref = C * Xref

# Generate training data
Xall = Vector{Matrix{Float64}}(undef, num_of_inputs)
Xdotall = Vector{Matrix{Float64}}(undef, num_of_inputs)
Uall = Vector{Matrix{Float64}}(undef, num_of_inputs)
Xstore = nothing  # store one data trajectory for plotting
for j in 1:num_of_inputs
    @info "Generating data for input $j"
    Urand = hcat(10*rand(chafeeinfante.time_dim), zeros(chafeeinfante.time_dim))'  # boundary condition → control input
    Uall[j] = Urand[:, 2:DS:end]
    states = chafeeinfante.integrate_model(A, B, E, Urand, chafeeinfante.tspan, chafeeinfante.IC; const_stepsize=true)
    Xall[j] = states[:, 2:DS:end]
    tmp = (states[:, 2:end] - states[:, 1:end-1]) / chafeeinfante.Δt
    Xdotall[j] = tmp[:, 1:DS:end]
    if j == 1
        Xstore = states
    end
end
X = reduce(hcat, Xall)
Xdot = reduce(hcat, Xdotall)
U = reduce(hcat, Uall)
Y = C * X

# compute the POD basis from the training data
Vrmax = svd(X).U[:, 1:rmax]


######################
## Plot Data to Check
######################
with_theme(theme_latexfonts()) do
    fig0 = Figure(fontsize=20, size=(1300,500))
    ax1 = Axis3(fig0[1, 1], xlabel="x", ylabel="t", zlabel="u(x,t)")
    surface!(ax1, chafeeinfante.xspan, chafeeinfante.tspan, Xref)
    ax2 = Axis(fig0[1, 2], xlabel="x", ylabel="t")
    hm = heatmap!(ax2, chafeeinfante.xspan, chafeeinfante.tspan, Xref)
    Colorbar(fig0[1, 3], hm)
    display(fig0)
end


#######################
## Intrusive-POD model
#######################
op_int = LnL.pod(op_chafeeinfante, Vrmax, options)


###############
## OpInf model
###############
# op_inf = LnL.opinf(X, Vrmax, options; U=U, Y=Y, Xdot=R)
op_inf, Rt = LnL.opinf(X, Vrmax, op_chafeeinfante, options; U=U, Y=Y, return_derivative=true)  # with reprojection

##############################
## Tikhonov Regularized OpInf
##############################
options.with_reg = true
options.λ = LnL.TikhonovParameter(
    lin = 1e-10,
    quad = 1e-10,
    ctrl = 1e-10,
    output = 1e-8
)
# op_inf_reg = LnL.opinf(X, Vrmax, options; U=U, Y=Y, Xdot=Xdot)
op_inf_reg = LnL.opinf(X, Vrmax, op_chafeeinfante, options; U=U, Y=Y)  # with reprojection


###################
## Streaming-OpInf
###################
# Construct batches of the training data
streamsize = 1

# Streamify the data based on the selected streamsizes
# INFO: Remember to make data matrices a tall matrix except X matrix
Xhat_stream = LnL.streamify(Vrmax' * X, streamsize)
U_stream = LnL.streamify(U', streamsize)
Y_stream = LnL.streamify(Y', streamsize)
# R_stream = LnL.streamify((Vrmax' * Xdot)', streamsize)
R_stream = LnL.streamify(Rt, streamsize)
num_of_streams = length(Xhat_stream)

# Initialize the stream
γs = 1e-5
γo = 1e-2
algo=:RLS
stream = LnL.StreamingOpInf(options, rmax, size(U,1), size(Y,1); variable_regularize=false, γs_k=γs, γo_k=γo, algorithm=algo)

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
    "TR-Streaming-OpInf" => op_stream
)
rse, roe = analysis_1(op_dict, chafeeinfante, Vrmax, Xref, Uref, Yref, [:A, :B, :E], chafeeinfante.integrate_model)

## Plot
fig1 = plot_rse(rse, roe, rmax, ace_light; provided_keys=["POD", "OpInf", "TR-OpInf", "TR-Streaming-OpInf"])
# fig1 = plot_rse(rse, roe, rmax, ace_light; provided_keys=["POD", "OpInf", "TR-OpInf"])
display(fig1)


##################################################
## (Analysis 2) Per stream quantities of interest
##################################################
r_select = 2:rmax
analysis_results = analysis_2( # Attention: This will take some time to run
    Xhat_stream, U_stream, Y_stream, R_stream, num_of_streams, 
    op_inf_reg, Xref, Vrmax, Uref, Yref, fisherkpp, r_select, options, 
    [:A, :B, :F], fisherkpp.integrate_model; VR=false, α=γs, β=γo, algo=algo
)

## Plot
fig2 = plot_rse_per_stream(analysis_results["rse_stream"], analysis_results["roe_stream"], 
                           analysis_results["streaming_error"], analysis_results["streaming_error_output"], 
                           [5,10,15], num_of_streams; ylimits=([1e-5,2e2], [1e-7,1e1]))
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
