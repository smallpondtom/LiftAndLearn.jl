"""
    Streaming-OpInf example of the Burgers equation.
"""

#############
## Packages
#############
using CairoMakie
using LinearAlgebra
using ProgressMeter
using Random: rand


###############
## My modules
###############
using LiftAndLearn
const LnL = LiftAndLearn


###################
## Global Settings
###################
CONST_STREAM = true


###############################
## Include functions and files
###############################
include("utilities/plot_theme.jl")
include("utilities/analysis.jl")
include("utilities/plotting.jl")


##########################
## Burgers equation setup
##########################
burgers = LnL.burgers(
    [0.0, 1.0], [0.0, 1.0], [0.5, 0.5],
    2^(-7), 1e-4, 1, "dirichlet"
)
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        is_lin=true,
        is_quad=true,
        has_control=true,
        has_output=true,
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=1e-4,
        deriv_type="SI"
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)
num_of_inputs = 3
rmax = 15


##########################
## Generate training data
##########################
μ = burgers.μs[1]
A, B, F = burgers.generateABFmatrix(burgers, μ)
C = ones(1, burgers.Xdim) / burgers.Xdim
op_burgers = LnL.operators(A=A, B=B, C=C, F=F)

# Reference solution
Uref = ones(burgers.Tdim - 1, 1)  # Reference input/boundary condition
Xref = burgers.semiImplicitEuler(A, B, F, Uref, burgers.t, burgers.IC)
Yref = C * Xref

Urand = rand(burgers.Tdim - 1, num_of_inputs)  # uniformly random input
Xall = Vector{Matrix{Float64}}(undef, num_of_inputs)
Xdotall = Vector{Matrix{Float64}}(undef, num_of_inputs)
Xstore = nothing  # store one data trajectory for plotting
for j in 1:num_of_inputs
    @info "Generating data for input $j"
    states = burgers.semiImplicitEuler(A, B, F, Urand[:, j], burgers.t, burgers.IC)
    Xall[j] = states[:, 2:end]
    Xdotall[j] = (states[:, 2:end] - states[:, 1:end-1]) / burgers.Δt
    if j == 1
        Xstore = states
    end
end
X = reduce(hcat, Xall)
Xdot = reduce(hcat, Xdotall)
U = reshape(Urand, (burgers.Tdim - 1) * num_of_inputs, 1)
Y = C * X

# compute the POD basis from the training data
Vrmax = svd(X).U[:, 1:rmax]


######################
## Plot Data to Check
######################
with_theme(theme_latexfonts()) do
    fig0 = Figure(fontsize=20, size=(1300,500), backgroundcolor="#FFFFFF")
    ax1 = Axis3(fig0[1, 1], xlabel="x", ylabel="t", zlabel="u(x,t)")
    surface!(ax1, burgers.x, burgers.t, Xstore)
    ax2 = Axis(fig0[1, 2], xlabel="x", ylabel="t")
    hm = heatmap!(ax2, burgers.x, burgers.t, Xstore)
    Colorbar(fig0[1, 3], hm)
    display(fig0)
end


#######################
## Intrusive-POD model
#######################
op_int = LnL.pod(op_burgers, Vrmax, options)


###############
## OpInf model
###############
op_inf = LnL.opinf(X, Vrmax, options; U=U, Y=Y, Xdot=Xdot)


##############################
## Tikhonov Regularized OpInf
##############################
options.with_reg = true
options.λ = LnL.TikhonovParameter(
    lin = 1e-7,
    ctrl = 1e-7,
    output = 1e-6
)
op_inf_reg = LnL.opinf(X, Vrmax, options; U=U, Y=Y, Xdot=Xdot)


###################
## Streaming-OpInf
###################
# Construct batches of the training data
if CONST_STREAM  # using a single constant batchsize
    global streamsize = 1
else  # initial batch updated with smaller batches
    init_streamsize = 200
    update_size = 1
    global streamsize = vcat([init_streamsize], [update_size for _ in 1:((size(X,2)-init_streamsize)÷update_size)])
end

# Streamify the data based on the selected streamsizes
# INFO: Remember to make data matrices a tall matrix except X matrix
Xhat_stream = LnL.streamify(Vrmax' * X, streamsize)
U_stream = LnL.streamify(U, streamsize)
Y_stream = LnL.streamify(Y', streamsize)
R_stream = LnL.streamify((Vrmax' * Xdot)', streamsize)
num_of_streams = length(Xhat_stream)

# Initialize the stream
γs = 1e-7
γo = 1e-6
stream = LnL.StreamingOpInf(options, rmax, 1, 1; variable_regularize=false, atol=[0.0,0.0], rtol=[0.0,0.0], γs_k=γs, γo_k=γo)
# D_k = stream.init!(stream, Xhat_stream[1], R_stream[1]; U_k=U_stream[1], Y_k=Y_stream[1], α_k=α[1], β_k=β[1])

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
    "Streaming-OpInf" => op_stream
)
rse, roe = analysis_1(op_dict, burgers, Vrmax, Xref, Uref, Yref, [:A, :B, :F], burgers.semiImplicitEuler)

## Plot
fig1 = plot_rse(rse, roe, rmax, ace_light; provided_keys=["POD", "OpInf", "TR-OpInf", "Streaming-OpInf"])
display(fig1)


##################################################
## (Analysis 2) Per stream quantities of interest
##################################################
r_select = 1:15
analysis_results = analysis_2( # Attention: This will take some time to run
    Xhat_stream, U_stream, Y_stream, R_stream, num_of_streams, 
    op_inf_reg, Xref, Vrmax, Uref, Yref, burgers, r_select, options, 
    [:A, :B, :F], burgers.semiImplicitEuler; VR=false, α=α, β=β
)

## Plot
fig2 = plot_rse_per_stream(analysis_results["rse_stream"], analysis_results["roe_stream"], 
                           analysis_results["streaming_error"], analysis_results["streaming_error_output"], 
                           [5,10,15], num_of_streams; ylimits=([1e-6,1e2], [1e-9,1e2]))
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
init_rse, init_roe = analysis_3(streamsizes, Vrmax, X, U, Y, Vrmax' * Xdot, op_inf_reg, r_select, options; 
                                tol=tol, α=α, β=β)

## Plot
fig5 = plot_initial_error(streamsizes, init_rse, init_roe, ace_light, 1:15)
display(fig5)