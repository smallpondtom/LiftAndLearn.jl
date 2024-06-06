"""
    Streaming-OpInf example of the Fisher-KPP equation.
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
CONST_STREAM = true
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
Ω = (0.0, 3.0)
fisherkpp = LnL.fisherkpp(
    spatial_domain=Ω, time_domain=(0,2), Δx=(Ω[2] + 2^(-6))*2^(-6), Δt=1e-4, 
    diffusion_coeffs=0.2, growth_rates=0.3, BC=:mixed
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
        Δt=1e-3,
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)
num_of_inputs = 5
rmax = 15


##########################
## Generate training data
##########################
A, B, F = fisherkpp.finite_diff_model(fisherkpp, fisherkpp.diffusion_coeffs, fisherkpp.growth_rates)
C = ones(1, fisherkpp.spatial_dim) / fisherkpp.spatial_dim
C = vcat(C, rand(1, fisherkpp.spatial_dim) .- 0.5)  # add a random row to C
op_fisherkpp = LnL.operators(A=A, B=B, C=C, F=F)
fisherkpp.IC = cos.(0.5π * fisherkpp.xspan)  # change IC
# fisherkpp.IC = zeros(fisherkpp.spatial_dim)  # change IC

# Reference solution
σ1 = 5.0; σ2 = 10.0; μ = 1.0
Uref = zeros(fisherkpp.time_dim - 1, 2)  # boundary condition → control input
g = (t) -> 5*sin.(2π * t)
Uref[:,1] .= μ 
Uref[:,2] = g(fisherkpp.tspan[1:end-1])
Xref = fisherkpp.integrate_model(A, B, F, Uref, fisherkpp.tspan, fisherkpp.IC; const_stepsize=true)
Yref = C * Xref

# Generate training data
Urand = zeros(fisherkpp.time_dim - 1, 2, num_of_inputs)  # boundary condition → control input
for j in 1:num_of_inputs
    Urand[:,1,j] = σ1*randn(fisherkpp.time_dim - 1) .+ μ  # random input for Dirichlet BC
    Urand[:,2,j] = g(fisherkpp.tspan[1:end-1]) .+ σ2*randn(fisherkpp.time_dim-1)  # another input for Neumann BC
end
Xall = Vector{Matrix{Float64}}(undef, num_of_inputs)
Xdotall = Vector{Matrix{Float64}}(undef, num_of_inputs)
Xstore = nothing  # store one data trajectory for plotting
for j in 1:num_of_inputs
    @info "Generating data for input $j"
    states = fisherkpp.integrate_model(A, B, F, Urand[:,:,j], fisherkpp.tspan, fisherkpp.IC; const_stepsize=true)
    Xall[j] = states[:, 2:end]
    Xdotall[j] = (states[:, 2:end] - states[:, 1:end-1]) / fisherkpp.Δt
    if j == 1
        Xstore = states
    end
end
X = reduce(hcat, Xall)
Xdot = reduce(hcat, Xdotall)
U = vcat([Urand[:,:,j] for j in 1:num_of_inputs]...)
Y = C * X

# compute the POD basis from the training data
Vrmax = svd(X).U[:, 1:rmax]


######################
## Plot Data to Check
######################
with_theme(theme_latexfonts()) do
    fig0 = Figure(fontsize=20, size=(1300,500), backgroundcolor="#FFFFFF")
    ax1 = Axis3(fig0[1, 1], xlabel="x", ylabel="t", zlabel="u(x,t)")
    surface!(ax1, fisherkpp.xspan, fisherkpp.tspan, Xstore)
    ax2 = Axis(fig0[1, 2], xlabel="x", ylabel="t")
    hm = heatmap!(ax2, fisherkpp.xspan, fisherkpp.tspan, Xstore)
    Colorbar(fig0[1, 3], hm)
    display(fig0)
end


#######################
## Intrusive-POD model
#######################
op_int = LnL.pod(op_fisherkpp, Vrmax, options)


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
algo=:RLS
stream = LnL.StreamingOpInf(options, rmax, size(U,2), size(Y,1); variable_regularize=false, γs_k=γs, γo_k=γo, algorithm=algo)

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
    "TRRLS-Streaming-OpInf" => op_stream
)
rse, roe = analysis_1(op_dict, fisherkpp, Vrmax, Xref, Uref, Yref, [:A, :B, :F], fisherkpp.integrate_model)

## Plot
fig1 = plot_rse(rse, roe, rmax, ace_light; provided_keys=["POD", "OpInf", "TR-OpInf", "TRRLS-Streaming-OpInf"])
display(fig1)