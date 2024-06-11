"""
    Streaming-OpInf example of the Kuramoto-Sivashinsky equation.
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


######################
## KSE equation setup
######################
kse = LnL.ks(
    [0.0, 60.0], [0.0, 300.0], [1.0, 1.0],
    512, 0.001, 1, "nc"
)

options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        is_lin=true,
        is_quad=true,
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

# Parameterized function for the initial condition
L = kse.Omega[2] - kse.Omega[1]  # length of the domain
u0 = (a,b) -> a * cos.((2*π*kse.x)/L) .+ b * cos.((4*π*kse.x)/L)  # initial condition


##########################
## Generate training data
##########################
A, F = kse.model_FD(kse, kse.μs[1])
C = ones(1, kse.Xdim) / kse.Xdim
op_kse = LnL.operators(A=A, C=C, F=F)
a = 0.8; b = 1.2

# Training data (which is also Reference solution)
X = kse.integrate_FD(A, F, kse.t, u0(a,b))
Y = C * X

# compute the POD basis from the training data
rmax = 150
Vrmax = svd(X).U[:, 1:rmax]
Σr = svd(X).S[1:rmax]

# Save the reference solution
Xref = copy(X)
Yref = copy(Y)


######################
## Plot Data to Check
######################
with_theme(theme_latexfonts()) do
    DS = 100
    fig0 = Figure(fontsize=20, size=(1300,500), backgroundcolor="#FFFFFF")
    ax1 = Axis3(fig0[1, 1], xlabel="x", ylabel="t", zlabel="u(x,t)")
    surface!(ax1, kse.x, kse.t[1:DS:end], X[:,1:DS:end])
    ax2 = Axis(fig0[1, 2], xlabel="x", ylabel="t")
    hm = heatmap!(ax2, kse.x, kse.t[1:DS:end], X[:,1:DS:end])
    Colorbar(fig0[1, 3], hm)
    display(fig0)
end


################################################
## Check the energy levels from singular values
################################################
nice_orders, energy_level = LnL.choose_ro(Σr; en_low=-12)
with_theme(theme_latexfonts()) do
    fig0 = Figure()
    ax1 = Axis(fig0[1, 1], 
        xlabel="number of retained modes", 
        ylabel="relative energy loss from truncation",
        yscale=log10
    )
    scatterlines!(ax1, nice_orders, energy_level[nice_orders], label="energy loss", lw=2)
    display(fig0)
end


##############################
## Assign selected basis size
##############################
r_select = nice_orders[1:8]
Vr = Vrmax[:, 1:maximum(r_select)]


#######################
## Intrusive-POD model
#######################
op_int = LnL.pod(op_kse, Vr, options)


###############
## OpInf model
###############
# Choose the correct indices corresponding to the derivative data
Xdot = (X[:, 2:end] - X[:, 1:end-1]) / kse.Δt
X = X[:, 2:end]
Y = Y[:, 2:end]
op_inf = LnL.opinf(X, Vr, options; Y=Y, Xdot=Xdot)


##############################
## Tikhonov Regularized OpInf
##############################
options.with_reg = true
options.λ = LnL.TikhonovParameter(
    lin = 1e-7,
    quad = 1e-7,
    ctrl = 1e-7,
    output = 1e-6
)
op_inf_reg = LnL.opinf(X, Vr, options; Y=Y, Xdot=Xdot)


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
Xhat_stream = LnL.streamify(Vr' * X, streamsize)
Y_stream = LnL.streamify(Y', streamsize)
R_stream = LnL.streamify((Vr' * Xdot)', streamsize)
num_of_streams = length(Xhat_stream)

# Initialize the stream
γs = 1e-7
γo = 1e-6
algo=:RLS
stream = LnL.StreamingOpInf(options, rmax, 0, size(Y,1); variable_regularize=false, γs_k=γs, γo_k=γo, algorithm=algo)

# Stream all at once
stream.stream!(stream, Xhat_stream, R_stream)
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
    # "iQRRLS-Streaming-OpInf" => op_stream
)
rse, roe = analysis_1(op_dict, kse, Vr, Xref, [], Yref, [:A, :F], kse.integrate_FD)

## Plot
# fig1 = plot_rse(rse, roe, rmax, ace_light; provided_keys=["POD", "OpInf", "TR-OpInf", "iQRRLS-Streaming-OpInf"])
fig1 = plot_rse(rse, roe, rmax, ace_light; provided_keys=["POD", "OpInf", "TR-OpInf"])
display(fig1)