"""
viscous Burgers' equation iOpInf example
"""

#################
## Load Packages
#################
using CairoMakie
using IncrementalPOD
using LinearAlgebra
using LiftAndLearn
const LnL = LiftAndLearn


###################
## Global Settings
###################
TOL = 1e-10
ALGO = :baker
SAVEFIG = true


###############################
## Include functions and files
###############################
include("utilities/plot_theme.jl")
include("utilities/analysis.jl")
include("utilities/plotting.jl")


##########################
## Burgers equation setup
##########################
Ω = (0.0, 1.0)
Nx = 2^7; dt = 1e-4
burgers = LnL.BurgersModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    diffusion_coeffs=0.5, BC=:dirichlet,
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
        Δt=dt,
        deriv_type="SI"
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)
num_of_inputs = 3
rmax = 15


#################
## Generate Data
#################
μ = burgers.diffusion_coeffs
A, B, F = burgers.finite_diff_model(burgers, μ)
C = ones(1, burgers.spatial_dim) / burgers.spatial_dim
op_burgers = LnL.Operators(A=A, B=B, C=C, F=F)

# Reference solution
Uref = ones(burgers.time_dim - 1, 1)  # Reference input/boundary condition
Xref = burgers.integrate_model(A, B, F, Uref, burgers.tspan, burgers.IC)
Yref = C * Xref

Urand = rand(burgers.time_dim - 1, num_of_inputs)  # uniformly random input
Xall = Vector{Matrix{Float64}}(undef, num_of_inputs)
Xdotall = Vector{Matrix{Float64}}(undef, num_of_inputs)
Xstore = nothing  # store one data trajectory for plotting
for j in 1:num_of_inputs
    @info "Generating data for input $j"
    states = burgers.integrate_model(A, B, F, Urand[:, j], burgers.tspan, burgers.IC)
    Xall[j] = states[:, 2:end]
    Xdotall[j] = (states[:, 2:end] - states[:, 1:end-1]) / burgers.Δt
    if j == 1
        Xstore = states
    end
end
X = reduce(hcat, Xall)
Xdot = reduce(hcat, Xdotall)
U = reshape(Urand, (burgers.time_dim - 1) * num_of_inputs, 1)
Y = C * X



####################################
## Compute the SVD for the POD basis
####################################
r = rmax  # order of the reduced form
V, Σ, _ = svd(X)
Vr = V[:, 1:r]
Σr = Σ[1:r]


#########################
## Compute the iPOD basis
#########################
if ALGO == :baker
    ipod = iPOD(x1=X[:,1], algo=ALGO, kselect=r)
    ipod.full_increment!(ipod, X[:,2:end], tol=1e-10)
    iVr = ipod.Q[:,1:r]
    iΣ = ipod.Σ
    iΣr = sort(iΣ, rev=true)[1:r]
elseif ALGO == :brand1
    ipod = iPOD(x1=X[:,1], algo=ALGO, reorth_method=:qr)
    ipod.full_increment!(ipod, X[:,2:end], tol=1e-11)
    iVr = ipod.Q[:,1:r]
    iΣr = ipod.Σ[1:r]
elseif ALGO == :brand2
    ipod = iPOD(x1=X[:,1], algo=ALGO, reorth_method=:qr)
    ipod.full_increment!(ipod, X[:,2:end], tol=1e-10)
    iVr = (ipod.Q * ipod.Qt)[:,1:r]
    iΣr = ipod.Σ[1:r]
elseif ALGO == :zhang1
    ipod = iPOD(x1=X[:,1], algo=ALGO)
    ipod.full_increment!(ipod, X[:,2:end], tol=1e-10)
    iVr = ipod.Q[:,1:r]
    iΣr = ipod.Σ[1:r]
elseif ALGO == :zhang2
    ipod = iPOD(x1=X[:,1], algo=ALGO)
    ipod.full_increment!(ipod, X[:,2:end], tol=1e-10)
    iVr = ipod.Q[:,1:r]
    iΣr = ipod.Σ[1:r]
else
    error("Invalid algorithm. Available options are :brand1, :brand2, :zhang1, :zhang2, and :baker.")
end

# Copy the data for later analysis
Xfull = copy(X)
Yfull = copy(Y)
Ufull = copy(U)

# Match the signs 
for i in 1:r
    if abs(Vr[1,i] + iVr[1,i]) < TOL
        iVr[:,i] .*= -1
    end
end

# Error
error = norm(Vr - iVr) / norm(Vr)


######################
## Plot Data to Check
######################
with_theme(theme_latexfonts()) do
    fig0 = Figure(fontsize=20, size=(1300,500), backgroundcolor="#FFFFFF")
    ax1 = Axis3(fig0[1, 1], xlabel="x", ylabel="t", zlabel="u(x,t)")
    surface!(ax1, burgers.xspan, burgers.tspan, Xstore)
    ax2 = Axis(fig0[1, 2], xlabel="x", ylabel="t")
    hm = heatmap!(ax2, burgers.xspan, burgers.tspan, Xstore)
    Colorbar(fig0[1, 3], hm)
    display(fig0)
end

########################
## Plot Singular Values
########################
fig0 = Figure()
ax = Axis(fig0[1,1], title="Singular Values", xlabel="Index", ylabel="Value")
scatterlines!(ax, 1:r, Σr, color=:black, linewidth=3, label="SVD")
scatterlines!(ax, 1:r, iΣr, color=:red, linewidth=2, linestyle=:dash, label="iSVD")
axislegend(ax, labelsize=20, position=:rt)
display(fig0)

##
fig0 = Figure()
ax = Axis(fig0[1,1], title="Singular Values", xlabel="Index", ylabel="Value", yscale=log10)
scatterlines!(ax, 1:r, Σr, color=:black, linewidth=3, label="SVD")
scatterlines!(ax, 1:r, iΣr, color=:red, linewidth=2, linestyle=:dash, label="iSVD")
axislegend(ax, labelsize=20, position=:rt)
display(fig0)


#######################
## Intrusive-POD model
#######################
op = LnL.pod(op_burgers, Vr, options)


###############
## OpInf model
###############
op_inf = LnL.opinf(X, Vr, options; U=U, Y=Y, Xdot=Xdot)


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
op_inf_reg = LnL.opinf(X, Vr, options; U=U, Y=Y, Xdot=Xdot)


################
## Naive iOpInf
################
# Streamify the data based on the selected streamsizes
streamsize = 1
Xhat_stream = LnL.streamify(iVr' * X, streamsize)
U_stream = LnL.streamify(U, streamsize)
Y_stream = LnL.streamify(Y', streamsize)
R_stream = LnL.streamify((iVr' * Xdot)', streamsize)
num_of_streams = length(Xhat_stream)

# Initialize the stream
# TR-Streaming-OpInf
# γs = 1e-7
# γo = 1e-6
# iQR/QR-Streaming-OpInf
γs = 1e-13
γo = 1e-10
algo=:iQRRLS
stream = LnL.StreamingOpInf(options, rmax, 1, 1; variable_regularize=false, γs_k=γs, γo_k=γo, algorithm=algo)

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
    "POD" => op,
    "OpInf" => op_inf,
    "TR-OpInf" => op_inf_reg,
)
op_ipod_dict = Dict(
    "Naive-iOpInf" => op_stream
)
rse, roe = analysis_1(op_dict, burgers, Vr, Xref, Uref, Yref, [:A, :B, :F], burgers.integrate_model)
rse_ipod, roe_ipod = analysis_1(op_ipod_dict, burgers, iVr, Xref, Uref, Yref, [:A, :B, :F], burgers.integrate_model)
rse = merge(rse, rse_ipod)
roe = merge(roe, roe_ipod)

## Plot
fig1 = plot_rse(rse, roe, r, ace_light; provided_keys=["POD", "OpInf", "TR-OpInf", "Naive-iOpInf"])
display(fig1)
