"""
1D heat equation iOpInf example
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
## 1D Heat equation setup
##########################
Nx = 2^7; dt = 1e-3
heat1d = LnL.Heat1DModel(  # define the model
    spatial_domain=(0.0, 1.0), time_domain=(0.0, 2.0), diffusion_coeffs=0.1,
    Δx=1/Nx, Δt=1e-3, BC=:dirichlet
)
foo = zeros(heat1d.spatial_dim)
foo[(Nx÷2+1):end] .= 1
heat1d.IC = foo .* (0.5 * sin.(2π * heat1d.xspan))  # change IC
U = ones(heat1d.time_dim)  # boundary condition → control input

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
        Δt=dt, # time step
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
μ = heat1d.diffusion_coeffs
A, B = heat1d.finite_diff_model(heat1d, μ)
C = ones(1, heat1d.spatial_dim) / heat1d.spatial_dim
op_heat = LnL.Operators(A=A, B=B, C=C)

# Compute the state snapshot data with backward Euler
X = LnL.backwardEuler(A, B, U, heat1d.tspan, heat1d.IC)

# Compute the output of the system
Y = C * X


####################################
## Compute the SVD for the POD basis
####################################
r = 12  # order of the reduced form
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
    ipod.full_increment!(ipod, X[:,2:end], tol=1e-12)
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
    surface!(ax1, heat1d.xspan, heat1d.tspan, X)
    ax2 = Axis(fig0[1, 2], xlabel="x", ylabel="t")
    hm = heatmap!(ax2, heat1d.xspan, heat1d.tspan, X)
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


################
## POD-Galerkin
################
op = LnL.pod(op_heat, Vr, options)


#########
## OpInf
#########
# Obtain derivative data
Xdot = (X[:, 2:end] - X[:, 1:end-1]) / heat1d.Δt
idx = 2:heat1d.time_dim
X = X[:, idx]  # fix the index of states
U = U[idx, :]  # fix the index of inputs
Y = Y[:, idx]  # fix the index of outputs
op_inf = LnL.opinf(X, Vr, options; U=U, Y=Y, Xdot=Xdot)


##############################
## Tikhonov Regularized OpInf
##############################
options.with_reg = true
options.λ = LnL.TikhonovParameter(
    lin = 1e-13,
    ctrl = 1e-13,
    output = 1e-10
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
# γs = 1e-10
# γo = 7.6e-9
# iQR/QR-Streaming-OpInf
γs = 1e-13
γo = 1e-10
rlsalgo = :QRRLS
stream = LnL.StreamingOpInf(options, r, size(U,2), size(Y,1); γs_k=γs, γo_k=γo, algorithm=rlsalgo)

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
rse, roe = analysis_1(op_dict, heat1d, Vr, Xfull, Ufull, Yfull, [:A, :B], LnL.backwardEuler)
rse_ipod, roe_ipod = analysis_1(op_ipod_dict, heat1d, iVr, Xfull, Ufull, Yfull, [:A, :B], LnL.backwardEuler)
rse = merge(rse, rse_ipod)
roe = merge(roe, roe_ipod)

## Plot
fig1 = plot_rse(rse, roe, r, ace_light; provided_keys=["POD", "OpInf", "TR-OpInf", "Naive-iOpInf"])
display(fig1)
