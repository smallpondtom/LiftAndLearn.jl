"""
2D heat equation iOpInf example
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
SAVEFIG = true


###############################
## Include functions and files
###############################
include("utilities/plot_theme.jl")
include("utilities/analysis.jl")
include("utilities/plotting.jl")


##########################
## 2D Heat equation setup
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
op_heat = LnL.Operators(A=A, B=B, C=C)

# Generate the input data
U = [1.0, 1.0, -1.0, -1.0]
U = repeat(U, 1, heat2d.time_dim)

# Compute the state snapshot data with backward Euler
X = heat2d.integrate_model(A, B, U, heat2d.tspan, heat2d.IC)

# Compute the output of the system
Y = C * X

####################################
## Compute the SVD for the POD basis
####################################
r = 15  # order of the reduced form
V, Σ, _ = svd(X)
Vr = V[:, 1:r]
Σr = Σ[1:r]


#########################
## Compute the iPOD basis
#########################
ipod = iPOD(x1=X[:,1], algo=:baker, kselect=r)
ipod.full_increment!(ipod, X[:,2:end], tol=1e-9, verbose=true)
iVr = ipod.Q[:,1:r]
iΣ = ipod.Σ
iΣr = sort(iΣ, rev=true)[1:r]

# Match the signs 
for i in 1:r
    if abs(Vr[1,i] + iVr[1,i]) < TOL
        iVr[:,i] .*= -1
    end
end

# Error
error = norm(Vr - iVr) / norm(Vr)


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


#########################
## Plot Singular Vectors
#########################
fig0 = Figure()
ax = Axis(fig0[1,1], yreversed=true)
ax.title = "Difference between POD and iPOD basis"
hm = heatmap!(ax, (Vr - iVr)')
Colorbar(fig0[1,2], hm)
display(fig0)


################
## POD-Galerkin
################
op = LnL.pod(op_heat, Vr, options)


#########
## OpInf
#########
# Obtain derivative data
op_inf = LnL.opinf(X, Vr, options; U=U, Y=Y)


##############################
## Tikhonov Regularized OpInf
##############################
options.with_reg = true
options.λ = LnL.TikhonovParameter(
    lin = 1e-6,
    ctrl = 1e-6,
    output = 1e-3
)
op_inf_reg = LnL.opinf(X, Vr, options; U=U, Y=Y)


################
## Naive iOpInf
################
# Save data 
Xfull = copy(X)
Yfull = copy(Y)
Ufull = copy(U)

# Obtain derivative data and adjust data
Xdot = (X[:, 2:end] - X[:, 1:end-1]) / heat2d.Δt
idx = 2:heat2d.time_dim
X = X[:, idx]  
U = U[:, idx]
Y = Y[:, idx] 

# Streamify the data based on the selected streamsizes
streamsize = 1
Xhat_stream = LnL.streamify(Vr' * X, streamsize)
U_stream = LnL.streamify(U, streamsize)
Y_stream = LnL.streamify(Y, streamsize)
Xhatdot_stream = LnL.streamify(Vr' * Xdot, streamsize)
num_of_streams = length(Xhat_stream)

# Initialize the stream
# TR-Streaming-OpInf
# γs = 0.0
# γo = 0.0
# iQR/QR-Streaming-OpInf
γs = 1e-9
γo = 1e-8
stream = LnL.StreamingOpInf(options, r, size(U,1), size(Y,1); γs_k=γs, γo_k=γo, algorithm=:QRRLS)

# Stream all at once
stream.stream!(stream, Xhat_stream, Xhatdot_stream; U_k=U_stream)
stream.stream_output!(stream, Xhat_stream, Y_stream)

# Unpack solution operators
op_inc = stream.unpack_operators(stream)



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
    "iOpInf" => op_inc
)
rse, roe = analysis_1(op_dict, heat2d, Vr, Xfull, Ufull, Yfull, [:A, :B], LnL.backwardEuler)
rse_ipod, roe_ipod = analysis_1(op_ipod_dict, heat2d, iVr, Xfull, Ufull, Yfull, [:A, :B], LnL.backwardEuler)
rse = merge(rse, rse_ipod)
roe = merge(roe, roe_ipod)

## Plot
fig1 = plot_rse(rse, roe, r, ace_light; provided_keys=["POD", "OpInf", "TR-OpInf", "iOpInf"])
display(fig1)
