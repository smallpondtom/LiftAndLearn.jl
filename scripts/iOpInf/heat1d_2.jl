"""
1D heat equation iOpInf example 2
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


#####################################################
## Compute the iPOD basis and learn at the same time
#####################################################
# Initialize ipod object
ipod = iPOD(x1=X[:,1], algo=:baker, kselect=r)

# Obtain derivative data and modify the data a little bit
Xdot = (X[:, 2:end] - X[:, 1:end-1]) / heat1d.Δt
idx = 2:heat1d.time_dim
Xcopy = X[:, idx]  # fix the index of states
Ucopy = U[idx, :]'  # fix the index of inputs
Ycopy = Y[:, idx]  # fix the index of outputs

# Initialize the RLS object
γs = 1e-10
γo = 7.6e-9
stream = LnL.StreamingOpInf(options, r, size(U,2), size(Y,1); γs_k=γs, γo_k=γo, algorithm=:QRRLS)

K = size(Xcopy,2)
reach_r = false
for i in 2:K
    ipod.Q, ipod.Σ, ipod.W = ipod.increment(ipod.Q, ipod.Σ, ipod.W, Xcopy[:,i], ipod.kselect, 1e-10)

    if !reach_r 
        rankQ = size(ipod.Q, 2)
        if rankQ == r
            @info "The POD-basis is rank $r at iteration $i. Start learning..."
            reach_r = true
            for j in 1:i  # learn each column data from the beginning
                stream.stream!(
                    stream, ipod.Q' * Xcopy[:,j], ipod.Q' * Xdot[:,j]; 
                    U_k=Ucopy[1:end, j], γs_k=γs
                )
                stream.stream_output!(stream, ipod.Q' * Xcopy[:,j], Ycopy[:,j]'; γo_k=γo)
            end
        else
            @info "The POD-basis is rank $(rankQ) at iteration $i. Waiting to reach rank $r..."
        end
    else
        @info "Learning at iteration $i..."
        stream.stream!(
            stream, ipod.Q[:,1:r]' * Xcopy[:,i], ipod.Q[:,1:r]' * Xdot[:,i]; 
            U_k=Ucopy[1:end, i], γs_k=γs
        )
        stream.stream_output!(stream, ipod.Q[:,1:r]' * Xcopy[:,i], Ycopy[:,i]'; γo_k=γo)
    end
end
@info "Learning is done"
iVr = ipod.Q[:,1:r]

# Unpack solution operators
op_inc = stream.unpack_operators(stream)


################
## POD-Galerkin
################
op = LnL.pod(op_heat, iVr, options)


#########
## OpInf
#########
op_inf = LnL.opinf(X, iVr, options; U=U, Y=Y)


##############################
## Tikhonov Regularized OpInf
##############################
options.with_reg = true
options.λ = LnL.TikhonovParameter(
    lin = 1e-13,
    ctrl = 1e-13,
    output = 1e-10
)
op_inf_reg = LnL.opinf(X, iVr, options; U=U, Y=Y)


###############################
## (Analysis 1) Relative Error 
###############################
# Collect all operators into a dictionary
op_dict = Dict(
    "POD" => op,
    "OpInf" => op_inf,
    "TR-OpInf" => op_inf_reg,
    "iOpInf" => op_stream
)
rse, roe = analysis_1(op_dict, heat1d, iVr, X, U, Y, [:A, :B], LnL.backwardEuler)

## Plot
fig1 = plot_rse(rse, roe, r, ace_light; provided_keys=["POD", "OpInf", "TR-OpInf", "iOpInf"])
display(fig1)

