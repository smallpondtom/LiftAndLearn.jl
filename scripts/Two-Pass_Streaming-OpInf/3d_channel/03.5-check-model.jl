"""
3D Channel flow: Check trained model
"""

#================#
## Load Packages
#================#
using DifferentialEquations
using FileIO
using JLD2
using LinearAlgebra
using ProgressMeter
using Printf
using Sundials
using UniqueKronecker
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
DATAPATH = "../../../../../DATA/NREL/3D_CHANNEL"
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"Two-Pass_Streaming-OpInf/3d_channel") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/3d_channel")
fn = "channel_5200_data_0_10000.h5"
datafile = joinpath(DATAPATH, fn)

#==========================================#
## Load struct to read data in HDF5 format 
#==========================================#
include(joinpath(FILEPATH, "datasource.jl"))

#========================#
## Additional functions ##
#========================#
include(joinpath(FILEPATH, "derivative.jl"))
include(joinpath(FILEPATH, "../utilities/extract_operators.jl"))
include(joinpath(FILEPATH, "../utilities/interpolate.jl"))

#=============================#
## Load the training dataset
#=============================#
ds = ChannelDataSource(datafile, ["z", "y", "x", "fields", "times"])
Nz, Ny, Nx, n_fields, n = ds.dims

#=================#
## Load the bases 
#=================#
basis_file = joinpath(FILEPATH, "data/streaming/basis.jld2")
basis_data = load(basis_file)
rmax = 100
iVrmax = basis_data["baker"].iVr[:,1:rmax];  # choose Baker's iSVD basis

#=========================#
## Load reduced data
#=========================#
Xhat = load(joinpath(FILEPATH, "data/streaming/reduced_data.jld2"))["Xhat"]
U = load(joinpath(FILEPATH, "data/streaming/reduced_data.jld2"))["U"]

#====================================#
## Simulate the learned reduced model
#====================================#
ops = load(joinpath(FILEPATH, "data/models/operators.jld2"))
op = ops["tropinf"]

##
function channel3d(dx, x, p, t)
    A = p[1]
    A2u = p[2]
    B = p[3]
    u = p[4]
    dx[:] = A * x + A2u * (x ⊘ x) + B * u
end

##
Tend = ds["times"][length(ds)] - ds["times"][1]
params = (op.A, op.A2u, op.B, U[1])
prob_3dchannel = ODEProblem(
    channel3d, iVrmax' * ds[1], (0.0, Tend), params
)

##
sol = solve(prob_3dchannel, CVODE_BDF(linear_solver = :GMRES));

## Integrate the model
Tend = ds["times"][length(ds)] - ds["times"][1]
tspan_rk4 = 0:0.001:Tend
Xrecon = zeros(rmax, length(tspan_rk4))
Xrecon[:,1] = iVrmax' * ds[1]
for k in 1:length(tspan_rk4)-1
    dt = tspan_rk4[k+1] - tspan_rk4[k]
    Xrecon[:,k+1] = rk4_step(Xrecon[:,k], U[1], dt, op.A, op.A2u, op.B)
    x1 = Xrecon[1,k+1]
    x_end = Xrecon[end,k+1]
    if isnan(x1) || isnan(x_end)
        println("Step $k of $(length(tspan_rk4)-1): x1 = $(x1), x_end = $(x_end)")
        println("NaN encountered in reduced state at step $k")
        break
    else
        println("Step $k of $(length(tspan_rk4)-1): x1 = $(x1), x_end = $(x_end)")
    end
end

## Compute relative state error (averaged over parameters)
Xhat_interp = cubic_interpolate_matrix(Xhat, ds["times"][:], tspan_rk4)
reduced_error += norm(Xhat_interp - Xrecon) / norm(Xhat_interp)

## Save the reconstructed reduced states 
save(joinpath(FILEPATH, "data/check/recon_states.jld2"), "Xrecon", Xrecon)