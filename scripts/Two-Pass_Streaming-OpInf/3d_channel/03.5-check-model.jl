"""
3D Channel flow: Check trained model
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using LinearAlgebra
using ProgressMeter
using Printf
using Random
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
iVrmax = basis_data["baker"].iVr  # choose Baker's iSVD basis
rmax = size(iVrmax,2)

#=========================#
## Load reduced data
#=========================#
Xhat = load(joinpath(FILEPATH, "data/streaming/reduced_data.jld2"))["Xhat"]
U = load(joinpath(FILEPATH, "data/streaming/reduced_data.jld2"))["U"]

#====================================#
## Simulate the learned reduced model
#====================================#
# Error analysis 
rspan = [25, 50, 100]
train_errors = Dict(
    # :pod           => zeros(length(rspan),1),
    :opinf         => zeros(length(rspan),1),
    :tropinf       => zeros(length(rspan),1),
    :stream_rls    => zeros(length(rspan),1),
    :stream_iqrrls => zeros(length(rspan),1),
    :stream_qrrls  => zeros(length(rspan),1)
)

ops = load(joinpath(FILEPATH, "data/streamwise/models/operators.jld2"))
op_rls = ops["stream_rls"]

# Integrate the model
tspan_rk4 = 0:0.001:tspan[end]
Xrecon = zeros(rmax, length(tspan_rk4))
Xrecon[:,1] = iVrmax' * ds[1]
for k in 1:length(tspan_rk4)-1
    dt = tspan_rk4[k+1] - tspan_rk4[k]
    Xrecon[:,k+1] = rk4_step(Xrecon[:,k], U[1], dt, op_rls.A, op_rls.A2u, op_rls.B)
end

# Compute relative state error (averaged over parameters)
Xhat_interp = cubic_interpolate_matrix(Xhat, ds["times"][:], tspan_rk4)
reduced_error += norm(Xhat_interp - Xrecon) / norm(Xhat_interp)

# Save the reconstructed reduced states 
save(joinpath(FILEPATH, "data/check/recon_states.jld2"), "Xrecon", Xrecon)