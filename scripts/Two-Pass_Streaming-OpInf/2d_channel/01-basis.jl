"""
2D Channel (wall-normal) flow: Compute basis
"""

#================#
## Load Packages
#================#
using CairoMakie
using FileIO
using HDF5
using JLD2
using IncrementalSVD
using LinearAlgebra
using ProgressMeter
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
DATAPATH = "../../../../../DATA/NREL/2D_CHANNEL"
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"Two-Pass_Streaming-OpInf/2d_channel") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/2d_channel")

#=============================#
## Load the training dataset
#=============================#
datafile = joinpath(DATAPATH, "2d_channel.h5")
X = h5read(datafile, "V")  # wall-normal StreamVelocity
xspan = h5read(datafile, "x") 
yspan = h5read(datafile, "y")
n, Ny, Nx = size(X)

#====================#
## Preprocess data  ##
#====================#
Xfold = zeros(Nx*Ny, n)
for i in 1:n
    Xfold[:,i] = reshape(X[i,:,:], :, 1)
end
X = Xfold

#=========================================================#
## Generate the POD basis using iSVD using all algorithms
#=========================================================#
rmax = 500
Xall = Array[]

# Execution times 
time_baker = []
time_brand = []
time_sketchy = []

## (Dry) Run it once due to JUlia's JIT compilation
baker = iSVD(x1=X[:,1], algo=:baker, max_rank=rmax) 
full_increment!(baker, X[:,2:3], verbose=true, runtime=true)
brand = iSVD(x1=X[:,1], algo=:brand1, reorth_method=:qr, max_rank=rmax)
full_increment!(brand, X[:,2:3], verbose=true, tol=1e-10, runtime=true)
sketchy = iSVD(algo=:sketchy; m=100, n=200, r=4, ReduxMap=:Sparse)
full_increment!(sketchy, X[1:100,1:200], verbose=true, runtime=true, dump_all=true)
svd(X[:,1:10])

## baker
tmp = @elapsed baker = iSVD(x1=X[:,1], algo=:baker, max_rank=rmax)
push!(time_baker, tmp)
tmp = full_increment!(baker, X[:,2:end], verbose=true, runtime=true)
push!(time_baker, tmp)

## brand
tmp = @elapsed brand = iSVD(x1=X[:,1], algo=:brand1, reorth_method=:gramschmidt, max_rank=rmax)
push!(time_brand, tmp)
tmp = full_increment!(brand, X[:,2:end], verbose=true, tol=1e-10, runtime=true)
push!(time_brand, tmp)

## sketchy
tmp = @elapsed sketchy = iSVD(algo=:sketchy; m=Nx*Ny, n=n, r=rmax, ReduxMap=:Sparse)
push!(time_sketchy, tmp)
tmp = full_increment!(sketchy, X, verbose=true, runtime=true, dump_all=true)
push!(time_sketchy, tmp.runtime)

#============================================#
## Compute the POD basis using the batch SVD 
#============================================#
time_batch = @elapsed F = svd(X)

#=====================================================================#
## Save the POD basis and singular values from the iSVD and batch SVD
#=====================================================================#
bases = Dict(
    "baker" => (iVr=baker.Q[:,1:rmax], iΣr=baker.Σ[1:rmax]),
    "brand" => (iVr=brand.Q[:,1:rmax], iΣr=brand.Σ[1:rmax]),
    "sketchy" => (iVr=sketchy.Q[:,1:rmax], iΣr=sketchy.Σ[1:rmax]),
    "batch" => (Vr=F.U[:,1:rmax], Σr=F.S[1:rmax]),
)
save(joinpath(FILEPATH, "data/streaming/basis.jld2"), bases)

#============================================================#
## Save the runtime of the iSVD algorithms over all streams
#============================================================#
time_baker = reduce(vcat, time_baker)
time_brand = reduce(vcat, time_brand)
# time_sketchy = reduce(vcat, time_sketchy)
save(
    joinpath(FILEPATH, "data/streaming/basis_runtime.jld2"),
    "baker", time_baker, "brand", time_brand,  
    "batch", time_batch,  "sketchy", time_sketchy, 
)

#================================#
## Compute the projection errors
#================================#
proj_error = Dict(
    "baker" => zeros(rmax),
    "brand" => zeros(rmax),
    "sketchy" => zeros(rmax),
    "batch" => zeros(rmax),
)
for i in 1:rmax
    proj_error["baker"][i] = norm(X - bases["baker"].iVr[:,1:i] * bases["baker"].iVr[:,1:i]' * X, 2) / norm(X, 2)
    proj_error["brand"][i] = norm(X - bases["brand"].iVr[:,1:i] * bases["brand"].iVr[:,1:i]' * X, 2) / norm(X, 2)
    proj_error["sketchy"][i] = norm(X - bases["sketchy"].iVr[:,1:i] * bases["sketchy"].iVr[:,1:i]' * X, 2) / norm(X, 2)
    proj_error["batch"][i] = norm(X - bases["batch"].Vr[:,1:i] * bases["batch"].Vr[:,1:i]' * X, 2) / norm(X, 2)
end

save(joinpath(FILEPATH, "data/projection_errors.jld2"), proj_error)            