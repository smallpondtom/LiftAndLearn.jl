"""
2D heat equation: Compute basis
"""

#================#
## Load Packages
#================#
using CairoMakie
using FileIO
using JLD2
using IncrementalSVD
using LinearAlgebra
using ProgressMeter
import LiftAndLearn as LnL
using PolynomialModelReductionDataset: Heat2DModel

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"Two-Pass_Streaming-OpInf/heat2d") : joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/heat2d")

#======================================#
## Obtain all the saved training files
#======================================#
training_data_files = readdir(joinpath(FILEPATH, "data/training"), join=true)

#=================#
## Load the setup
#=================#
setup_file = joinpath(FILEPATH, "data/setup.jld2")
setup = load(setup_file)
heat2d = setup["heat2d"]

#=========================================================#
## Generate the POD basis using iSVD using all algorithms
#=========================================================#
rmax = 10
Xall = Array[]

# Execution times 
time_baker = []
time_brand = []
time_sketchy = []
time_mergingsketchy = []

# Initialize the iSVD object with the first dataset
data = load(training_data_files[1])

## (Dry) Run it once due to JUlia's JIT compilation
baker = iSVD(x1=data["X"][:,1], algo=:baker, max_rank=rmax) 
full_increment!(baker, data["X"][:,2:3], verbose=true, runtime=true)
brand = iSVD(x1=data["X"][:,1], algo=:brand1, reorth_method=:qr, max_rank=rmax)
full_increment!(brand, data["X"][:,2:3], verbose=true, tol=1e-10, runtime=true)
sketchy = iSVD(algo=:sketchy; m=prod(heat2d.spatial_dim), n=(heat2d.time_dim-1), r=rmax, ReduxMap=:Sparse)
full_increment!(sketchy, data["X"][:,2:3], verbose=true, runtime=true)
mergingsketchy = iSVD(algo=:mergingsketchy; m=prod(heat2d.spatial_dim), b=200, r=rmax, ReduxMap=:Sparse)
full_increment!(mergingsketchy, data["X"][:,2:201], verbose=true, runtime=true)
svd(data["X"][:,1:10])

## baker
@info "Processing file 1 out of $(length(training_data_files))"
tmp = @elapsed baker = iSVD(x1=data["X"][:,1], algo=:baker, max_rank=rmax)
push!(time_baker, tmp)
tmp = full_increment!(baker, data["X"][:,2:end], verbose=true, runtime=true)
push!(time_baker, tmp)
# brand
tmp = @elapsed brand = iSVD(x1=data["X"][:,1], algo=:brand1, reorth_method=:qr, max_rank=rmax)
push!(time_brand, tmp)
tmp = full_increment!(brand, data["X"][:,2:end], verbose=true, tol=1e-10, runtime=true)
push!(time_brand, tmp)
# sketchy
tmp = @elapsed sketchy = iSVD(algo=:sketchy; m=prod(heat2d.spatial_dim), n=(heat2d.time_dim-1)*heat2d.param_dim, 
               r=rmax, ReduxMap=:Sparse)
push!(time_sketchy, tmp)
tmp = full_increment!(sketchy, data["X"], verbose=true, runtime=true)
push!(time_sketchy, tmp.runtime)
# mergingsketchy
blk = 10
blksize = size(data["X"],2) ÷ blk
tmp = @elapsed mergingsketchy = iSVD(algo=:mergingsketchy; m=prod(heat2d.spatial_dim), b=blksize, r=rmax, ReduxMap=:Sparse)
push!(time_mergingsketchy, tmp)
tmp = full_increment!(mergingsketchy, data["X"], verbose=true, runtime=true)
push!(time_mergingsketchy, tmp)

push!(Xall, data["X"])

# Increment for the rest of the data
for (i,data_file) in enumerate(training_data_files[2:end])
    @info "Processing file $(i+1) out of $(length(training_data_files))"
    jldopen(data_file, "r") do data
        # Load the data
        X = data["X"]
        # Compute the POD basis using Baker's algorithm
        tmp = full_increment!(baker, X, verbose=true, runtime=true)
        push!(time_baker, tmp)
        # Comput the POD basis using Brand's algorithm
        tmp = full_increment!(brand, X, verbose=true, tol=1e-12, runtime=true)
        push!(time_brand, tmp)
        # Compute the POD basis using SketchySVD
        tmp = full_increment!(sketchy, X, verbose=true, runtime=true)
        push!(time_sketchy, tmp.runtime)
        # Compute the POD basis using MergingSketchySVD
        tmp = full_increment!(mergingsketchy, X, verbose=true, runtime=true)
        push!(time_mergingsketchy, tmp)
        # Save the data for batch SVD
        push!(Xall, X)
    end
end

#============================================#
## Compute the POD basis using the batch SVD 
#============================================#
time_batch = @elapsed F = svd(reduce(hcat, Xall))

#=====================================================================#
## Save the POD basis and singular values from the iSVD and batch SVD
#=====================================================================#
bases = Dict(
    "baker" => (iVr=baker.Q[:,1:rmax], iΣr=baker.Σ[1:rmax]),
    "brand" => (iVr=brand.Q[:,1:rmax], iΣr=brand.Σ[1:rmax]),
    "sketchy" => (iVr=sketchy.Q[:,1:rmax], iΣr=sketchy.Σ[1:rmax]),
    "mergingsketchy" => (iVr=mergingsketchy.Q[:,1:rmax], iΣr=mergingsketchy.Σ[1:rmax]),
    "batch" => (Vr=F.U[:,1:rmax], Σr=F.S[1:rmax]),
)
save(joinpath(FILEPATH, "data/streaming/basis.jld2"), bases)

#============================================================#
## Save the runtime of the iSVD algorithms over all streams
#============================================================#
time_baker = reduce(vcat, time_baker)
time_brand = reduce(vcat, time_brand)
time_sketchy = reduce(vcat, time_sketchy)
time_mergingsketchy = reduce(vcat, time_mergingsketchy)
save(
    joinpath(FILEPATH, "data/streaming/basis_runtime.jld2"),
    "baker", time_baker, "brand", time_brand, "sketchy", time_sketchy, 
    "mergingsketchy", time_mergingsketchy, "batch", time_batch,
)

#================================#
## Compute the projection errors
#================================#
X = reduce(hcat, Xall)
proj_error = Dict(
    "baker" => zeros(rmax),
    "brand" => zeros(rmax),
    "sketchy" => zeros(rmax),
    "mergingsketchy" => zeros(rmax),
    "batch" => zeros(rmax),
)
for i in 1:rmax
    proj_error["baker"][i] = norm(X - bases["baker"].iVr[:,1:i] * bases["baker"].iVr[:,1:i]' * X, 2) / norm(X, 2)
    proj_error["brand"][i] = norm(X - bases["brand"].iVr[:,1:i] * bases["brand"].iVr[:,1:i]' * X, 2) / norm(X, 2)
    proj_error["sketchy"][i] = norm(X - bases["sketchy"].iVr[:,1:i] * bases["sketchy"].iVr[:,1:i]' * X, 2) / norm(X, 2)
    proj_error["mergingsketchy"][i] = norm(X - bases["mergingsketchy"].iVr[:,1:i] * bases["mergingsketchy"].iVr[:,1:i]' * X, 2) / norm(X, 2)
    proj_error["batch"][i] = norm(X - bases["batch"].Vr[:,1:i] * bases["batch"].Vr[:,1:i]' * X, 2) / norm(X, 2)
end

save(joinpath(FILEPATH, "data/projection_errors.jld2"), proj_error)            