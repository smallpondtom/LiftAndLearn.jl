"""
2D heat equation: generate data
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
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"Streaming-OpInf/heat2d") : joinpath(pwd(), "scripts/Streaming-OpInf/heat2d")

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

# #============================================================#
# ## Generate the POD basis using iSVD using Baker's algorithm
# #============================================================#
# rmax = 12
# Xall = Array[]

# # Initialize the iSVD object with the first dataset
# data = load(training_data_files[1])
# isvd = iSVD(x1=data["X"][:,1], algo=:baker, max_rank=rmax)
# # isvd = iSVD(x1=data["X"][:,1], algo=:brand1, reorth_method=:qr, max_rank=rmax)
# # isvd = iSVD(x1=data["X"][:,1], algo=:sketchy; m=32*40, n=10010, r=rmax, ReduxMap=:Sparse)
# full_increment!(isvd, data["X"][:,2:end], verbose=true, tol=1e-12)
# push!(Xall, data["X"])

# # Increment for the rest of the data
# for (i,data_file) in enumerate(training_data_files[2:end])
#     jldopen(data_file, "r") do data
#         # Load the data
#         X = data["X"]
#         # Compute the POD basis using the incremental SVD
#         full_increment!(isvd, X, verbose=true, tol=1e-12)
#         # Save the data for batch SVD
#         push!(Xall, X)
#     end
# end

# #============================================#
# ## Compute the POD basis using the batch SVD 
# #============================================#
# F = svd(reduce(hcat, Xall))

# #=====================================================================#
# ## Save the POD basis and singular values from the iSVD and batch SVD
# #=====================================================================#
# save(
#     joinpath(FILEPATH, "data/basis.jld2"),
#     "iVr", isvd.Q[:,1:rmax], "iΣr", isvd.Σ[1:rmax], 
#     "Vr", F.U[:,1:rmax], "Σr", F.S[1:rmax], "r", rmax
# )

#=========================================================#
## Generate the POD basis using iSVD using all algorithms
#=========================================================#
rmax = 10
Xall = Array[]

# Execution times 
time_baker = []
time_brand = []
time_sketchy = []

# Initialize the iSVD object with the first dataset
data = load(training_data_files[1])
# baker
baker = iSVD(x1=data["X"][:,1], algo=:baker, max_rank=rmax)
tmp = full_increment!(baker, data["X"][:,2:end], verbose=true, runtime=true)
push!(time_baker, tmp)
# brand
brand = iSVD(x1=data["X"][:,1], algo=:brand1, reorth_method=:qr, max_rank=rmax)
tmp = full_increment!(brand, data["X"][:,2:end], verbose=true, tol=1e-12, runtime=true)
push!(time_brand, tmp)
# sketchy
# sketchy = iSVD(x1=data["X"][:,1], algo=:sketchy; m=prod(heat2d.spatial_dim), n=heat2d.time_dim*heat2d.param_dim, 
#                r=rmax, ReduxMap=:Sparse)
# _, tmp = full_increment!(sketchy, data["X"][:,2:end], verbose=true, runtime=true)
# push!(time_sketchy, tmp)

## mergesketchy
X = data["X"]
K = size(X,2)
blks = 4
m = size(X,2) ÷ blks
Vmerge = nothing
Smerge = nothing
Wmerge = nothing
for i in 1:blks
    isvd2 = iSVD(algo=:sketchy, m=prod(heat2d.spatial_dim), n=m, r=rmax, ReduxMap=:Sparse)
    Xi = X[:,m*(i-1)+1:m*i]
    full_increment!(isvd2, Xi, verbose=false)
    if i == 1
        Vmerge = isvd2.Q 
        Smerge = isvd2.Σ
        Wmerge = isvd2.W
    else
        Vmerge, Smerge = IncrementalSVD.opt_merge_eigenspace(
            Vmerge, isvd2.Q, Smerge, isvd2.Σ, rmax
        )
    end
    @info "MergingSketchySVD: block #$i"
end
# Vr[:sketchy] = Vmerge
# Σr[:sketchy] = Smerge

##
push!(Xall, data["X"])

# Increment for the rest of the data
for (i,data_file) in enumerate(training_data_files[2:end])
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
        _, tmp = full_increment!(sketchy, X, verbose=true, runtime=true)
        push!(time_sketchy, tmp)
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
    "batch" => (Vr=F.U[:,1:rmax], Σr=F.S[1:rmax]),
)
save(joinpath(FILEPATH, "data/streaming/basis.jld2"), bases)

#============================================================#
## Save the runtime of the iSVD algorithms over all streams
#============================================================#
time_baker = reduce(vcat, time_baker)
time_brand = reduce(vcat, time_brand)
time_sketchy = reduce(vcat, time_sketchy)
save(
    joinpath(FILEPATH, "data/streaming/basis_runtime.jld2"),
    "baker", time_baker, "brand", time_brand, "sketchy", time_sketchy, "batch", time_batch,
)

#================================#
## Compute the projection errors
#================================#
X = reduce(hcat, Xall)
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