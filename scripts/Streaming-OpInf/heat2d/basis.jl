"""
2D heat equation: generate data
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using IncrementalSVD
using LinearAlgebra
using ProgressMeter
using LiftAndLearn
const LnL = LiftAndLearn

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"Streaming-OpInf/heat2d") : joinpath(pwd(), "scripts/Streaming-OpInf/heat2d")

#======================================#
## Obtain all the saved training files
#======================================#
training_data_files = readdir(joinpath(FILEPATH, "data/training"), join=true)

#====================================#
## Generate the POD basis using iSVD
#====================================#
rmax = 12
# Xall = Array[]

# Initialize the iSVD object with the first dataset
data = load(training_data_files[1])
isvd = iSVD(x1=data["X"][:,1], algo=:baker, max_rank=rmax)
# isvd = iSVD(x1=data["X"][:,1], algo=:brand1, reorth_method=:qr, max_rank=rmax)
# isvd = iSVD(x1=data["X"][:,1], algo=:sketchy; m=32*40, n=10010, r=rmax, ReduxMap=:Sparse)
full_increment!(isvd, data["X"][:,2:end], verbose=true, tol=1e-12)
push!(Xall, data["X"])

# Increment for the rest of the data
for (i,data_file) in enumerate(training_data_files[2:end])
    jldopen(data_file, "r") do data
        # Load the data
        X = data["X"]
        # Compute the POD basis using the incremental SVD
        full_increment!(isvd, X, verbose=true, tol=1e-12)
        # Save the data for batch SVD
        push!(Xall, X)
    end
end

#============================================#
## Compute the POD basis using the batch SVD 
#============================================#
F = svd(reduce(hcat, Xall))

#=====================================================================#
## Save the POD basis and singular values from the iSVD and batch SVD
#=====================================================================#
save(
    joinpath(FILEPATH, "data/basis.jld2"),
    "iVr", isvd.Q[:,1:rmax], "iΣr", isvd.Σ[1:rmax], 
    "Vr", F.U[:,1:rmax], "Σr", F.S[1:rmax], "r", rmax
)