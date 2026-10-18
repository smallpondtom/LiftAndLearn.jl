"""
Kuramoto–Sivashinsky equation: generate data
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
using PolynomialModelReductionDataset: KuramotoSivashinskyModel

#================================#
## Configure filepath for saving
#================================#
FILEPATH = @__DIR__

#======================================#
## Obtain all the saved training files
#======================================#
training_data_files = readdir(joinpath(FILEPATH, "data/training"), join=true)

#=================#
## Load the setup
#=================#
setup_file = joinpath(FILEPATH, "data/setup.jld2")
setup = load(setup_file)
kse = setup["kse"]
num_of_streams = Int(setup["num_of_streams"])

#=========================================================#
## Generate the POD basis using iSVD using all algorithms
#=========================================================#
@info "Generating the POD basis using iSVD algorithms..."
rmin = 9
rmax = 24
Xall = Array[]

# Initialize the iSVD object with the first dataset
data = load(training_data_files[1])

## (Dry) Run it once due to JUlia's JIT compilation
baker = iSVD(x1=data["X"][:,1], algo=:baker, max_rank=rmax) 
full_increment!(baker, data["X"][:,2:3], verbose=true)
brand = iSVD(x1=data["X"][:,1], algo=:brand1, reorth_method=:qr, max_rank=rmax)
full_increment!(brand, data["X"][:,2:3], verbose=true, tol=1e-10)
sketchy = iSVD(algo=:sketchy; m=kse.spatial_dim, n=1000, r=rmax, ReduxMap=:Sparse)
full_increment!(sketchy, data["X"][:,2:3], verbose=true)
svd(data["X"][:,1:10])

## baker
@info "Processing file 1 out of $(length(training_data_files))"
baker = iSVD(x1=data["X"][:,1], algo=:baker, max_rank=rmax)
full_increment!(baker, data["X"][:,2:end], verbose=true, runtime=false)

# brand
brand = iSVD(x1=data["X"][:,1], algo=:brand1, reorth_method=:qr, max_rank=rmax)
full_increment!(brand, data["X"][:,2:end], verbose=true, tol=1e-10, runtime=false)

# sketchy
sketchy = iSVD(
    algo=:sketchy; m=kse.spatial_dim, n=num_of_streams, 
    r=rmax, ReduxMap=:Sparse
)
full_increment!(sketchy, data["X"], verbose=true, runtime=false, dump_all=true)

push!(Xall, data["X"])

# Increment for the rest of the data
for (i,data_file) in enumerate(training_data_files[2:end])
    @info "Processing file $(i+1) out of $(length(training_data_files))"
    jldopen(data_file, "r") do data
        # Load the data
        X = data["X"]
        # Compute the POD basis using Baker's algorithm
        full_increment!(baker, X, verbose=true, runtime=false)
        # Comput the POD basis using Brand's algorithm
        full_increment!(brand, X, verbose=true, tol=1e-12, runtime=false)
        # Compute the POD basis using SketchySVD
        full_increment!(sketchy, X, verbose=true, runtime=false, dump_all=true)
        # Save the data for batch SVD
        push!(Xall, X)
    end
end

# Terminate the SketchySVD SVD computation
IncrementalSVD.terminate!(sketchy, false, false)

@info "Done generating the POD basis using iSVD algorithms."

#============================================#
## Compute the POD basis using the batch SVD 
#============================================#
F = svd(reduce(hcat, Xall))

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

#================================#
## Compute the projection errors
#================================#
@info "Computing the projection errors..."
X = reduce(hcat, Xall)
proj_error = Dict(
    "baker" => zeros(rmax),
    "brand" => zeros(rmax),
    "sketchy" => zeros(rmax),
    "batch" => zeros(rmax),
)
for i in 1:rmax
    proj_error["baker"][i] = norm(
        X - bases["baker"].iVr[:,1:i] * bases["baker"].iVr[:,1:i]' * X, 2
    ) / norm(X, 2)
    proj_error["brand"][i] = norm(
        X - bases["brand"].iVr[:,1:i] * bases["brand"].iVr[:,1:i]' * X, 2
    ) / norm(X, 2)
    proj_error["sketchy"][i] = norm(
        X - bases["sketchy"].iVr[:,1:i] * bases["sketchy"].iVr[:,1:i]' * X, 2
    ) / norm(X, 2)
    proj_error["batch"][i] = norm(
        X - bases["batch"].Vr[:,1:i] * bases["batch"].Vr[:,1:i]' * X, 2
    ) / norm(X, 2)
    println("Computed projection error for r = $i")
    println(
        "  Baker: $(proj_error["baker"][i]), Brand: $(proj_error["brand"][i]),"
    )
    println(
        "  Sketchy: $(proj_error["sketchy"][i]), Batch: $(proj_error["batch"][i])"
    )
end

save(joinpath(FILEPATH, "data/projection_errors.jld2"), proj_error)           
@info "Done computing the projection errors."