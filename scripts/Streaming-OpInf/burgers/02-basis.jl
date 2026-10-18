"""
1D Viscous Burgers' equation: Compute the POD basis using iSVD
"""

#=================#
## Load Packages ##
#=================#
using CairoMakie
using FileIO
using JLD2
using IncrementalSVD
using LinearAlgebra
using ProgressMeter
import LiftAndLearn as LnL
using PolynomialModelReductionDataset: BurgersModel

#=================================#
## Configure filepath for saving ##
#=================================#
FILEPATH = @__DIR__

#=======================================#
## Obtain all the saved training files ##
#=======================================#
training_data_files = readdir(joinpath(FILEPATH, "data/training"), join=true)
n_files = length(training_data_files)

#==================#
## Load the setup ##
#==================#
setup_file = joinpath(FILEPATH, "data/setup.jld2")
setup = load(setup_file)
burgers = setup["burgers"]
options = setup["options"]
n_t = burgers.time_dim

#==========================================================#
## Generate the POD basis using iSVD using all algorithms ##
#==========================================================#
@info "Generating POD basis using iSVD..."
rmax = 14
Xall = Array[]

# Initialize the iSVD object with the first dataset
data = load(training_data_files[1])
n_inputs = data["num_inputs"]

## Run all algorithms for the first data file (first parameter value)
@info "Processing file 1 out of $(n_files)"
baker = iSVD(x1=data["X"][:,1,1], algo=:baker, max_rank=rmax)
brand = iSVD(x1=data["X"][:,1,1], algo=:brand1, reorth_method=:qr, max_rank=rmax)
sketchy = iSVD(
    algo=:sketchy; m=burgers.spatial_dim, 
    n=n_t * n_inputs * n_files,
    r=rmax, ReduxMap=:Sparse
)
for j in 1:n_inputs
    # Baker 
    full_increment!(baker, 
        j == 1 ? data["X"][:,2:end,j] : data["X"][:, :, j],
        verbose=true
    )

    # Brand
    full_increment!(brand, 
        j == 1 ? data["X"][:,2:end,j] : data["X"][:, :, j],
        verbose=true
    )

    # SketchySVD
    full_increment!(sketchy, data["X"][:, :, j], verbose=true)

    # Batch (just storing data)
    push!(Xall, data["X"][:, :, j])
end

## Increment for the rest of the data
for (i,data_file) in enumerate(training_data_files[2:end])
    @info "Processing file $(i+1) out of $(n_files)"
    jldopen(data_file, "r") do data
        for j in 1:n_inputs
            # Load the data
            X = data["X"][:,:,j]
            # Baker
            full_increment!(baker, X, verbose=true)
            # Brand
            full_increment!(brand, X, verbose=true)
            # SketchySVD
            full_increment!(sketchy, X, verbose=true)
            # Save the data for batch SVD
            push!(Xall, X)
        end
    end
end
@info "Done."

#=============================================#
## Compute the POD basis using the batch SVD ##
#=============================================#
F = svd(reduce(hcat, Xall))

#===============================##
## Approximate Xdothat and Xhat ##
#===============================##
@info "Computing Xhat and Xdothat for all training data..."
IncrementalSVD.terminate!(sketchy, false, false)
@assert size(sketchy.W, 1) == n_t * n_inputs * n_files
@assert size(sketchy.W, 2) == rmax
for i in 1:n_files
    @info "Processing file $(i) out of $(n_files) for Xhat and Xhatdot"
    data = load(training_data_files[i])
    X = data["X"]
    Xhat_save = Array{Float64,3}(undef, rmax, size(X,2)-1, n_inputs)
    Xhatdot_save = Array{Float64,3}(undef, rmax, size(X,2)-1, n_inputs)
    for j in 1:n_inputs 
        @info "  Processing input $(j) out of $(n_inputs)"
        idx_start = (i-1) * n_t * n_inputs + (j-1) * n_t + 1
        idx_end = idx_start + n_t - 1
        Xhat = Diagonal(sketchy.Σ) * sketchy.W[idx_start:idx_end, 1:rmax]'
        E, Δidx = LnL.finite_diff_matrix(
            options.data.deriv_type, n_t, options.data.Δt
        )
        Xhatdot = Xhat * E
        Xhat = Xhat[:, Δidx]

        # Check accuracy of the reduced data 
        Xj = X[:, :, j]
        Xhat_true = sketchy.Q[:, 1:rmax]' * Xj
        Xhatdot_true = Xhat_true * E
        Xhat_true = Xhat_true[:, Δidx]
        @info "    || Xhat - Xhat_true ||_2 / ||X||_2 = $(norm(Xhat - Xhat_true, 2) / norm(X, 2))"
        @info "    || Xdothat - Xdothat_true ||_2 / ||X||_2 = $(norm(Xhatdot - Xhatdot_true, 2) / norm(X, 2))"

        # Store values
        Xhat_save[:, :, j] = Xhat
        Xhatdot_save[:, :, j] = Xhatdot
    end
    # Save the approximated data
    data["Xhat"] = Xhat_save
    data["Xhatdot"] = Xhatdot_save
    save(training_data_files[i], data)
end
@info "Done." 

#=====================================================================#
## Save the POD basis and singular values from the iSVD and batch SVD
#=====================================================================#
bases = Dict(
    "baker"   => (iVr=baker.Q[:,1:rmax], iΣr=baker.Σ[1:rmax]),
    "brand"   => (iVr=brand.Q[:,1:rmax], iΣr=brand.Σ[1:rmax]),
    "sketchy" => (iVr=sketchy.Q[:,1:rmax], iΣr=sketchy.Σ[1:rmax], iWr=sketchy.W[:,1:rmax]),
    "batch"   => (Vr=F.U[:,1:rmax], Σr=F.S[1:rmax], Wr=F.V[:,1:rmax]),
)
save(joinpath(FILEPATH, "data/streaming/basis.jld2"), bases)

#================================#
## Compute the projection errors
#================================#
@info "Computing projection errors..."
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
end
save(joinpath(FILEPATH, "data/projection_errors.jld2"), proj_error)
@info "Done."