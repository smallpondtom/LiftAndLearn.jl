"""
Generate the reduced data using the reduced basis from the iSVD.
"""

#================#
## Load Packages
#================#
using FileIO
using FLoops
using JLD2
using LinearAlgebra
using ProgressMeter
using BlockDiagonals
using SparseArrays
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
DATAPATH = "../../../../../DATA/NREL/ABL"
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"Two-Pass_Streaming-OpInf/ABL") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/ABL")
fn = "ABL_0_10000.h5"
datafile = joinpath(DATAPATH, fn)

#==========================================#
## Load struct to read data in HDF5 format 
#==========================================#
include(joinpath(FILEPATH, "datasource.jl"))

#========================#
## Additional functions
#========================#
include(joinpath(FILEPATH, "derivative.jl"))
include(joinpath(FILEPATH, "preprocess.jl"))

#=============================#
## Load the training dataset
#=============================#
ds = ChannelDataSource(datafile, ["z", "y", "x", "fields", "times"])
Nz, Ny, Nx, n_fields, n = ds.dims
dim_per_field = Nz * Ny * Nx
n_test = 2000
n_train = n - n_test

#============================#
## Load the mean and scaling
#============================#
means  = load(joinpath(FILEPATH, "data/mean.jld2"))["xbar"]
shifts = load(joinpath(FILEPATH, "data/minmax.jld2"))["minmax"]["shifts"]
scales = load(joinpath(FILEPATH, "data/minmax.jld2"))["minmax"]["scales"]

#=================#
## Load the bases 
#=================#
basis_file = joinpath(FILEPATH, "data/bases/basis.jld2")
iVrmax = load(basis_file)["bases"]["baker"].iVr

#=============================#
## Generate the reduced data 
#=============================#
Xhat = nothing     # put on global scope just in case
Xhatdot = nothing  # put on global scope just in case

# LnL options
options = LnL.LSOpInfOption(
    data=LnL.DataStructure(
        Δt=sum(diff(ds["times"][1:n_train])) / (length(ds["times"][1:n_train])-1),
        deriv_type="FBCT4"
    ),
)
CONTINUOUS_TINE = true

for r in [100, 150, 200, 250, 300, 350, 400]
    # Process snapshots in parallel batches
    BATCH_SIZE = 100  # Adjust based on available memory
    num_batches = ceil(Int, n_train / BATCH_SIZE)
    Xhat = Array{Float64}(undef, r, n_train)  # Preallocate the reduced data matrix

    # Compose basis
    iVr = view(iVrmax, :, 1:r) 

    @info "Processing $n_train snapshots in $num_batches batches using $(Threads.nthreads()) threads"

    # Use FLoops for better thread management
    @floop ThreadedEx() for batch in 1:num_batches
        start_idx = (batch-1) * BATCH_SIZE + 1
        end_idx = min(start_idx + BATCH_SIZE - 1, n_train)
        
        # Pre-allocate batch array for this thread
        batch_size = end_idx - start_idx + 1
        batch_xhat = Matrix{Float64}(undef, r, batch_size)

        # Process batch sequentially within each thread
        for (local_idx, i) in enumerate(start_idx:end_idx)
            # Each thread processes its batch independently
            snapshot = preprocess!(ds[i], means, shifts, scales)
            batch_xhat[:, local_idx] = iVr' * snapshot
        end
        
        # Thread-safe assignment to non-overlapping region
        @views Xhat[:, start_idx:end_idx] .= batch_xhat
    end

    reduced_data_file = joinpath(FILEPATH, "data/streaming/reduced_data_r$r.jld2")
    @info "Saving reduced data for r=$(r) to $reduced_data_file"

    # Compute the reduced time derivative data from the reduced data
    if CONTINUOUS_TINE
        @info "Computing time derivative for continuous time"
        Xhatdot, idx = LnL.time_derivative_approx(Xhat, options)
        Xhat = Xhat[:, idx]  # Keep only the valid indices
        save(reduced_data_file, "Xhat", Xhat, "Xhatdot", Xhatdot)
    else
        save(reduced_data_file, "Xhat", Xhat)
    end
end