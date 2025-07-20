"""
3D Channel flow: Preprocessing step
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using LinearAlgebra

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
## Additional functions
#========================#
include(joinpath(FILEPATH, "preprocess.jl"))

#=============================#
## Load the training dataset
#=============================#
ds = ChannelDataSource(datafile, ["z", "y", "x", "fields", "times"])
Nz, Ny, Nx, n_fields, n = ds.dims
dim_per_field = Nz * Ny * Nx
dPdx = 0.001722

#================================================================#
## Compute preprocessing parameters (mean and/or minmax scaling)
#================================================================#
COMPUTE_MEAN = true
COMPUTE_MINMAX = true
SCALE_TYPE = :minmaxsym

if COMPUTE_MEAN && COMPUTE_MINMAX
    @info "Computing mean and min/max for preprocessing"
    mean_file = joinpath(FILEPATH, "data/mean.jld2")
    if isfile(mean_file)
        @info "Loading existing mean from file"
        xbar = load(mean_file, "xbar")
    else
        @info "Starting mean computation with $(Threads.nthreads()) threads"
        @time begin
            xbar = compute_mean_parallel_threads_locked(ds, (Nz*Ny*Nx*4,); batch_size=100)
        end
        @info "Mean computation complete"
        save(mean_file, "xbar", xbar)
    end

    minmax_file = joinpath(FILEPATH, "data/minmax.jld2")
    if isfile(minmax_file)
        @info "Loading existing minmax parameters from file"
        minmax_data = load(minmax_file, "minmax")
        shifts = minmax_data["shifts"]
        scales = minmax_data["scales"]
    else
        @info "Starting min/max computation with $(Threads.nthreads()) threads"
        @time begin
            x_min, x_max = compute_minmax_parallel_threads(ds, batch_size=100)
        end
        # Compute minmax scaling parameters
        @info "Min/max computation complete"
        if SCALE_TYPE == :minmaxsym
            @info "Using symmetric minmax scaling in range [-1, 1]"
            scales = (x_max .- x_min) ./ 2
            shifts = (x_max .+ x_min) ./ 2
        elseif SCALE_TYPE == :minmax
            @info "Using minmax scaling in range [0, 1]"
            scales = x_max .- x_min
            shifts = x_min
        else
            error("Unknown scaling type: $SCALE_TYPE")
        end
        save(minmax_file, "minmax", Dict(
            "min" => x_min, "max" => x_max,
            "shifts" => shifts, "scales" => scales
        ))
    end
elseif COMPUTE_MEAN
    @info "Computing mean for mean-shift preprocessing"
    mean_file = joinpath(FILEPATH, "data/mean.jld2")
    if isfile(mean_file)
        @info "Loading existing mean from file"
        xbar = load(mean_file, "xbar")
    else
        @info "Starting mean computation with $(Threads.nthreads()) threads"
        @time begin
            xbar = compute_mean_parallel_threads_locked(ds, (Nz*Ny*Nx*4,); batch_size=100)
        end
        @info "Mean computation complete"
        save(mean_file, "xbar", xbar)
    end
elseif COMPUTE_MINMAX
    @info "Computing min/max for minmax shift-and-scale preprocessing"
    minmax_file = joinpath(FILEPATH, "data/minmax.jld2")
    if isfile(minmax_file)
        @info "Loading existing minmax parameters from file"
        minmax_data = load(minmax_file, "minmax")
        xbar = minmax_data["xbar"]
        shifts = minmax_data["shifts"]
        scales = minmax_data["scales"]
    else
        @info "Starting min/max computation with $(Threads.nthreads()) threads"
        @time begin
            x_min, x_max = compute_minmax_parallel_threads(ds, batch_size=100)
        end
        # Compute minmax scaling parameters
        @info "Min/max computation complete"
        if SCALE_TYPE == :minmaxsym
            @info "Using symmetric minmax scaling in range [-1, 1]"
            scales = (x_max .- x_min) ./ 2
            shifts = (x_max .+ x_min) ./ 2
        elseif SCALE_TYPE == :minmax
            @info "Using minmax scaling in range [0, 1]"
            scales = x_max .- x_min
            shifts = x_min
        else
            error("Unknown scaling type: $SCALE_TYPE")
        end
        save(minmax_file, "minmax", Dict(
            "min" => x_min, "max" => x_max,
            "shifts" => shifts, "scales" => scales
        ))
    end
else
    @info "Skipping preprocessing, using default scaling and zero mean"
    xbar = 0.0
    scales = 1.0
    shifts = 0.0
end

# NOTES: Attempts
# 1) No preprocessing 
# 2) Mean shift only 
# 3) Mean shift + scaling with [sqrt(dPdx), sqrt(dPdx), sqrt(dPdx), dPdx]
# 4) Mean shift + scaling with [1.0, 0.01, 0.01, dPdx]