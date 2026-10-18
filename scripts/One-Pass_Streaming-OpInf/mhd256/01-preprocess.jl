"""
MHD256: Preprocessing the data
"""

#=================#
## Load packages ##
#=================#
using LinearAlgebra
using FileIO
using JLD2
using Statistics

#=============#
## Load data ##
#=============#
# Get paths and data file name
DATAPATH = "../../../../../scratch1/1/tkoike3"
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(), "One-Pass_Streaming-OpInf/mhd256") : 
           joinpath(pwd(), "scripts/One-Pass_Streaming-OpInf/mhd256")
fn = "MHD_Ma_0.7_Ms_0.5.hdf5"

# Include the data sourcing module for data access
include(joinpath(FILEPATH, "datasource.jl"))

# Load data source and define dimensions 
ds = DataSource(fn)
nx, ny, nz, n_fields, n_time, n_traj = ds.dims
n = n_time * n_traj
nxyz = nx * ny * nz

# Include the preprocessing functions
include(joinpath(FILEPATH, "preprocess.jl"))


#=================================================================#
## Compute preprocessing parameters (mean and/or minmax scaling) ##
#=================================================================#
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
            xbar = compute_mean_parallel_threads_locked(ds, (nz*ny*nx*n_fields,), 
                                                        n; batch_size=100)
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
            x_min, x_max = compute_minmax_parallel_threads(ds, n, 
                                                           batch_size=100,
                                                           means=nothing)
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
            xbar = compute_mean_parallel_threads_locked(ds, (nz*ny*nx*n_fields,), 
                                                        n; batch_size=100)
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
            x_min, x_max = compute_minmax_parallel_threads(ds, n, 
                                                           batch_size=100)
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