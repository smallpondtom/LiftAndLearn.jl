"""
Supernova 64^3 example: Preprocessing the data
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
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(), "Two-Pass_Streaming-OpInf/supernova64") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/supernova64")
DATAPATH = "../../../../DATA/THE_WELL/supernova_explosion_64/train"
train_files = readdir(DATAPATH, join=true)
fn = train_files[1]

# Include the data sourcing module for data access
include(joinpath(FILEPATH, "datasource.jl"))

# Load data source 
ds = DataSource(fn)
nx, ny, nz, n_fields, n_time, n_traj = ds.dims
nxyz = nx * ny * nz

# Flag for float64
USE_FLOAT64 = true

# Flag to center
CENTER_DATA = true
SCALE_DATA = true

#====================#
## Preprocess data  ##
#====================#
n = n_time * n_traj
if USE_FLOAT64
    @info "Using Float64 for data"
    Xp_orig = Float64.(reshape(ds["p"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xz_orig = Float64.(reshape(ds["z"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xu_orig = Float64.(reshape(ds["u"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xv_orig = Float64.(reshape(ds["v"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xw_orig = Float64.(reshape(ds["w"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
else
    @info "Using Float32 for data"
    Xp_orig = Float32.(reshape(ds["p"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xz_orig = Float32.(reshape(ds["z"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xu_orig = Float32.(reshape(ds["u"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xv_orig = Float32.(reshape(ds["v"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xw_orig = Float32.(reshape(ds["w"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
end
X_orig = vcat(Xp_orig, Xz_orig, Xu_orig, Xv_orig, Xw_orig)

## Save unscaled/unshifted data
original_file = joinpath(FILEPATH, "data/original_data.jld2")
if !isfile(original_file)
    @info "Saving original data to file"
    save(original_file, 
        "X", Dict(
            "p" => Xp_orig, "z" => Xz_orig, "u" => Xu_orig, 
            "v" => Xv_orig, "w" => Xw_orig,
            "all" => X_orig
        ),
    )
end

## Copy original data
Xp = copy(Xp_orig)
Xz = copy(Xz_orig)
Xu = copy(Xu_orig)
Xv = copy(Xv_orig)
Xw = copy(Xw_orig)

## Center the data
Xpbar = mean(Xp, dims=2)
Xzbar = mean(Xz, dims=2)
Xubar = mean(Xu, dims=2)
Xvbar = mean(Xv, dims=2)
Xwbar = mean(Xw, dims=2)

if CENTER_DATA
    @info "Centering the data"
    Xp .-= Xpbar
    Xz .-= Xzbar
    Xu .-= Xubar
    Xv .-= Xvbar
    Xw .-= Xwbar
end

## Save mean data
if !isfile(joinpath(FILEPATH, "data/mean.jld2"))
    @info "Saving mean data to file"
    save(joinpath(FILEPATH, "data/mean.jld2"),
        "mean", Dict(
            "p" => Xpbar, "z" => Xzbar, 
            "u" => Xubar, "v" => Xvbar, "w" => Xwbar
        )
    )
end

## Normalize to [0,1] with (row-wise) minmax scaling
SCALE_TYPE = :minmaxsym
function minmax_shift_scale(X)
    X_min = minimum(X, dims=2)
    X_max = maximum(X, dims=2)
    X .-= X_min 
    X ./= (X_max - X_min)
    return X, X_min, X_max
end

function minmaxsym_shift_scale(X)
    X_min = minimum(X, dims=2)
    X_max = maximum(X, dims=2)
    X .-= (0.5 * (X_max + X_min))
    X ./= (0.5 * (X_max - X_min))
    return X, X_min, X_max
end

if SCALE_DATA
    if SCALE_TYPE == :minmax
        @info "Scaling the data to [0, 1] with minmax scaling"
        Xp, Xp_min, Xp_max = minmax_shift_scale(Xp)
        Xz, Xz_min, Xz_max = minmax_shift_scale(Xz)
        Xu, Xu_min, Xu_max = minmax_shift_scale(Xu)
        Xv, Xv_min, Xv_max = minmax_shift_scale(Xv)
        Xw, Xw_min, Xw_max = minmax_shift_scale(Xw)
    elseif SCALE_TYPE == :minmaxsym
        @info "Scaling the data to [-1, 1] with minmax scaling"
        Xp, Xp_min, Xp_max = minmaxsym_shift_scale(Xp)
        Xz, Xz_min, Xz_max = minmaxsym_shift_scale(Xz)
        Xu, Xu_min, Xu_max = minmaxsym_shift_scale(Xu)
        Xv, Xv_min, Xv_max = minmaxsym_shift_scale(Xv)
        Xw, Xw_min, Xw_max = minmaxsym_shift_scale(Xw)
    else
        error("Unknown scaling type: $SCALE_TYPE")
    end
end

## Save preprocessed data
preprocessed_file = joinpath(FILEPATH, "data/preprocessed_data.jld2")
if !isfile(preprocessed_file)
    @info "Saving preprocessed data to file"
    save(preprocessed_file, 
        "X", Dict(
            "p" => Xp, "z" => Xz, "u" => Xu, "v" => Xv, "w" => Xw,
            "all" => vcat(Xp, Xz, Xu, Xv, Xw)
        )
    )
end

## Save shift and scaling 
if SCALE_DATA
    minmax_file = joinpath(FILEPATH, "data/minmax.jld2")
    if !isfile(minmax_file)
        if SCALE_TYPE == :minmax
            @info "Saving min/max [0, 1] scaling parameters to file"
            save(minmax_file, 
                "shift", Dict(
                "p" => Xp_min, "z" => Xz_min,
                "u" => Xu_min, "v" => Xv_min, "w" => Xw_min
                ),
                "scale", Dict(
                "p" => Xp_max - Xp_min, "z" => Xz_max - Xz_min,
                "u" => Xu_max - Xu_min, "v" => Xv_max - Xv_min, 
                "w" => Xw_max - Xw_min
                ),
            )
        elseif SCALE_TYPE == :minmaxsym
            @info "Saving min/max [-1, 1] scaling parameters to file"
            save(minmax_file, 
                "shift", Dict(
                "p" => 0.5 * (Xp_max + Xp_min), 
                "z" => 0.5 * (Xz_max + Xz_min),
                "u" => 0.5 * (Xu_max + Xu_min), 
                "v" => 0.5 * (Xv_max + Xv_min), 
                "w" => 0.5 * (Xw_max + Xw_min)
                ),
                "scale", Dict(
                "p" => 0.5 * (Xp_max - Xp_min), 
                "z" => 0.5 * (Xz_max - Xz_min),
                "u" => 0.5 * (Xu_max - Xu_min), 
                "v" => 0.5 * (Xv_max - Xv_min), 
                "w" => 0.5 * (Xw_max - Xw_min)
                ),
            )
        else
            error("Unknown scaling type: $SCALE_TYPE")
        end
    end
end