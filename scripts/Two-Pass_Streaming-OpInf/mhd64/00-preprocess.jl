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
           joinpath(pwd(), "Two-Pass_Streaming-OpInf/mhd64") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/mhd64")
DATAPATH = "../../../../DATA/THE_WELL/mhd64"
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
    Xr_orig  = Float64.(reshape(ds["rho"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xz_orig  = Float64.(reshape(ds["z"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xmx_orig = Float64.(reshape(ds["mx"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xmy_orig = Float64.(reshape(ds["my"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xmz_orig = Float64.(reshape(ds["mz"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xbx_orig = Float64.(reshape(ds["Bx"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xby_orig = Float64.(reshape(ds["By"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xbz_orig = Float64.(reshape(ds["Bz"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
else
    @info "Using Float32 for data"
    Xr_orig  = Float32.(reshape(ds["rho"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xz_orig  = Float32.(reshape(ds["z"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xmx_orig = Float32.(reshape(ds["mx"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xmy_orig = Float32.(reshape(ds["my"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xmz_orig = Float32.(reshape(ds["mz"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xbx_orig = Float32.(reshape(ds["Bx"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xby_orig = Float32.(reshape(ds["By"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xbz_orig = Float32.(reshape(ds["Bz"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
end
X_orig = vcat(Xr_orig, Xz_orig, Xmx_orig, Xmy_orig, Xmz_orig, 
              Xbx_orig, Xby_orig, Xbz_orig)

## Save unscaled/unshifted data
original_file = joinpath(FILEPATH, "data/original_data.jld2")
if !isfile(original_file)
    @info "Saving original data to file"
    save(original_file, 
        "X", Dict(
            "rho" => Xr_orig, "z" => Xz_orig, 
            "mx" => Xmx_orig, "my" => Xmy_orig, "mz" => Xmz_orig,
            "Bx" => Xbx_orig, "By" => Xby_orig, "Bz" => Xbz_orig,
            "all" => X_orig
        ),
    )
end

## Copy original data
Xr = copy(Xr_orig)
Xz = copy(Xz_orig)
Xmx = copy(Xmx_orig)
Xmy = copy(Xmy_orig)
Xmz = copy(Xmz_orig)
Xbx = copy(Xbx_orig)
Xby = copy(Xby_orig)
Xbz = copy(Xbz_orig)

## Center the data
Xrbar = mean(Xr, dims=2)
Xzbar = mean(Xz, dims=2)
Xmxbar = mean(Xmx, dims=2)
Xmybar = mean(Xmy, dims=2)
Xmzbar = mean(Xmz, dims=2)
Xbxbar = mean(Xbx, dims=2)
Xbybar = mean(Xby, dims=2)
Xbzbar = mean(Xbz, dims=2)

if CENTER_DATA
    @info "Centering the data"
    Xr .-= Xrbar
    Xz .-= Xzbar
    Xmx .-= Xmxbar
    Xmy .-= Xmybar
    Xmz .-= Xmzbar
    Xbx .-= Xbxbar
    Xby .-= Xbybar
    Xbz .-= Xbzbar
end

## Save mean data
if !isfile(joinpath(FILEPATH, "data/mean.jld2"))
    @info "Saving mean data to file"
    save(joinpath(FILEPATH, "data/mean.jld2"),
        "mean", Dict(
            "rho" => Xrbar, "z" => Xzbar,
            "mx" => Xmxbar, "my" => Xmybar, "mz" => Xmzbar,
            "Bx" => Xbxbar, "By" => Xbybar, "Bz" => Xbzbar,
            "all" => vcat(Xrbar, Xzbar, Xmxbar, Xmybar, Xmzbar, 
                          Xbxbar, Xbybar, Xbzbar)
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
        Xr, Xr_min, Xr_max = minmax_shift_scale(Xr)
        Xz, Xz_min, Xz_max = minmax_shift_scale(Xz)
        Xmx, Xmx_min, Xmx_max = minmax_shift_scale(Xmx)
        Xmy, Xmy_min, Xmy_max = minmax_shift_scale(Xmy)
        Xmz, Xmz_min, Xmz_max = minmax_shift_scale(Xmz)
        Xbx, Xbx_min, Xbx_max = minmax_shift_scale(Xbx)
        Xby, Xby_min, Xby_max = minmax_shift_scale(Xby)
        Xbz, Xbz_min, Xbz_max = minmax_shift_scale(Xbz)
    elseif SCALE_TYPE == :minmaxsym
        @info "Scaling the data to [-1, 1] with minmax scaling"
        Xr, Xr_min, Xr_max = minmaxsym_shift_scale(Xr)
        Xz, Xz_min, Xz_max = minmaxsym_shift_scale(Xz)
        Xmx, Xmx_min, Xmx_max = minmaxsym_shift_scale(Xmx)
        Xmy, Xmy_min, Xmy_max = minmaxsym_shift_scale(Xmy)
        Xmz, Xmz_min, Xmz_max = minmaxsym_shift_scale(Xmz)
        Xbx, Xbx_min, Xbx_max = minmaxsym_shift_scale(Xbx)
        Xby, Xby_min, Xby_max = minmaxsym_shift_scale(Xby)
        Xbz, Xbz_min, Xbz_max = minmaxsym_shift_scale(Xbz)
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
            "rho" => Xr, "z" => Xz,
            "mx" => Xmx, "my" => Xmy, "mz" => Xmz,
            "Bx" => Xbx, "By" => Xby, "Bz" => Xbz,
            "all" => vcat(Xr, Xz, Xmx, Xmy, Xmz, Xbx, Xby, Xbz)
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
                    "rho" => Xr_min, "z" => Xz_min,
                    "mx" => Xmx_min, "my" => Xmy_min, "mz" => Xmz_min,
                    "Bx" => Xbx_min, "By" => Xby_min, "Bz" => Xbz_min,
                    "all" => vcat(
                        Xr_min, Xz_min, Xmx_min, Xmy_min, Xmz_min, 
                        Xbx_min, Xby_min, Xbz_min
                    )
                ),
                "scale", Dict(
                    "rho" => Xr_max - Xr_min, "z" => Xz_max - Xz_min,
                    "mx" => Xmx_max - Xmx_min, "my" => Xmy_max - Xmy_min,
                    "mz" => Xmz_max - Xmz_min,
                    "Bx" => Xbx_max - Xbx_min, "By" => Xby_max - Xby_min,
                    "Bz" => Xbz_max - Xbz_min,
                    "all" => vcat(
                        Xr_max - Xr_min,   Xz_max - Xz_min, 
                        Xmx_max - Xmx_min, Xmy_max - Xmy_min, 
                        Xmz_max - Xmz_min, Xbx_max - Xbx_min, 
                        Xby_max - Xby_min, Xbz_max - Xbz_min
                    )
                ),
            )
        elseif SCALE_TYPE == :minmaxsym
            @info "Saving min/max [-1, 1] scaling parameters to file"
            save(minmax_file, 
                "shift", Dict(
                    "rho" => 0.5 * (Xr_max + Xr_min), 
                    "z" => 0.5 * (Xz_max + Xz_min),
                    "mx" => 0.5 * (Xmx_max + Xmx_min),
                    "my" => 0.5 * (Xmy_max + Xmy_min),
                    "mz" => 0.5 * (Xmz_max + Xmz_min),
                    "Bx" => 0.5 * (Xbx_max + Xbx_min),
                    "By" => 0.5 * (Xby_max + Xby_min),
                    "Bz" => 0.5 * (Xbz_max + Xbz_min),
                    "all" => vcat(
                        0.5 * (Xr_max + Xr_min),   0.5 * (Xz_max + Xz_min), 
                        0.5 * (Xmx_max + Xmx_min), 0.5 * (Xmy_max + Xmy_min), 
                        0.5 * (Xmz_max + Xmz_min), 0.5 * (Xbx_max + Xbx_min), 
                        0.5 * (Xby_max + Xby_min), 0.5 * (Xbz_max + Xbz_min)
                    )
                ),
                "scale", Dict(
                    "rho" => 0.5 * (Xr_max - Xr_min), 
                    "z" => 0.5 * (Xz_max - Xz_min),
                    "mx" => 0.5 * (Xmx_max - Xmx_min),
                    "my" => 0.5 * (Xmy_max - Xmy_min),
                    "mz" => 0.5 * (Xmz_max - Xmz_min),
                    "Bx" => 0.5 * (Xbx_max - Xbx_min),
                    "By" => 0.5 * (Xby_max - Xby_min),
                    "Bz" => 0.5 * (Xbz_max - Xbz_min),
                    "all" => vcat(
                        0.5 * (Xr_max - Xr_min),   0.5 * (Xz_max - Xz_min), 
                        0.5 * (Xmx_max - Xmx_min), 0.5 * (Xmy_max - Xmy_min), 
                        0.5 * (Xmz_max - Xmz_min), 0.5 * (Xbx_max - Xbx_min), 
                        0.5 * (Xby_max - Xby_min), 0.5 * (Xbz_max - Xbz_min)
                    )
                ),
            )
        else
            error("Unknown scaling type: $SCALE_TYPE")
        end
    end
end

#=============#
## Test data ##
#=============#
test_files = readdir(joinpath(DATAPATH, "test"), join=true)
fn_test = test_files[1] 
ds_test = DataSource(fn_test)
nx, ny, nz, n_fields, n_time, n_traj = ds_test.dims
nxyz = nx * ny * nz
n = n_time * n_traj
if USE_FLOAT64
    @info "Using Float64 for data"
    Xr_test  = Float64.(reshape(ds_test["rho"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xz_test  = Float64.(reshape(ds_test["z"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xmx_test = Float64.(reshape(ds_test["mx"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xmy_test = Float64.(reshape(ds_test["my"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xmz_test = Float64.(reshape(ds_test["mz"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xbx_test = Float64.(reshape(ds_test["Bx"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xby_test = Float64.(reshape(ds_test["By"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xbz_test = Float64.(reshape(ds_test["Bz"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
else
    @info "Using Float32 for data"
    Xr_test  = Float32.(reshape(ds_test["rho"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xz_test  = Float32.(reshape(ds_test["z"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xmx_test = Float32.(reshape(ds_test["mx"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xmy_test = Float32.(reshape(ds_test["my"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xmz_test = Float32.(reshape(ds_test["mz"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xbx_test = Float32.(reshape(ds_test["Bx"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xby_test = Float32.(reshape(ds_test["By"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xbz_test = Float32.(reshape(ds_test["Bz"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
end
X_test = vcat(Xr_test, Xz_test, Xmx_test, Xmy_test, Xmz_test, 
              Xbx_test, Xby_test, Xbz_test)
save(joinpath(FILEPATH, "data/test_data.jld2"), 
    "X", Dict(
        "rho" => Xr_test, "z" => Xz_test,
        "mx" => Xmx_test, "my" => Xmy_test, "mz" => Xmz_test,
        "Bx" => Xbx_test, "By" => Xby_test, "Bz" => Xbz_test,
        "all" => X_test
    )
)
