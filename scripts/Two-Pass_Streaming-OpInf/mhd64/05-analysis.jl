"""
Compute 2PCF and 3PCF analysis for the original data and reduced model results
"""

#=================#
## Load packages ##
#=================#
using LinearAlgebra
using FileIO
using JLD2
using IncrementalSVD
import LiftAndLearn as LnL

#============#
## Settings ##
#============#
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(), "Two-Pass_Streaming-OpInf/mhd64") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/mhd64")
DATAPATH = "../../../../DATA/THE_WELL/mhd64"
train_files = readdir(DATAPATH, join=true)
test_files = readdir(joinpath(DATAPATH, "test"), join=true)
fn = train_files[1]
fn_test = test_files[1]

# Include the data sourcing module for data access
include(joinpath(FILEPATH, "datasource.jl"))

# Load data source (training)
ds = DataSource(fn)
nx, ny, nz, n_fields, n_time, n_traj = ds.dims
nxyz = nx * ny * nz
n = n_time * n_traj

# Load data source (testing)
ds_test = DataSource(fn_test)

#=======================================#
## Compute the velocity power spectrum ##
#=======================================#
#------- Training -------#

# Load the original density data
Xrho = load(joinpath(FILEPATH, "data/original_data.jld2"))["X"]["rho"]

# Load the original velocities
Xmx = load(joinpath(FILEPATH, "data/original_data.jld2"))["X"]["mx"]
Xmy = load(joinpath(FILEPATH, "data/original_data.jld2"))["X"]["my"]
Xmz = load(joinpath(FILEPATH, "data/original_data.jld2"))["X"]["mz"]
Xvx = Xmx ./ Xrho
Xvy = Xmy ./ Xrho
Xvz = Xmz ./ Xrho

# Load the original magnetic field data
Xbx = load(joinpath(FILEPATH, "data/original_data.jld2"))["X"]["Bx"]
Xby = load(joinpath(FILEPATH, "data/original_data.jld2"))["X"]["By"]
Xbz = load(joinpath(FILEPATH, "data/original_data.jld2"))["X"]["Bz"]

## Load the necessary functions 
include(joinpath(FILEPATH, "analysis.jl"))

# Obtain the grids
xspan = ds.grid["x"][:]
yspan = ds.grid["y"][:]
zspan = ds.grid["z"][:]

# Compute grid spacing
dx = sum(xspan[2:end] - xspan[1:end-1]) / (length(xspan) - 1)
dy = sum(yspan[2:end] - yspan[1:end-1]) / (length(yspan) - 1)
dz = sum(zspan[2:end] - zspan[1:end-1]) / (length(zspan) - 1)

## Compute the horizontal velocity power spectrum of the original data 
k_orig = zeros(8)
Pv_orig = zeros(8)
Pb_orig = zeros(8)
for i in 1:n 
    Xvx_3d = reshape(view(Xvx, :, i), nx, ny, nz)
    Xvy_3d = reshape(view(Xvy, :, i), nx, ny, nz)
    Xvz_3d = reshape(view(Xvz, :, i), nx, ny, nz)

    Xbx_3d = reshape(view(Xbx, :, i), nx, ny, nz)
    Xby_3d = reshape(view(Xby, :, i), nx, ny, nz)
    Xbz_3d = reshape(view(Xbz, :, i), nx, ny, nz)

    k, Pv, Pb = mhd_energy_spectra(
        (Xvx_3d, Xvy_3d, Xvz_3d),
        (Xbx_3d, Xby_3d, Xbz_3d),
        dx, dy, dz, n_bins=8
    )
    k_orig += k
    Pv_orig += Pv
    Pb_orig += Pb
end
k_orig ./= n
Pv_orig ./= n
Pb_orig ./= n

## Free some memory 
Xvx = nothing; Xvy = nothing; Xvz = nothing
Xbx = nothing; Xby = nothing; Xbz = nothing
GC.gc()

## Load the training ROM simulation trajectory data
Xrom_train = load(joinpath(
    FILEPATH, "data/results/stream_rom_training_states.jld2"))["states"][:iqrrls]

## Load the basis
baker = load(joinpath(FILEPATH, "data/bases/baker_basis.jld2"))["baker"]
V = baker.Q

## Reproject the data back to full space
Xrom_train_full = V * Xrom_train

## Load the means and scales 
means = load(joinpath(FILEPATH, "data/mean.jld2"))["mean"]
scales = load(joinpath(FILEPATH, "data/minmax.jld2"))["scale"]
shifts = load(joinpath(FILEPATH, "data/minmax.jld2"))["shift"]

## Unprocess the data 
include("preprocess.jl")
Xrom_train_full = unprocess!(
    Xrom_train_full, vec(means["all"]), vec(shifts["all"]), vec(scales["all"])
)

## Extract the velocity
Xr_rom_train_full = Xrom_train_full[1:nxyz, :]
Xmx_rom_train_full = Xrom_train_full[2*nxyz+1:3*nxyz, :]
Xmy_rom_train_full = Xrom_train_full[3*nxyz+1:4*nxyz, :]
Xmz_rom_train_full = Xrom_train_full[4*nxyz+1:5*nxyz, :]
Xvx_rom_train_full = Xmx_rom_train_full ./ Xr_rom_train_full
Xvy_rom_train_full = Xmy_rom_train_full ./ Xr_rom_train_full
Xvz_rom_train_full = Xmz_rom_train_full ./ Xr_rom_train_full

# Extract the magnetic field
Xbx_rom_train_full = Xrom_train_full[5*nxyz+1:6*nxyz, :]
Xby_rom_train_full = Xrom_train_full[6*nxyz+1:7*nxyz, :]
Xbz_rom_train_full = Xrom_train_full[7*nxyz+1:8*nxyz, :]

## Compute the velocity and magnetic power spectrum of the ROM training data
k_rom_train = zeros(8)
Pv_rom_train = zeros(8)
Pb_rom_train = zeros(8)
for i in 1:n 
    Xvx_rom_train_3d = reshape(view(Xvx_rom_train_full, :, i), nx, ny, nz)
    Xvy_rom_train_3d = reshape(view(Xvy_rom_train_full, :, i), nx, ny, nz)
    Xvz_rom_train_3d = reshape(view(Xvz_rom_train_full, :, i), nx, ny, nz)

    Xbx_rom_train_3d = reshape(view(Xbx_rom_train_full, :, i), nx, ny, nz)
    Xby_rom_train_3d = reshape(view(Xby_rom_train_full, :, i), nx, ny, nz)
    Xbz_rom_train_3d = reshape(view(Xbz_rom_train_full, :, i), nx, ny, nz)

    k, Pv, Pb = mhd_energy_spectra(
        (Xvx_rom_train_3d, Xvy_rom_train_3d, Xvz_rom_train_3d),
        (Xbx_rom_train_3d, Xby_rom_train_3d, Xbz_rom_train_3d),
        dx, dy, dz, n_bins=8
    )
    k_rom_train += k
    Pv_rom_train += Pv
    Pb_rom_train += Pb
end
k_rom_train ./= n
Pv_rom_train ./= n
Pb_rom_train ./= n

## Free some memory
Xvx_rom_train_full = nothing; Xvy_rom_train_full = nothing; Xvz_rom_train_full = nothing
Xbx_rom_train_full = nothing; Xby_rom_train_full = nothing; Xbz_rom_train_full = nothing
GC.gc()


##-------- Test --------#
# Load the original density data
Xrho_test = load(joinpath(FILEPATH, "data/test_data.jld2"))["X"]["rho"]

# Load the test velocities
Xmx = load(joinpath(FILEPATH, "data/test_data.jld2"))["X"]["mx"]
Xmy = load(joinpath(FILEPATH, "data/test_data.jld2"))["X"]["my"]
Xmz = load(joinpath(FILEPATH, "data/test_data.jld2"))["X"]["mz"]
Xvx = Xmx ./ Xrho_test
Xvy = Xmy ./ Xrho_test
Xvz = Xmz ./ Xrho_test

# Load the test magnetic field data
Xbx = load(joinpath(FILEPATH, "data/test_data.jld2"))["X"]["Bx"]
Xby = load(joinpath(FILEPATH, "data/test_data.jld2"))["X"]["By"]
Xbz = load(joinpath(FILEPATH, "data/test_data.jld2"))["X"]["Bz"]

## Compute the horizontal velocity power spectrum of the original data 
k_orig_test = zeros(8)
Pv_orig_test = zeros(8)
Pb_orig_test = zeros(8)
for i in 1:size(Xvx, 2) 
    Xvx_3d = reshape(view(Xvx, :, i), nx, ny, nz)
    Xvy_3d = reshape(view(Xvy, :, i), nx, ny, nz)
    Xvz_3d = reshape(view(Xvz, :, i), nx, ny, nz)

    Xbx_3d = reshape(view(Xbx, :, i), nx, ny, nz)
    Xby_3d = reshape(view(Xby, :, i), nx, ny, nz)
    Xbz_3d = reshape(view(Xbz, :, i), nx, ny, nz)

    k, Pv, Pb = mhd_energy_spectra(
        (Xvx_3d, Xvy_3d, Xvz_3d),
        (Xbx_3d, Xby_3d, Xbz_3d),
        dx, dy, dz, n_bins=8
    )
    k_orig_test += k
    Pv_orig_test += Pv
    Pb_orig_test += Pb
end
k_orig_test ./= size(Xvx, 2)
Pv_orig_test ./= size(Xvx, 2)
Pb_orig_test ./= size(Xvx, 2)

## Free some memory 
Xvx = nothing; Xvy = nothing; Xvz = nothing
Xbx = nothing; Xby = nothing; Xbz = nothing
GC.gc()

## Load the testing ROM simulation trajectory data
Xrom_test = load(joinpath(
    FILEPATH, "data/results/stream_rom_testing_states.jld2"))["states"][:iqrrls]

# Reproject the data back to full space
Xrom_test_full = V * Xrom_test

## Unprocess the data
Xrom_test_full = unprocess!(
    Xrom_test_full, vec(means["all"]), vec(shifts["all"]), vec(scales["all"])
)

## Extract the velocity
Xr_rom_test_full = Xrom_test_full[1:nxyz, :]
Xmx_rom_test_full = Xrom_test_full[2*nxyz+1:3*nxyz, :]
Xmy_rom_test_full = Xrom_test_full[3*nxyz+1:4*nxyz, :]
Xmz_rom_test_full = Xrom_test_full[4*nxyz+1:5*nxyz, :]
Xvx_rom_test_full = Xmx_rom_test_full ./ Xr_rom_test_full
Xvy_rom_test_full = Xmy_rom_test_full ./ Xr_rom_test_full
Xvz_rom_test_full = Xmz_rom_test_full ./ Xr_rom_test_full
# Extract the magnetic field
Xbx_rom_test_full = Xrom_test_full[5*nxyz+1:6*nxyz, :]
Xby_rom_test_full = Xrom_test_full[6*nxyz+1:7*nxyz, :]
Xbz_rom_test_full = Xrom_test_full[7*nxyz+1:8*nxyz, :]

## Compute the velocity and magnetic power spectrum of the ROM testing data
k_rom_test = zeros(8)
Pv_rom_test = zeros(8)
Pb_rom_test = zeros(8)
for i in 1:size(Xrom_test_full, 2) 
    Xvx_rom_test_3d = reshape(view(Xvx_rom_test_full, :, i), nx, ny, nz)
    Xvy_rom_test_3d = reshape(view(Xvy_rom_test_full, :, i), nx, ny, nz)
    Xvz_rom_test_3d = reshape(view(Xvz_rom_test_full, :, i), nx, ny, nz)

    Xbx_rom_test_3d = reshape(view(Xbx_rom_test_full, :, i), nx, ny, nz)
    Xby_rom_test_3d = reshape(view(Xby_rom_test_full, :, i), nx, ny, nz)
    Xbz_rom_test_3d = reshape(view(Xbz_rom_test_full, :, i), nx, ny, nz)

    k, Pv, Pb = mhd_energy_spectra(
        (Xvx_rom_test_3d, Xvy_rom_test_3d, Xvz_rom_test_3d),
        (Xbx_rom_test_3d, Xby_rom_test_3d, Xbz_rom_test_3d),
        dx, dy, dz, n_bins=8
    )
    k_rom_test += k
    Pv_rom_test += Pv
    Pb_rom_test += Pb
end
k_rom_test ./= size(Xrom_test_full, 2)
Pv_rom_test ./= size(Xrom_test_full, 2)
Pb_rom_test ./= size(Xrom_test_full, 2)

## Free some memory
Xvx_rom_test_full = nothing; Xvy_rom_test_full = nothing; Xvz_rom_test_full = nothing
Xbx_rom_test_full = nothing; Xby_rom_test_full = nothing; Xbz_rom_test_full = nothing
GC.gc()

## Save results 
save(joinpath(FILEPATH, "data/results/power_spectrum.jld2"), 
     "k_orig_train", k_orig, "Pv_orig_train", Pv_orig, "Pb_orig_train", Pb_orig,
     "k_rom_train", k_rom_train, "Pv_rom_train", Pv_rom_train, "Pb_rom_train", Pb_rom_train,
     "k_orig_test", k_orig_test, "Pv_orig_test", Pv_orig_test, "Pb_orig_test", Pb_orig_test,
     "k_rom_test", k_rom_test, "Pv_rom_test", Pv_rom_test, 
     "Pb_rom_test", Pb_rom_test)

#====================#
## Compute the 3PCF ##
#====================#
using Statistics
# Compute the density fluctuation
density_fluct = (X) -> (log.(X) .- mean(log.(X), dims=2)) ./ std(log.(X), dims=2)

which_traj = 1
i1 = (which_traj - 1) * n_time + 1
i2 = which_traj * n_time

Xrho_fluct = density_fluct(Xrho[:, i1:i2])
Xrho_test_fluct = density_fluct(Xrho_test[:, i1:i2])
Xrho_rom_train_fluct = density_fluct(Xr_rom_train_full[:, i1:i2])
Xrho_rom_test_fluct = density_fluct(Xr_rom_test_full[:, i1:i2])

## Free some memory
Xrho = nothing; Xrho_test = nothing
Xr_rom_train_full = nothing; Xr_rom_test_full = nothing
GC.gc()

## Initialize the grid values array
grid_vals = zeros(nxyz, 4)
idx = 1
for k in 1:nz, j in 1:ny, i in 1:nx 
    grid_vals[idx, 1] = ds.grid["x"][i]
    grid_vals[idx, 2] = ds.grid["y"][j]
    grid_vals[idx, 3] = ds.grid["z"][k]
    idx += 1
end

## Load some packages for parallel computing
using Distributed 
addprocs(150)
using NPCFs
@everywhere using NPCFs

## Initialize the 3PCF object
npcf3 = NPCFs.NPCF(
    N=3, D=3, periodic=true, volume=1.0^3, verb=true,
    coords="cartesian", r_min=0.1, r_max=0.4, nbins=8, lmax=5,
    complete=true
)

## Compute the grid values assembled as [x, y, z, fluctuation]
time_idx = [5]
for tidx in time_idx
    # Original data
    grid_vals[:, 4] .= vec(Xrho_fluct[:, tidx])
    t1 = time()
    npcf3_orig = NPCFs.compute_npcf_pairwise_complete(grid_vals, npcf3)
    t2 = time()
    @info "3PCF for original data done. Took $(t2 - t1) seconds"

    # ROM training data
    grid_vals[:, 4] .= vec(Xrho_rom_train_fluct[ :, tidx])
    t1 = time()
    npcf3_rom_train = NPCFs.compute_npcf_pairwise_complete(grid_vals, npcf3)
    t2 = time()
    @info "3PCF for ROM training data done. Took $(t2 - t1) seconds"

    ## Oiriginal test data 
    grid_vals[:, 4] .= vec(Xrho_test_fluct[:, tidx])
    t1 = time()
    npcf3_orig_test = NPCFs.compute_npcf_pairwise_complete(grid_vals, npcf3)
    t2 = time()
    @info "3PCF for original testing data done. Took $(t2 - t1) seconds"

    # ROM testing data
    grid_vals[:, 4] .= vec(Xrho_rom_test_fluct[:, tidx])
    t1 = time()
    npcf3_rom_test = NPCFs.compute_npcf_pairwise_complete(grid_vals, npcf3)
    t2 = time()
    @info "3PCF for ROM testing data done. Took $(t2 - t1) seconds"

    # Free some memory 
    GC.gc()

    # Save
    save(joinpath(FILEPATH, "data/results/3pcf_t$(tidx).jld2"), 
         "npcf3_orig", npcf3_orig,
         "npcf3_rom_train", npcf3_rom_train,
         "npcf3_orig_test", npcf3_orig_test,
         "npcf3_rom_test", npcf3_rom_test)

    @info "Finished 3PCF computation for time $(tidx)"
end
