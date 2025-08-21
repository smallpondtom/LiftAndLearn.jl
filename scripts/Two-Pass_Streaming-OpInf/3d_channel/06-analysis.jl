"""
3D Channel flow: Compute the QoIs of the reduced model
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using LinearAlgebra
using BlockDiagonals
using SparseArrays
using Statistics
using Revise
import LiftAndLearn as LnL

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


#=============================#
## Load the training dataset
#=============================#
ds = ChannelDataSource(datafile, ["z", "y", "x", "fields", "times"])
nz, ny, nx, n_fields, n_time = ds.dims
nxyz = nz * ny * nx
n_test = 2000
n_train = n_time - n_test

#===================#
## Setup the options
#===================#
batch_or_stream = "batch"
rmax = 200
rls_algo = :iqrrls


#=================#
## Load the basis 
#=================#
# Standard basis 
basis_file = joinpath(FILEPATH, "data/bases/basis_0_8000_r400.jld2")
iVrmax = load(basis_file)["bases"]["baker"].iVr[:, 1:rmax]


#============================#
## Load the mean and scaling
#============================#
means  = load(joinpath(FILEPATH, "data/mean.jld2"))["xbar"]
shifts = load(joinpath(FILEPATH, "data/minmax.jld2"))["minmax"]["shifts"]
scales = load(joinpath(FILEPATH, "data/minmax.jld2"))["minmax"]["scales"]


#=================================#
## Load the training reduced data
#=================================#
states = load(joinpath(FILEPATH, "data/results", 
             "$(batch_or_stream)_rom_train_sim_states_0_8000_r$(rmax).jld2")
             )["states"]
if batch_or_stream == "stream"
    states = states[rls_algo]
end

#====================================================#
## Compute the QoIs of the reduced model (training)
#====================================================#
include(joinpath(FILEPATH, "preprocess.jl"))
include(joinpath(FILEPATH, "qoi.jl"))
zprof_train = zeros(nz, n_train)
utau_train = zeros(n_train)
zprof_train_rom = zeros(nz, n_train)
utau_train_rom = zeros(n_train)
Threads.@threads for i in 1:n_train
    qois = get_qois(ds, i)
    qois_rom = get_qois_rom(states, iVrmax, ds["z"][:], 
                            means, shifts, scales, ds.dims, i)
    zprof_train[:, i] = qois.zprof
    utau_train[i] = qois.utau
    zprof_train_rom[:, i] = qois_rom.zprof
    utau_train_rom[i] = qois_rom.utau
end

## Save the QoIs for training 
save(joinpath(FILEPATH, 
     "data/results/$(batch_or_stream)_rom_train_qois_0_8000_r$(rmax).jld2"), 
     "zprof", zprof_train, "utau", utau_train,
     "zprof_rom", zprof_train_rom, "utau_rom", utau_train_rom)

#=================================#
## Load the testing reduced data
#=================================#
test_states = load(joinpath(FILEPATH, "data/results", 
             "$(batch_or_stream)_rom_test_sim_states_0_8000_r$(rmax).jld2")
             )["states"]

#====================================================#
## Compute the QoIs of the reduced model (test)
#====================================================#
zprof_test = zeros(nz, n_test)
utau_test = zeros(n_test)
zprof_test_rom = zeros(nz, n_test)
utau_test_rom = zeros(n_test)
Threads.@threads for i in 1:n_test
    qois = get_qois(ds, n_train + i)
    qois_rom = get_qois_rom(test_states, iVrmax, ds["z"][:], 
                            means, shifts, scales, ds.dims, i)
    zprof_test[:, i] = qois.zprof
    utau_test[i] = qois.utau
    zprof_test_rom[:, i] = qois_rom.zprof
    utau_test_rom[i] = qois_rom.utau
end

## Save the QoIs for testing
save(joinpath(FILEPATH, 
     "data/results/$(batch_or_stream)_rom_test_qois_0_8000_r$(rmax).jld2"), 
     "zprof", zprof_test, "utau", utau_test,
     "zprof_rom", zprof_test_rom, "utau_rom", utau_test_rom)