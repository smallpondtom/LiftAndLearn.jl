"""
3D Channel flow: Compute the QoIs of the reduced model
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using LinearAlgebra
using SparseArrays
using ProgressMeter
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
DATAPATH = "../../../../../DATA/NREL/3D_CHANNEL"
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"One-Pass_Streaming-OpInf/3d_channel") : 
           joinpath(pwd(), "scripts/One-Pass_Streaming-OpInf/3d_channel")
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
rmax = 300

#=================#
## Load the basis 
#=================#
basis_file = joinpath(FILEPATH, "data/results/onepass_stream.jld2")
iVrmax = load(basis_file)["stream"].V[:, 1:rmax]


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
             "stream_rom_train_sim_states_0_8000_r$(rmax).jld2")
             )["states"]

#====================================================#
## Compute the QoIs of the reduced model (training)
#====================================================#
COMPUTE_ORIGINAL = false
include(joinpath(FILEPATH, "preprocess.jl"))
include(joinpath(FILEPATH, "qoi.jl"))
zprof_train = zeros(nz, n_train)
utau_train = zeros(n_train)
zprof_train_rom = zeros(nz, n_train)
utau_train_rom = zeros(n_train)
zcoord = ds["z"][:]
@showprogress for i in 1:n_train
    if COMPUTE_ORIGINAL
        qois = get_qois(ds, i)
        zprof_train[:, i] = qois.zprof
        utau_train[i] = qois.utau
    end
    qois_rom = get_qois_rom(states, iVrmax, zcoord, 
                            means, shifts, scales, ds.dims, i)
    zprof_train_rom[:, i] = qois_rom.zprof
    utau_train_rom[i] = qois_rom.utau
end

## Save the QoIs for training 
if COMPUTE_ORIGINAL
    save(joinpath(FILEPATH, "data/results/original_qois_train.jld2"), 
         "zprof", zprof_train, "utau", utau_train)
end
save(joinpath(FILEPATH, 
     "data/results/stream_rom_train_qois_0_8000_r$(rmax).jld2"), 
     "zprof", zprof_train, "utau", utau_train,
     "zprof_rom", zprof_train_rom, "utau_rom", utau_train_rom)

#=================================#
## Load the testing reduced data
#=================================#
test_states = load(joinpath(FILEPATH, "data/results", 
             "stream_rom_test_sim_states_0_8000_r$(rmax).jld2")
             )["states"]

#====================================================#
## Compute the QoIs of the reduced model (test)
#====================================================#
zprof_test = zeros(nz, n_test)
utau_test = zeros(n_test)
zprof_test_rom = zeros(nz, n_test)
utau_test_rom = zeros(n_test)
@showprogress for i in 1:n_test
    if COMPUTE_ORIGINAL
        qois = get_qois(ds, n_train + i)
        zprof_test[:, i] = qois.zprof
        utau_test[i] = qois.utau
    end
    qois_rom = get_qois_rom(test_states, iVrmax, zcoord, 
                            means, shifts, scales, ds.dims, i)
    zprof_test_rom[:, i] = qois_rom.zprof
    utau_test_rom[i] = qois_rom.utau
end

## Save the QoIs for testing
if COMPUTE_ORIGINAL
    save(joinpath(FILEPATH, "data/results/original_qois_test.jld2"), 
         "zprof", zprof_test, "utau", utau_test)
end
save(joinpath(FILEPATH, 
     "data/results/stream_rom_test_qois_0_8000_r$(rmax).jld2"), 
     "zprof", zprof_test, "utau", utau_test,
     "zprof_rom", zprof_test_rom, "utau_rom", utau_test_rom)
