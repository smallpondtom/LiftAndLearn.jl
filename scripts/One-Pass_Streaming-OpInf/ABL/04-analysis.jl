"""
ABL: Compute the QoIs of the reduced model
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
DATAPATH = "../../../../../DATA/NREL/ABL"
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"One-Pass_Streaming-OpInf/ABL") : 
           joinpath(pwd(), "scripts/One-Pass_Streaming-OpInf/ABL")
fn = "ABL_0_10000.h5"
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
rmax = 200

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
COMPUTE_ORIGINAL = true
include(joinpath(FILEPATH, "preprocess.jl"))
include(joinpath(FILEPATH, "qoi.jl"))

# horizontal wind speed profile 
wprof_train = zeros(nz, n_train)
wang_train = zeros(nz, n_train)
wprof_train_rom = zeros(nz, n_train)
wang_train_rom = zeros(nz, n_train)
zcoord = ds["z"][:]

@showprogress for i in 1:n_train
    if COMPUTE_ORIGINAL
        qois = get_qois(ds, i)
        wprof_train[:, i] = qois.wprof
        wang_train[:, i] = qois.wang
    end
    qois_rom = get_qois_rom(states, iVrmax, zcoord, 
                            means, shifts, scales, ds.dims, i)
    wprof_train_rom[:, i] = qois_rom.wprof
    wang_train_rom[:, i] = qois_rom.wang
end

## Save the QoIs for training 
if COMPUTE_ORIGINAL
    save(joinpath(FILEPATH, "data/results/original_qois_train.jld2"), 
         "wprof", wprof_train, "wang", wang_train)
end
save(joinpath(FILEPATH, 
     "data/results/stream_rom_train_qois_0_8000_r$(rmax).jld2"), 
     "wprof", wprof_train, "wang", wang_train,
     "wprof_rom", wprof_train_rom, "wang_rom", wang_train_rom)

#=================================#
## Load the testing reduced data
#=================================#
test_states = load(joinpath(FILEPATH, "data/results", 
             "stream_rom_test_sim_states_0_8000_r$(rmax).jld2")
             )["states"]

#====================================================#
## Compute the QoIs of the reduced model (test)
#====================================================#
wprof_test = zeros(nz, n_test)
wang_test = zeros(nz, n_test)
wprof_test_rom = zeros(nz, n_test)
wang_test_rom = zeros(nz, n_test)
@showprogress for i in 1:n_test
    if COMPUTE_ORIGINAL
        qois = get_qois(ds, n_train + i)
        wprof_test[:, i] = qois.wprof
        wang_test[:, i] = qois.wang
    end
    qois_rom = get_qois_rom(test_states, iVrmax, zcoord, 
                            means, shifts, scales, ds.dims, i)
    wprof_test_rom[:, i] = qois_rom.wprof
    wang_test_rom[:, i] = qois_rom.wang
end

## Save the QoIs for testing
if COMPUTE_ORIGINAL
    save(joinpath(FILEPATH, "data/results/original_qois_test.jld2"), 
         "wprof", wprof_test, "wang", wang_test)
end
save(joinpath(FILEPATH, 
     "data/results/stream_rom_test_qois_0_8000_r$(rmax).jld2"), 
     "wprof", wprof_test, "wang", wang_test,
     "wprof_rom", wprof_test_rom, "wang_rom", wang_test_rom)

