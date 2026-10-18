"""
ABL: plotting results
"""

#================#
## Load Packages
#================#
using CairoMakie
using FileIO
using JLD2
using LinearAlgebra
using Statistics
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
batch_or_stream = "stream"
rmax = 200


#=================#
## Plot the QoIs
#=================#
if batch_or_stream == "batch"
    qois_train = load(joinpath(FILEPATH, 
        "data/results/$(batch_or_stream)_rom_train_qois_0_8000_r$(rmax).jld2"))
    qois_test = load(joinpath(FILEPATH, 
        "data/results/$(batch_or_stream)_rom_test_qois_0_8000_r$(rmax).jld2"))
else
    qois_train = load(joinpath(FILEPATH, 
        "data/results/$(batch_or_stream)_rom_train_qois_0_8000_r$(rmax).jld2"))
    qois_test = load(joinpath(FILEPATH, 
        "data/results/$(batch_or_stream)_rom_test_qois_0_8000_r$(rmax).jld2"))
end
original_qois_train = load(joinpath(FILEPATH, 
    "data/results/original_qois_train.jld2"))

original_qois_test = load(joinpath(FILEPATH, 
    "data/results/original_qois_test.jld2"))


#===========================================#
## Extract data from loaded QoI files
#===========================================#
# train
wprof_train = original_qois_train["wprof"]
wang_train = original_qois_train["wang"]
wprof_train_rom = qois_train["wprof_rom"]
wang_train_rom = qois_train["wang_rom"]
# test
wprof_test = original_qois_test["wprof"]
wang_test = original_qois_test["wang"]
wprof_test_rom = qois_test["wprof_rom"]
wang_test_rom = qois_test["wang_rom"]

# Get z coordinates
z_coords = ds["z"][:]

## Compute the temporal mean of the QoIs
# wprof_train_mean = mean(wprof_train, dims=2) |> vec
# wang_train_mean = mean(wang_train, dims=2) |> vec
# wprof_train_rom_mean = mean(wprof_train_rom, dims=2) |> vec
# wang_train_rom_mean = mean(wang_train_rom, dims=2) |> vec

# wprof_test_mean = mean(wprof_test, dims=2) |> vec
# wang_test_mean = mean(wang_test, dims=2) |> vec
# wprof_test_rom_mean = mean(wprof_test_rom, dims=2) |> vec
# wang_test_rom_mean = mean(wang_test_rom, dims=2) |> vec

idx = 1500
wprof_train_mean = wprof_train[:, idx]
wang_train_mean =  wang_train[:, idx]
wprof_train_rom_mean = wprof_train_rom[:, idx]
wang_train_rom_mean = wang_train_rom[:, idx]

wprof_test_mean = wprof_test[:, idx]
wang_test_mean = wang_test[:, idx]
wprof_test_rom_mean = wprof_test_rom[:, idx]
wang_test_rom_mean = wang_test_rom[:, idx]


## Wind profile and angle
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1200, 800))

    # Wind horizontal speed profile (training)
    ax11 = Axis(fig[1, 1], 
        xlabel = L"$\langle \sqrt{ \tilde{u}_1^2 + \tilde{u}_2^2} \rangle$",
        ylabel = L"$z$ coordinate",
        title = "Training",
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )
    lines!(ax11, wprof_train_mean, z_coords, label="Original", linewidth=8, linestyle=:dash)
    lines!(ax11, wprof_train_rom_mean, z_coords, label="ROM", linewidth=3)
    axislegend(ax11, position=:lt, labelsize=25, patchsize=(120, 20))

    # Wind horizontal speed profile (testing)
    ax12 = Axis(fig[1, 2], 
        xlabel = L"$\langle \sqrt{\tilde{u}_1^2 + \tilde{u}_2^2} \rangle$",
        title = "Testing",
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )
    lines!(ax12, wprof_test_mean, z_coords, linewidth=8, linestyle=:dash)
    lines!(ax12, wprof_test_rom_mean, z_coords, linewidth=3)

    # Wind angle profile (training)
    ax21 = Axis(fig[2, 1], 
        xlabel = L"$\varphi = \langle \mathrm{atan2}(\tilde{u}_2,\tilde{u}_1) \rangle$ (deg)",
        ylabel = L"$z$ coordinate",
        title = "Training",
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )
    lines!(ax21, 
        wang_train_mean .- wang_train_mean[end], 
        z_coords, label="Original", linewidth=8, linestyle=:dash
    )
    lines!(ax21, 
        wang_train_rom_mean .- wang_train_rom_mean[end], 
        z_coords, label="ROM", linewidth=3
    )

    # Wind angle profile (testing)
    ax22 = Axis(fig[2, 2], 
        xlabel = L"$\varphi = \langle \mathrm{atan2}(\tilde{u}_2,\tilde{u}_1) \rangle$ (deg)",
        title = "Testing",
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )
    lines!(ax22, 
        wang_test_mean .- wang_test_mean[end],
        z_coords, linewidth=8, linestyle=:dash
    )
    lines!(ax22, 
        wang_test_rom_mean .- wang_test_rom_mean[end],
        z_coords, linewidth=3
    )

    save(joinpath(FILEPATH, "plots/$(batch_or_stream)_rom_qois_r$(rmax).pdf"), fig)
    display(fig)
end

