"""
3D Channel: plotting results
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


#=================#
## Plot the QoIs
#=================#
qois_train = load(joinpath(FILEPATH, 
      "data/results/$(batch_or_stream)_rom_train_qois_0_8000_r$(rmax).jld2"))
qois_test = load(joinpath(FILEPATH, 
      "data/results/$(batch_or_stream)_rom_test_qois_0_8000_r$(rmax).jld2"))


#===========================================#
## Extract data from loaded QoI files
#===========================================#
zprof_train = qois_train["zprof"]
utau_train = qois_train["utau"]
zprof_train_rom = qois_train["zprof_rom"]
utau_train_rom = qois_train["utau_rom"]

zprof_test = qois_test["zprof"]
utau_test = qois_test["utau"]
zprof_test_rom = qois_test["zprof_rom"]
utau_test_rom = qois_test["utau_rom"]

# Get z coordinates
z_coords = ds["z"][:]

#===========================================#
## Plot 1: Z Profile - Training Data
#===========================================#
with_theme(theme_latexfonts()) do 
    fig1 = Figure(size=(1200, 600))
    # Left subplot: Original vs ROM z profiles
    ax1_left = Axis(fig1[1, 1], 
        xlabel = "Z Coordinate",
        ylabel = L"Time-Averaged $u_z$ velocity",
        xticks=round.(vcat(0, ds["z"][8:8:nz]), digits=3), 
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )

    # Plot mean profiles over time
    zprof_train_mean = mean(zprof_train, dims=2)[:]
    zprof_train_rom_mean = mean(zprof_train_rom, dims=2)[:]

    l1 = lines!(ax1_left, z_coords, zprof_train_mean, label="Original", linewidth=6)
    l2 = lines!(ax1_left, z_coords, zprof_train_rom_mean, label="ROM", linewidth=6, 
                linestyle=:dash)
    axislegend(ax1_left, position=:rb, labelsize=28, patchsize=(80, 30))

    # Right subplot: Error in z profile
    ax1_right = Axis(fig1[1, 2],
        xlabel = "Z Coordinate",
        ylabel = "Relative Error",
        xticks=round.(vcat(0, ds["z"][8:8:nz]), digits=3), 
        yscale=log10,
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )

    zprof_error_train = abs.(zprof_train_rom_mean - zprof_train_mean) ./ 
                            abs.(zprof_train_mean)
    lines!(ax1_right, z_coords, zprof_error_train, color=:red, linewidth=6)

    # Add super title
    Label(fig1[0, :], 
        text=L"Time-Averaged $u_z$ Profile (left) and Relative Errors (right) for Training Data", 
        fontsize = 30)

    display(fig1)
    # save(joinpath(FILEPATH, 
    #     "plots/$(batch_or_stream)_zprofile_training_comparison.png"), fig1)
end

#===========================================#
## Plot 2: Z Profile - Testing Data
#===========================================#
with_theme(theme_latexfonts()) do 
    fig2 = Figure(size=(1200, 600))

    # Left subplot: Original vs ROM z profiles
    ax2_left = Axis(fig2[1, 1], 
        xlabel = "Z Coordinate",
        ylabel = L"Time-Averaged $u_z$ velocity",
        xticks=round.(vcat(0, ds["z"][8:8:nz]), digits=3), 
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )

    # Plot mean profiles over time
    zprof_test_mean = mean(zprof_test, dims=2)[:]
    zprof_test_rom_mean = mean(zprof_test_rom, dims=2)[:]

    lines!(ax2_left, z_coords, zprof_test_mean, label="Original", linewidth=6)
    lines!(ax2_left, z_coords, zprof_test_rom_mean, label="ROM", linewidth=6, linestyle=:dash)
    axislegend(ax2_left, position=:rb, labelsize=28, patchsize=(80, 30))

    # Right subplot: Error in z profile
    ax2_right = Axis(fig2[1, 2],
        xlabel = "Z Coordinate",
        ylabel = "Relative Error",
        xticks=round.(vcat(0, ds["z"][8:8:nz]), digits=3), 
        yscale=log10,
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )

    zprof_error_test = abs.(zprof_test_rom_mean - zprof_test_mean) ./ 
                        abs.(zprof_test_mean)
    lines!(ax2_right, z_coords, zprof_error_test, color=:red, linewidth=6)

    # Add super title
    Label(fig2[0, :], 
        text=L"Time-Averaged $u_z$ Profile (left) and Relative Errors (right) for Testing Data", 
        fontsize = 30)

    display(fig2)
    # save(joinpath(FILEPATH, 
    #     "plots/$(batch_or_stream)_zprofile_testing_comparison.png"), fig2)
end


#===========================================#
## Plot 3: Wall Shear Flow - Training Data
#===========================================#
with_theme(theme_latexfonts()) do 
    fig3 = Figure(size=(1200, 600))

    # Left subplot: Original vs ROM wall shear flow
    ax3_left = Axis(fig3[1, 1], 
        xlabel = "Time Step",
        ylabel = L"$u_\tau$",
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )

    time_steps_train = 1:n_train
    lines!(ax3_left, time_steps_train, utau_train, label="Original", linewidth=3)
    lines!(ax3_left, time_steps_train, utau_train_rom, label="ROM", linewidth=3, 
           linestyle=:dash)
    axislegend(ax3_left, position=:rt, labelsize=28, patchsize=(80, 30))

    # Right subplot: Error in wall shear flow
    ax3_right = Axis(fig3[1, 2],
        xlabel = "Time Step",
        ylabel = "Relative Error",
        yscale=log10,
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )

    utau_error_train = abs.(utau_train_rom - utau_train) ./ abs.(utau_train)
    lines!(ax3_right, time_steps_train, utau_error_train, color=:red, linewidth=3)

    # Add super title
    Label(fig3[0, :], 
        text=L"Wall Shear Flow $u_\tau$ (left) and Relative Errors (right) for Training Data", 
        fontsize = 30)

    display(fig3)
    # save(joinpath(FILEPATH, 
    #     "plots/$(batch_or_stream)_utau_training_comparison.png"), fig3)
end


#===========================================#
## Plot 4: Wall Shear Flow - Testing Data
#===========================================#
with_theme(theme_latexfonts()) do 
    fig4 = Figure(size=(1200, 600))

    # Left subplot: Original vs ROM wall shear flow
    ax4_left = Axis(fig4[1, 1], 
        xlabel = "Time Step",
        ylabel = L"$u_\tau$",
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )

    time_steps_test = 1:n_test
    lines!(ax4_left, time_steps_test, utau_test, label="Original", linewidth=3)
    lines!(ax4_left, time_steps_test, utau_test_rom, label="ROM", linewidth=3, 
           linestyle=:dash)
    axislegend(ax4_left, position=:rb, labelsize=28, patchsize=(80, 30))

    # Right subplot: Error in wall shear flow
    ax4_right = Axis(fig4[1, 2],
        xlabel = "Time Step",
        ylabel = "Relative Error",
        yscale=log10,
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )

    utau_error_test = abs.(utau_test_rom - utau_test) ./ abs.(utau_test)
    lines!(ax4_right, time_steps_test, utau_error_test, color=:red, linewidth=3)

    # Add super title
    Label(fig4[0, :], 
        text=L"Wall Shear Flow $u_\tau$ (left) and Relative Errors (right) for Testing Data", 
        fontsize = 30)

    display(fig4)
    # save(joinpath(FILEPATH, 
    #     "plots/$(batch_or_stream)_utau_testing_comparison.png"), fig4)
end