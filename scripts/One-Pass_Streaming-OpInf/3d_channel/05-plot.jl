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
batch_or_stream = "stream"
rmax = 300


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
zprof_train = original_qois_train["zprof"]
utau_train = original_qois_train["utau"]
zprof_train_rom = qois_train["zprof_rom"]
utau_train_rom = qois_train["utau_rom"]
# test
zprof_test = original_qois_test["zprof"]
utau_test = original_qois_test["utau"]
zprof_test_rom = qois_test["zprof_rom"]
utau_test_rom = qois_test["utau_rom"]

# Get z coordinates
z_coords = ds["z"][:]


#=======================================================#
## Plot 1: Z Profile Error
#=======================================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1400, 600))
    
    # Calculate errors first to determine common y-axis range
    zprof_error_train = norm.(eachcol(zprof_train_rom - zprof_train), 2) ./ 
                        norm.(eachcol(zprof_train), 2)
    zprof_error_test = norm.(eachcol(zprof_test_rom - zprof_test), 2) ./ 
                       norm.(eachcol(zprof_test), 2)
    
    # Determine common y-axis limits
    y_min = min(minimum(zprof_error_train), minimum(zprof_error_test))
    y_max = max(maximum(zprof_error_train), maximum(zprof_error_test))
    y_limits = (y_min * 0.9, y_max * 1.1)  # Add some padding
    
    # Left subplot: Training error
    c1, c2 = Makie.wong_colors()[1:2]
    ax1 = Axis(fig[1, 1], 
        xlabel = L"Time, $s$",
        ylabel = "Relative Error",
        title = "Training",
        yscale = log10,
        limits = (nothing, nothing, y_limits...),
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )

    # Right subplot: Testing error
    ax2 = Axis(fig[1, 2], 
        xlabel = L"Time, $s$",
        ylabel = "Relative Error",
        title = "Testing",
        yscale = log10,
        limits = (nothing, nothing, y_limits...),
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )

    # Time spans
    tspan_train = ds["times"][1:n_train] .- ds["times"][1]
    tspan_test = ds["times"][n_train+1:n_train+n_test] .- ds["times"][n_train+1]
    
    # Add lines
    l1 = lines!(ax1, tspan_train, zprof_error_train, 
                linewidth=3, color=c1)
    l2 = lines!(ax2, tspan_test, zprof_error_test, 
                linewidth=3, color=c2)
    
    # Add super title
    Label(fig[0, :], 
        text=L"Error of Wall Normal Profile of $u$", 
        fontsize = 32)

    display(fig)
    # save(joinpath(FILEPATH, 
    #     "plots/$(batch_or_stream)_zprofile_error_r$(rmax).png"), fig)
end

#===========================================#
## Plot 2: Z Profile heatmap
#===========================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1800, 1200))
    
    # First row - Training data
    # Left subplot: Original training data heatmap
    ax1 = Axis(fig[1, 1], 
        ylabel = "Z Coordinate",
        title = "Original (Training)",
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )
    
    # Center subplot: ROM training data heatmap
    ax2 = Axis(fig[1, 2], 
        title = "Streaming-OpInf (Training)",
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )
    
    # Right subplot: Training error heatmap
    ax3 = Axis(fig[1, 4], 
        title = "Pointwise Abs. Error (Training)",
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )
    
    # Second row - Testing data
    # Left subplot: Original testing data heatmap
    ax4 = Axis(fig[2, 1], 
        xlabel = L"Time, $s$",
        ylabel = "Z Coordinate",
        title = "Original (Testing)",
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )
    
    # Center subplot: ROM testing data heatmap
    ax5 = Axis(fig[2, 2], 
        xlabel = L"Time, $s$",
        title = "Streaming-OpInf (Testing)",
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )
    
    # Right subplot: Testing error heatmap
    ax6 = Axis(fig[2, 4], 
        xlabel = L"Time, $s$",
        title = "Pointwise Abs. Error (Testing)",
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )
    
    # Get time spans
    tspan_train = ds["times"][1:n_train] .- ds["times"][1]
    tspan_test = ds["times"][n_train+1:n_train+n_test] .- ds["times"][n_train+1]

    # Calculate global data range for uniform scaling across all velocity plots
    data_range = (min(minimum(zprof_train), minimum(zprof_train_rom),
                      minimum(zprof_test), minimum(zprof_test_rom)), 
                  max(maximum(zprof_train), maximum(zprof_train_rom),
                      maximum(zprof_test), maximum(zprof_test_rom)))
    
    # Calculate global error range for uniform error scaling
    zprof_error_train = abs.(zprof_train_rom - zprof_train)
    zprof_error_test = abs.(zprof_test_rom - zprof_test)
    error_range = (0, max(maximum(zprof_error_train), maximum(zprof_error_test)))
    
    # Create training heatmaps
    hm1 = heatmap!(ax1, tspan_train, z_coords, zprof_train', colorrange=data_range)
    hm2 = heatmap!(ax2, tspan_train, z_coords, zprof_train_rom', colorrange=data_range)
    hm3 = heatmap!(ax3, tspan_train, z_coords, zprof_error_train', 
                   colormap=:matter, colorrange=error_range)
    
    # Create testing heatmaps
    hm4 = heatmap!(ax4, tspan_test, z_coords, zprof_test', colorrange=data_range)
    hm5 = heatmap!(ax5, tspan_test, z_coords, zprof_test_rom', colorrange=data_range)
    hm6 = heatmap!(ax6, tspan_test, z_coords, zprof_error_test', 
                   colormap=:matter, colorrange=error_range)
    
    # Add colorbars
    # Shared colorbar for velocity data (between columns 2 and 4)
    cb1 = Colorbar(fig[1:2, 3], hm2, label = L"$u$ velocity", 
                   labelsize=25, ticklabelsize=20)
    
    # Colorbar for error plots
    cb2 = Colorbar(fig[1:2, 5], hm3, label = "Absolute Error", 
                   labelsize=25, ticklabelsize=20)
    
    display(fig)
    save(joinpath(FILEPATH, 
        "plots/$(batch_or_stream)_zprofile_heatmap_r$(rmax).png"), fig)
end


#===========================================#
## Plot 3: Wall Shear Flow
#===========================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(2000, 1200))

    # normalize utau values for better visualization
    utau_train_norm = utau_train ./ mean(utau_train)
    utau_train_rom_norm = utau_train_rom ./ mean(utau_train_rom)
    utau_test_norm = utau_test ./ mean(utau_test)
    utau_test_rom_norm = utau_test_rom ./ mean(utau_test_rom)

    # Calculate errors and data ranges for consistent scaling across all subplots
    utau_error_train = abs.(utau_train_rom_norm - utau_train_norm) ./ abs.(utau_train_norm)
    utau_error_test = abs.(utau_test_rom_norm - utau_test_norm) ./ abs.(utau_test_norm)
    
    # Common y-axis limits for utau values
    utau_min = min(minimum(utau_train_norm), minimum(utau_train_rom_norm), 
                   minimum(utau_test_norm), minimum(utau_test_rom_norm))
    utau_max = max(maximum(utau_train_norm), maximum(utau_train_rom_norm),
                   maximum(utau_test_norm), maximum(utau_test_rom_norm))
    utau_limits = (utau_min * 0.999, utau_max * 1.001)
    
    # Common y-axis limits for error values
    error_min = min(minimum(utau_error_train), minimum(utau_error_test))
    error_max = max(maximum(utau_error_train), maximum(utau_error_test))
    error_limits = (error_min * 0.5, error_max * 2.0)

    c1, c2 = Makie.wong_colors()[1:2]

    # First row - Training data
    # Left subplot: Training utau values
    ax1 = Axis(fig[1, 1], 
        ylabel = L"$v_{x,\tau} \,/\, \langle v_{x,\tau} \rangle$",
        title = "Training Data",
        limits = (nothing, nothing, utau_limits...),
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )

    # Right subplot: Training error
    ax2 = Axis(fig[1, 2],
        ylabel = "Relative Error",
        title = "Training Error", 
        yscale=log10,
        limits = (nothing, nothing, error_limits...),
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )

    # Second row - Testing data
    # Left subplot: Testing utau values
    ax3 = Axis(fig[2, 1], 
        xlabel = "Time Step",
        ylabel = L"$v_{x,\tau} \,/\, \langle v_{x,\tau} \rangle$",
        title = "Testing Data",
        limits = (nothing, nothing, utau_limits...),
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )

    # Right subplot: Testing error
    ax4 = Axis(fig[2, 2],
        xlabel = "Time Step",
        ylabel = "Relative Error",
        title = "Testing Error",
        yscale=log10,
        limits = (nothing, nothing, error_limits...),
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25
    )

    # Time steps
    time_steps_train = 1:n_train
    time_steps_test = 1:n_test

    # Training data plots
    lines!(ax1, time_steps_train, utau_train_norm, 
           label="Original", linewidth=3, color=:black)
    lines!(ax1, time_steps_train, utau_train_rom_norm, 
           label="Streaming-OpInf", linewidth=3, color=c2, linestyle=:dash)
    axislegend(ax1, position=:rt, labelsize=25, patchsize=(80, 20))

    lines!(ax2, time_steps_train, utau_error_train, 
           linewidth=3, color=c1)

    # Testing data plots
    lines!(ax3, time_steps_test, utau_test_norm, 
           label="Original", linewidth=3, color=:black)
    lines!(ax3, time_steps_test, utau_test_rom_norm, 
           label="Streaming-OpInf", linewidth=3, color=c2, linestyle=:dash)
    axislegend(ax3, position=:rt, labelsize=25, patchsize=(80, 20))

    lines!(ax4, time_steps_test, utau_error_test, 
           linewidth=3, color=c1)

    # Add super title
    Label(fig[0, :], 
        text=L"Wall Shear Flow $v_{x,\tau}$ and Relative Errors", 
        fontsize = 32)

    display(fig)
    # save(joinpath(FILEPATH, 
    #     "plots/$(batch_or_stream)_utau_r$(rmax).png"), fig)
end


#===========================================#
## Plot 4: Z Profile 
#===========================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1440, 800))

    # Calculate error ranges first to ensure consistent scaling
    zprof_error_train = abs.(mean(zprof_train_rom, dims=2)[:] - mean(zprof_train, dims=2)[:]) ./ 
                        abs.(mean(zprof_train, dims=2)[:])
    zprof_error_test = abs.(mean(zprof_test_rom, dims=2)[:] - mean(zprof_test, dims=2)[:]) ./ 
                       abs.(mean(zprof_test, dims=2)[:])
    
    # Find common error range for both training and testing
    error_min = min(minimum(zprof_error_train), minimum(zprof_error_test))
    error_max = max(maximum(zprof_error_train), maximum(zprof_error_test))
    common_error_limits = (error_min * 0.8, error_max * 1.2)
    
    # Find common velocity range for both training and testing
    vel_min = min(minimum(mean(zprof_train, dims=2)), minimum(mean(zprof_train_rom, dims=2)),
                  minimum(mean(zprof_test, dims=2)), minimum(mean(zprof_test_rom, dims=2)))
    vel_max = max(maximum(mean(zprof_train, dims=2)), maximum(mean(zprof_train_rom, dims=2)),
                  maximum(mean(zprof_test, dims=2)), maximum(mean(zprof_test_rom, dims=2)))
    common_vel_limits = (vel_min * 0.95, vel_max * 1.05)

    # Add super title
    Label(fig[1, 1:2], text="Training", fontsize = 40)

    # Left subplot: Original vs ROM z profiles
    ax1_left = Axis(fig[2, 1], 
        ylabel = "wall-normal profile",
        yticks=round.(vcat(0, ds["z"][6:6:nz]), digits=3), 
        titlesize=30, xlabelsize=38, ylabelsize=38, 
        xticklabelsize=35, yticklabelsize=35,
        width=620,
        yscale=log10,
        limits = (common_vel_limits..., nothing, nothing)  # Set common x-axis limits
    )

    # Plot mean profiles over time
    zprof_train_mean = mean(zprof_train, dims=2)[2:end]
    zprof_train_rom_mean = mean(zprof_train_rom, dims=2)[2:end]
    l1 = lines!(ax1_left, z_coords[2:end], zprof_train_mean,
                label="Original", linewidth=8)
    l2 = lines!(ax1_left, z_coords[2:end], zprof_train_rom_mean, 
                label="Streaming OpInf", 
                linewidth=8, linestyle=:dash)
    axislegend(ax1_left, position=:lt, labelsize=40, patchsize=(120, 30))

    # Right subplot: Error in z profile
    ax1_right = Axis(fig[2, 2],
        yticks=round.(vcat(0, ds["z"][6:6:nz]), digits=3), 
        xscale=log10,
        titlesize=30, xlabelsize=38, ylabelsize=38, 
        xticklabelsize=35, yticklabelsize=35,
        width=620,
        yticksvisible=false, yticklabelsvisible=false, ygridvisible=true,
        limits = (common_error_limits..., nothing, nothing)  # Set common error limits
    )

    lines!(ax1_right, zprof_error_train, z_coords, color=:red, linewidth=8)

    # Add super title
    Label(fig[3, 1:2], text="Testing", fontsize = 40)

    # Left subplot: Original vs ROM z profiles
    ax2_left = Axis(fig[4, 1], 
        xlabel = "Z coordinate",
        ylabel = "wall-normal profile",
        yticks=round.(vcat(0, ds["z"][6:6:nz]), digits=3), 
        titlesize=30, xlabelsize=38, ylabelsize=38, 
        xticklabelsize=35, yticklabelsize=35,
        width=620,
        yscale=log10,
        limits = (common_vel_limits..., nothing, nothing)  # Set common x-axis limits
    )

    # Plot mean profiles over time
    zprof_test_mean = mean(zprof_test, dims=2)[2:end]
    zprof_test_rom_mean = mean(zprof_test_rom, dims=2)[2:end]

    lines!(ax2_left, z_coords[2:end], zprof_test_mean, label="Original", linewidth=6)
    lines!(ax2_left, z_coords[2:end], zprof_test_rom_mean, label="Streaming OpInf", 
           linewidth=6, linestyle=:dash)

    # Right subplot: Error in z profile
    ax2_right = Axis(fig[4, 2],
        xlabel = "Relative Error",
        yticks=round.(vcat(0, ds["z"][6:6:nz]), digits=3), 
        xscale=log10,
        titlesize=30, xlabelsize=38, ylabelsize=38, 
        xticklabelsize=35, yticklabelsize=35,
        width=620,
        yticksvisible=false, yticklabelsvisible=false, ygridvisible=true,
        limits = (common_error_limits..., nothing, nothing)  # Set common error limits
    )

    lines!(ax2_right, zprof_error_test, z_coords, color=:red, linewidth=6)
    
    display(fig)
    # save(joinpath(FILEPATH, 
    #     "plots/$(batch_or_stream)_zprofile_r$(rmax).pdf"), fig)
end

#===========================================#
## Plot 4: Z Profile 
#===========================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1500, 900))

    # Calculate error ranges first to ensure consistent scaling
    zprof_error_train = abs.(mean(zprof_train_rom, dims=2)[:] - mean(zprof_train, dims=2)[:]) ./ 
                        abs.(mean(zprof_train, dims=2)[:])
    zprof_error_test = abs.(mean(zprof_test_rom, dims=2)[:] - mean(zprof_test, dims=2)[:]) ./ 
                       abs.(mean(zprof_test, dims=2)[:])
    
    # Find common error range for both training and testing
    error_min = min(minimum(zprof_error_train[2:end]), minimum(zprof_error_test[2:end]))
    error_max = max(maximum(zprof_error_train[2:end]), maximum(zprof_error_test[2:end]))
    common_error_limits = (error_min * 0.5, error_max * 2.0)
    
    # Find common velocity range for both training and testing
    vel_min = min(minimum(mean(zprof_train, dims=2)[2:end]), minimum(mean(zprof_train_rom, dims=2)[2:end]),
                  minimum(mean(zprof_test, dims=2)[2:end]), minimum(mean(zprof_test_rom, dims=2)[2:end]))
    vel_max = max(maximum(mean(zprof_train, dims=2)[2:end]), maximum(mean(zprof_train_rom, dims=2)[2:end]),
                  maximum(mean(zprof_test, dims=2)[2:end]), maximum(mean(zprof_test_rom, dims=2)[2:end]))
    common_vel_limits = (vel_min * 0.9, vel_max * 1.1)

    # Add column titles
    Label(fig[1, 1], text="Training", fontsize = 45)
    Label(fig[1, 2], text="Testing", fontsize = 45)

    c1, c2 = Makie.wong_colors()[1:2]

    # Top left: Original vs ROM z profiles (Training)
    ax1_train = Axis(fig[2, 1], 
        ylabel = L"Wall-Normal $u_1$",
        yticks=round.([vel_min, (vel_min+vel_max)/2, vel_max], digits=3), 
        titlesize=40, xlabelsize=40, ylabelsize=43, 
        xticklabelsize=35, yticklabelsize=35,
        xticksvisible=false, xticklabelsvisible=false, xgridvisible=true,
        width=640,
        xscale=log10,
        limits = (nothing, nothing, common_vel_limits...)
    )

    # Top right: Original vs ROM z profiles (Testing)
    ax1_test = Axis(fig[2, 2], 
        yticks=round.([vel_min, (vel_min+vel_max)/2, vel_max], digits=3), 
        titlesize=40, xlabelsize=40, ylabelsize=40, 
        xticklabelsize=35, yticklabelsize=35,
        width=640,
        xscale=log10,
        xticksvisible=false, xticklabelsvisible=false, xgridvisible=true,
        yticksvisible=false, yticklabelsvisible=false, ygridvisible=true,
        limits = (nothing, nothing, common_vel_limits...)
    )

    # Plot mean profiles over time - Training
    zprof_train_mean = mean(zprof_train, dims=2)[2:end]
    zprof_train_rom_mean = mean(zprof_train_rom, dims=2)[2:end]
    lines!(ax1_train, z_coords[2:end], zprof_train_mean, color=:black,
                label="Original", linewidth=10)
    lines!(ax1_train, z_coords[2:end], zprof_train_rom_mean,
                label="Streaming OpInf", color=c2,
                linewidth=14, linestyle=:dash)
    axislegend(ax1_train, position=:lt, labelsize=40, patchsize=(120, 30))

    # Plot mean profiles over time - Testing
    zprof_test_mean = mean(zprof_test, dims=2)[2:end]
    zprof_test_rom_mean = mean(zprof_test_rom, dims=2)[2:end]
    lines!(ax1_test, z_coords[2:end], zprof_test_mean, color=:black,
           label="Original", linewidth=10)
    lines!(ax1_test, z_coords[2:end], zprof_test_rom_mean, 
           label="Streaming OpInf", color=c2,
           linewidth=14, linestyle=:dash)

    # Bottom left: Error in z profile (Training)
    ax2_train = Axis(fig[3, 1],
        xlabel = L"$\omega_3$ coordinate ($\log_{10}$ scale)",
        ylabel = "Relative Error",
        xticks=round.(logrange(z_coords[2], z_coords[end], 4), digits=2), 
        yscale=log10,
        titlesize=40, xlabelsize=40, ylabelsize=43, 
        xticklabelsize=35, yticklabelsize=35,
        xscale=log10, ylabelpadding=15,
        limits = (nothing, nothing, common_error_limits...)
    )

    lines!(ax2_train, z_coords[2:end], zprof_error_train[2:end], 
           color=c1, linewidth=10)

    # Bottom right: Error in z profile (Testing)
    ax2_test = Axis(fig[3, 2],
        xlabel = L"$\omega_3$ coordinate ($\log_{10}$ scale)",
        xticks=round.(logrange(z_coords[2], z_coords[end], 4), digits=2), 
        yscale=log10,
        titlesize=40, xlabelsize=40, ylabelsize=40, 
        xticklabelsize=35, yticklabelsize=35,
        xscale=log10,
        yticksvisible=false, yticklabelsvisible=false, ygridvisible=true,
        limits = (nothing, nothing, common_error_limits...)
    )

    lines!(ax2_test, z_coords[2:end], zprof_error_test[2:end], 
           color=c1, linewidth=10)
    
    display(fig)
    save(joinpath(FILEPATH, 
        "plots/$(batch_or_stream)_zprofile_r$(rmax).pdf"), fig)
end

#==============================================================#
## Plot 5: Wall Shear Flow combined (train + test) horizontal ##
#==============================================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(2400, 650))

    # normalize utau values for better visualization
    utau_train_norm = utau_train ./ mean(utau_train)
    utau_train_rom_norm = utau_train_rom ./ mean(utau_train_rom)
    utau_test_norm = utau_test ./ mean(utau_test)
    utau_test_rom_norm = utau_test_rom ./ mean(utau_test_rom)

    # Calculate errors
    utau_error_train = abs.(utau_train_rom_norm - utau_train_norm) ./ abs.(utau_train_norm)
    utau_error_test = abs.(utau_test_rom_norm - utau_test_norm) ./ abs.(utau_test_norm)
    
    # Combine training and testing data for continuous plotting
    utau_combined_orig = vcat(utau_train_norm, utau_test_norm)
    utau_combined_rom = vcat(utau_train_rom_norm, utau_test_rom_norm)
    utau_error_combined = vcat(utau_error_train, utau_error_test)
    
    # Common y-axis limits for utau values
    utau_min = minimum(utau_combined_orig)
    utau_max = maximum(utau_combined_orig)
    utau_limits = (utau_min * 0.999, utau_max * 1.001)
    
    # Common y-axis limits for error values
    error_min = minimum(utau_error_combined)
    error_max = maximum(utau_error_combined)
    error_limits = (error_min * 0.5, error_max * 5.0)

    c1, c2 = Makie.wong_colors()[1:2]

    # Left subplot: Combined utau values
    ax1 = Axis(fig[1, 1], 
        xlabel = "Time Step",
        ylabel = L"$v_{x,\tau} \,/\, \langle v_{x,\tau} \rangle$",
        title = "Friction Velocity over Time (Training + Testing)",
        limits = (nothing, nothing, utau_limits...),
        titlesize=40, xlabelsize=38, ylabelsize=38, 
        xticklabelsize=35, yticklabelsize=35
    )

    # Right subplot: Combined error
    ax2 = Axis(fig[1, 2],
        xlabel = "Time Step",
        ylabel = "Relative Error",
        title = "Relative Error",
        yscale=log10,
        limits = (nothing, nothing, error_limits...),
        titlesize=40, xlabelsize=38, ylabelsize=38, 
        xticklabelsize=35, yticklabelsize=35
    )

    # Combined time steps
    time_steps_combined = 1:(n_train + n_test)
    
    # Plot combined data
    lines!(ax1, time_steps_combined, utau_combined_orig, 
           label="Original", linewidth=3, color=:black)
    lines!(ax1, time_steps_combined, utau_combined_rom, 
           label="Streaming-OpInf", linewidth=3, color=c2, linestyle=:solid)
    
    # Add vertical line to separate training and testing
    vlines!(ax1, n_train + 0.5, color=:red, linewidth=4, linestyle=:solid)
    
    lines!(ax2, time_steps_combined, utau_error_combined, 
           linewidth=3, color=c1)
    
    # Add vertical line to separate training and testing
    vlines!(ax2, n_train + 0.5, color=:red, linewidth=4, linestyle=:solid)
    
    # Add legends and annotations
    axislegend(ax1, position=:lb, labelsize=35, patchsize=(80, 20))
    
    # Add text annotations to indicate training and testing regions
    text!(ax1, n_train/2, utau_limits[2]*(1-5e-4), text="Training", 
          fontsize=32, color=:black, align=(:center, :top))
    text!(ax1, n_train + n_test/2, utau_limits[2]*(1-5e-4), text="Testing", 
          fontsize=32, color=:black, align=(:center, :top))
    
    text!(ax2, n_train/2, error_limits[2]*0.5, text="Training", 
          fontsize=32, color=:black, align=(:center, :center))
    text!(ax2, n_train + n_test/2, error_limits[2]*0.5, text="Testing", 
          fontsize=32, color=:black, align=(:center, :center))

    display(fig)
    # save(joinpath(FILEPATH, 
    #     "plots/$(batch_or_stream)_utau_combined_r$(rmax).pdf"), fig)
end

#============================================================#
## Plot 5: Wall Shear Flow combined (train + test) vertical ##
#============================================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1500, 1000))

    # normalize utau values for better visualization
    utau_train_norm = utau_train ./ mean(utau_train)
    utau_train_rom_norm = utau_train_rom ./ mean(utau_train_rom)
    utau_test_norm = utau_test ./ mean(utau_test)
    utau_test_rom_norm = utau_test_rom ./ mean(utau_test_rom)

    # Calculate errors
    utau_error_train = abs.(utau_train_rom_norm - utau_train_norm) ./ abs.(utau_train_norm)
    utau_error_test = abs.(utau_test_rom_norm - utau_test_norm) ./ abs.(utau_test_norm)
    
    # Combine training and testing data for continuous plotting
    utau_combined_orig = vcat(utau_train_norm, utau_test_norm)
    utau_combined_rom = vcat(utau_train_rom_norm, utau_test_rom_norm)
    utau_error_combined = vcat(utau_error_train, utau_error_test)
    
    # Calculate means for the combined data
    utau_combined_orig_mean = mean(utau_combined_orig)
    utau_combined_rom_mean = mean(utau_combined_rom)
    
    # Common y-axis limits for utau values
    utau_min = minimum(utau_combined_orig)
    utau_max = maximum(utau_combined_orig)
    utau_limits = (utau_min * 0.999, utau_max * 1.001)
    
    # Common y-axis limits for error values
    error_min = minimum(utau_error_combined)
    error_max = maximum(utau_error_combined)
    error_limits = (error_min * 0.5, error_max * 5.0)

    c1, c2 = Makie.wong_colors()[1:2]

    # Top subplot: Combined utau values
    ax1 = Axis(fig[1, 1], 
        ylabel = L"$u_{\tau} \,/\, \overline{u_{\tau}}$",
        title = "Friction Velocity over Time (Training + Testing)",
        limits = (nothing, nothing, utau_limits...),
        titlesize=40, xlabelsize=38, ylabelsize=45, 
        xticklabelsize=35, yticklabelsize=35,
        xticksvisible=false, xticklabelsvisible=false
    )

    # Bottom subplot: Combined error
    ax2 = Axis(fig[2, 1],
        xlabel = "Time Step",
        ylabel = "Relative Error",
        title = "Relative Error",
        yscale=log10, ylabelpadding=15,
        xticks=0:2000:10000,
        limits = (nothing, nothing, error_limits...),
        titlesize=40, xlabelsize=40, ylabelsize=42, 
        xticklabelsize=39, yticklabelsize=35
    )

    # Combined time steps
    time_steps_combined = 1:(n_train + n_test)
    
    # Plot combined data
    lines!(ax1, time_steps_combined, utau_combined_orig, 
           label="Original", linewidth=3.5, color=:black)
    lines!(ax1, time_steps_combined, utau_combined_rom, 
           label="Streaming OpInf", linewidth=3.5, color=c2, linestyle=:solid)
    
    # # Add mean lines
    # hlines!(ax1, utau_combined_orig_mean, 
    #         color=:black, linewidth=6, linestyle=:solid, alpha=0.9)
    # hlines!(ax1, utau_combined_rom_mean, 
    #         color=c2, linewidth=6, linestyle=:dash, alpha=0.9)

    # # Add arrow and text label for the target mean line
    # arrow_x = n_train + n_test - 20
    # arrow_y_start = utau_combined_orig_mean * 0.995
    # arrow_y_end = utau_combined_orig_mean 
    
    # arrows!(ax1, [arrow_x], [arrow_y_start], 
    #         [0], [arrow_y_end - arrow_y_start - 0.0007],
    #         color=:black, linewidth=5, arrowsize=28)
    
    # text!(ax1, arrow_x, arrow_y_start * 0.9998, 
    #       text=L"$\overline{u_\tau} /\langle \overline{u_\tau} \rangle$", 
    #       fontsize=43, color=:black, align=(:center, :top))

    # Print mean values in terminal 
    @info "Mean of training original utau: $(mean(utau_train))"
    @info "Mean of training ROM utau: $(mean(utau_train_rom))"
    @info "Mean of testing original utau: $(mean(utau_test))"
    @info "Mean of testing ROM utau: $(mean(utau_test_rom))"
    
    # Add vertical line to separate training and testing
    vlines!(ax1, n_train + 0.5, color=:red, linewidth=6, linestyle=:dash)
    
    lines!(ax2, time_steps_combined, utau_error_combined, 
           linewidth=3.5, color=c1)
    
    # Add vertical line to separate training and testing
    vlines!(ax2, n_train + 0.5, color=:red, linewidth=6, linestyle=:dash)
    
    # Add legends and annotations
    axislegend(ax1, position=:lb, labelsize=40, patchsize=(80, 20), nbanks=2)
    
    # Add text annotations to indicate training and testing regions
    text!(ax1, n_train/2, utau_limits[2]*(1-5e-4), text="Training", 
          fontsize=40, color=:black, align=(:center, :top))
    text!(ax1, n_train + n_test/2, utau_limits[2]*(1-5e-4), text="Testing", 
          fontsize=40, color=:black, align=(:center, :top))
    
    text!(ax2, n_train/2, error_limits[2]*0.4, text="Training", 
          fontsize=40, color=:black, align=(:center, :center))
    text!(ax2, n_train + n_test/2, error_limits[2]*0.4, text="Testing", 
          fontsize=40, color=:black, align=(:center, :center))

    display(fig)
    save(joinpath(FILEPATH, 
        "plots/$(batch_or_stream)_utau_combined_r$(rmax).pdf"), fig)
end

#============================================================#
## Plot 6: Friction Velocity Statistics (Mean & Variance) ##
#============================================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1800, 1000))

    # normalize utau values for better visualization
    utau_train_norm = utau_train 
    utau_train_rom_norm = utau_train_rom 
    utau_test_norm = utau_test 
    utau_test_rom_norm = utau_test_rom 

    # Calculate statistics
    # Training
    mean_train_orig = mean(utau_train_norm)
    mean_train_rom = mean(utau_train_rom_norm)
    var_train_orig = var(utau_train_norm)
    var_train_rom = var(utau_train_rom_norm)
    std_train_orig = sqrt(var_train_orig)
    std_train_rom = sqrt(var_train_rom)
    
    # Testing
    mean_test_orig = mean(utau_test_norm)
    mean_test_rom = mean(utau_test_rom_norm)
    var_test_orig = var(utau_test_norm)
    var_test_rom = var(utau_test_rom_norm)
    std_test_orig = sqrt(var_test_orig)
    std_test_rom = sqrt(var_test_rom)

    # Common y-axis limits for utau values
    utau_min = min(minimum(utau_train_norm), minimum(utau_train_rom_norm), 
                   minimum(utau_test_norm), minimum(utau_test_rom_norm))
    utau_max = max(maximum(utau_train_norm), maximum(utau_train_rom_norm),
                   maximum(utau_test_norm), maximum(utau_test_rom_norm))
    utau_limits = (utau_min * 0.998, utau_max * 1.002)

    c1, c2 = Makie.wong_colors()[1:2]
    
    # Time steps
    time_steps_train = 1:n_train
    time_steps_test = 1:n_test

    # ===== ROW 1: TRAINING DATA =====
    # Left: Time series with mean and variance bands
    ax1 = Axis(fig[1, 1], 
        ylabel = L"$u_{\tau}$",
        title = "Training: Time Series with Mean ± Std Dev",
        limits = (nothing, nothing, utau_limits...),
        titlesize=32, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=28, yticklabelsize=28,
        xticksvisible=false, xticklabelsvisible=false
    )

    # Plot time series
    lines!(ax1, time_steps_train, utau_train_norm, 
           label="Original", linewidth=2, color=:black, alpha=0.6)
    lines!(ax1, time_steps_train, utau_train_rom_norm, 
           label="Streaming-OpInf", linewidth=2, color=c2, alpha=0.6)
    
    # Plot mean lines
    hlines!(ax1, mean_train_orig, color=:black, linewidth=3, linestyle=:dash)
    hlines!(ax1, mean_train_rom, color=c2, linewidth=3, linestyle=:dash)
    
    # Plot variance bands (mean ± std)
    band!(ax1, time_steps_train, 
          fill(mean_train_orig - std_train_orig, n_train),
          fill(mean_train_orig + std_train_orig, n_train),
          color=(:black, 0.2))
    band!(ax1, time_steps_train, 
          fill(mean_train_rom - std_train_rom, n_train),
          fill(mean_train_rom + std_train_rom, n_train),
          color=(c2, 0.2))
    
    axislegend(ax1, position=:rt, labelsize=28, patchsize=(60, 15))

    # Right: Bar chart comparing statistics
    ax2 = Axis(fig[1, 2],
        ylabel = "Value",
        title = "Training: Statistical Comparison",
        xticks = (1:4, ["Mean\n(Orig)", "Mean\n(ROM)", "Var\n(Orig)", "Var\n(ROM)"]),
        titlesize=32, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=26, yticklabelsize=28,
        limits=(nothing, nothing, 0, 0.05)
    )

    # Bar plot for statistics
    barplot!(ax2, [1, 2], [mean_train_orig, mean_train_rom], 
             color=[:black, c2], strokewidth=2, strokecolor=:black)
    barplot!(ax2, [3, 4], [var_train_orig, var_train_rom], 
             color=[:black, c2], strokewidth=2, strokecolor=:black)
    
    # Add value labels on bars
    text!(ax2, 1, mean_train_orig + 0.0005, 
          text=string(round(mean_train_orig, digits=4)), 
          fontsize=24, align=(:center, :bottom))
    text!(ax2, 2, mean_train_rom + 0.0005, 
          text=string(round(mean_train_rom, digits=4)), 
          fontsize=24, align=(:center, :bottom))
    text!(ax2, 3, var_train_orig + 0.00002, 
          text=string(round(var_train_orig, digits=6)), 
          fontsize=24, align=(:center, :bottom))
    text!(ax2, 4, var_train_rom + 0.00002, 
          text=string(round(var_train_rom, digits=6)), 
          fontsize=24, align=(:center, :bottom))

    # ===== ROW 2: TESTING DATA =====
    # Left: Time series with mean and variance bands
    ax3 = Axis(fig[2, 1], 
        xlabel = "Time Step",
        ylabel = L"$u_{\tau}$",
        title = "Testing: Time Series with Mean ± Std Dev",
        limits = (nothing, nothing, utau_limits...),
        titlesize=32, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=28, yticklabelsize=28
    )

    # Plot time series
    lines!(ax3, time_steps_test, utau_test_norm, 
           label="Original", linewidth=2, color=:black, alpha=0.6)
    lines!(ax3, time_steps_test, utau_test_rom_norm, 
           label="Streaming-OpInf", linewidth=2, color=c2, alpha=0.6)
    
    # Plot mean lines
    hlines!(ax3, mean_test_orig, color=:black, linewidth=3, linestyle=:dash)
    hlines!(ax3, mean_test_rom, color=c2, linewidth=3, linestyle=:dash)
    
    # Plot variance bands (mean ± std)
    band!(ax3, time_steps_test, 
          fill(mean_test_orig - std_test_orig, n_test),
          fill(mean_test_orig + std_test_orig, n_test),
          color=(:black, 0.2))
    band!(ax3, time_steps_test, 
          fill(mean_test_rom - std_test_rom, n_test),
          fill(mean_test_rom + std_test_rom, n_test),
          color=(c2, 0.2))
    
    axislegend(ax3, position=:rt, labelsize=28, patchsize=(60, 15))

    # Right: Bar chart comparing statistics
    ax4 = Axis(fig[2, 2],
        xlabel = "Statistic",
        ylabel = "Value",
        title = "Testing: Statistical Comparison",
        xticks = (1:4, ["Mean\n(Orig)", "Mean\n(ROM)", "Var\n(Orig)", "Var\n(ROM)"]),
        titlesize=32, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=26, yticklabelsize=28,
        limits=(nothing, nothing, 0, 0.05)
    )

    # Bar plot for statistics
    barplot!(ax4, [1, 2], [mean_test_orig, mean_test_rom], 
             color=[:black, c2], strokewidth=2, strokecolor=:black)
    barplot!(ax4, [3, 4], [var_test_orig, var_test_rom], 
             color=[:black, c2], strokewidth=2, strokecolor=:black)
    
    # Add value labels on bars
    text!(ax4, 1, mean_test_orig + 0.0005, 
          text=string(round(mean_test_orig, digits=4)), 
          fontsize=24, align=(:center, :bottom))
    text!(ax4, 2, mean_test_rom + 0.0005, 
          text=string(round(mean_test_rom, digits=4)), 
          fontsize=24, align=(:center, :bottom))
    text!(ax4, 3, var_test_orig + 0.00002, 
          text=string(round(var_test_orig, digits=6)), 
          fontsize=24, align=(:center, :bottom))
    text!(ax4, 4, var_test_rom + 0.00002, 
          text=string(round(var_test_rom, digits=6)), 
          fontsize=24, align=(:center, :bottom))

    # Add super title
    Label(fig[0, :], 
        text="Friction Velocity Statistics: Mean and Variance Analysis", 
        fontsize=38, font=:bold)

    # Print statistics to terminal
    println("\n" * "="^60)
    println("FRICTION VELOCITY STATISTICS")
    println("="^60)
    println("\nTRAINING:")
    println("  Original    - Mean: $(round(mean_train_orig, digits=6)), Variance: $(round(var_train_orig, digits=8)), Std: $(round(std_train_orig, digits=6))")
    println("  ROM         - Mean: $(round(mean_train_rom, digits=6)), Variance: $(round(var_train_rom, digits=8)), Std: $(round(std_train_rom, digits=6))")
    println("  Mean Error  : $(round(abs(mean_train_rom - mean_train_orig)/abs(mean_train_orig) * 100, digits=3))%")
    println("  Var Error   : $(round(abs(var_train_rom - var_train_orig)/abs(var_train_orig) * 100, digits=3))%")
    
    println("\nTESTING:")
    println("  Original    - Mean: $(round(mean_test_orig, digits=6)), Variance: $(round(var_test_orig, digits=8)), Std: $(round(std_test_orig, digits=6))")
    println("  ROM         - Mean: $(round(mean_test_rom, digits=6)), Variance: $(round(var_test_rom, digits=8)), Std: $(round(std_test_rom, digits=6))")
    println("  Mean Error  : $(round(abs(mean_test_rom - mean_test_orig)/abs(mean_test_orig) * 100, digits=3))%")
    println("  Var Error   : $(round(abs(var_test_rom - var_test_orig)/abs(var_test_orig) * 100, digits=3))%")
    println("="^60 * "\n")

    display(fig)
    # save(joinpath(FILEPATH, 
    #     "plots/$(batch_or_stream)_utau_statistics_r$(rmax).pdf"), fig)
end