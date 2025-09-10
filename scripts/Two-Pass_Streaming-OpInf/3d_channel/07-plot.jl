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
batch_or_stream = "stream"
rmax = 350


#=================#
## Plot the QoIs
#=================#
if batch_or_stream == "batch"
    qois_train = load(joinpath(FILEPATH, 
        "data/results/$(batch_or_stream)_rom_train_qois_0_8000_r$(rmax).jld2"))
    qois_test = load(joinpath(FILEPATH, 
        "data/results/$(batch_or_stream)_rom_test_qois_0_8000_r$(rmax).jld2"))
else
    algo = "rls"
    qois_train = load(joinpath(FILEPATH, 
        "data/results/$(batch_or_stream)_rom_train_qois_0_8000_r$(rmax)_$(algo).jld2"))
    qois_test = load(joinpath(FILEPATH, 
        "data/results/$(batch_or_stream)_rom_test_qois_0_8000_r$(rmax)_$(algo).jld2"))
end


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
    save(joinpath(FILEPATH, 
        "plots/$(batch_or_stream)_zprofile_error_r$(rmax)_$(algo).png"), fig)
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
        "plots/$(batch_or_stream)_zprofile_heatmap_r$(rmax)_$(algo).png"), fig)
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
    save(joinpath(FILEPATH, 
        "plots/$(batch_or_stream)_utau_r$(rmax)_$(algo).png"), fig)
end


#===========================================#
## Plot 4: Z Profile 
#===========================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1440, 1200))

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
        ylabel = "Z Coordinate",
        yticks=round.(vcat(0, ds["z"][4:4:nz]), digits=3), 
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=28, yticklabelsize=28,
        width=600,
        limits = (common_vel_limits..., nothing, nothing)  # Set common x-axis limits
    )

    # Plot mean profiles over time
    zprof_train_mean = mean(zprof_train, dims=2)[:]
    zprof_train_rom_mean = mean(zprof_train_rom, dims=2)[:]
    l1 = lines!(ax1_left, zprof_train_mean, z_coords, 
                label="Original", linewidth=8)
    l2 = lines!(ax1_left, zprof_train_rom_mean, z_coords, 
                label="Streaming-OpInf", 
                linewidth=8, linestyle=:dash)
    axislegend(ax1_left, position=:lt, labelsize=40, patchsize=(120, 30))

    # Right subplot: Error in z profile
    ax1_right = Axis(fig[2, 2],
        yticks=round.(vcat(0, ds["z"][4:4:nz]), digits=3), 
        xscale=log10,
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=28, yticklabelsize=28,
        width=600,
        yticksvisible=false, yticklabelsvisible=false, ygridvisible=true,
        limits = (common_error_limits..., nothing, nothing)  # Set common error limits
    )

    lines!(ax1_right, zprof_error_train, z_coords, color=:red, linewidth=8)

    # Add super title
    Label(fig[3, 1:2], text="Testing", fontsize = 40)

    # Left subplot: Original vs ROM z profiles
    ax2_left = Axis(fig[4, 1], 
        xlabel = "Time-Averaged Wall Normal Profile",
        ylabel = "Z Coordinate",
        yticks=round.(vcat(0, ds["z"][4:4:nz]), digits=3), 
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25,
        width=600,
        limits = (common_vel_limits..., nothing, nothing)  # Set common x-axis limits
    )

    # Plot mean profiles over time
    zprof_test_mean = mean(zprof_test, dims=2)[:]
    zprof_test_rom_mean = mean(zprof_test_rom, dims=2)[:]

    lines!(ax2_left, zprof_test_mean, z_coords, label="Original", linewidth=6)
    lines!(ax2_left, zprof_test_rom_mean, z_coords, label="ROM", 
           linewidth=6, linestyle=:dash)

    # Right subplot: Error in z profile
    ax2_right = Axis(fig[4, 2],
        xlabel = "Relative Error",
        yticks=round.(vcat(0, ds["z"][4:4:nz]), digits=3), 
        xscale=log10,
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25,
        width=600,
        yticksvisible=false, yticklabelsvisible=false, ygridvisible=true,
        limits = (common_error_limits..., nothing, nothing)  # Set common error limits
    )

    lines!(ax2_right, zprof_error_test, z_coords, color=:red, linewidth=6)
    
    display(fig)
    save(joinpath(FILEPATH, 
        "plots/$(batch_or_stream)_zprofile_r$(rmax)_$(algo).png"), fig)
end


#=============================#
## Plot the streaming errors ##
#=============================#
streaming_errors = load(
    joinpath(FILEPATH, "data/results/streaming_errors_0_8000_r$(rmax).jld2"),
    "stream_error"
)

with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 600))
    ytick_vals = 10.0 .^ (-14:2:0)
    ax = Axis(
        fig[1, 1], 
        xlabel=L"$k$-th stream", 
        ylabel="relative streaming error",
        yscale=log10,
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25,
        yticks=(ytick_vals, [L"10^{%$(Int(log10(y)))}" for y in ytick_vals]),
    )
    
    lines = []
    labels = ["RLS"]
    methods = ["rls"]
    # labels = ["RLS", "iQRRLS", "QRRLS"]
    # methods = ["rls", "iqrrls", "qrrls"]
    marker_styles = [:rect, :star5, :hexagon]
    line_styles = [:dot, :dash, :dashdot]
    colors = Makie.wong_colors()[1:3]
    
    for (i, method) in enumerate(methods)
        num_streams = length(streaming_errors[Symbol(method)])
        l = scatterlines!(
            ax, 1:num_streams, streaming_errors[Symbol(method)],
            marker=marker_styles[i], markersize=(35-(i-1)*5),
            linestyle=line_styles[i], linewidth=7,
            color=colors[i]
        )
        push!(lines, l)
    end
    
    axislegend(
        ax, lines, labels,
        position=:lb,
        labelsize=30,
        patchsize=(80,20)
    )
    
    display(fig)
    save(joinpath(FILEPATH, "plots/streaming_errors_r$(rmax).pdf"), fig)
end


# #===========================================#
# ## Plot 2: Z Profile - Testing Data
# #===========================================#
# with_theme(theme_latexfonts()) do 
#     fig2 = Figure(size=(1200, 600))

#     # Left subplot: Original vs ROM z profiles
#     ax2_left = Axis(fig2[1, 1], 
#         xlabel = "Z Coordinate",
#         ylabel = L"Time-Averaged $u_z$ velocity",
#         xticks=round.(vcat(0, ds["z"][8:8:nz]), digits=3), 
#         titlesize=30, xlabelsize=30, ylabelsize=30, 
#         xticklabelsize=25, yticklabelsize=25
#     )

#     # Plot mean profiles over time
#     zprof_test_mean = mean(zprof_test, dims=2)[:]
#     zprof_test_rom_mean = mean(zprof_test_rom, dims=2)[:]

#     lines!(ax2_left, z_coords, zprof_test_mean, label="Original", linewidth=6)
#     lines!(ax2_left, z_coords, zprof_test_rom_mean, label="ROM", linewidth=6, linestyle=:dash)
#     axislegend(ax2_left, position=:rb, labelsize=28, patchsize=(80, 30))

#     # Right subplot: Error in z profile
#     ax2_right = Axis(fig2[1, 2],
#         xlabel = "Z Coordinate",
#         ylabel = "Relative Error",
#         xticks=round.(vcat(0, ds["z"][8:8:nz]), digits=3), 
#         yscale=log10,
#         titlesize=30, xlabelsize=30, ylabelsize=30, 
#         xticklabelsize=25, yticklabelsize=25
#     )

#     zprof_error_test = abs.(zprof_test_rom_mean - zprof_test_mean) ./ 
#                         abs.(zprof_test_mean)
#     lines!(ax2_right, z_coords, zprof_error_test, color=:red, linewidth=6)

#     # Add super title
#     Label(fig2[0, :], 
#         text=L"Time-Averaged $u_z$ Profile (left) and Relative Errors (right) for Testing Data", 
#         fontsize = 30)

#     display(fig2)
#     # save(joinpath(FILEPATH, 
#     #     "plots/$(batch_or_stream)_zprofile_testing_comparison.png"), fig2)
# end



# #===========================================#
# ## Plot 3: Wall Shear Flow - Training Data
# #===========================================#
# with_theme(theme_latexfonts()) do 
#     fig3 = Figure(size=(1200, 600))

#     # Left subplot: Original vs ROM wall shear flow
#     ax3_left = Axis(fig3[1, 1], 
#         xlabel = "Time Step",
#         ylabel = L"$u_\tau$",
#         titlesize=30, xlabelsize=30, ylabelsize=30, 
#         xticklabelsize=25, yticklabelsize=25
#     )

#     time_steps_train = 1:n_train
#     lines!(ax3_left, time_steps_train, utau_train, label="Original", linewidth=3)
#     lines!(ax3_left, time_steps_train, utau_train_rom, label="ROM", linewidth=3, 
#            linestyle=:dash)
#     axislegend(ax3_left, position=:rt, labelsize=28, patchsize=(80, 30))

#     # Right subplot: Error in wall shear flow
#     ax3_right = Axis(fig3[1, 2],
#         xlabel = "Time Step",
#         ylabel = "Relative Error",
#         yscale=log10,
#         titlesize=30, xlabelsize=30, ylabelsize=30, 
#         xticklabelsize=25, yticklabelsize=25
#     )

#     utau_error_train = abs.(utau_train_rom - utau_train) ./ abs.(utau_train)
#     lines!(ax3_right, time_steps_train, utau_error_train, color=:red, linewidth=3)

#     # Add super title
#     Label(fig3[0, :], 
#         text=L"Wall Shear Flow $u_\tau$ (left) and Relative Errors (right) for Training Data", 
#         fontsize = 30)

#     display(fig3)
#     # save(joinpath(FILEPATH, 
#     #     "plots/$(batch_or_stream)_utau_training_comparison.png"), fig3)
# end


# #===========================================#
# ## Plot 4: Wall Shear Flow - Testing Data
# #===========================================#
# with_theme(theme_latexfonts()) do 
#     fig4 = Figure(size=(1200, 600))

#     # Left subplot: Original vs ROM wall shear flow
#     ax4_left = Axis(fig4[1, 1], 
#         xlabel = "Time Step",
#         ylabel = L"$u_\tau$",
#         titlesize=30, xlabelsize=30, ylabelsize=30, 
#         xticklabelsize=25, yticklabelsize=25
#     )

#     time_steps_test = 1:n_test
#     lines!(ax4_left, time_steps_test, utau_test, label="Original", linewidth=3)
#     lines!(ax4_left, time_steps_test, utau_test_rom, label="ROM", linewidth=3, 
#            linestyle=:dash)
#     axislegend(ax4_left, position=:rb, labelsize=28, patchsize=(80, 30))

#     # Right subplot: Error in wall shear flow
#     ax4_right = Axis(fig4[1, 2],
#         xlabel = "Time Step",
#         ylabel = "Relative Error",
#         yscale=log10,
#         titlesize=30, xlabelsize=30, ylabelsize=30, 
#         xticklabelsize=25, yticklabelsize=25
#     )

#     utau_error_test = abs.(utau_test_rom - utau_test) ./ abs.(utau_test)
#     lines!(ax4_right, time_steps_test, utau_error_test, color=:red, linewidth=3)

#     # Add super title
#     Label(fig4[0, :], 
#         text=L"Wall Shear Flow $u_\tau$ (left) and Relative Errors (right) for Testing Data", 
#         fontsize = 30)

#     display(fig4)
#     # save(joinpath(FILEPATH, 
#     #     "plots/$(batch_or_stream)_utau_testing_comparison.png"), fig4)
# end

# #===========================================#
# ## Plot 4: Wall Shear Flow 
# #===========================================#
# with_theme(theme_latexfonts()) do 
#     fig = Figure(size=(1400, 600))

#     # Left subplot: Original vs ROM wall shear flow for both training and testing
#     ax_left = Axis(fig[1, 1], 
#         xlabel = "Time Step",
#         ylabel = L"$u_\tau$",
#         titlesize=30, xlabelsize=30, ylabelsize=30, 
#         xticklabelsize=25, yticklabelsize=25
#     )

#     # Training data
#     c1, c2, c3 = Makie.wong_colors()[2:4]
#     time_steps_train = 1:n_train
#     lines!(ax_left, time_steps_train, utau_train, 
#            label="Original (Training)", linewidth=3, color=:black)
#     lines!(ax_left, time_steps_train, utau_train_rom, 
#            label="ROM (Training)", linewidth=3, color=c1)
    
#     # Testing data (continue time steps from where training ended)
#     time_steps_test = (n_train+1):(n_train+n_test)
#     lines!(ax_left, time_steps_test, utau_test, 
#            label="Original (Testing)", linewidth=3, 
#            color=:black, linestyle=:dash)
#     lines!(ax_left, time_steps_test, utau_test_rom, 
#            label="ROM (Testing)", linewidth=3, 
#            color=c1, linestyle=:dash)
    
#     axislegend(ax_left, position=:rt, labelsize=20, patchsize=(60, 20))

#     # Add vertical line to separate training/testing regions
#     vlines!(ax_left, [n_train], color=:gray, linestyle=:dot, linewidth=6)

#     # Right subplot: Error in wall shear flow for both training and testing
#     ax_right = Axis(fig[1, 2],
#         xlabel = "Time Step",
#         ylabel = "Relative Error",
#         yscale=log10,
#         titlesize=30, xlabelsize=30, ylabelsize=30, 
#         xticklabelsize=25, yticklabelsize=25
#     )

#     # Training error
#     utau_error_train = abs.(utau_train_rom - utau_train) ./ abs.(utau_train)
#     lines!(ax_right, time_steps_train, utau_error_train, 
#            label="Training", linewidth=3, color=c2)
    
#     # Testing error
#     utau_error_test = abs.(utau_test_rom - utau_test) ./ abs.(utau_test)
#     lines!(ax_right, time_steps_test, utau_error_test, 
#            label="Testing", linewidth=3, linestyle=:dash, color=c3)

#     axislegend(ax_right, position=:rb, labelsize=20, patchsize=(60, 20))

#     # Add vertical line to separate training/testing regions
#     vlines!(ax_right, [n_train], color=:gray, linestyle=:dot, linewidth=6)

#     # Add super title
#     Label(fig[0, :], 
#         text=L"Wall Shear Flow $u_\tau$ (left) and Relative Errors (right)", 
#         fontsize = 30)

#     display(fig)
#     # save(joinpath(FILEPATH, 
#     #     "plots/$(batch_or_stream)_utau_comparison.png"), fig)
# end


# #=======================================================#
# ## Plot 1: Z Profile Error - Training and Testing Data
# #=======================================================#
# with_theme(theme_latexfonts()) do 
#     fig = Figure(size=(700, 600))
#     ax = Axis(fig[1, 1], 
#         xlabel = L"Time, $s$",
#         ylabel = "Relative Error",
#         title = L"Error of Wall Normal Profile of $u$ (Training)",
#         yscale = log10,
#         titlesize=30, xlabelsize=30, ylabelsize=30, 
#         xticklabelsize=25, yticklabelsize=25
#     )

#     # Plot error profiles over time
#     zprof_error = norm.(eachcol(zprof_train_rom - zprof_train), 2) ./ 
#                   norm.(eachcol(zprof_train), 2)
#     tspan = ds["times"][1:n_train] .- ds["times"][1]
#     l1 = lines!(ax, tspan, zprof_error, linewidth=3)

#     display(fig)
#     save(joinpath(FILEPATH, 
#         "plots/$(batch_or_stream)_zprofile_training_error.png"), fig)
# end


# #===========================================#
# ## Plot 2: Z Profile Error - Testing Data
# #===========================================#
# with_theme(theme_latexfonts()) do 
#     fig = Figure(size=(700, 600))
#     # Left subplot: Original vs ROM z profiles
#     ax = Axis(fig[1, 1], 
#         xlabel = L"Time, $s$",
#         ylabel = "Relative Error",
#         title = L"Error of Wall Normal Profile of $u$ (Test)",
#         yscale = log10,
#         titlesize=30, xlabelsize=30, ylabelsize=30, 
#         xticklabelsize=25, yticklabelsize=25
#     )

#     # Plot error profiles over time
#     zprof_error = norm.(eachcol(zprof_test_rom - zprof_test), 2) ./ 
#                   norm.(eachcol(zprof_test), 2)
#     tspan = ds["times"][n_train+1:n_train+n_test] .- ds["times"][n_train+1]
#     l1 = lines!(ax, tspan, zprof_error, linewidth=3)

#     display(fig)
#     save(joinpath(FILEPATH, 
#         "plots/$(batch_or_stream)_zprofile_testing_error.png"), fig)
# end


# #===========================================#
# ## Plot 3: Z Profile - Training Data
# #===========================================#
# with_theme(theme_latexfonts()) do 
#     fig = Figure(size=(1800, 600))
    
#     # Left subplot: Original data heatmap
#     ax1 = Axis(fig[1, 1], 
#         xlabel = L"Time, $s$",
#         ylabel = "Z Coordinate",
#         title = "Original",
#         titlesize=30, xlabelsize=30, ylabelsize=30, 
#         xticklabelsize=25, yticklabelsize=25
#     )
    
#     # Center subplot: ROM data heatmap
#     ax2 = Axis(fig[1, 2], 
#         xlabel = L"Time, $s$",
#         title = "Reduced Model",
#         titlesize=30, xlabelsize=30, ylabelsize=30, 
#         xticklabelsize=25, yticklabelsize=25
#     )
    
#     # Right subplot: Error heatmap
#     ax3 = Axis(fig[1, 4], 
#         xlabel = L"Time, $s$",
#         title = "Pointwise Abs. Error",
#         titlesize=30, xlabelsize=30, ylabelsize=30, 
#         xticklabelsize=25, yticklabelsize=25
#     )
    
#     # Get time span for training data
#     tspan = ds["times"][1:n_train] .- ds["times"][1]

#     # Calculate data range for consistent scaling
#     data_range = (min(minimum(zprof_train), minimum(zprof_train_rom)), 
#                   max(maximum(zprof_train), maximum(zprof_train_rom)))
    
#     # Create heatmaps with consistent color ranges
#     hm1 = heatmap!(ax1, tspan, z_coords, zprof_train', colorrange=data_range)
#     hm2 = heatmap!(ax2, tspan, z_coords, zprof_train_rom', colorrange=data_range)
    
#     # Calculate pointwise error
#     zprof_error_pointwise = abs.(zprof_train_rom - zprof_train)
#     hm3 = heatmap!(ax3, tspan, z_coords, zprof_error_pointwise', 
#                    colormap=:matter)
    
#     # Add colorbars (without limits parameter)
#     cb1 = Colorbar(fig[1, 3], hm2, label = L"$u$ velocity", 
#                    labelsize=25, ticklabelsize=20)
    
#     # Colorbar for error plot
#     cb2 = Colorbar(fig[1, 5], hm3, label = "Abs. Error", 
#                    labelsize=25, ticklabelsize=20)
    
#     display(fig)
#     save(joinpath(FILEPATH, 
#         "plots/$(batch_or_stream)_zprofile_training_heatmap.png"), fig)
# end

# #===========================================#
# ## Plot 4: Z Profile - Testing Data
# #===========================================#
# with_theme(theme_latexfonts()) do 
#     fig = Figure(size=(1800, 600))
    
#     # Left subplot: Original data heatmap
#     ax1 = Axis(fig[1, 1], 
#         xlabel = L"Time, $s$",
#         ylabel = "Z Coordinate",
#         title = "Original",
#         titlesize=30, xlabelsize=30, ylabelsize=30, 
#         xticklabelsize=25, yticklabelsize=25
#     )
    
#     # Center subplot: ROM data heatmap
#     ax2 = Axis(fig[1, 2], 
#         xlabel = L"Time, $s$",
#         title = "Reduced Model",
#         titlesize=30, xlabelsize=30, ylabelsize=30, 
#         xticklabelsize=25, yticklabelsize=25
#     )
    
#     # Right subplot: Error heatmap
#     ax3 = Axis(fig[1, 4], 
#         xlabel = L"Time, $s$",
#         title = "Pointwise Abs. Error",
#         titlesize=30, xlabelsize=30, ylabelsize=30, 
#         xticklabelsize=25, yticklabelsize=25
#     )
    
#     # Get time span for testing data
#     tspan = ds["times"][n_test+1:n_test+n_test] .- ds["times"][n_test+1]

#     # Calculate data range for consistent scaling
#     data_range = (min(minimum(zprof_test), minimum(zprof_test_rom)), 
#                   max(maximum(zprof_test), maximum(zprof_test_rom)))
    
#     # Create heatmaps with consistent color ranges
#     hm1 = heatmap!(ax1, tspan, z_coords, zprof_test', colorrange=data_range)
#     hm2 = heatmap!(ax2, tspan, z_coords, zprof_test_rom', colorrange=data_range)
    
#     # Calculate pointwise error
#     zprof_error_pointwise = abs.(zprof_test_rom - zprof_test)
#     hm3 = heatmap!(ax3, tspan, z_coords, zprof_error_pointwise', 
#                    colormap=:matter)
    
#     # Add colorbars (without limits parameter)
#     cb1 = Colorbar(fig[1, 3], hm2, label = L"$u$ velocity", 
#                    labelsize=25, ticklabelsize=20)
    
#     # Colorbar for error plot
#     cb2 = Colorbar(fig[1, 5], hm3, label = "Abs. Error", 
#                    labelsize=25, ticklabelsize=20)
    
#     display(fig)
#     # save(joinpath(FILEPATH, 
#     #     "plots/$(batch_or_stream)_zprofile_testing_heatmap.png"), fig)
# end