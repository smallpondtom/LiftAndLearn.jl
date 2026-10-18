"""
ABL: Reconstruct the states using the ROM
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using LinearAlgebra
using ProgressMeter
using Printf
using UniqueKronecker
using BlockDiagonals
using SparseArrays
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
nz, ny, nx, n_fields, n = ds.dims
nxyz = nz * ny * nx
n_test = 2000
n_train = n - n_test
rmax = 200

#=======================#
## Load the Stream model 
#=======================#
GRID_SEARCH = true
op_filename = GRID_SEARCH ? 
    joinpath(FILEPATH, "data/models/op_stream_r$(rmax)_lamGS.jld2") : 
    joinpath(FILEPATH, "data/models/op_stream_r$(rmax).jld2")
op_stream = load(op_filename)["op_stream"]

#=================#
## Load the bases 
#=================#
basis_file = joinpath(FILEPATH, "data/results/onepass_stream.jld2")
stream = load(basis_file)["stream"]
iVrmax = stream.V[:, 1:rmax]
iΣrmax = Diagonal(stream.Σ[1:rmax])
iWrmax = stream.W[:, 1:rmax]
stream = nothing  # free memory
GC.gc()

#============================#
## Load the mean and scaling
#============================#
means  = load(joinpath(FILEPATH, "data/mean.jld2"))["xbar"]
shifts = load(joinpath(FILEPATH, "data/minmax.jld2"))["minmax"]["shifts"]
scales = load(joinpath(FILEPATH, "data/minmax.jld2"))["minmax"]["scales"]

#===========================#
## Simulate ROM (training) ##
#===========================#
include(joinpath(FILEPATH, "integrate.jl"))
include(joinpath(FILEPATH, "preprocess.jl"))
tspan = ds["times"][1:n_train] .- ds["times"][1]
x0 = iVrmax' * preprocess!(ds[1], means, shifts, scales)
states, _ = rk4_integrate(x0, tspan, op_stream.A, op_stream.A2u, op_stream.K)

## Save the training states
save(joinpath(FILEPATH, "data/results", 
     "stream_rom_train_sim_states_0_8000_r$(rmax).jld2"), 
     "states", states)

## Load the training states (if needed)
states = load(joinpath(FILEPATH, "data/results", 
     "stream_rom_train_sim_states_0_8000_r$(rmax).jld2"))["states"]

#==========================#
## Simulate ROM (testing) ##
#==========================#
tspan = ds["times"][n_train+1:n_train+n_test] .- ds["times"][n_train+1]
# Make sure to preprocess the first state
x0 = iVrmax' * preprocess!(ds[n_train+1], means, shifts, scales)
test_states, _ = rk4_integrate(x0, tspan, op_stream.A, op_stream.A2u, op_stream.K)

## Save the test states
save(joinpath(FILEPATH, "data/results", 
     "stream_rom_test_sim_states_0_8000_r$(rmax).jld2"), 
     "states", test_states)

## Load the test states (if needed)
test_states = load(joinpath(FILEPATH, "data/results", 
     "stream_rom_test_sim_states_0_8000_r$(rmax).jld2"))["states"]

#========================================#
## Plot the field with projection error ##
#========================================#
using CairoMakie

with_theme(theme_latexfonts()) do 
    train_or_test = "train"
    fig = Figure(size=(1200, 1200)) # Increased height for 5 rows

    fld = "u"
    if fld == "u"
        i_s = 1
        i_f = nxyz
    elseif fld == "v"
        i_s = nxyz + 1
        i_f = 2 * nxyz
    elseif fld == "w"
        i_s = 2 * nxyz + 1
        i_f = 3 * nxyz
    elseif fld == "p"
        i_s = 3 * nxyz + 1
        i_f = 4 * nxyz
    end

    # Get midpoint index for z-direction
    z_mid = nz ÷ 2

    # Select 3 time steps (beginning, middle, end)
    if train_or_test == "train"
        tspan = ds["times"][1:n_train] .- ds["times"][1]
        time_indices = [50, 1000, length(tspan)÷2, length(tspan)]
        n_shift = 0
    else  # if test data you need to shift
        tspan = ds["times"][n_train+1:n_train+n_test] .- ds["times"][n_train+1]
        time_indices = [10, 500, 1000, length(tspan)-10] 
        n_shift = n_train
    end

    # Pre-calculate all data for colorbar scaling
    all_full_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_rom_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_error_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_pod_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_pod_error_data = Vector{Matrix{Float64}}(undef, length(time_indices))

    # Collect all data first
    for (i, t_idx) in enumerate(time_indices)
        # Get full data
        u_full_field = reshape(ds[t_idx+n_shift][i_s:i_f], nx, ny, nz)
        all_full_data[i] = u_full_field[:, :, z_mid]
        
        # Get ROM data
        if train_or_test == "train"
            x_rom_t = iVrmax * states[:, t_idx]
            x_rom_t = unprocess!(x_rom_t, means, shifts, scales)
        else
            x_rom_t = iVrmax * test_states[:, t_idx]
            x_rom_t = unprocess!(x_rom_t, means, shifts, scales)
        end
        u_rom_field = reshape(x_rom_t[i_s:i_f], nx, ny, nz)
        all_rom_data[i] = u_rom_field[:, :, z_mid]

        # Get POD reconstruction (iVrmax * iVrmax' * ds[i])
        x_full_preprocessed = preprocess!(copy(ds[t_idx+n_shift]), means, shifts, scales)
        x_pod_t = iVrmax * (iVrmax' * x_full_preprocessed)
        x_pod_t = unprocess!(x_pod_t, means, shifts, scales)
        u_pod_field = reshape(x_pod_t[i_s:i_f], nx, ny, nz)
        all_pod_data[i] = u_pod_field[:, :, z_mid]

        # Compute errors
        all_error_data[i] = abs.(all_full_data[i] - all_rom_data[i]) 
        all_pod_error_data[i] = abs.(all_full_data[i] - all_pod_data[i])
    end

    # Calculate global min/max for each row type
    full_min, full_max = extrema(vcat(all_full_data...))
    rom_min, rom_max = extrema(vcat(all_rom_data...))
    pod_min, pod_max = extrema(vcat(all_pod_data...))
    error_min, error_max = extrema(vcat(all_error_data...))
    pod_error_min, pod_error_max = extrema(vcat(all_pod_error_data...))

    # Align the color ranges for first three rows (full, ROM, and POD)
    common_min = min(full_min, rom_min, pod_min)
    common_max = max(full_max, rom_max, pod_max)
    
    # Align error color ranges
    error_common_min = min(error_min, pod_error_min)
    error_common_max = max(error_max, pod_error_max)

    # Create axes and heatmaps
    hm_full = nothing
    hm_rom = nothing
    hm_error = nothing
    hm_pod = nothing
    hm_pod_error = nothing

    for (i, t_idx) in enumerate(time_indices)
        ds_t = ds["times"][t_idx+n_shift] .- ds["times"][1+n_shift]
        n_label = train_or_test == "train" ? n_train : n_test
        # Create axes
        ax_full = Axis(fig[1, i], 
            title = i == 1 ? 
                    L"$t$=%$(round(ds_t, digits=2)) \n snapshot %$(t_idx)/%$(n_label)" : 
                    L"$t$=%$(round(ds_t, digits=2)) \n %$(t_idx)/%$(n_label)",
            ylabel = i == 1 ? L"$y$" : "", 
            xlabelsize=30, ylabelsize=30, 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            titlesize=30,
        )
        ax_rom = Axis(fig[2, i], 
            ylabel = i == 1 ? L"$y$" : "", 
            xlabelsize=30, ylabelsize=30, 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
        )
        ax_pod = Axis(fig[3, i], 
            ylabel = i == 1 ? L"$y$" : "", 
            xlabelsize=30, ylabelsize=30, 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
        )
        ax_error = Axis(fig[4, i], 
            ylabel = i == 1 ? L"$y$" : "", 
            xlabelsize=30, ylabelsize=30, 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
        )
        ax_pod_error = Axis(fig[5, i], 
            ylabel = i == 1 ? L"$y$" : "", 
            xlabel = L"$x$",
            xlabelsize=30, ylabelsize=30, 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
        )

        # Create heatmaps with aligned color ranges
        hm_full = heatmap!(ax_full, ds["x"][:], ds["y"][:], all_full_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_rom = heatmap!(ax_rom, ds["x"][:], ds["y"][:], all_rom_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_error = heatmap!(ax_error, ds["x"][:], ds["y"][:], all_error_data[i], 
            colormap = :matter, colorrange = (error_common_min, error_common_max))
        hm_pod = heatmap!(ax_pod, ds["x"][:], ds["y"][:], all_pod_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_pod_error = heatmap!(ax_pod_error, ds["x"][:], ds["y"][:], all_pod_error_data[i], 
            colormap = :matter, colorrange = (error_common_min, error_common_max))
    end
    
    # Add colorbars at the end of each row
    Colorbar(fig[1, length(time_indices) + 1], hm_full, label="Original", labelsize=23)
    Colorbar(fig[2, length(time_indices) + 1], hm_rom, label="Streaming OpInf", labelsize=23)
    Colorbar(fig[3, length(time_indices) + 1], hm_pod, label="Projection", labelsize=23)
    Colorbar(fig[4, length(time_indices) + 1], hm_error, label="Streaming OpInf Error", labelsize=23)
    Colorbar(fig[5, length(time_indices) + 1], hm_pod_error, label="Projection Error", labelsize=23)
    
    save(joinpath(FILEPATH, "plots", 
         "$(fld)_slice_comparison_$(train_or_test)_$(rmax).png"), fig)
    display(fig)
end

#==================#
## Plot the field ##
#==================#
with_theme(theme_latexfonts()) do 
    train_or_test = "train"
    fig = Figure(size=(1200, 940))

    fld = "p"
    if fld == "u"
        i_s = 1
        i_f = nxyz
    elseif fld == "v"
        i_s = nxyz + 1
        i_f = 2 * nxyz
    elseif fld == "w"
        i_s = 2 * nxyz + 1
        i_f = 3 * nxyz
    elseif fld == "p"
        i_s = 3 * nxyz + 1
        i_f = 4 * nxyz
    end

    # Get midpoint index for z-direction
    z_mid = nz ÷ 2

    # Select 3 time steps (beginning, middle, end)
    if train_or_test == "train"
        tspan = ds["times"][1:n_train] .- ds["times"][1]
        time_indices = [50, 1000, length(tspan)÷2, length(tspan)]
        n_shift = 0
    else  # if test data you need to shift
        tspan = ds["times"][n_train+1:n_train+n_test] .- ds["times"][n_train+1]
        time_indices = [10, 500, 1000, length(tspan)-10] 
        n_shift = n_train
    end

    # Pre-calculate all data for colorbar scaling
    all_full_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_rom_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_error_data = Vector{Matrix{Float64}}(undef, length(time_indices))

    # Collect all data first
    for (i, t_idx) in enumerate(time_indices)
        # Get full data
        u_full_field = reshape(ds[t_idx+n_shift][i_s:i_f], nx, ny, nz)
        all_full_data[i] = u_full_field[:, :, z_mid]
        
        # Get ROM data
        if train_or_test == "train"
            x_rom_t = iVrmax * states[:, t_idx]
            x_rom_t = unprocess!(x_rom_t, means, shifts, scales)
        else
            x_rom_t = iVrmax * test_states[:, t_idx]
            x_rom_t = unprocess!(x_rom_t, means, shifts, scales)
        end
        u_rom_field = reshape(x_rom_t[i_s:i_f], nx, ny, nz)
        all_rom_data[i] = u_rom_field[:, :, z_mid]

        # Compute error
        all_error_data[i] = abs.(all_full_data[i] - all_rom_data[i]) 
    end

    # Calculate global min/max for each row type
    full_min, full_max = extrema(vcat(all_full_data...))
    rom_min, rom_max = extrema(vcat(all_rom_data...))
    error_min, error_max = extrema(vcat(all_error_data...))

    # Align the color ranges for first and second rows (full and ROM)
    common_min = min(full_min, rom_min)
    common_max = max(full_max, rom_max)

    # Create axes and heatmaps
    hm_full = nothing
    hm_rom = nothing
    hm_error = nothing

    for (i, t_idx) in enumerate(time_indices)
        ds_t = ds["times"][t_idx+n_shift] .- ds["times"][1+n_shift]
        n_label = train_or_test == "train" ? n_train : n_test
        # Create axes
        ax_full = Axis(fig[1, i], 
            title = i == 1 ? 
                    L"$t$=%$(round(ds_t, digits=2)) \n snapshot %$(t_idx)/%$(n_label)" : 
                    L"$t$=%$(round(ds_t, digits=2)) \n %$(t_idx)/%$(n_label)",
            ylabel = i == 1 ? L"$y$" : "", 
            xlabelsize=30, ylabelsize=30, 
            # xticklabelsize=25, yticklabelsize=25,
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            titlesize=30,
        )
        ax_rom = Axis(fig[2, i], 
            ylabel = i == 1 ? L"$y$" : "", 
            xlabelsize=30, ylabelsize=30, 
            # xticklabelsize=25, yticklabelsize=25,
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
        )
        ax_error = Axis(fig[3, i], 
            ylabel = i == 1 ? L"$y$" : "", 
            xlabel = L"$x$",
            xlabelsize=30, ylabelsize=30, 
            # xticklabelsize=25, yticklabelsize=25,
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
        )

        # Create heatmaps with aligned color ranges
        hm_full = heatmap!(ax_full, ds["x"][:], ds["y"][:], all_full_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_rom = heatmap!(ax_rom, ds["x"][:], ds["y"][:], all_rom_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_error = heatmap!(ax_error, ds["x"][:], ds["y"][:], all_error_data[i], 
            colormap = :matter, colorrange = (error_min, error_max))
    end
    
    # Add colorbars at the end of each row
    Colorbar(fig[1, length(time_indices) + 1], hm_full, label="Full", labelsize=20)
    Colorbar(fig[2, length(time_indices) + 1], hm_rom, label="ROM", labelsize=20)
    Colorbar(fig[3, length(time_indices) + 1], hm_error, label="Abs. Error", labelsize=20)
    
    save(joinpath(FILEPATH, "plots", 
         "$(fld)_slice_comparison_$(train_or_test)_$(rmax).png"), fig)
    display(fig)
end

#=======================================================#
## Plot the field with projection error (train + test) ##
#=======================================================#
using CairoMakie

with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1500, 950))

    fld = "p"
    if fld == "u"
        i_s = 1
        i_f = nxyz
    elseif fld == "v"
        i_s = nxyz + 1
        i_f = 2 * nxyz
    elseif fld == "w"
        i_s = 2 * nxyz + 1
        i_f = 3 * nxyz
    elseif fld == "p"
        i_s = 3 * nxyz + 1
        i_f = 4 * nxyz
    end

    # Get midpoint index for z-direction
    z_mid = nz ÷ 2

    # Define time indices for train (3) and test (2)
    tspan_train = ds["times"][1:n_train] .- ds["times"][1]
    tspan_test = ds["times"][n_train+1:n_train+n_test] .- ds["times"][n_train+1]
    
    train_time_indices = [50, 1000, length(tspan_train)-10]
    test_time_indices = [50, length(tspan_test)-10]
    
    # Combine all time indices with their shifts
    all_time_data = [
        (train_time_indices[1], 0, "train"),
        (train_time_indices[2], 0, "train"),
        (train_time_indices[3], 0, "train"),
        (test_time_indices[1], n_train, "test"),
        (test_time_indices[2], n_train, "test")
    ]

    # Pre-calculate all data for colorbar scaling
    all_full_data = Vector{Matrix{Float64}}(undef, 5)
    all_rom_data = Vector{Matrix{Float64}}(undef, 5)
    all_error_data = Vector{Matrix{Float64}}(undef, 5)
    all_pod_data = Vector{Matrix{Float64}}(undef, 5)
    all_pod_error_data = Vector{Matrix{Float64}}(undef, 5)

    # Collect all data first
    for (col_idx, (t_idx, n_shift, data_type)) in enumerate(all_time_data)
        # Get full data
        u_full_field = reshape(ds[t_idx+n_shift][i_s:i_f], nx, ny, nz)
        all_full_data[col_idx] = u_full_field[:, :, z_mid]
        
        # Get ROM data
        if data_type == "train"
            x_rom_t = iVrmax * states[:, t_idx]
            x_rom_t = unprocess!(x_rom_t, means, shifts, scales)
        else
            x_rom_t = iVrmax * test_states[:, t_idx]
            x_rom_t = unprocess!(x_rom_t, means, shifts, scales)
        end
        u_rom_field = reshape(x_rom_t[i_s:i_f], nx, ny, nz)
        all_rom_data[col_idx] = u_rom_field[:, :, z_mid]

        # Get POD reconstruction
        x_full_preprocessed = preprocess!(copy(ds[t_idx+n_shift]), means, shifts, scales)
        x_pod_t = iVrmax * (iVrmax' * x_full_preprocessed)
        x_pod_t = unprocess!(x_pod_t, means, shifts, scales)
        u_pod_field = reshape(x_pod_t[i_s:i_f], nx, ny, nz)
        all_pod_data[col_idx] = u_pod_field[:, :, z_mid]

        # Compute errors
        all_error_data[col_idx] = abs.(all_full_data[col_idx] - all_rom_data[col_idx]) 
        all_pod_error_data[col_idx] = abs.(all_full_data[col_idx] - all_pod_data[col_idx])
    end

    # Calculate global min/max for each row type
    full_min, full_max = extrema(vcat(all_full_data...))
    rom_min, rom_max = extrema(vcat(all_rom_data...))
    pod_min, pod_max = extrema(vcat(all_pod_data...))
    error_min, error_max = extrema(vcat(all_error_data...))
    pod_error_min, pod_error_max = extrema(vcat(all_pod_error_data...))

    # Align the color ranges for first three rows
    common_min = min(full_min, rom_min, pod_min)
    common_max = max(full_max, rom_max, pod_max)
    
    # Align error color ranges
    error_common_min = min(error_min, pod_error_min)
    error_common_max = max(error_max, pod_error_max)

    # Create axes and heatmaps
    hm_full = nothing
    hm_rom = nothing
    hm_error = nothing
    hm_pod = nothing
    hm_pod_error = nothing

    for (col_idx, (t_idx, n_shift, data_type)) in enumerate(all_time_data)
        ds_t = ds["times"][t_idx+n_shift] .- ds["times"][1+n_shift]
        ds_t = data_type == "train" ? 
            ds["times"][t_idx] .- ds["times"][1] : 
            ds["times"][t_idx+n_shift] .- ds["times"][1]
        
        # Column title showing time and train/test
        col_title = col_idx <= 3 ? 
                    L"Train: $t$=%$(round(ds_t, digits=2))" : 
                    L"Test: $t$=%$(round(ds_t, digits=2))"
        
        # Row 1: Original
        ax_full = Axis(fig[1, col_idx], 
            title = col_title,
            ylabel = col_idx == 1 ? L"$\omega_2$" : "", 
            xlabelsize=30, ylabelsize=30, 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            titlesize=30,
        )
        
        # Row 2: Streaming OpInf
        ax_rom = Axis(fig[2, col_idx], 
            ylabel = col_idx == 1 ? L"$\omega_2$" : "", 
            xlabelsize=30, ylabelsize=30, 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
        )
        
        # Row 3: POD Projection
        ax_pod = Axis(fig[3, col_idx], 
            ylabel = col_idx == 1 ? L"$\omega_2$" : "", 
            xlabelsize=30, ylabelsize=30, 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
        )
        
        # Row 4: Streaming OpInf Error
        ax_error = Axis(fig[4, col_idx], 
            ylabel = col_idx == 1 ? L"$\omega_2$" : "", 
            xlabelsize=30, ylabelsize=30, 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
        )
        
        # Row 5: POD Error
        ax_pod_error = Axis(fig[5, col_idx], 
            ylabel = col_idx == 1 ? L"$\omega_2$" : "", 
            xlabel = L"$\omega_1$",
            xlabelsize=30, ylabelsize=30, 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
        )

        # Create heatmaps with aligned color ranges
        hm_full = heatmap!(ax_full, ds["x"][:], ds["y"][:], all_full_data[col_idx], 
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_rom = heatmap!(ax_rom, ds["x"][:], ds["y"][:], all_rom_data[col_idx], 
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_pod = heatmap!(ax_pod, ds["x"][:], ds["y"][:], all_pod_data[col_idx], 
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_error = heatmap!(ax_error, ds["x"][:], ds["y"][:], all_error_data[col_idx], 
            colormap = :matter, colorrange = (error_common_min, error_common_max))
        hm_pod_error = heatmap!(ax_pod_error, ds["x"][:], ds["y"][:], all_pod_error_data[col_idx], 
            colormap = :matter, colorrange = (error_common_min, error_common_max))
    end
    
    # Add colorbars at the end of each row
    Colorbar(fig[1, 6], hm_full, 
             label="Original", labelsize=26, ticklabelsize=25)
    Colorbar(fig[2, 6], hm_rom, 
             label="Streaming\nOpInf", labelsize=26, ticklabelsize=25)
    Colorbar(fig[3, 6], hm_pod, 
             label="Projection", labelsize=26, ticklabelsize=25)
    Colorbar(fig[4, 6], hm_error,
             label="Streaming\nOpInf Error", labelsize=26, ticklabelsize=25)
    Colorbar(fig[5, 6], hm_pod_error,
             label="Projection\nError", labelsize=26, ticklabelsize=25)
    
    save(joinpath(FILEPATH, "plots", 
         "$(fld)_slice_comparison_train_test_$(rmax).png"),
         dpi=600, px_per_unit=4.0, fig)
    display(fig)
end


#=============================================================================#
## Plot one reconstructed state over time for each field w/ projection error ##
#=============================================================================#
with_theme(theme_latexfonts()) do
    fig = Figure(size=(1200, 800))

    # Pick the first spatial point for each field
    nrow = 10000
    idx_u = nrow            # First point in u field
    idx_v = nxyz + nrow     # First point in v field
    idx_w = 2*nxyz + nrow   # First point in w field
    idx_p = 3*nxyz + nrow   # First point in p field

    field_indices = [idx_u, idx_v, idx_w, idx_p]
    field_names = ["u", "v", "w", "p"]
    field_colors = [Makie.wong_colors()[2] for _ in 1:4]
    pod_colors = [Makie.wong_colors()[5] for _ in 1:4]

    # Combined timeline: training followed by test
    tspan = ds["times"][1:n_train+n_test] .- ds["times"][1]
    train_end_time = ds["times"][n_train] - ds["times"][1]

    # Precompute reduced coefficients once (shared across all 4 fields).
    # Training POD coeffs come from the SVD factors directly (Σ * W').
    reduced_coeffs_train = iΣrmax * iWrmax'                     # rmax × n_train
    # Test POD coeffs: read+preprocess+project each snapshot once (not 4×).
    reduced_coeffs_test = zeros(size(iVrmax, 2), n_test)         # rmax × n_test
    @showprogress desc="Projecting test snapshots..." for t in 1:n_test
        x_full_preprocessed = preprocess!(copy(ds[t+n_train]), means, shifts, scales)
        mul!(view(reduced_coeffs_test, :, t), iVrmax', x_full_preprocessed)
    end

    # Extract all 4 field rows at once, then do 4 large matmuls instead of
    # 4 * (n_train + n_test) inner dot products.
    V_rows       = iVrmax[field_indices, :]              # 4 × rmax
    rom_train_all = V_rows * states                      # 4 × n_train
    rom_test_all  = V_rows * test_states                 # 4 × n_test
    pod_train_all = V_rows * reduced_coeffs_train        # 4 × n_train
    pod_test_all  = V_rows * reduced_coeffs_test         # 4 × n_test

    # Create legend at the top spanning all columns
    Legend(fig[0, 1],
        [LineElement(color=:black, linewidth=2),
        LineElement(color=field_colors[1], linewidth=2),
        LineElement(color=pod_colors[1], linewidth=2, linestyle=:dash),
        LineElement(color=:gray, linewidth=2, linestyle=:dot)],
        ["True", "Streaming-OpInf", "POD", "Train/Test split"],
        orientation=:horizontal,
        labelsize=25,
        patchsize=(60, 30),
        tellwidth=false,
        tellheight=true,
        framevisible=false,
    )

    for (i, (idx, name, color)) in enumerate(zip(field_indices, field_names, field_colors))
        ax = Axis(fig[i, 1],
            ylabel = L"%$(name)",
            xlabel = i == 4 ? "Time" : "",
            xlabelsize = 28,
            ylabelsize = 28,
            xticklabelsize = 22,
            yticklabelsize = 18,
            xticksvisible = i == 4 ? true : false,
            xticklabelsvisible = i == 4 ? true : false,
            # title = i == 1 ? "Reconstructed vs True States" : "",
            titlesize = 20
        )

        # Get factors to unprocess data
        mean_val = means[idx]
        shift_val = shifts[idx]
        scale_val = scales[idx]

        # Concatenate true training and test data
        true_field = vcat(ds[name][nrow, 1:n_train], ds[name][nrow, n_train+1:n_train+n_test])
        rom_field  = vcat(rom_train_all[i, :], rom_test_all[i, :])
        pod_field  = vcat(pod_train_all[i, :], pod_test_all[i, :])

        # Unprocess ROM field
        rom_field .*= scale_val
        rom_field .+= shift_val
        rom_field .+= mean_val

        # Unprocess POD field
        pod_field .*= scale_val
        pod_field .+= shift_val
        pod_field .+= mean_val

        # Plot true vs reconstructed (no labels since legend is at top)
        lines!(ax, tspan, true_field, color=:black, linewidth=4)
        lines!(ax, tspan, rom_field, color=color, linewidth=3)
        lines!(ax, tspan, pod_field, color=pod_colors[i], linewidth=3, linestyle=:dash)

        # Mark train/test boundary
        vlines!(ax, [train_end_time], color=:gray, linestyle=:dot, linewidth=3)
    end
    save(joinpath(FILEPATH, "plots",
         "state_evolution_train_test_$(rmax).png"), fig)
    display(fig)
end


#=========================================================#
## Plot one reconstructed state over time for each field ##
#=========================================================#
with_theme(theme_latexfonts()) do
    fig = Figure(size=(1200, 800))

    # Pick the first spatial point for each field
    nrow = 10000
    idx_u = nrow            # First point in u field
    idx_v = nxyz + nrow     # First point in v field
    idx_w = 2*nxyz + nrow   # First point in w field
    idx_p = 3*nxyz + nrow   # First point in p field

    field_indices = [idx_u, idx_v, idx_w, idx_p]
    field_names = ["u", "v", "w", "p"]
    field_colors = [:orange, :orange, :orange, :orange]

    # Pre-extract basis rows for each field (much more efficient)
    basis_rows = [iVrmax[idx, :] for idx in field_indices]

    # Combined timeline: training followed by test
    tspan = ds["times"][1:n_train+n_test] .- ds["times"][1]
    train_end_time = ds["times"][n_train] - ds["times"][1]

    for (i, (idx, name, color)) in enumerate(zip(field_indices, field_names, field_colors))
        ax = Axis(fig[i, 1],
            ylabel = L"%$(name)",
            xlabel = i == 4 ? "Time" : "",
            xlabelsize = 28,
            ylabelsize = 28,
            xticklabelsize = 22,
            yticklabelsize = 18,
            xticksvisible = i == 4 ? true : false,
            xticklabelsvisible = i == 4 ? true : false,
            # title = i == 1 ? "Reconstructed vs True States" : "",
            titlesize = 20
        )

        # Extract basis row once for this field
        basis_row = basis_rows[i]

        # Get factors to unprocess data
        mean_val = means[idx]
        shift_val = shifts[idx]
        scale_val = scales[idx]

        # Concatenate true training and test data
        true_field = vcat(ds[name][nrow, 1:n_train], ds[name][nrow, n_train+1:n_train+n_test])
        rom_field = zeros(n_train + n_test)

        Vrow = view(iVrmax, idx, :)

        # Training portion
        for t in 1:n_train
            states_col = view(states, :, t)
            rom_field[t] = dot(Vrow, states_col)
        end

        # Test portion
        for t in 1:n_test
            states_col = view(test_states, :, t)
            rom_field[n_train + t] = dot(Vrow, states_col)
        end

        rom_field .*= scale_val
        rom_field .+= shift_val
        rom_field .+= mean_val

        # Plot true vs reconstructed
        lines!(ax, tspan, true_field, color=:black, linewidth=2, label="True")
        lines!(ax, tspan, rom_field, color=color, linewidth=2, label="ROM")

        # Mark train/test boundary
        vlines!(ax, [train_end_time], color=:gray, linestyle=:dot, linewidth=2,
                label = nothing)

        # Add legend only to the bottom subplot
        if i == 4
            axislegend(ax, position=:rt, labelsize=25)
        end
    end
    save(joinpath(FILEPATH, "plots",
         "state_evolution_train_test_$(rmax).png"), fig)
    display(fig)
end


#=============================================#
## Create animation of field reconstruction ##
## (Training + Testing combined)            ##
#=============================================#
using CairoMakie

"""
    create_field_animation(fld; frame_skip=1, max_time=nothing, slice_dir="z", slice_idx=nothing)

Create an animation comparing original data, streaming-OpInf reconstruction, and error
for both training and testing data in a continuous timeline.

# Arguments
- `fld::String`: Field to animate ("u", "v", "w", or "p")
- `frame_skip::Int`: Skip frames to reduce file size (default=1, use every frame)
- `max_time::Union{Float64, Nothing}`: Maximum time to animate up to (default=nothing, animates all data)
- `slice_dir::String`: Direction to slice ("x", "y", or "z", default="z")
- `slice_idx::Union{Int, Nothing}`: Index along slice direction (default=nothing, uses midpoint)
"""
function create_field_animation(fld::String; frame_skip::Int=1, max_time::Union{Float64, Nothing}=nothing, 
                                slice_dir::String="z", slice_idx::Union{Int, Nothing}=nothing)
    @assert fld in ["u", "v", "w", "p"] "Field must be one of: u, v, w, p"
    @assert slice_dir in ["x", "y", "z"] "slice_dir must be one of: x, y, z"
    
    # Determine field indices
    if fld == "u"
        i_s = 1
        i_f = nxyz
    elseif fld == "v"
        i_s = nxyz + 1
        i_f = 2 * nxyz
    elseif fld == "w"
        i_s = 2 * nxyz + 1
        i_f = 3 * nxyz
    elseif fld == "p"
        i_s = 3 * nxyz + 1
        i_f = 4 * nxyz
    end
    
    # Determine slice index and plot dimensions based on slice direction
    if slice_dir == "z"
        slice_idx = slice_idx !== nothing ? slice_idx : nz ÷ 2
        @assert 1 <= slice_idx <= nz "slice_idx must be between 1 and $nz for z-direction"
        plot_x_coords = ds["x"][:]
        plot_y_coords = ds["y"][:]
        dim1, dim2 = nx, ny  # Width, Height of plot data
        use_vertical_layout = false
    elseif slice_dir == "y"
        slice_idx = slice_idx !== nothing ? slice_idx : ny ÷ 2
        @assert 1 <= slice_idx <= ny "slice_idx must be between 1 and $ny for y-direction"
        plot_x_coords = ds["x"][:]
        plot_y_coords = ds["z"][:]
        dim1, dim2 = nx, nz  # Dimensions for heatmap: (nx, nz)
        use_vertical_layout = true
    else  # slice_dir == "x"
        slice_idx = slice_idx !== nothing ? slice_idx : nx ÷ 2
        @assert 1 <= slice_idx <= nx "slice_idx must be between 1 and $nx for x-direction"
        plot_x_coords = ds["y"][:]
        plot_y_coords = ds["z"][:]
        dim1, dim2 = ny, nz  # Dimensions for heatmap: (ny, nz)
        use_vertical_layout = false
    end
    
    # Calculate aspect ratio for the plot
    aspect_ratio = dim1 / dim2
    
    # Set figure size based on layout
    if use_vertical_layout
        # For y slices: 3 rows, 2 columns (plot + colorbar)
        base_width = 1600
        fig_width = base_width
        fig_height_raw = base_width / aspect_ratio * 2.5 # 3 rows
        fig_height = Int(round(fig_height_raw / 2) * 2)
        fig_width = Int(round(fig_width / 2) * 2)
    else
        if slice_dir == "x"
            # Constrain to reasonable limits
            max_width = 2000
            fig_width = max_width
            fig_height = Int(round(max_width / aspect_ratio / 2.5 / 2) * 2)
        else  
            # For z slice: 1 row, 5 columns (3 plots + 2 colorbars)
            base_height = 720
            fig_height = base_height
            fig_width_raw = base_height * aspect_ratio * 2.4
            fig_width = Int(round(fig_width_raw / 2) * 2)
            fig_height = Int(round(fig_height / 2) * 2)
            
            # Constrain to reasonable limits
            max_width = 2400
            if fig_width > max_width
                fig_width = max_width
                fig_height = Int(round(max_width / aspect_ratio / 2.4 / 2) * 2)
            end
        end
    end
    
    # Create combined time span
    tspan_combined = ds["times"][1:n_train+n_test] .- ds["times"][1]
    
    # Determine the maximum index based on max_time
    if max_time !== nothing
        max_idx = findfirst(t -> t >= max_time, tspan_combined)
        if max_idx === nothing
            max_idx = n_train + n_test
            @warn "max_time ($max_time) exceeds available data. Using all data."
        end
    else
        max_idx = n_train + n_test
    end
    
    time_indices = 1:frame_skip:max_idx
    
    println("Animating from time 0.0 to $(tspan_combined[max_idx]) (index 1 to $max_idx)")
    println("Slice direction: $slice_dir, index: $slice_idx")
    println("Plot dimensions: $(dim1) x $(dim2), aspect ratio: $(round(aspect_ratio, digits=2))")
    println("Figure size: $(fig_width) x $(fig_height)")
    println("Layout: $(use_vertical_layout ? "3×2 (vertical)" : "1×5 (horizontal)")")
    
    # Helper function to extract slice based on direction
    function extract_slice(field_3d, slice_dir, slice_idx)
        if slice_dir == "z"
            # field_3d is (nx, ny, nz), return (nx, ny)
            return field_3d[:, :, slice_idx]
        elseif slice_dir == "y"
            # field_3d is (nx, ny, nz), extract y-slice and transpose
            # We want (nx, nz) to match dim1=nx, dim2=nz
            return field_3d[:, slice_idx, :]
        else  # x
            # field_3d is (nx, ny, nz), extract x-slice and transpose
            # We want (ny, nz) to match dim1=ny, dim2=nz  
            return field_3d[slice_idx, :, :]
        end
    end 

    # Pre-calculate global min/max for consistent color scaling
    println("Calculating global color ranges for field: $fld")
    all_full_min, all_full_max = Inf, -Inf
    all_rom_min, all_rom_max = Inf, -Inf
    all_error_min, all_error_max = Inf, -Inf
    
    # Sample frames from the range we're animating
    sample_step = max(1, max_idx ÷ 100)
    sample_indices = 1:sample_step:max_idx
    
    for t_idx in sample_indices
        # Get full data
        u_full_field = reshape(ds[t_idx][i_s:i_f], nx, ny, nz)
        full_slice = extract_slice(u_full_field, slice_dir, slice_idx)
        
        # Get ROM data
        if t_idx <= n_train
            x_rom_t = iVrmax * states[:, t_idx]
        else
            x_rom_t = iVrmax * test_states[:, t_idx - n_train]
        end
        x_rom_t = unprocess!(x_rom_t, means, shifts, scales)
        u_rom_field = reshape(x_rom_t[i_s:i_f], nx, ny, nz)
        rom_slice = extract_slice(u_rom_field, slice_dir, slice_idx)
        
        # Compute error
        error_slice = abs.(full_slice - rom_slice)
        
        # Update min/max
        all_full_min = min(all_full_min, minimum(full_slice))
        all_full_max = max(all_full_max, maximum(full_slice))
        all_rom_min = min(all_rom_min, minimum(rom_slice))
        all_rom_max = max(all_rom_max, maximum(rom_slice))
        all_error_min = min(all_error_min, minimum(error_slice))
        all_error_max = max(all_error_max, maximum(error_slice))
    end
    
    # Align color ranges for data fields
    common_min = min(all_full_min, all_rom_min)
    common_max = max(all_full_max, all_rom_max)
    
    println("Creating animation...")
    
    with_theme(theme_latexfonts()) do
        fig = Figure(size=(fig_width, fig_height))
        
        if use_vertical_layout
            # 3×2 layout for x and y slices
            # Row 1: Original
            ax_full = Axis(fig[1, 1],
                xlabelsize=30, ylabelsize=30,
                xticklabelsvisible=false, xticksvisible=false,
                yticklabelsvisible=false, yticksvisible=false,
                ylabel = "Original",
                ylabelrotation = 0,
                titlesize=30,
                aspect = DataAspect()
            )
            
            # Row 2: Streaming OpInf
            ax_rom = Axis(fig[2, 1],
                xlabelsize=30, ylabelsize=30,
                xticklabelsvisible=false, xticksvisible=false,
                yticklabelsvisible=false, yticksvisible=false,
                ylabel =  "Streaming OpInf",
                ylabelrotation = 0,
                titlesize=30,
                aspect = DataAspect()
            )
            
            # Row 3: Absolute Error
            ax_error = Axis(fig[3, 1],
                xlabelsize=30, ylabelsize=30,
                xticklabelsvisible=false, xticksvisible=false,
                yticklabelsvisible=false, yticksvisible=false,
                ylabel = "Absolute Error",
                ylabelrotation = 0,
                titlesize=30,
                aspect = DataAspect()
            )
            
            # Initialize observables for the data
            full_data = Observable(zeros(dim1, dim2))
            rom_data = Observable(zeros(dim1, dim2))
            error_data = Observable(zeros(dim1, dim2))
            
            # Create heatmaps
            hm_full = heatmap!(ax_full, plot_x_coords, plot_y_coords, full_data,
                colormap = :viridis, colorrange = (common_min, common_max))
            hm_rom = heatmap!(ax_rom, plot_x_coords, plot_y_coords, rom_data,
                colormap = :viridis, colorrange = (common_min, common_max))
            hm_error = heatmap!(ax_error, plot_x_coords, plot_y_coords, error_data,
                colormap = :matter, colorrange = (all_error_min, all_error_max))
            
            # Add colorbars spanning rows
            Colorbar(fig[1:2, 2], hm_full,
                label="Field Value", labelsize=26, ticklabelsize=22,
                )
            Colorbar(fig[3, 2], hm_error,
                label="Error", labelsize=26, ticklabelsize=22)
            
            # Add time and phase label at top
            time_label = Label(fig[0, 1:2], "", fontsize=32, tellwidth=false)
            
        else
            # 1×5 layout for z slice (original horizontal layout)
            ax_full = Axis(fig[1, 1],
                xlabelsize=30, ylabelsize=30,
                xticklabelsvisible=false, xticksvisible=false,
                yticklabelsvisible=false, yticksvisible=false,
                title = "Original",
                titlesize=30,
                aspect = DataAspect()
            )
            
            ax_rom = Axis(fig[1, 2],
                xlabelsize=30, ylabelsize=30,
                xticklabelsvisible=false, xticksvisible=false,
                yticklabelsvisible=false, yticksvisible=false,
                title = "Streaming OpInf",
                titlesize=30,
                aspect = DataAspect()
            )
            
            ax_error = Axis(fig[1, 3],
                xlabelsize=30, ylabelsize=30,
                xticklabelsvisible=false, xticksvisible=false,
                yticklabelsvisible=false, yticksvisible=false,
                title = "Absolute Error",
                titlesize=30,
                aspect = DataAspect()
            )
            
            # Initialize observables for the data
            full_data = Observable(zeros(dim1, dim2))
            rom_data = Observable(zeros(dim1, dim2))
            error_data = Observable(zeros(dim1, dim2))
            
            # Create heatmaps
            hm_full = heatmap!(ax_full, plot_x_coords, plot_y_coords, full_data,
                colormap = :viridis, colorrange = (common_min, common_max))
            hm_rom = heatmap!(ax_rom, plot_x_coords, plot_y_coords, rom_data,
                colormap = :viridis, colorrange = (common_min, common_max))
            hm_error = heatmap!(ax_error, plot_x_coords, plot_y_coords, error_data,
                colormap = :matter, colorrange = (all_error_min, all_error_max))
            
            # Add colorbars
            Colorbar(fig[1, 4], hm_full,
                label="Field Value", labelsize=26, ticklabelsize=22)
            Colorbar(fig[1, 5], hm_error,
                label="Error", labelsize=26, ticklabelsize=22)
            
            # Add time and phase label at top
            time_label = Label(fig[0, 1:3], "", fontsize=32, tellwidth=false)
        end
        
        # Create animation filename with time suffix if specified
        time_suffix = max_time !== nothing ? "_t$(Int(round(max_time)))" : "_full"
        output_path = joinpath(FILEPATH, "plots",
            "$(fld)_animation_combined_$(slice_dir)_slice_$(rmax)$(time_suffix).mp4")
        
        # Use explicit format and codec settings to avoid issues
        record(fig, output_path, time_indices; 
               framerate=30, format="mp4", compression=5,
               px_per_unit=1.0) do t_idx
            
            # Determine if we're in training or testing phase
            phase = t_idx <= n_train ? "Training" : "Testing"
            
            # Update time label
            ds_t = tspan_combined[t_idx]
            time_label.text = @sprintf(
                "%s: Time = %.3f, %d/%d = %d× reduction", 
                phase, ds_t, nxyz*4, rmax, div(nxyz*4, rmax))
            
            # Get full data
            u_full_field = reshape(ds[t_idx][i_s:i_f], nx, ny, nz)
            full_data[] = extract_slice(u_full_field, slice_dir, slice_idx)
            
            # Get ROM data (from training or testing states)
            if t_idx <= n_train
                x_rom_t = iVrmax * states[:, t_idx]
            else
                x_rom_t = iVrmax * test_states[:, t_idx - n_train]
            end
            x_rom_t = unprocess!(x_rom_t, means, shifts, scales)
            u_rom_field = reshape(x_rom_t[i_s:i_f], nx, ny, nz)
            rom_data[] = extract_slice(u_rom_field, slice_dir, slice_idx)
            
            # Compute error
            error_data[] = abs.(full_data[] - rom_data[])
        end
        
        println("Animation saved to: $output_path")
    end
end

## Animate streamwise velocity with z-direction slice (default)
create_field_animation("u", frame_skip=1, max_time=0.01, slice_dir="x")

## Create animations for all fields 
for dir in ["x", "y", "z"]
    println("\nCreating animations for slice direction: $dir")
    for fld in ["u", "v", "w", "p"]
        println("  Field: $fld")
        create_field_animation(fld, frame_skip=1, max_time=100.0, slice_dir=dir)
    end
end