"""
3D Channel flow: Reconstruct the states using the ROM
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
nz, ny, nx, n_fields, n = ds.dims
nxyz = nz * ny * nx
n_test = 2000
n_train = n - n_test
rmax = 400

#=======================#
## Load the Stream model 
#=======================#
op_stream = load(joinpath(FILEPATH, 
        "data/results/op_stream_r$(rmax).jld2"))["op_stream_r$(rmax)"]

#=================#
## Load the bases 
#=================#
basis_file = joinpath(FILEPATH, "data/results/onepass_stream.jld2")
iVrmax = load(basis_file)["stream"].V[:, 1:rmax]

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

#======================#
## Plot the u-velocity 
#======================#
using CairoMakie

with_theme(theme_latexfonts()) do 
    train_or_test = "test"
    fig = Figure(size=(1200, 940))

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


#=========================================================#
## Plot one reconstructed state over time for each field 
#=========================================================#
with_theme(theme_latexfonts()) do 
    train_or_test = "test"

    fig = Figure(size=(1200, 800))
    
    # Pick the first spatial point for each field
    nrow = 1
    idx_u = nrow            # First point in u field
    idx_v = nxyz + nrow     # First point in v field  
    idx_w = 2*nxyz + nrow   # First point in w field
    idx_p = 3*nxyz + nrow   # First point in p field
    
    field_indices = [idx_u, idx_v, idx_w, idx_p]
    field_names = ["u", "v", "w", "p"]
    field_colors = [:orange, :orange, :orange, :orange]
    
    # Pre-extract basis rows for each field (much more efficient)
    basis_rows = [iVrmax[idx, :] for idx in field_indices]

    # Reconstruct full states for 
    if train_or_test == "train"
        tspan = ds["times"][1:n_train] .- ds["times"][1]
    else
        tspan = ds["times"][n_train+1:n_train+n_test] .- ds["times"][n_train+1]
    end
    
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
        if train_or_test == "train"
            mean_val = means[idx]
            shift_val = shifts[idx]
            scale_val = scales[idx]
        else
            mean_val = means[idx]
            shift_val = shifts[idx]
            scale_val = scales[idx]
        end

        if train_or_test == "train"
            true_field = ds[name][nrow, 1:n_train]
            rom_field = zeros(n_train)
        else
            true_field = ds[name][nrow, n_train+1:n_train+n_test]
            rom_field = zeros(n_test)
        end
        
        if train_or_test == "train"
            for t in 1:n_train
                Vrow = view(iVrmax, idx, :)
                states_col = view(states, :, t)
                rom_field[t] = dot(Vrow, states_col)
            end
        else
            for t in 1:n_test
                Vrow = view(iVrmax, idx, :)
                states_col = view(test_states, :, t)
                rom_field[t] = dot(Vrow, states_col)
            end
        end
        rom_field .*= scale_val
        rom_field .+= shift_val
        rom_field .+= mean_val
        
        # Plot true vs reconstructed
        lines!(ax, tspan, true_field, color=:black, linewidth=2, label="True")
        lines!(ax, tspan, rom_field, color=color, linewidth=2, label="ROM")
        
        # Add legend only to the top subplot
        if i == 4
            axislegend(ax, position=:rt, labelsize=25)
        end
    end
    save(joinpath(FILEPATH, "plots", 
         "state_evolution_$(train_or_test)_$(rmax).png"), fig)
    display(fig)
end