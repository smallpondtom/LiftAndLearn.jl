"""
3D Channel flow: training models
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

#========================#
## Additional functions ##
#========================#
include(joinpath(FILEPATH, "derivative.jl"))
include(joinpath(FILEPATH, "preprocess.jl"))

#=============================#
## Load the training dataset
#=============================#
ds = ChannelDataSource(datafile, ["z", "y", "x", "fields", "times"])
nz, ny, nx, n_fields, n = ds.dims
nxyz = nz * ny * nx
n_test = 2000
n_train = n - n_test

#===================#
## Setup the options
#===================#
# Some options for operator inference
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=[1,2],
        control=0,
        constant=1,
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
    use_backslash=true,
)
rmax = 200

#=======================#
## Load the batch model 
#=======================#
GRID_SEARCH = true
if GRID_SEARCH
    op = load(joinpath(FILEPATH, "data/models", 
            "batch_operators_0_8000_r$(rmax)_lamGS.jld2"))["op"]
    # load the optimal regularization parameters 
    beta1 = load(joinpath(FILEPATH, "data/results",
                 "reg_grid_search.jld2"))["beta1"]  
    beta2 = load(joinpath(FILEPATH, "data/results",
                 "reg_grid_search.jld2"))["beta2"]  
else
    op = load(joinpath(FILEPATH, "data/models", 
            "batch_operators_0_8000_r$(rmax)_lam1e12.jld2"))["op"]
    beta1 = 1e12
    beta2 = 1e12
end


#=========================#
## Load reduced data
#=========================#
CONTINUOUS_TIME = true
if CONTINUOUS_TIME
    Xhat = load(joinpath(FILEPATH, "data/streaming/reduced_data_r$(rmax).jld2"))["Xhat"]
    Xhatdot = load(joinpath(FILEPATH, "data/streaming/reduced_data_r$(rmax).jld2"))["Xhatdot"]
else
    Xhat = load(joinpath(FILEPATH, "data/streaming/reduced_data_r$(rmax).jld2"))["Xhat"]
    Xhatdot = Xhat[:, 1:end-1]  
    Xhat = Xhat[:, 2:end]
end
size(Xhat,1) != rmax && @warn "Xhat has a different number of \
    rows than the basis. This might lead to unexpected results."


#=========================#
## Train streaming model
#=========================#
options.with_reg = true
options.λ = LnL.TikhonovParameter(A=beta1, A2=beta2, K=beta1)

stream_error = Dict(
    :rls    => zeros(n_train),
    :iqrrls => zeros(n_train),
    :qrrls  => zeros(n_train),
)

## RLS
rls_stream  = LnL.TwoPassStreamingOpInf(
    options=options, n=rmax, m=0, algorithm=:RLS, qr_method=:givens, use_gpu=false) 
Ostar = op.O'
Ostar_norm = norm(Ostar, 2)  

##
tmp = nothing  # temporary variable to store previous stream error
@showprogress for i in 1:n_train
    xhat_i = @views Xhat[:,i]
    xhatdot_i = @views Xhatdot[:,i]  
    LnL.stream!(rls_stream, xhat_i, xhatdot_i)
    stream_error[:rls][i] = norm(rls_stream.cache.O - Ostar) / Ostar_norm
end
op_rls = LnL.terminate_stream(rls_stream)
rls_stream = nothing
GC.gc()

## iQRRLS
iqrrls_stream  = LnL.TwoPassStreamingOpInf(
    options=options, n=rmax, m=0, algorithm=:iQRRLS, qr_method=:givens) 
tmp = nothing  # temporary variable to store previous stream error
@showprogress for i in 1:n_train
    xhat_i = @views Xhat[:,i]
    xhatdot_i = @views Xhatdot[:,i]  
    LnL.stream!(iqrrls_stream, xhat_i, xhatdot_i)
    stream_error[:iqrrls][i] = norm(iqrrls_stream.cache.O - Ostar) / Ostar_norm
end
op_iqrrls = LnL.terminate_stream(iqrrls_stream)
iqrrls_stream = nothing
GC.gc()

## QRRLS
qrrls_stream  = LnL.TwoPassStreamingOpInf(
    options=options, n=rmax, m=0, algorithm=:QRRLS, qr_method=:givens) 
tmp = nothing  # temporary variable to store previous stream error
@showprogress for i in 1:n_train
    xhat_i = @views Xhat[:,i]
    xhatdot_i = @views Xhatdot[:,i]  
    LnL.stream!(qrrls_stream, xhat_i, xhatdot_i)
    stream_error[:qrrls][i] = norm(qrrls_stream.cache.O - Ostar) / Ostar_norm
end
op_qrrls = LnL.terminate_stream(qrrls_stream)
qrrls_stream = nothing
GC.gc()

## Save operators
save(joinpath(
        FILEPATH, "data/models", 
        "stream_rls_operators_0_8000_r$(rmax)_lam1e12.jld2"
    ), 
    "op_rls", op_rls,
    "op_iqrrls", op_iqrrls,
    "op_qrrls", op_qrrls,
)

#============================#
## Save the streaming errors
#============================#
save(joinpath(FILEPATH, "data/results", 
     "streaming_errors_0_8000_r$(rmax).jld2"), 
     "stream_error", stream_error)

#===========================#
## Simulate ROM (training) ##
#===========================#
include(joinpath(FILEPATH, "integrate.jl"))
states = Dict(
    :rls    => zeros(size(Xhat,1), n_train),
    :iqrrls => zeros(size(Xhat,1), n_train),
    :qrrls  => zeros(size(Xhat,1), n_train),
)
for (algo, op_stream) in zip([:rls, :iqrrls, :qrrls], 
                             [op_rls, op_iqrrls, op_qrrls])
    if CONTINUOUS_TIME
        tspan = ds["times"][1:n_train] .- ds["times"][1]
        x0 = Xhat[:,1]
        states[algo] = rk4_integrate(x0, tspan, op_stream.A, 
                                     op_stream.A2u, op_stream.K)
    else
        states[algo][:,1] = x0
        for j in 2:n_train
            states[algo][:,j] = reduced_model(
                states[algo][:,j-1], 
                op_stream.A, op_stream.A2u, op_stream.K)
            if any(isnan.(states[algo][:,j]))
                @warn "NaN detected in trajectory $i at time step $j"
                break
            end
        end
    end
end

## Save the training states
save(joinpath(FILEPATH, "data/results", 
     "stream_rom_train_sim_states_0_8000_r$(rmax).jld2"), 
     "states", states)

## Load the state states
states = load(joinpath(FILEPATH, "data/results", 
              "stream_rom_train_sim_states_0_8000_r400.jld2"))["states"]


#=================#
## Load the bases 
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

#==========================#
## Simulate ROM (testing) ##
#==========================#
include(joinpath(FILEPATH, "preprocess.jl"))
test_states = Dict(
    :rls    => zeros(size(Xhat,1), n_test),
    :iqrrls => zeros(size(Xhat,1), n_test),
    :qrrls  => zeros(size(Xhat,1), n_test),
)
for (alg, op_stream) in zip([:rls, :iqrrls, :qrrls], 
                            [op_rls, op_iqrrls, op_qrrls])
    if CONTINUOUS_TIME
        tspan = ds["times"][n_train+1:n_train+n_test] .- ds["times"][n_train+1]
        # Make sure to preprocess the first state
        x0 = iVrmax' * preprocess!(ds[n_train+1], means, shifts, scales)
        # x0 = iVrmax' * preprocess!(ds[n_train+1], means_test, shifts_test, scales_test)
        test_states[alg] = rk4_integrate(
            x0, tspan, op_stream.A, op_stream.A2u, op_stream.K)
    else
        test_states[alg][:,1] = iVrmax * preprocess!(
            ds[n_train+1], means_test, shifts_test, scales_test)
        for j in 2:n_test
            test_states[alg][:,j] = reduced_model(
                test_states[alg][:,j-1], op_stream.A, op_stream.A2u, op_stream.K)
            if any(isnan.(test_states[alg][:,j]))
                @warn "NaN detected in trajectory $i at time step $j"
                break
            end
        end
    end
end

## Save the test states
save(joinpath(FILEPATH, "data/results", 
     "stream_rom_test_sim_states_0_8000_r$(rmax).jld2"), 
     "states", test_states)

## Load the test states
test_states = load(joinpath(FILEPATH, "data/results", 
                   "stream_rom_test_sim_states_0_8000_r$(rmax).jld2"))["states"]


#======================#
## Plot the u-velocity 
#======================#
using CairoMakie

with_theme(theme_latexfonts()) do 
    train_or_test = "test"
    alg = :rls

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
            x_rom_t = iVrmax * states[alg][:, t_idx]
            x_rom_t = unprocess!(x_rom_t, means, shifts, scales)
        else
            x_rom_t = iVrmax * test_states[alg][:, t_idx]
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
         "$(fld)_slice_comparison_$(train_or_test)_$(String(alg)).png"), fig)
    display(fig)
end


#=========================================================#
## Plot one reconstructed state over time for each field 
#=========================================================#
with_theme(theme_latexfonts()) do 
    train_or_test = "test"

    fig = Figure(size=(1200, 800))
    
    # Pick the first spatial point for each field
    nrow = 50
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
            xlabelsize = 20, 
            ylabelsize = 20,
            xticklabelsize = 15, 
            yticklabelsize = 15,
            title = i == 1 ? "Reconstructed vs True States" : "",
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
            # mean_val = means_test[idx]
            # shift_val = shifts_test[idx]
            # scale_val = scales_test[idx]
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
        if i == 1
            axislegend(ax, position=:rt, labelsize=25)
        end
    end
    
    # save(joinpath(FILEPATH, "plots", "state_evolution_$(train_or_test).png"), fig)
    display(fig)
end


#====================================================#
## Plot the mean flow of each field and their errors
#====================================================#
# Compute mean flows of ROM
include(joinpath(FILEPATH, "preprocess.jl"))
means_rom = compute_mean_parallel_threads_fixed_rom(states, iVrmax, 4*nxyz, 
                                                    n_time; batch_size=100)
save(joinpath(FILEPATH, "data/results/mean_rom.jld2"), "means_rom", means_rom)

##

# 3D Mean Flow Comparison Plot
using GLMakie
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1800, 1200))
    
    # Unprocess the ROM mean flows
    means_rom_unprocessed = copy(means_rom)
    means_rom_unprocessed = unprocess!(means_rom_unprocessed, means, shifts, scales)
    
    # Field names and their corresponding indices
    field_names = ["u", "v", "w", "p"]
    
    for (row, field) in enumerate(field_names)
        # Extract field data
        i_s = nxyz * (row - 1) + 1
        i_f = nxyz * row
        
        # Get mean data for this field
        mean_orig = means[i_s:i_f]
        mean_rom = means_rom_unprocessed[i_s:i_f]
        mean_error = abs.(mean_orig - mean_rom)
        
        # Reshape data to 3D
        mean_orig_3d = reshape(mean_orig, nx, ny, nz)
        mean_rom_3d = reshape(mean_rom, nx, ny, nz)
        mean_error_3d = reshape(mean_error, nx, ny, nz)
        
        # Get coordinate spans and convert to ranges
        x_span = ds["x"][:]
        y_span = ds["y"][:]
        z_span = ds["z"][:]
        
        # Convert to endpoint ranges
        x_range = x_span[1]..x_span[end]
        y_range = y_span[1]..y_span[end]
        z_range = z_span[1]..z_span[end]
        
        # Calculate unified color range for original and ROM data
        combined_min = min(minimum(mean_orig), minimum(mean_rom))
        combined_max = max(maximum(mean_orig), maximum(mean_rom))
        error_min = minimum(mean_error)
        error_max = maximum(mean_error)
        
        # Create 3D axes for each column
        # Column 1: Original mean flow
        ax1 = Axis3(fig[row, 1],
            xlabel = "x", ylabel = "y", zlabel = "z",
            xlabelsize = 20, ylabelsize = 20, zlabelsize = 20,
            xticklabelsvisible = false, yticklabelsvisible = false, zticklabelsvisible = false,
            xticksvisible = false, yticksvisible = false, zticksvisible = false,
            title = row == 1 ? "Original" : "", aspect = (3,2,1)
        )
        
        # Column 2: ROM mean flow
        ax2 = Axis3(fig[row, 2],
            xlabel = "x", ylabel = "y", zlabel = "z",
            xlabelsize = 20, ylabelsize = 20, zlabelsize = 20,
            xticklabelsvisible = false, yticklabelsvisible = false, zticklabelsvisible = false,
            xticksvisible = false, yticksvisible = false, zticksvisible = false,
            title = row == 1 ? "ROM" : "",aspect = (3,2,1)
        )
        
        # Column 3: Error
        ax3 = Axis3(fig[row, 4],
            xlabel = "x", ylabel = "y", zlabel = "z",
            xlabelsize = 20, ylabelsize = 20, zlabelsize = 20,
            xticklabelsvisible = false, yticklabelsvisible = false, zticklabelsvisible = false,
            xticksvisible = false, yticksvisible = false, zticksvisible = false,
            title = row == 1 ? "Error" : "", aspect = (3,2,1)
        )
        
        # Create volume plots with endpoint ranges
        vol1 = volume!(ax1, x_range, y_range, z_range, mean_orig_3d,
            colorrange = (combined_min, combined_max),
            colormap = :viridis)
            
        vol2 = volume!(ax2, x_range, y_range, z_range, mean_rom_3d,
            colorrange = (combined_min, combined_max),
            colormap = :viridis)
            
        vol3 = volume!(ax3, x_range, y_range, z_range, mean_error_3d,
            colorrange = (error_min, error_max),
            colormap = :matter)
        
        # Add field name as row label
        Label(fig[row, 0], field, rotation = π/2, fontsize = 30, 
              tellheight = false, tellwidth = true)
        
        # Add colorbars
        if row == 1
            # Unified colorbar for original and ROM (after column 2)
            Colorbar(fig[row, 3], vol2, 
                labelsize = 20,
                ticklabelsize = 15)
            
            # Error colorbar (after column 3)  
            Colorbar(fig[row, 5], vol3,
                labelsize = 20,
                ticklabelsize = 15)
        else
            # For other rows, create invisible colorbars to maintain spacing
            Colorbar(fig[row, 3], vol2, 
                label = "", 
                labelsize = 20,
                ticklabelsize = 15)
            
            Colorbar(fig[row, 5], vol3,
                label = "",
                labelsize = 20,
                ticklabelsize = 15)
        end
    end

    colsize!(fig.layout, 1, Fixed(370))
    colsize!(fig.layout, 2, Fixed(370))
    colsize!(fig.layout, 3, Fixed(45))
    colsize!(fig.layout, 4, Fixed(370))
    colsize!(fig.layout, 5, Fixed(45))
    
    # Add overall title
    Label(fig[0, 1:5], "3D Mean Flow Comparison", fontsize = 35, tellwidth = false)
    save(joinpath(FILEPATH, "plots", "3d_mean_flow_comparison.png"), fig)
    display(fig)
end

##

with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1200, 800))

    means_rom_unprocessed = copy(means_rom)
    means_rom_unprocessed = unprocess!(means_rom_unprocessed, means, shifts, scales)
    
    # Plot each field and its error
    for (i, field) in enumerate(["u", "v", "w", "p"])
        # First column: Mean flows
        ax1 = Axis(fig[i, 1], 
            ylabel = L"%$(field)", 
            xlabel = i == 4 ? "grid point" : "",
            xlabelsize = 20, 
            ylabelsize = 20,
            xticklabelsize = 15, 
            yticklabelsize = 15,
            title = i == 1 ? "Mean Flows" : "",
            titlesize = 20
        )
        
        # Second column: Errors
        ax2 = Axis(fig[i, 2], 
            ylabel = i == 1 ? "Error" : "",
            xlabel = i == 4 ? "grid point" : "",
            xlabelsize = 20, 
            ylabelsize = 20,
            xticklabelsize = 15, 
            yticklabelsize = 15,
            title = i == 1 ? "Absolute Errors" : "",
            titlesize = 20,
        )

        i1 = (i - 1) * nxyz + 1
        i2 = i * nxyz

        full = @view means[i1:i2]
        rom = @view means_rom_unprocessed[i1:i2]
        error = abs.(full - rom)
        
        # Plot mean fields in first column
        lines!(ax1, 1:nxyz, full, color=:black, linewidth=3, label="Original")
        lines!(ax1, 1:nxyz, rom, color=:orange, linewidth=0.5, linestyle=:dash, label="ROM")
        
        # Plot error in second column
        lines!(ax2, 1:nxyz, error, color=:black, linewidth=1, label="Abs. Error")
        
        # Add legend only to the top subplot of first column
        if i == 1
            axislegend(ax1, position=:rt, labelsize=15)
            axislegend(ax2, position=:rt, labelsize=15)
        end
    end
    
    save(joinpath(FILEPATH, "plots", "mean_flow_and_errors.png"), fig)
    display(fig)
end
