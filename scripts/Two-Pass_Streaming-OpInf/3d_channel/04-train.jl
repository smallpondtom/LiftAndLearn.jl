"""
3D Channel flow: training models
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
    # use_backslash=true,
    use_svd_truncation=true,
    # tolerance=1e-22
)
rmax = 350

#=========================#
## Load reduced data
#=========================#
CONTINUOUS_TIME = true
if CONTINUOUS_TIME
    Xhat = load(joinpath(FILEPATH, 
                "data/streaming/reduced_data_r$(rmax).jld2"))["Xhat"]
    Xhatdot = load(joinpath(FILEPATH, 
                   "data/streaming/reduced_data_r$(rmax).jld2"))["Xhatdot"]
else
    Xhat = load(joinpath(FILEPATH, 
                "data/streaming/reduced_data_r$(rmax).jld2"))["Xhat"]
    Xhatdot = Xhat[:, 1:end-1]  
    Xhat = Xhat[:, 2:end]
end
size(Xhat,1) != rmax && @warn "Xhat has a different number of \
    rows than the basis. This might lead to unexpected results."

#========================================#
## Grid Search Regularization Parameters
#========================================#
function simulate_opinf(x0, n_time, op, tspan=nothing, continuous=true)
    contains_nan = false
    final_idx = 0
    if continuous
        states, final_idx = rk4_integrate(x0, tspan, op.A, op.A2u, op.K)
        contains_nan = final_idx < n_time ? true : false
    else
        states = zeros(size(op.A, 1), n_time)
        states[:, 1] = x0
        for j in 2:n_time
            states[:, j] = reduced_model(states[:, j-1], op.A, op.A2u, op.K)
            fidx = j
            if any(isnan.(states[:, j]))
                @warn "NaN detected in trajectory at time step $j"
                contains_nan = true
                break
            end
        end
    end
    return contains_nan, states, final_idx
end


function find_best_opinf_model(
    reg_pairs, Xhat, Xhat1, Xhat2, 
    n_time, n_time_pred, max_growth, opinf_options, 
    tspan=nothing, continuous=true)

    @assert options.with_reg == true "Regularization must be enabled in options."
    
    best_train_err = 1e20
    best_beta1, best_beta2 = nothing, nothing
    best_final_idx = 0
    Xtilde_opt = nothing
    eval_time_opt = nothing
    best_model = nothing

    mean_Xhat = mean(Xhat, dims=2)
    max_diff_Xhat = maximum(abs.(Xhat .- mean_Xhat), dims=2)
    
    # Loop over all regularization pairs
    for (beta1, beta2) in reg_pairs
        
        # Construct a regularizer that penalizes the linear and constant reduced
        # operators using beta1 and the quadratic operator using beta2
        reg = LnL.TikhonovParameter(A=beta1, A2=beta2, K=beta1)
        opinf_options.λ = reg
        
        # Solve the regularized OpInf problem
        ops = LnL.opinf(Xhat1, opinf_options; Xhatdot=Xhat2)
        
        # Extract the reduced initial condition from Qhat_1
        xhat0 = Xhat1[:,1]
        
        # Compute the reduced solution over the trial time horizon
        start_eval_time = time()
        contains_nans, Xtilde, fidx = simulate_opinf(
            xhat0, n_time_pred, ops, tspan, continuous)
        end_eval_time = time()
        time_opinf_eval = end_eval_time - start_eval_time
        
        # If the model produced an unstable solution, move on to the next
        # regularization candidates
        if contains_nans
            ops = nothing
            GC.gc() 
            continue
        end
        
        # If the ratio of the maximum coefficient growth exceeds the allowed
        # threshold, move on to the next regularization candidates
        max_diff_Xhat_trial = maximum(abs.(Xtilde .- mean_Xhat), dims=2)
        max_growth_trial = maximum(max_diff_Xhat_trial) / maximum(max_diff_Xhat)
        if max_growth_trial > max_growth
            ops = nothing
            GC.gc() 
            continue
        end
        
        # At this point we know the model produced a stable solution without too
        # much growth. Compute the training error and, if it's better than the
        # current best error, save the regularization, reduced solution, and
        # the learning times
        train_err = norm(
                Xhat[:, 1:n_time] - Xtilde[:, 1:n_time]
            )^2 / norm(Xhat[:, 1:n_time])^2
        if train_err < best_train_err
            best_beta1 = beta1
            best_beta2 = beta2
            best_train_err = train_err
            Xtilde_opt = Xtilde
            eval_time_opt = time_opinf_eval
            best_model = ops
        end

        if best_final_idx < fidx
            best_final_idx = fidx
        end

        @info "Regularization pair (β1, β2) = ($beta1, $beta2): \
               training error = $train_err, evaluation time = $time_opinf_eval, \
               max growth = $max_growth_trial, final index = $fidx"
        ops = nothing
        GC.gc() 
    end

    if isnothing(Xtilde_opt)
        @error "No suitable OpInf model found with the given regularization pairs."
    else
        @info "Best OpInf model found with β1 = $best_beta1, β2 = $best_beta2, \
               training error = $best_train_err, evaluation time = $eval_time_opt"
    end

    return (best_model, best_beta1, best_beta2, best_train_err, 
            Xtilde_opt, eval_time_opt, best_final_idx)
end

## Run grid Search
B1 = 10.0 .^ range(11.0, 13.0, length=10)
B2 = 10.0 .^ range(11.0, 13.0, length=10)
reg_pairs_global = vec([(b1, b2) for b1 in B1, b2 in B2])
n_reg_global = length(reg_pairs_global)
max_growth = 1.2
options.with_reg = true
op, best_beta1, best_beta2, best_train_err, states, eval_time, fidx = 
    find_best_opinf_model(reg_pairs_global, Xhat, Xhat, Xhatdot,
                          n_train, n_train, max_growth, options,
                          ds["times"][1:n_train], CONTINUOUS_TIME)

## Save results
save(joinpath(FILEPATH, "data/results", 
     "reg_grid_search.jld2"), 
     "beta1", best_beta1, "beta2", best_beta2, 
     "train_err", best_train_err, "states", states, 
     "eval_time", eval_time, "final_idx", fidx)

## Save the best model
save(joinpath(FILEPATH, "data/models", 
     "batch_operators_0_8000_r$(rmax)_lamGS.jld2"), 
     "op", op)

## Load the batch model
op = load(joinpath(FILEPATH, "data/models", 
          "batch_operators_0_8000_r$(rmax)_lamGS.jld2"))["op"]


#=========================#
## Train Batch model
#=========================#
# Tikhonov Regularized OpInf
options.with_reg = true
options.λ = LnL.TikhonovParameter(A=1e12, A2=1e12, K=1e12)
op = LnL.opinf(Xhat, options; Xhatdot=Xhatdot)

## Save the model
save(joinpath(FILEPATH, "data/models", 
     "batch_operators_0_8000_r$(rmax)_lam1e12.jld2"), 
     "op", op)

#===========================#
## Simulate ROM (training) ##
#===========================#
include(joinpath(FILEPATH, "integrate.jl"))

if CONTINUOUS_TIME
    t1 = time()
    tspan = ds["times"][1:n_train] .- ds["times"][1]
    x0 = Xhat[:,1]
    states = rk4_integrate(x0, tspan, op.A, op.A2u, op.K)
    t2 = time()
    @info "ROM integration time: $(t2 - t1) seconds"
else
    states = zeros(size(V,2), n_time)
    states[:,1] = x0
    for j in 2:n_train
        states[:,j] = reduced_model(states[:,j-1], op.A, op.A2u, op.K)
        if any(isnan.(states[:,j]))
            @warn "NaN detected in trajectory $i at time step $j"
            break
        end
    end
    Xrom[i] = states
end

##
save(joinpath(FILEPATH, "data/results", 
     "batch_rom_train_sim_states_0_8000_r$(rmax).jld2"), 
     "states", states)

## Load the state data
states = load(joinpath(FILEPATH, "data/results", 
              "batch_rom_train_sim_states_0_8000_r$(rmax).jld2"))["states"]


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

if CONTINUOUS_TIME
    tspan = ds["times"][n_train+1:n_train+n_test] .- ds["times"][n_train+1]
    # Make sure to preprocess the first state
    x0 = iVrmax' * preprocess!(ds[n_train+1], means, shifts, scales)
    t1 = time()
    # x0 = iVrmax' * preprocess!(ds[n_train+1], means_test, shifts_test, scales_test)
    test_states = rk4_integrate(x0, tspan, op.A, op.A2u, op.K)
    t2 = time()
    @info "ROM integration time for test data: $(t2 - t1) seconds"
else
    test_states = zeros(size(V,2), n_test)
    test_states[:,1] = iVrmax * preprocess!(ds[n_train+1], means_test, 
                                            shifts_test, scales_test)
    for j in 2:n_test
        test_states[:,j] = reduced_model(test_states[:,j-1], op.A, op.A2u, op.K)
        if any(isnan.(test_states[:,j]))
            @warn "NaN detected in trajectory $i at time step $j"
            break
        end
    end
end

##
save(joinpath(FILEPATH, "data/results", 
     "batch_rom_test_sim_states_0_8000_r200.jld2"), 
     "states", test_states)

## Load the test states
test_states = load(joinpath(FILEPATH, "data/results", 
                   "batch_rom_test_sim_states_0_8000_r400.jld2"))["states"]


#======================#
## Plot the u-velocity 
#======================#
using CairoMakie

with_theme(theme_latexfonts()) do 
    train_or_test = "train"

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
            # x_rom_t = unprocess!(x_rom_t, means_test, shifts_test, scales_test)
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
    
    # save(joinpath(FILEPATH, "plots", 
    #      "$(fld)_slice_comparison_$(train_or_test).png"), fig)
    display(fig)
end


#=========================================================#
## Plot one reconstructed state over time for each field 
#=========================================================#
with_theme(theme_latexfonts()) do 
    train_or_test = "train"

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
                                                    n_train; batch_size=100)
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


#===================================#
## Compute the relative state error
#===================================#
rse = zeros(length(tspan))
for i in 1:length(tspan)
    x_full_i = iVrmax * states[:, i]
    x_full_i = unscale(x_full_i, dim_per_field, scale_factors) + xbar
    # x_full_i = x_full_i[1:Nz*Ny*Nx*3] 
    rse[i] = norm(x_full_i - ds[i], 2)
end
rse_tot = sum(rse) / length(rse)

## Compute the relative state error efficiently with parallelization
@info "Computing relative state error with $(Threads.nthreads()) threads..."

@time begin
    rse = zeros(length(tspan))
    den = zeros(length(tspan))
    
    # Pre-allocate thread-local temporary arrays to avoid allocations in the loop
    temp_arrays = [zeros(size(iVrmax, 1)) for _ in 1:Threads.nthreads()]
    true_states = [zeros(size(iVrmax, 1)) for _ in 1:Threads.nthreads()]
    
    # Parallelize the RSE computation across time steps
    Threads.@threads for i in 1:length(tspan)
        tid = Threads.threadid()
        temp_full = temp_arrays[tid]
        true_state = true_states[tid]
        
        # Reconstruct full state for time step i (reuse pre-allocated array)
        mul!(temp_full, iVrmax, states[:, i])  # More efficient matrix-vector multiplication
        
        # Apply scaling and mean (in-place operations)
        temp_full .= unscale(temp_full, dim_per_field, scale_factors) .+ xbar
        
        # Load true state (reuse pre-allocated array)
        copyto!(true_state, ds[i])
        
        # Compute relative state error using efficient norm computation
        temp_full .-= true_state  # Compute difference in-place
        rse[i] = norm(temp_full) 
        den[i] = norm(true_state)
    end
    
    # Compute total relative state error
    rse_tot = sum(rse) / sum(den)
    
    @info "Relative state error computation completed"
    @info "Average RSE: $(rse_tot)"
    @info "Max RSE: $(maximum(rse))"
    @info "Min RSE: $(minimum(rse))"
end
