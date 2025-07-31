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
rmax = 400


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
## Train Batch model
#=========================#
# Tikhonov Regularized OpInf
options.with_reg = true
options.λ = LnL.TikhonovParameter(A=1e12, A2=1e12, K=1e12)
op = LnL.opinf(Xhat, options; Xhatdot=Xhatdot)

##
save(joinpath(FILEPATH, "data/models", "operators_tol1e-12.jld2"), 
    "opinf", op_inf)


#========================================#
## Grid Search Regularization Parameters
#========================================#
function solve_opinf_difference_model(init_cond, n_steps, reduced_model)
    Qhat = zeros(length(init_cond), n_steps)
    contains_nan = false
    Qhat[:, 1] = init_cond
    final_idx = 0
    for i in 2:n_steps
        Qhat[:, i] = reduced_model(Qhat[:, i-1])
        if any(isnan.(Qhat[:, i]))
            contains_nan = true
            final_idx = i - 1
            break
        end
    end
    return contains_nan, Qhat, final_idx
end

function find_best_opinf_model(
    reg_pairs, Xhat, Xhat1, Xhat2, 
    n_time, n_time_pred, max_growth, opinf_options)

    @assert options.with_reg == true "Regularization must be enabled in options."
    
    best_train_err = 1e20
    best_beta1, best_beta2 = nothing, nothing
    best_final_idx = 0
    Xtilde_opt = nothing
    eval_time_opt = nothing

    mean_Xhat = mean(Xhat, dims=2)
    max_diff_Xhat = maximum(abs.(Xhat .- mean_Xhat), dims=2)
    
    # Loop over all regularization pairs
    @showprogress for (beta1, beta2) in reg_pairs
        
        # Construct a regularizer that penalizes the linear and constant reduced
        # operators using beta1 and the quadratic operator using beta2
        reg = LnL.TikhonovParameter(A=beta1, A2=beta2, K=beta1)
        opinf_options.λ = reg
        
        # Solve the regularized OpInf problem
        ops = LnL.opinf(Xhat1, opinf_options; Xhatdot=Xhat2)
        
        # Define the OpInf reduced model
        opinf_reduced_model = x -> ops.A * x + ops.A2u * (x ⊘ x) + ops.K

        # Extract the reduced initial condition from Qhat_1
        xhat0 = Xhat1[:,1]
        
        # Compute the reduced solution over the trial time horizon
        start_eval_time = time()
        contains_nans, Xtilde, fidx = solve_opinf_difference_model(
            xhat0, n_time_pred, opinf_reduced_model)
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
        end

        if best_final_idx < fidx
            best_final_idx = fidx
        end

        ops = nothing
        GC.gc() 
    end

    if isnothing(Xtilde_opt)
        @error "No suitable OpInf model found with the given regularization pairs."
    else
        @info "Best OpInf model found with β1 = $best_beta1, β2 = $best_beta2, \
               training error = $best_train_err, evaluation time = $eval_time_opt"
    end

    return best_beta1, best_beta2, best_train_err, Xtilde_opt, eval_time_opt, best_final_idx
end

##
B1 = 10.0 .^ range(-24.0, -20.0, length=8)
B2 = 10.0 .^ range(-20.0, -8.0, length=8)
reg_pairs_global = vec([(b1, b2) for b1 in B1, b2 in B2])
n_reg_global = length(reg_pairs_global)
max_growth = 1.2
options.with_reg = true
best_beta1, best_beta2, best_train_err, op_trinf, eval_time, fidx = 
    find_best_opinf_model(reg_pairs_global, Xhat, Xhat1, Xhat2,
                          n, Int(n+(n // 10)), max_growth, options)


#================#
## Simulate ROM ##
#================#
include(joinpath(FILEPATH, "integrate.jl"))

if CONTINUOUS_TIME
    tspan = ds["times"][:] .- ds["times"][1]
    x0 = Xhat[:,1]
    states = rk4_integrate(x0, tspan, op.A, op.A2u, op.K)
else
    states = zeros(size(V,2), n_time)
    states[:,1] = x0
    for j in 2:n_time
        states[:,j] = reduced_model(states[:,j-1], op.A, op.A2u, op.K)
        if any(isnan.(states[:,j]))
            @warn "NaN detected in trajectory $i at time step $j"
            break
        end
    end
    Xrom[i] = states
end


#============================#
## Load the mean and scaling
#============================#
means  = load(joinpath(FILEPATH, "data/mean.jld2"))["xbar"]
shifts = load(joinpath(FILEPATH, "data/minmax.jld2"))["minmax"]["shifts"]
scales = load(joinpath(FILEPATH, "data/minmax.jld2"))["minmax"]["scales"]


#=================#
## Load the bases 
#=================#
# Standard basis
basis_file = joinpath(FILEPATH, "data/bases/basis.jld2")
iVrmax = load(basis_file)["bases"]["baker"].iVr[:, 1:rmax]


#======================#
## Plot the u-velocity 
#======================#
using CairoMakie
include(joinpath(FILEPATH, "preprocess.jl"))

with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1200, 900))

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
    time_indices = [50, 1000, length(tspan)÷2, length(tspan)]

    # Pre-calculate all data for colorbar scaling
    all_full_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_rom_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_error_data = Vector{Matrix{Float64}}(undef, length(time_indices))

    # Collect all data first
    for (i, t_idx) in enumerate(time_indices)
        # Get full data
        u_full_field = reshape(ds[t_idx][i_s:i_f], nx, ny, nz)
        all_full_data[i] = u_full_field[:, :, z_mid]
        
        # Get ROM data
        x_rom_t = iVrmax * states[:, t_idx]
        x_rom_t = unprocess!(x_rom_t, means, shifts, scales)
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
        # Create axes
        ax_full = Axis(fig[1, i], 
            title = L"$t$ = %$(round(tspan[t_idx], digits=2))",
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
    
    save(joinpath(FILEPATH, "plots", "$(fld)_slice_comparison.png"), fig)
    display(fig)
end


#=========================================================#
## Plot one reconstructed state over time for each field 
#=========================================================#
with_theme(theme_latexfonts()) do 
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
        mean_val = means[idx]
        shift_val = shifts[idx]
        scale_val = scales[idx]

        true_field = ds[name][nrow, 1:n_time]
        rom_field = zeros(n_time)
        for t in 1:n_time
            Vrow = view(iVrmax, idx, :)
            states_col = view(states, :, t)
            rom_field[t] = dot(Vrow, states_col)
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
    
    save(joinpath(FILEPATH, "plots", "state_evolution.png"), fig)
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
