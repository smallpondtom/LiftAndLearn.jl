"""
Vortex-Shedding Flow Past a Cylinder (2D) - Discrete-Time Operator Inference
"""

#===============#
## Load packages
#===============#
using LinearAlgebra
using HDF5
using UniqueKronecker: ⊘
using Statistics
using ProgressMeter: @showprogress

#==========#
## Load LnL
#==========#
using LiftAndLearn
const LnL = LiftAndLearn

#==================#
## Load data file
#==================#
# Some settings
n_fields = 2
n_state = 9477
n_time = 300
n_time_predictions = 600
n = n_fields * n_state
state_variables = ["u_x", "u_y"]

# Get data file path
data_filename = "velocity_training_snapshots.h5"
if occursin("OpInf", @__DIR__)
    data_path = joinpath(@__DIR__, "data")
else
    data_path = joinpath(@__DIR__, "OpInf", "data")
end
data_filepath = joinpath(data_path, data_filename)

# Load the data as a unfolded matrix 
Q_glob = zeros(n, n_time)
h5open(data_filepath, "r") do hf
    for i in 1:n_fields
        Q_glob[(i-1)*n_state+1:i*n_state, :] .= hf[state_variables[i]][:,:]'
    end
end

#======================#
## Preprocess the data 
#======================#
# Center the data with the mean
temporal_mean_glob = mean(Q_glob, dims=2)
Q_glob .-= temporal_mean_glob

#=========================================#
## Compute the POD basis the reduced data
#=========================================#
V, Σ, _ = svd(Q_glob)
target_ret_energy = 0.999
ret_energy = cumsum(Σ.^2) / sum(Σ.^2)
r = findfirst(x -> x > target_ret_energy, ret_energy) + 1
V_r = V[:, 1:r]
Qhat_glob = V_r' * Q_glob

#=================================================================================#
## Perform the discrete-time OpInf while grid-searching regularization parameters 
#=================================================================================#
# Setup the OpInf problem 
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=[1,2],
        constant=1
    ),
    vars=LnL.VariableStructure(
        N=2,
    ),
    data=LnL.DataStructure(),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
    with_reg = true,
)

# Define ranges for the regularization parameter pairs
B1 = 10.0 .^ range(-10.0, 0.0, length=8)
B2 = 10.0 .^ range(-4.0, 4.0, length=8)

# Get the Cartesian product of all regularization pairs (beta1, beta2)
reg_pairs_global = vec([(b1, b2) for b1 in B1, b2 in B2])
n_reg_global = length(reg_pairs_global)

# Set the threshold for the maximum growth of the inferred reduced
# coefficients, used for selecting the optimal regularization parameter pair
max_growth = 1.2

# Extract left and right shifted reduced data matrices for discrete OpInf
Qhat1 = Qhat_glob[:, 1:end-1]
Qhat2 = Qhat_glob[:, 2:end]

# Compute the temporal mean and maximum deviation of the reduced training data
mean_Qhat_train = mean(Qhat_glob, dims=2)
max_diff_Qhat_train = maximum(abs.(Qhat_glob .- mean_Qhat_train), dims=2)

## Supplemental function to solve the discrete-time model
function solve_opinf_difference_model(init_cond, n_steps, reduced_model)
    Qhat = zeros(length(init_cond), n_steps)
    contains_nan = false
    Qhat[:, 1] = init_cond
    for i in 2:n_steps
        Qhat[:, i] = reduced_model(Qhat[:, i-1])
        if any(isnan.(Qhat[:, i]))
            contains_nan = true
            break
        end
    end
    return contains_nan, Qhat
end

##
"""
Function to find the best model and regularization parameters for OpInf.

Parameters:
- reg_pairs: Vector of tuples containing (beta1, beta2) regularization pairs
- Qhat1: Left-shifted reduced data matrix
- Qhat2: Right-shifted reduced data matrix
- mean_Qhat_train: Temporal mean of training data
- max_diff_Qhat_train: Maximum deviation of training data
- nt: Number of training time steps
- nt_p: Number of prediction time steps
- max_growth: Maximum allowed growth threshold
- Qhat_global: Global reduced data matrix
- opinf_options: Options for the OpInf problem

Returns:
- best_beta1, best_beta2: Optimal regularization parameters
- best_train_err: Best training error achieved
- Qtilde_opt: Optimal reduced solution
- opinf_rom_wtime_opt: Wall-clock time for optimal evaluation
"""
function find_best_opinf_model(
    reg_pairs, Qhat1, Qhat2, 
    mean_Qhat_train, max_diff_Qhat_train, 
    nt, nt_p, max_growth, Qhat_global, opinf_options)
    
    best_train_err = 1e20
    best_beta1, best_beta2 = nothing, nothing
    Qtilde_opt = nothing
    opinf_rom_wtime_opt = nothing
    
    # Loop over all regularization pairs
    @showprogress for (beta1, beta2) in reg_pairs
        
        # Construct a regularizer that penalizes the linear and constant reduced
        # operators using beta1 and the quadratic operator using beta2
        reg = LnL.TikhonovParameter(A=beta1, A2=beta2, K=beta1)
        opinf_options.λ = reg
        
        # Solve the regularized OpInf problem
        ops = LnL.opinf(Qhat1, opinf_options; Xhatdot=Qhat2)
        
        # Define the OpInf reduced model
        opinf_reduced_model = x -> ops.A * x + ops.A2u * (x ⊘ x) + ops.K

        # Extract the reduced initial condition from Qhat_1
        qhat0 = Qhat1[:,1]
        
        # Compute the reduced solution over the trial time horizon
        start_eval_time = time()
        contains_nans, Qtilde = solve_opinf_difference_model(
            qhat0, nt_p, opinf_reduced_model)
        end_eval_time = time()
        time_opinf_eval = end_eval_time - start_eval_time
        
        # If the model produced an unstable solution, move on to the next
        # regularization candidates
        if contains_nans
            continue
        end
        
        # If the ratio of the maximum coefficient growth exceeds the allowed
        # threshold, move on to the next regularization candidates
        max_diff_Qhat_trial = maximum(abs.(Qtilde .- mean_Qhat_train), dims=2)
        max_growth_trial = maximum(max_diff_Qhat_trial) / maximum(max_diff_Qhat_train)
        if max_growth_trial > max_growth
            continue
        end
        
        # At this point we know the model produced a stable solution without too
        # much growth. Compute the training error and, if it's better than the
        # current best error, save the regularization, reduced solution, and
        # the learning times
        train_err = norm(Qhat_global[:, 1:nt] - Qtilde[:, 1:nt])^2 / norm(Qhat_global[:, 1:nt])^2
        if train_err < best_train_err
            best_beta1 = beta1
            best_beta2 = beta2
            best_train_err = train_err
            Qtilde_opt = Qtilde
            opinf_rom_wtime_opt = time_opinf_eval
        end
    end
    
    return best_beta1, best_beta2, best_train_err, Qtilde_opt, opinf_rom_wtime_opt
end

## Call the function to find the best model
best_beta1, best_beta2, best_train_err, Qtilde_opt, OpInf_ROM_wtime_opt = 
    find_best_opinf_model(reg_pairs_global, Qhat1, Qhat2,
                          mean_Qhat_train, max_diff_Qhat_train,
                          n_time, n_time_predictions, max_growth, 
                          Qhat_glob, options)


#===========================#
## Plot the Trajectories
#===========================#
using CairoMakie 

Q_recon = V_r * Qtilde_opt

# Create time vector for plotting
t_plot_glob = 1:size(Q_glob, 2)
t_plot = 1:size(Q_recon, 2)

# Select 4 states from each field for plotting
n_plot_states = 4
state_indices = 1:div(n_state, n_plot_states):n_state

# Create 4x4 subplot figure (4 trajectories x 2 variables x 2 columns for errors)
fig = Figure(size=(1600, 1200))

# Plot u_x trajectories and errors
for i in 1:n_plot_states
    state_idx = (i-1)*div(n_state, n_plot_states) + 1
    
    # u_x trajectory
    ax_traj = Axis(fig[i, 1], 
        title="u_x State $(state_idx)", 
        xlabel="Time Step", 
        ylabel="u_x",
        titlesize=14, xlabelsize=12, ylabelsize=12
    )
    
    lines!(ax_traj, t_plot_glob, Q_glob[state_idx, :] .+ temporal_mean_glob[state_idx], 
           color=:blue, linewidth=2, label="Original")
    lines!(ax_traj, t_plot, Q_recon[state_idx, :] .+ temporal_mean_glob[state_idx], 
           color=:orange, linewidth=2, linestyle=:dash, label="ROM")
    
    if i == 1
        axislegend(ax_traj, position=:rt)
    end
    
    # u_x error
    ax_err = Axis(fig[i, 2], 
        title="u_x Error State $(state_idx)", 
        xlabel="Time Step", 
        ylabel="Error",
        titlesize=14, xlabelsize=12, ylabelsize=12
    )
    
    error = Q_recon[state_idx, 1:n_time] - Q_glob[state_idx, :]
    lines!(ax_err, t_plot_glob, error, color=:black, linewidth=2)
end

# Plot u_y trajectories and errors
for i in 1:n_plot_states
    state_idx = n_state + (i-1)*div(n_state, n_plot_states) + 1
    
    # u_y trajectory
    ax_traj = Axis(fig[i, 3], 
        title="u_y State $(state_idx)", 
        xlabel="Time Step", 
        ylabel="u_y",
        titlesize=14, xlabelsize=12, ylabelsize=12
    )
    
    lines!(ax_traj, t_plot_glob, Q_glob[state_idx, :] .+ temporal_mean_glob[state_idx], 
           color=:blue, linewidth=2, label="Original")
    lines!(ax_traj, t_plot, Q_recon[state_idx, :] .+ temporal_mean_glob[state_idx], 
           color=:orange, linewidth=2, linestyle=:dash, label="ROM")
    
    if i == 1
        axislegend(ax_traj, position=:rt)
    end
    
    # u_y error
    ax_err = Axis(fig[i, 4], 
        title="u_y Error State $(state_idx)", 
        xlabel="Time Step", 
        ylabel="Error",
        titlesize=14, xlabelsize=12, ylabelsize=12
    )
    
    error = Q_recon[state_idx, 1:n_time] - Q_glob[state_idx, :]
    lines!(ax_err, t_plot_glob, error, color=:black, linewidth=2)
end

display(fig)



#===========================#
## Plot the results
#===========================#
using NPZ

# Parameters (should match your Python version)
t_start = 4.0
t_end   = 10.0
t_train = 7.0
dt      = 1e-2

# Build time vector (Python’s arange excludes the endpoint)
t = collect(t_start:dt:(t_end - dt))

# Load the reference and ROM probe data
ref_data    = npzread(joinpath(data_path, "ref_vorticity_full_time_domain.npy"))
OpInf_data  = npzread(joinpath(data_path, "ROM_vorticity_probes.npy"))

# Slice out the three probe locations
ref_loc1  = ref_data[401:end, 1]   # Python ref_data[400:,0]
ref_loc2  = ref_data[401:end, 2]   # Python ref_data[400:,1]
ref_loc3  = ref_data[401:end, 3]   # Python ref_data[400:,2]

rom_loc1  = OpInf_data[:, 1]       # Python [:,0]
rom_loc2  = OpInf_data[:, 2]       # Python [:,1]
rom_loc3  = OpInf_data[:, 3]       # Python [:,2]

# Plot styling
charcoal = RGBAf(0.1, 0.1, 0.1, 1)
color1   = "#D55E00"

# Create a 1×3 figure, size ≃ 8″×2″ @ 400 dpi → 3200×800 px
fig = Figure(size=(800, 1200))

ax1 = Axis(fig[1, 1],
    title = "probe 1 (0.4, 0.2)",
    ylabel = L"\omega",
    limits = (t_start, t_end, nothing, nothing),
    xticks = (t_start:1:t_end, string.(t_start:1:t_end)),
    rightspinevisible=false, topspinevisible=false,
    xlabelsize=25, ylabelsize=25,
    titlesize=25, xticksize=20, yticksize=20,
)

ax2 = Axis(fig[2, 1],
    title = "probe 2 (0.6, 0.2)",
    ylabel = L"\omega",
    limits = (t_start, t_end, nothing, nothing),
    xticks = (t_start:1:t_end, string.(t_start:1:t_end)),
    rightspinevisible=false, topspinevisible=false,
    xlabelsize=25, ylabelsize=25,
    titlesize=25, xticksize=20, yticksize=20,
)

ax3 = Axis(fig[3, 1],
    title = "probe 3 (1.0, 0.2)",
    ylabel = L"\omega",
    xlabel = "target time horizon (seconds)",
    limits = (t_start, t_end, nothing, nothing),
    xticks = (t_start:1:t_end, string.(t_start:1:t_end)),
    rightspinevisible=false, topspinevisible=false,
    xlabelsize=25, ylabelsize=25,
    titlesize=25, xticksize=20, yticksize=20,
)

# Draw the two time-series on each axis
lines!(ax1, t, ref_loc1,  linestyle = :solid, linewidth = 1, color = charcoal, label = "reference")
lines!(ax1, t, rom_loc1,  linestyle = :dash,  linewidth = 1, color = color1, label = "ROM")

lines!(ax2, t, ref_loc2,  linestyle = :solid, linewidth = 1, color = charcoal)
lines!(ax2, t, rom_loc2,  linestyle = :dash,  linewidth = 1, color = color1)

lines!(ax3, t, ref_loc3,  linestyle = :solid, linewidth = 1, color = charcoal)
lines!(ax3, t, rom_loc3,  linestyle = :dash,  linewidth = 1, color = color1)

# Vertical line at end of training window
for ax in (ax1, ax2, ax3)
    vlines!(ax, [t_train], linestyle = :dash, linewidth = 1, color = :gray)
end

# Shade the training region [0, t_train] - get actual y limits after plotting
for ax in (ax1, ax2, ax3)
    ylims = ax.finallimits[].widths[2]
    ymin = ax.finallimits[].origin[2]
    ymax = ymin + ylims
    poly!(ax, [(t_start, ymin), (t_train, ymin), (t_train, ymax), (t_start, ymax)];
        color = (:gray, 0.2), strokewidth = 0)
end

axislegend(ax1, labelsize=25)

display(fig)