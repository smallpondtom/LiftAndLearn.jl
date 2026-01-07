"""
MHD64 example: Training rSSM-based Distributed OpInf model
"""

#=================#
## Load packages ##
#=================#
using LinearAlgebra
using BlockDiagonals
using FileIO
using JLD2
using Statistics
using Revise
using IncrementalSVD
import LiftAndLearn as LnL

#=============#
## Load data ##
#=============#
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(), "rSSM/mhd64") : 
           joinpath(pwd(), "scripts/rSSM/mhd64")
DATAPATH = "../../../../DATA/THE_WELL/mhd64"
train_files = readdir(DATAPATH, join=true)
test_files = readdir(joinpath(DATAPATH, "test"), join=true)
fn = train_files[1]
fn_test = test_files[1] 
X = load(joinpath(FILEPATH, "data/preprocessed_data.jld2"))["X"]["all"]
basis = load(joinpath(FILEPATH, "data/bases/basis_ssm.jld2"))
V = basis["Vrhat"]
S = basis["Srhat"]
W = basis["Wrhat"]
basis = nothing
GC.gc()

# Include the data sourcing module for data access
include(joinpath(FILEPATH, "datasource.jl"))

# Load data source 
ds = DataSource(fn)
nx, ny, nz, n_fields, n_time, n_traj = ds.dims
nxyz = nx * ny * nz
n = n_time * n_traj

#===============#
## Set Options ##
#===============#
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=[1,2,3],
        control=0,
        constant=1,
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
    data=LnL.DataStructure(
        Δt=sum(diff(ds.grid["time"])) / (length(ds.grid["time"])-1),
        deriv_type="FBCT4"
    ),
    use_backslash=true,
)

#===================#
## Train Operators ##
#===================#
# Compute the reduced data matrix
@info "Generate finite difference data for continuous-time "
# Compute finite difference approximation
Xhat = Vector{Matrix{Float64}}(undef, n_traj)
Xhatdot = Vector{Matrix{Float64}}(undef, n_traj)
# Do it individually for each trajectory
for i in 1:n_traj
    idx_start = (i-1) * n_time + 1
    idx_end = i * n_time
    E, Δidx = LnL.finite_diff_matrix(
        options.data.deriv_type, n_time, options.data.Δt
    )
    Xtmp = S .* W'[ :, idx_start:idx_end]
    Xhatdot[i] = Xtmp * E
    Xhat[i] = Xtmp[:, Δidx]
end
Xhat = reduce(hcat, Xhat)
Xhatdot = reduce(hcat, Xhatdot)

#========================================#
## Grid Search Regularization Parameters
#========================================#
include(joinpath(FILEPATH, "grid_search.jl"))
include(joinpath(FILEPATH, "integrate.jl"))

GRID_SEARCH = false
if GRID_SEARCH
    ## Run grid Search
    # B1 = logrange(10^5, 10^6, length=20)
    # B2 = logrange(9*10^9, 2*10^10, length=20)
    # B3 = logrange(10^14, 3*10^14, length=20)  
    B1 = 1.4384498882876628e5
    B2 = 1.0209357890016825e10
    B3 = 1.189419173187856e14
    reg_pairs_global = vec([(b1, b2, b3) for b1 in B1, b2 in B2, b3 in B3])
    max_growth = 3.0
    options.with_reg = true
    op, best_betas, best_train_err, states, eval_time, fidx =
        find_best_opinf_model(reg_pairs_global, Xhat, Xhatdot,
                            n_time, n_time, max_growth, options,
                            ds.grid["time"])

    ## Save results
    save(joinpath(FILEPATH, "data/results", 
        "reg_grid_search.jld2"), 
        "betas", best_betas,
        "train_err", best_train_err, "states", states, 
        "eval_time", eval_time, "final_idx", fidx)
else 
    @info "No grid search, using fixed regularization parameters"
    ## Solve the OpInf problem 
    options.with_reg = true
    beta1 = 1.0e10
    # beta1 = 1.4384498882876628e5
    beta2 = 1.0e10
    # beta2 = 1.0209357890016825e10
    beta3 = 1.189419173187856e14
    options.λ = LnL.TikhonovParameter(A=beta1, K=beta1, A2=beta2, A3=beta3)
    op = LnL.opinf(Xhat, options; Xhatdot=Xhatdot)
end

## Save the trained model
save(joinpath(FILEPATH, "data/models/ssm_opinf.jld2"), "op", op)

#================#
## Simulate ROM ##
#================#
# Integrate a single trajectory 
Xrom_train = Vector{Matrix{Float64}}(undef, n_traj)
for i in 1:n_traj
    idx_start = (i-1) * n_time + 1
    idx_end = i * n_time
    x0 = V' * X[:, idx_start:idx_end][:,1]
    Xrom_train[i], _ = rk4_integrate(x0, ds.grid["time"], op.A, op.A2u, op.A3u, op.K)
end
Xrom_train = reduce(hcat, Xrom_train)

## Save the ROM's training data 
save(joinpath(FILEPATH, "data/results/rom_training_states.jld2"), 
     "states", Xrom_train)

## Load instead
Xrom_train = load(joinpath(FILEPATH, "data/results/rom_training_states.jld2"), "states") 

#=============================#
## Load scaling and shifting ##
#=============================#
include("preprocess.jl")
means = load(joinpath(FILEPATH, "data/mean.jld2"))["mean"]
scales = load(joinpath(FILEPATH, "data/minmax.jld2"))["scale"]
shifts = load(joinpath(FILEPATH, "data/minmax.jld2"))["shift"]

#=================#
## Simulate Test ##
#=================#
ds_test = DataSource(fn_test)
Xtest = load(joinpath(FILEPATH, "data/test_data.jld2"))["X"]["all"]
# Integrate a single trajectory 
x0 = V' * preprocess!(Xtest[:,1], vec(means["all"]), vec(shifts["all"]), 
                      vec(scales["all"]))
Xrom_test, _ = rk4_integrate(x0, ds_test.grid["time"], op.A, op.A2u, op.A3u, op.K)

## Save the ROM's testing data 
save(joinpath(FILEPATH, "data/results/rom_testing_states.jld2"), 
     "states", Xrom_test)

## Load instead
Xrom_test = load(joinpath(FILEPATH, "data/results/rom_testing_states.jld2"), "states")

#===========================#
## Plot the sliced density ##
#===========================#
using CairoMakie
include(joinpath(FILEPATH, "..", "..", "custom_theme.jl"))

fig = themed_figure(dark=true, size=(1200,940)) do fig
    # Pick trajectory
    traj_idx = 5
    # Get midpoint index for z-direction
    x_slice = nx ÷ 2
    y_slice = 1:ny
    z_slice = 1:nz
    # Pick state 
    var = "rho"  # "rho", "z", "mx", "my", "mz", "Bx", "By", "Bz"
    # Pick training or testing 
    train_or_test = "test"
    if train_or_test == "train"
        Xrom = Xrom_train
        ds = DataSource(fn)
    else
        Xrom = Xrom_test
        ds = DataSource(fn_test)
        traj_idx = 1  # Only one trajectory in test set
    end

    if x_slice isa Int 
        axis_label = [L"$y$", L"$z$"] 
        horz_span = ds.grid["y"]
        vert_span = ds.grid["z"]
    elseif y_slice isa Int
        axis_label = [L"$x$", L"$z$"]
        horz_span = ds.grid["x"]
        vert_span = ds.grid["z"]
    else
        axis_label = [L"$x$", L"$y$"]
        horz_span = ds.grid["x"]
        vert_span = ds.grid["y"]
    end

    is_momentum = false
    is_magnetic = false
    if var == "rho"
        start_idx = 1
        end_idx = nxyz
    elseif var == "z"
        start_idx = nxyz + 1
        end_idx = nxyz*2
    elseif var == "mx"
        start_idx = nxyz*2 + 1
        end_idx = nxyz*3
        c = 1
        is_momentum = true
    elseif var == "my"
        start_idx = nxyz*3 + 1
        end_idx = nxyz*4
        c = 2
        is_momentum = true
    elseif var == "mz"
        start_idx = nxyz*4 + 1
        end_idx = nxyz*5
        c = 3
        is_momentum = true
    elseif var == "Bx"
        start_idx = nxyz*5 + 1
        end_idx = nxyz*6
        c = 1
        is_magnetic = true
    elseif var == "By"
        start_idx = nxyz*6 + 1
        end_idx = nxyz*7
        c = 2
        is_magnetic = true
    elseif var == "Bz"
        start_idx = nxyz*7 + 1
        end_idx = nxyz*8
        c = 3
        is_magnetic = true
    end

    # Select 3 time steps (beginning, middle, end)
    time_indices = Int.([ceil(n_time / 3), ceil(n_time * 2 / 3), n_time-10])

    # Pre-calculate all data for colorbar scaling
    all_full_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_rom_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_error_data = Vector{Matrix{Float64}}(undef, length(time_indices))

    unscale = (X, scale, shift) -> (scale .* X) .+ shift
    uncenter = (X, Xbar) -> X .+ Xbar

    # Collect all data first
    for (i, t_idx) in enumerate(time_indices)
        # Get full data
        if is_momentum || is_magnetic
            full_field = ds[var][c, :, :, :, t_idx, traj_idx]
        else
            full_field = ds[var][:, :, :, t_idx, traj_idx]
        end
        all_full_data[i] = full_field[x_slice, y_slice, z_slice]

        # Get ROM data
        Xrom_traj = Xrom[:, (traj_idx-1) * n_time + t_idx]
        Xrecon = V * Xrom_traj
        Xrecon = Xrecon[start_idx:end_idx]
        Xrecon = unscale(Xrecon, scales[var], shifts[var])
        Xrecon = uncenter(Xrecon, means[var])
        rom_field = reshape(Xrecon[1:nxyz], nx, ny, nz)
        all_rom_data[i] = rom_field[x_slice, y_slice, z_slice]

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
        time_value = ds.grid["time"][t_idx]
        n_label = length(ds) ÷ ds.dims[end]
        ax_full = Axis(fig[1, i], 
            title = i == 1 ? 
                    L"$t$=%$(round(time_value, digits=2)) \n snapshot %$(t_idx)/%$(n_label)" : 
                    L"$t$=%$(round(time_value, digits=2)) \n %$(t_idx)/%$(n_label)",
            ylabel = i == 1 ? axis_label[2] : "",
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            xlabelsize=30, ylabelsize=30, 
            titlesize=30, 
        )
        ax_rom = Axis(fig[2, i], 
            ylabel = i == 1 ? axis_label[2] : "", 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            xlabelsize=30, ylabelsize=30, 
        )
        ax_error = Axis(fig[3, i], 
            ylabel = i == 1 ? axis_label[2] : "", 
            xlabel = axis_label[1],
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            xlabelsize=30, ylabelsize=30, 
        )

        # Create heatmaps with aligned color ranges
        hm_full = heatmap!(ax_full, horz_span, vert_span, all_full_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max),
            colorscale = is_momentum || is_magnetic ? identity : log10)
        hm_rom = heatmap!(ax_rom, horz_span, vert_span, all_rom_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max),
            colorscale = is_momentum || is_magnetic ? identity : log10)
        hm_error = heatmap!(ax_error, horz_span, vert_span, all_error_data[i], 
            colormap = :matter, colorrange = (error_min, error_max),
            colorscale = is_momentum || is_magnetic ? identity : log10)
    end
    
    # Add colorbars at the end of each row
    Colorbar(fig[1, length(time_indices) + 1], hm_full, label="Full", 
             labelsize=30, ticklabelsize=20)
    Colorbar(fig[2, length(time_indices) + 1], hm_rom, label="SSM OpInf", 
             labelsize=30, ticklabelsize=20)
    Colorbar(fig[3, length(time_indices) + 1], hm_error, label="Abs. Error", 
             labelsize=30, ticklabelsize=20)
    
    # save(joinpath(FILEPATH, "plots/$(train_or_test)_sliced_$(var).png"), fig)
end

#==========================#
## Create video (density) ##
#==========================#
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(1900,700))
    ax1 = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y",
              xlabelsize=20, ylabelsize=20, title=L"x-y",
              xticklabelsize=15, yticklabelsize=15, titlesize=35)
    ax2 = Axis(fig[1, 2], xlabel=L"y", ylabel=L"z", 
              xlabelsize=20, ylabelsize=20, title=L"y-z",
              xticklabelsize=15, yticklabelsize=15, titlesize=35)
    ax3 = Axis(fig[1, 3], xlabel=L"x", ylabel=L"z", 
              xlabelsize=20, ylabelsize=20, title=L"x-z",
              xticklabelsize=15, yticklabelsize=15, titlesize=35)

    hidedecorations!(ax1)
    hidedecorations!(ax2)
    hidedecorations!(ax3)
    hidespines!(ax1)
    hidespines!(ax2)
    hidespines!(ax3)

    Xtmp = ds["rho"][:, :, :, 1:n_time, 1] # pressure field data
    xspan = ds.grid["x"]
    yspan = ds.grid["y"]
    zspan = ds.grid["z"]

    mid = nx ÷ 2
    hm1 = heatmap!(ax1, xspan, yspan, Xtmp[:,:,mid,1], colormap=:plasma,
                   colorscale=log10, colorrange=(extrema(Xtmp[:,:,mid,:])))
    hm2 = heatmap!(ax2, yspan, zspan, Xtmp[mid,:,:,1], colormap=:plasma, colorscale=log10)
    hm3 = heatmap!(ax3, xspan, zspan, Xtmp[:,mid,:,1], colormap=:plasma, colorscale=log10)
    cb = Colorbar(fig[1, 4], hm1, label=L"density$$", labelsize=40, ticklabelsize=15)
    tight_ticklabel_spacing!(cb)
    record(fig, joinpath(FILEPATH, "plots/density_dist.mp4"), 1:n_time) do i
        hm1[3] = Xtmp[:,:,mid,i]
        hm2[3] = Xtmp[mid,:,:,i]
        hm3[3] = Xtmp[:,mid,:,i]
        autolimits!(ax1) # update limits
        autolimits!(ax2) # update limits
        autolimits!(ax3) # update limits
    end
end

#=========================================#
## Create video Reconstruction (density) ##
#=========================================#
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(1900,700))
    ax1 = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y",
              xlabelsize=20, ylabelsize=20, title=L"x-y",
              xticklabelsize=15, yticklabelsize=15, titlesize=35)
    ax2 = Axis(fig[1, 2], xlabel=L"y", ylabel=L"z", 
              xlabelsize=20, ylabelsize=20, title=L"y-z",
              xticklabelsize=15, yticklabelsize=15, titlesize=35)
    ax3 = Axis(fig[1, 3], xlabel=L"x", ylabel=L"z", 
              xlabelsize=20, ylabelsize=20, title=L"x-z",
              xticklabelsize=15, yticklabelsize=15, titlesize=35)

    hidedecorations!(ax1)
    hidedecorations!(ax2)
    hidedecorations!(ax3)
    hidespines!(ax1)
    hidespines!(ax2)
    hidespines!(ax3)

    Xtmp = V * Xrom[:, 1:n_time]
    Xtmp = Xtmp[1:nxyz, :] # pressure field data
    Xtmp = unscale(Xtmp, scale["rho"], shift["rho"])
    Xtmp = uncenter(Xtmp, mean["rho"])
    Xtmp = reshape(Xtmp, nx, ny, nz, n_time)
    min_clip = minimum(abs.(Xtmp))
    Xtmp = max.(Xtmp, min_clip)
    xspan = ds.grid["x"]
    yspan = ds.grid["y"]
    zspan = ds.grid["z"]

    mid = nx ÷ 2
    hm1 = heatmap!(ax1, xspan, yspan, Xtmp[:,:,mid,1], colormap=:plasma,
                   colorscale=log10, colorrange=(extrema(Xtmp[:,:,mid,:])))
    hm2 = heatmap!(ax2, yspan, zspan, Xtmp[mid,:,:,1], colormap=:plasma, 
                   colorscale=log10)
    hm3 = heatmap!(ax3, xspan, zspan, Xtmp[:,mid,:,1], colormap=:plasma, 
                   colorscale=log10)
    cb = Colorbar(fig[1, 4], hm1, label=L"density$$", labelsize=40, ticklabelsize=15)
    tight_ticklabel_spacing!(cb)
    record(fig, joinpath(FILEPATH, "plots/rom_density_dist.mp4"), 1:n_time) do i
        hm1[3] = Xtmp[:,:,mid,i]
        hm2[3] = Xtmp[mid,:,:,i]
        hm3[3] = Xtmp[:,mid,:,i]
        autolimits!(ax1) # update limits
        autolimits!(ax2) # update limits
        autolimits!(ax3) # update limits
    end
end

#===========================#
## Create video (pressure) ##
#===========================#
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(1900,700))
    ax1 = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y",
              xlabelsize=20, ylabelsize=20, title=L"x-y",
              xticklabelsize=15, yticklabelsize=15, titlesize=35)
    ax2 = Axis(fig[1, 2], xlabel=L"y", ylabel=L"z", 
              xlabelsize=20, ylabelsize=20, title=L"y-z",
              xticklabelsize=15, yticklabelsize=15, titlesize=35)
    ax3 = Axis(fig[1, 3], xlabel=L"x", ylabel=L"z", 
              xlabelsize=20, ylabelsize=20, title=L"x-z",
              xticklabelsize=15, yticklabelsize=15, titlesize=35)

    hidedecorations!(ax1)
    hidedecorations!(ax2)
    hidedecorations!(ax3)
    hidespines!(ax1)
    hidespines!(ax2)
    hidespines!(ax3)

    Xtmp = ds["u"][1, :, :, :, 1:n_time, 1] # pressure field data
    xspan = ds.grid["x"]
    yspan = ds.grid["y"]
    zspan = ds.grid["z"]

    mid = nx ÷ 2
    hm1 = heatmap!(ax1, xspan, yspan, Xtmp[:,:,mid,1], colormap=:plasma,
                   colorrange=(extrema(Xtmp[:,:,mid,:])))
    hm2 = heatmap!(ax2, yspan, zspan, Xtmp[mid,:,:,1], colormap=:plasma)
    hm3 = heatmap!(ax3, xspan, zspan, Xtmp[:,mid,:,1], colormap=:plasma)
    cb = Colorbar(fig[1, 4], hm1, label=L"pressure$$", labelsize=40, ticklabelsize=15)
    tight_ticklabel_spacing!(cb)
    record(fig, joinpath(FILEPATH, "plots/u_velocity_dist.mp4"), 1:n_time) do i
        hm1[3] = Xtmp[:,:,mid,i]
        hm2[3] = Xtmp[mid,:,:,i]
        hm3[3] = Xtmp[:,mid,:,i]
        autolimits!(ax1) # update limits
        autolimits!(ax2) # update limits
        autolimits!(ax3) # update limits
    end
end

#==========================================#
## Create video Reconstruction (pressure) ##
#==========================================#
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(1900,700))
    ax1 = Axis(fig[1, 1], xlabel=L"x", ylabel=L"y",
              xlabelsize=20, ylabelsize=20, title=L"x-y",
              xticklabelsize=15, yticklabelsize=15, titlesize=35)
    ax2 = Axis(fig[1, 2], xlabel=L"y", ylabel=L"z", 
              xlabelsize=20, ylabelsize=20, title=L"y-z",
              xticklabelsize=15, yticklabelsize=15, titlesize=35)
    ax3 = Axis(fig[1, 3], xlabel=L"x", ylabel=L"z", 
              xlabelsize=20, ylabelsize=20, title=L"x-z",
              xticklabelsize=15, yticklabelsize=15, titlesize=35)

    hidedecorations!(ax1)
    hidedecorations!(ax2)
    hidedecorations!(ax3)
    hidespines!(ax1)
    hidespines!(ax2)
    hidespines!(ax3)

    Xtmp = V * Xrom[:, 1:n_time]
    Xtmp = Xtmp[nxyz*2+1:nxyz*3, :] # pressure field data
    Xtmp = unscale(Xtmp, scale["u"], shift["u"])
    Xtmp = uncenter(Xtmp, mean["u"])
    Xtmp = reshape(Xtmp, nx, ny, nz, n_time)
    min_clip = minimum(abs.(Xtmp))
    Xtmp = max.(Xtmp, min_clip)
    xspan = ds.grid["x"]
    yspan = ds.grid["y"]
    zspan = ds.grid["z"]

    mid = nx ÷ 2
    hm1 = heatmap!(ax1, xspan, yspan, Xtmp[:,:,mid,1], colormap=:plasma,
                   colorrange=(extrema(Xtmp[:,:,mid,:])))
    hm2 = heatmap!(ax2, yspan, zspan, Xtmp[mid,:,:,1], colormap=:plasma)
    hm3 = heatmap!(ax3, xspan, zspan, Xtmp[:,mid,:,1], colormap=:plasma)
    cb = Colorbar(fig[1, 4], hm1, label=L"pressure$$", labelsize=40, ticklabelsize=15)
    tight_ticklabel_spacing!(cb)
    record(fig, joinpath(FILEPATH, "plots/rom_u_velocity_dist.mp4"), 1:n_time) do i
        hm1[3] = Xtmp[:,:,mid,i]
        hm2[3] = Xtmp[mid,:,:,i]
        hm3[3] = Xtmp[:,mid,:,i]
        autolimits!(ax1) # update limits
        autolimits!(ax2) # update limits
        autolimits!(ax3) # update limits
    end
end
