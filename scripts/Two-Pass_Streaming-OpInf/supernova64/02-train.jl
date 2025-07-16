#=================#
## Load packages ##
#=================#
using LinearAlgebra
using BlockDiagonals
using FileIO
using JLD2
using Revise
import LiftAndLearn as LnL

#=============#
## Load data ##
#=============#
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(), "Two-Pass_Streaming-OpInf/supernova64") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/supernova64")
DATAPATH = "../../../../DATA/THE_WELL/supernova_explosion_64/train"
train_files = readdir(DATAPATH, join=true)
fn = train_files[1]
X_all = load(joinpath(FILEPATH, "data/preprocessed_data.jld2"))["X"]
V = load(joinpath(FILEPATH, "data/streaming/basis.jld2"))["V"]

## Include the data sourcing module for data access
include(joinpath(FILEPATH, "datasource.jl"))

## Load data source 
ds = DataSource(fn)
nx, ny, nz, n_fields, n_time, n_traj = ds.dims
nxyz = nx * ny * nz

#===============#
## Set Options ##
#===============#
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=[1,2],
        control=0,
        constant=1,
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
    data=LnL.DataStructure(
        Δt=sum(diff(ds.grid["time"])) / (length(ds.grid["time"])-1),
        deriv_type="CTD4"
    ),
    use_svd_truncation=true,  # use SVD-based truncation
    tolerance=1e-5
)

#===================#
## Train Operators ##
#===================#
# Compose the data matrix
X = vcat(
    X_all["p"],
    X_all["z"],
    X_all["u"],
    X_all["v"],
    X_all["w"]
)

# Compute the reduced data matrix
Xhat = V' * X
# Shift the data for discrete-time data
Xhat1 = Xhat[:, 1:end-1]
Xhat2 = Xhat[:, 2:end]

## Train the operators
options.with_reg = true
options.λ = LnL.TikhonovParameter(A=1e-4, A2=1e-2, K=1e-4)
op = LnL.opinf(Xhat1, options; Xhatdot=Xhat2)

## Save the model
save(joinpath(FILEPATH, "data/models/r" * string(size(V,2)) * 
                            "_svd_reg_A1e-4_A21e-2_K1e-4.jld2"),
    "op", op
)

## Load the model if saved 
op = load(joinpath(FILEPATH, "data/models/r" * string(size(V,2)) * 
                            "_svd_reg_A1e-4_A21e-2_K1e-4.jld2"))["op"]

#===================#
## Check Operators ##
#===================#
include("integrate.jl")

# Integrate a single trajectory 
Xrom = rk4_integrate(Xhat[:,1], ds.grid["time"], op.A, op.A2u, op.K)


## Load the shift and scaling
shift  = load(joinpath(FILEPATH, "data/minmax.jld2"))["shift"]
scale  = load(joinpath(FILEPATH, "data/minmax.jld2"))["scale"]
mean   = load(joinpath(FILEPATH, "data/mean.jld2"))["mean"]

unscale = (X, scale, shift) -> (scale .* X) .+ shift
uncenter = (X, Xbar) -> X .+ Xbar

# function unnormalize(x, dim_per_field, scales, shifts)
#     x_unscaled = zeros(size(x))
#     i = 1
#     for (field, val) in scales
#         start_idx = (i - 1) * dim_per_field + 1
#         end_idx = i * dim_per_field
#         x_unscaled[start_idx:end_idx, :] = x[start_idx:end_idx, :] .* val .+ shifts[field]
#         i += 1
#     end
#     return x_unscaled
# end

# function normalize(x, dim_per_field, scales, shifts)
#     x_scaled = zeros(size(x))
#     i = 1
#     for (field, val) in scales
#         start_idx = (i - 1) * dim_per_field + 1
#         end_idx = i * dim_per_field
#         x_scaled[start_idx:end_idx, :] = (x[start_idx:end_idx, :] .- shifts[field]) ./ val
#         i += 1
#     end
#     return x_scaled
# end


# ## Load the original data 
# using HDF5
# DATAPATH = "../../../../DATA/THE_WELL/supernova_explosion_64/train"
# train_files = readdir(DATAPATH, join=true)
# fn = train_files[1]
# Xp = h5read(fn, "t0_fields")["pressure"] # pressure field data
# Xd = h5read(fn, "t0_fields")["density"]  # density field data
# Xvel = h5read(fn, "t1_fields")["velocity"] # velocity field data
# Xu = Xvel[1, :, :, :, :, :]
# Xv = Xvel[2, :, :, :, :, :]
# Xw = Xvel[3, :, :, :, :, :]
# xspan = h5read(fn, "dimensions")["x"]
# yspan = h5read(fn, "dimensions")["y"]
# zspan = h5read(fn, "dimensions")["z"]
# tspan = h5read(fn, "dimensions")["time"]
# nx, ny, nz, n_time, n_traj = size(Xp)
# Xvel = nothing
# n = n_time * n_traj
# Xp = reshape(Xp, nx, ny, nz, n)
# Xp = reshape(Xp, :, n)
# Xd = reshape(Xd, nx, ny, nz, n)
# Xd = reshape(Xd, :, n)
# Xz = 1 ./ Xd # specific volume
# Xu = reshape(Xu, nx, ny, nz, n)
# Xu = reshape(Xu, :, n)
# Xv = reshape(Xv, nx, ny, nz, n)
# Xv = reshape(Xv, :, n)
# Xw = reshape(Xw, nx, ny, nz, n)
# Xw = reshape(Xw, :, n)
# X_orig = vcat(
#     Xp,
#     Xz,
#     Xu,
#     Xv,
#     Xw
# )

# ## Compute the relative state error over all trajectories
# rse = 0.0
# X_single_traj = nothing
# X_rom_unscaled = nothing
# for i in 1:n_traj
#     X_single_traj = X_orig[:, (i-1)*n_time+1:i*n_time]
#     X_rom = rk4_integrate(Xhat[:,1], tspan, op.A, op.A2u, op.K)
#     X_rom = V * X_rom
#     X_rom_unscaled = unnormalize(X_rom, nx * ny * nz, scale, shift)
#     rse += norm(X_single_traj - X_rom_unscaled) / norm(X_single_traj)
#     println("Relative state error for trajectory $i: $rse")
# end
# rse /= n_traj

##
# Xrom = rk4_integrate(Xhat[:,1], tspan, op.A, op.A2u, op.K)
# norm(Xrom - Xhat[:,1:n_time]) / norm(Xhat[:,1:n_time])

##
# X_orig_processed = normalize(X_orig, nx * ny * nz, scale, shift) 
# norm(X_orig_processed - V * V' * X_orig_processed) / norm(X_orig_processed)

#============================#
## Plot the sliced pressure ##
#============================#
using CairoMakie
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1200, 900))
    # Get midpoint index for z-direction
    x_slice = nx ÷ 2
    y_slice = 1:ny
    z_slice = 1:nz

    # Select 3 time steps (beginning, middle, end)
    time_indices = Int.([ceil(n_time / 3), ceil(n_time * 2 / 3), n_time])

    # Pre-calculate all data for colorbar scaling
    all_full_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_rom_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_error_data = Vector{Matrix{Float64}}(undef, length(time_indices))

    # Collect all data first
    for (i, t_idx) in enumerate(time_indices)
        # Get full data
        full_field = ds["p"][:, :, :, t_idx, 1]
        all_full_data[i] = full_field[x_slice, y_slice, z_slice]
        
        # Get ROM data
        Xrecon = V * Xrom[:, t_idx]
        Xrecon = Xrecon[1:nxyz]
        Xrecon = unscale(Xrecon, scale["p"], shift["p"])
        Xrecon = uncenter(Xrecon, mean["p"])
        min_clip = minimum(abs.(Xrecon))
        Xrecon = max.(Xrecon, min_clip)
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
        ax_full = Axis(fig[1, i], 
            title = L"$t$ = %$(round(time_value, digits=2))",
            ylabel = i == 1 ? L"$x$" : "",
            xticklabelsvisible=false, xticksvisible=false,
            xlabelsize=30, ylabelsize=30, 
            xticklabelsize=25, yticklabelsize=25,
            titlesize=30, 
        )
        ax_rom = Axis(fig[2, i], 
            ylabel = i == 1 ? L"$x$" : "", 
            xticklabelsvisible=false, xticksvisible=false,
            xlabelsize=30, ylabelsize=30, 
            xticklabelsize=25, yticklabelsize=25,
        )
        ax_error = Axis(fig[3, i], 
            ylabel = i == 1 ? L"$x$" : "", 
            xlabel = L"$z$",
            xlabelsize=30, ylabelsize=30, 
            xticklabelsize=25, yticklabelsize=25,
        )

        # Create heatmaps with aligned color ranges
        xspan = ds.grid["x"]
        zspan = ds.grid["z"]
        hm_full = heatmap!(ax_full, zspan, xspan, all_full_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max),
            colorscale=log10)
        hm_rom = heatmap!(ax_rom, zspan, xspan, all_rom_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max),
            colorscale=log10)
        hm_error = heatmap!(ax_error, zspan, xspan, all_error_data[i], 
            colormap = :matter, colorrange = (error_min, error_max),
            colorscale=log10)
    end
    
    # Add colorbars at the end of each row
    Colorbar(fig[1, length(time_indices) + 1], hm_full, label="Full", labelsize=20)
    Colorbar(fig[2, length(time_indices) + 1], hm_rom, label="ROM", labelsize=20)
    Colorbar(fig[3, length(time_indices) + 1], hm_error, label="Abs. Error", labelsize=20)
    
    display(fig)
end


#============================#
## Plot the sliced velocity ##
#============================#
using CairoMakie
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1200, 900))
    # Get midpoint index for z-direction
    x_slice = nx ÷ 2
    y_slice = 1:ny
    z_slice = 1:nz

    # Select 3 time steps (beginning, middle, end)
    time_indices = Int.([ceil(n_time / 3), ceil(n_time * 2 / 3), n_time])

    # Pre-calculate all data for colorbar scaling
    all_full_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_rom_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_error_data = Vector{Matrix{Float64}}(undef, length(time_indices))

    # Collect all data first
    velocity = "w"
    if velocity == "u"
        start_idx = nxyz*2 + 1
        end_idx = nxyz*3
    elseif velocity == "v"
        start_idx = nxyz*3 + 1
        end_idx = nxyz*4
    elseif velocity == "w"
        start_idx = nxyz*4 + 1
        end_idx = nxyz*5
    end
    for (i, t_idx) in enumerate(time_indices)
        # Get full data
        full_field = ds[velocity][1, :, :, :, t_idx, 1]
        all_full_data[i] = full_field[x_slice, y_slice, z_slice]
        
        # Get ROM data
        Xrecon = V * Xrom[:, t_idx]
        Xrecon = Xrecon[start_idx:end_idx]
        Xrecon = unscale(Xrecon, scale[velocity], shift[velocity])
        Xrecon = uncenter(Xrecon, mean[velocity])
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
        ax_full = Axis(fig[1, i], 
            title = L"$t$ = %$(round(time_value, digits=2))",
            ylabel = i == 1 ? L"$x$" : "",
            xticklabelsvisible=false, xticksvisible=false,
            xlabelsize=30, ylabelsize=30, 
            xticklabelsize=25, yticklabelsize=25,
            titlesize=30, 
        )
        ax_rom = Axis(fig[2, i], 
            ylabel = i == 1 ? L"$x$" : "", 
            xticklabelsvisible=false, xticksvisible=false,
            xlabelsize=30, ylabelsize=30, 
            xticklabelsize=25, yticklabelsize=25,
        )
        ax_error = Axis(fig[3, i], 
            ylabel = i == 1 ? L"$x$" : "", 
            xlabel = L"$z$",
            xlabelsize=30, ylabelsize=30, 
            xticklabelsize=25, yticklabelsize=25,
        )

        # Create heatmaps with aligned color ranges
        xspan = ds.grid["x"]
        zspan = ds.grid["z"]
        hm_full = heatmap!(ax_full, zspan, xspan, all_full_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_rom = heatmap!(ax_rom, zspan, xspan, all_rom_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_error = heatmap!(ax_error, zspan, xspan, all_error_data[i], 
            colormap = :matter, colorrange = (error_min, error_max))
    end
    
    # Add colorbars at the end of each row
    Colorbar(fig[1, length(time_indices) + 1], hm_full, label="Full", labelsize=20)
    Colorbar(fig[2, length(time_indices) + 1], hm_rom, label="ROM", labelsize=20)
    Colorbar(fig[3, length(time_indices) + 1], hm_error, label="Abs. Error", labelsize=20)
    
    display(fig)
end

#================#
## Create video ##
#================#
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

    Xtmp = ds["p"][:, :, :, 1:n_time, 1] # pressure field data
    xspan = ds.grid["x"]
    yspan = ds.grid["y"]
    zspan = ds.grid["z"]

    mid = nx ÷ 2
    hm1 = heatmap!(ax1, xspan, yspan, Xtmp[:,:,mid,1], colormap=:plasma,
                   colorscale=log10, colorrange=(extrema(Xtmp[:,:,mid,:])))
    hm2 = heatmap!(ax2, yspan, zspan, Xtmp[mid,:,:,1], colormap=:plasma, colorscale=log10)
    hm3 = heatmap!(ax3, xspan, zspan, Xtmp[:,mid,:,1], colormap=:plasma, colorscale=log10)
    cb = Colorbar(fig[1, 4], hm1, label=L"pressure$$", labelsize=40, ticklabelsize=15)
    tight_ticklabel_spacing!(cb)
    record(fig, joinpath(FILEPATH, "plots/pressure_dist.mp4"), 1:n_time) do i
        hm1[3] = Xtmp[:,:,mid,i]
        hm2[3] = Xtmp[mid,:,:,i]
        hm3[3] = Xtmp[:,mid,:,i]
        autolimits!(ax1) # update limits
        autolimits!(ax2) # update limits
        autolimits!(ax3) # update limits
    end
end

#===============================#
## Create video Reconstruction ##
#===============================#
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

    Xtmp = V * Xrom
    Xtmp = Xtmp[1:nxyz, :] # pressure field data
    Xtmp = unscale(Xtmp, scale["p"], shift["p"])
    Xtmp = uncenter(Xtmp, mean["p"])
    Xtmp = reshape(Xtmp, nx, ny, nz, n_time)
    xspan = ds.grid["x"]
    yspan = ds.grid["y"]
    zspan = ds.grid["z"]

    mid = nx ÷ 2
    hm1 = heatmap!(ax1, xspan, yspan, Xtmp[:,:,mid,1], colormap=:plasma,
                   colorscale=NaNMath.log10, colorrange=(extrema(Xtmp[:,:,mid,:])))
    hm2 = heatmap!(ax2, yspan, zspan, Xtmp[mid,:,:,1], colormap=:plasma, 
                   colorscale=NaNMath.log10)
    hm3 = heatmap!(ax3, xspan, zspan, Xtmp[:,mid,:,1], colormap=:plasma, 
                   colorscale=NaNMath.log10)
    cb = Colorbar(fig[1, 4], hm1, label=L"pressure$$", labelsize=40, ticklabelsize=15)
    tight_ticklabel_spacing!(cb)
    record(fig, joinpath(FILEPATH, "plots/rom_pressure_dist.mp4"), 1:n_time) do i
        hm1[3] = Xtmp[:,:,mid,i]
        hm2[3] = Xtmp[mid,:,:,i]
        hm3[3] = Xtmp[:,mid,:,i]
        autolimits!(ax1) # update limits
        autolimits!(ax2) # update limits
        autolimits!(ax3) # update limits
    end
end