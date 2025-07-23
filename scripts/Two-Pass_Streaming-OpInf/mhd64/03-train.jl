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
           joinpath(pwd(), "Two-Pass_Streaming-OpInf/mhd64") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/mhd64")
DATAPATH = "../../../../DATA/THE_WELL/mhd64"
train_files = readdir(DATAPATH, join=true)
fn = train_files[1]
X = load(joinpath(FILEPATH, "data/preprocessed_data.jld2"))["X"]["all"]
V = load(joinpath(FILEPATH, "data/bases/basis.jld2"))["V"]

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
        state=[1,2],
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
    # use_normal_equations=false,
    # use_svd_truncation=true,  # use SVD-based truncation
    # tolerance=1e-3,
)

#===================#
## Train Operators ##
#===================#
CONTINUOUS_TIME = true
# Compute the reduced data matrix
if CONTINUOUS_TIME
    @info "Generate finite difference data for continuous-time "
    # Compute finite difference approximation
    Xhat = Vector{Matrix{Float64}}(undef, n_traj)
    Xdot = Vector{Matrix{Float64}}(undef, n_traj)
    Xhatdot = Vector{Matrix{Float64}}(undef, n_traj)
    # Do it individually for each trajectory
    for i in 1:n_traj
        idx_start = (i-1) * n_time + 1
        idx_end = i * n_time
        Xdot[i], idx = LnL.time_derivative_approx(
            X[:, idx_start:idx_end], options)
        Xhat[i] = V' * X[:, idx_start:idx_end][:, idx]

        # Mutliply the density data to the time-derivatives of velocities
        for j in 2:4
            Xdot[i][nxyz*(j-1)+1:nxyz*j, :] .*= X[1:nxyz, idx_start:idx_end]
        end
        Xhatdot[i] = V' * Xdot[i]
    end

    Xhat = reduce(hcat, Xhat)
    Xhatdot = reduce(hcat, Xhatdot)
else
    @info "Generate shifted data for discrete-time "
    # Shift the data for discrete-time data
    Xhat = Vector{Matrix{Float64}}(undef, n_traj)
    Xdot = Vector{Matrix{Float64}}(undef, n_traj)
    Xhatdot = Vector{Matrix{Float64}}(undef, n_traj)
    # Do it individually for each trajectory
    for i in 1:n_traj
        idx_start = (i-1) * n_time + 1
        idx_end = i * n_time
        Xhat[i] = X[:, idx_start:idx_end-1]
        Xdot[i] = X[:, idx_start+1:idx_end]

        # Mutliply the density data to the time-derivatives of velocities
        for i in 2:4
            Xdot[i][nxyz*(i-1)+1:nxyz*i, :] .*= Xhat[i][1:nxyz, :]
        end
        Xhat[i] = V' * Xhat[i]
        Xhatdot[i] = V' 
    end

    Xhat = reduce(hcat, Xhat)
    Xhatdot = reduce(hcat, Xhatdot)
end

## Train the operators
# options.with_reg = true
# options.λ = LnL.TikhonovParameter(A=1e-9, A2=1e-9, K=1e-9)
op = LnL.opinf(Xhat, options; Xhatdot=Xhatdot)

## Save the model
save(joinpath(FILEPATH, "data/models/r" * string(size(V,2)) * 
                            "_svd_reg_A1e-4_A21e-2_K1e-4.jld2"),
    "op", op
)

## Load the model if saved 
op = load(joinpath(FILEPATH, "data/models/r" * string(size(V,2)) * 
                            "_svd_reg_A1e-4_A21e-2_K1e-4.jld2"))["op"]

#================#
## Simulate ROM ##
#================#
include("integrate.jl")

# Integrate a single trajectory 
Xrom = Vector{Matrix{Float64}}(undef, n_traj)
for i in 1:n_traj
    idx_start = (i-1) * n_time + 1
    idx_end = i * n_time
    x0 = Xtmp[:, idx_start:idx_end][:,1]
    if CONTINUOUS_TIME
        Xrom[i] = rk4_integrate(x0, ds.grid["time"], op.A, op.A2u, op.K)
    else
        states = zeros(size(Xtmp,1), n_time)
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
end
Xrom = reduce(hcat, Xrom)

#=============================#
## Compute projection errors ##
#=============================#
## Original data (unscaled and uncentered)
X_orig = load(joinpath(FILEPATH, "data/original_data.jld2"))["X"]
shift  = load(joinpath(FILEPATH, "data/minmax.jld2"))["shift"]
scale  = load(joinpath(FILEPATH, "data/minmax.jld2"))["scale"]
mean   = load(joinpath(FILEPATH, "data/mean.jld2"))["mean"]

unscale = (X, scale, shift) -> (scale .* X) .+ shift
uncenter = (X, Xbar) -> X .+ Xbar

## Compute rse
begin
    # Processed data
    X_recon = V * Xrom
    X_error = X - X_recon
    rse_processed = Dict(
        fld => 0.0 for fld in [ds.fields, "all"]
    )
    for i in eachindex(ds.fields)
        fld = ds.fields[i]
        idx_start = (i-1) * nxyz + 1
        idx_end = i * nxyz
        num = norm(X_error[idx_start:idx_end, :], 2)
        den = norm(X[idx_start:idx_end, :], 2)
        rse_fld = num / den 
        rse_processed[ds.fields[i]] = rse_fld
        println("Reconstruction error for field $(fld): $rse_fld")
    end
    tmp = norm(X_error, 2) / norm(X, 2)
    rse_processed["all"] = tmp
    println("Overall reconstruction error: $(tmp)")

    scale_all = reduce(vcat, [scale[fld] for fld in ds.fields])
    shift_all = reduce(vcat, [shift[fld] for fld in ds.fields])
    mean_all  = reduce(vcat, [mean[fld] for fld in ds.fields])

    X_recon = unscale(X_recon, scale_all, shift_all)
    X_recon = uncenter(X_recon, mean_all)
    X_error = X_orig["all"] - X_recon

    rse_orig = Dict(
        fld => 0.0 for fld in [ds.fields, "all"]
    )

    for i in eachindex(ds.fields)
        fld = ds.fields[i]
        idx_start = (i-1) * nxyz + 1
        idx_end = i * nxyz
        X_error_field = X_error[idx_start:idx_end, :] 
        num = norm(X_error_field, 2)
        den = norm(X_orig[fld], 2)
        rse_fld = num / den
        rse_orig[fld] = rse_fld
        println("Relative reconstruction error for $fld: $rse_fld")
    end
    rse_orig["all"] = norm(X_recon) / norm(X_orig["all"])
    println("Overall relative reconstruction error: $(rse_orig["all"])")

    ## Save relative errors
    rse_file = joinpath(FILEPATH, "data/results/rse.jld2")
    if !isfile(rse_file)
        @info "Saving relative projection errors to file"
        save(rse_file, "rse", Dict("processed" => rse_processed, "original" => rse_orig))
    end
end

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

    unscale = (X, scale, shift) -> (scale .* X) .+ shift
    uncenter = (X, Xbar) -> X .+ Xbar

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
            yticklabelsvisible=i==1 ? true : false,
            yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            xticklabelsize=25, yticklabelsize=25,
            titlesize=30, 
        )
        ax_rom = Axis(fig[2, i], 
            ylabel = i == 1 ? L"$x$" : "", 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=i==1 ? true : false,
            yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            xticklabelsize=25, yticklabelsize=25,
        )
        ax_error = Axis(fig[3, i], 
            ylabel = i == 1 ? L"$x$" : "", 
            xlabel = L"$z$",
            yticklabelsvisible=i==1 ? true : false,
            yticksvisible=i==1 ? true : false,
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


#===================================#
## Plot the sliced specific volume ##
#===================================#
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

    unscale = (X, scale, shift) -> (scale .* X) .+ shift
    uncenter = (X, Xbar) -> X .+ Xbar

    # Collect all data first
    for (i, t_idx) in enumerate(time_indices)
        # Get full data
        full_field = ds["z"][:, :, :, t_idx, 1]
        all_full_data[i] = full_field[x_slice, y_slice, z_slice]
        
        # Get ROM data
        Xrecon = V * Xrom[:, t_idx]
        Xrecon = Xrecon[1:nxyz]
        Xrecon = unscale(Xrecon, scale["z"], shift["z"])
        Xrecon = uncenter(Xrecon, mean["z"])
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
            yticklabelsvisible=i==1 ? true : false,
            yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            xticklabelsize=25, yticklabelsize=25,
            titlesize=30, 
        )
        ax_rom = Axis(fig[2, i], 
            ylabel = i == 1 ? L"$x$" : "", 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=i==1 ? true : false,
            yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            xticklabelsize=25, yticklabelsize=25,
        )
        ax_error = Axis(fig[3, i], 
            ylabel = i == 1 ? L"$x$" : "", 
            xlabel = L"$z$",
            yticklabelsvisible=i==1 ? true : false,
            yticksvisible=i==1 ? true : false,
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
    x_slice = 1:nx
    y_slice = ny ÷ 2
    z_slice = 1:nz

    # Select 3 time steps (beginning, middle, end)
    time_indices = Int.([ceil(n_time / 3), ceil(n_time * 2 / 3), n_time])

    # Pre-calculate all data for colorbar scaling
    all_full_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_rom_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_error_data = Vector{Matrix{Float64}}(undef, length(time_indices))

    # Collect all data first
    velocity = "u"
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
            yticklabelsvisible=i==1 ? true : false,
            yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            xticklabelsize=25, yticklabelsize=25,
            titlesize=30, 
        )
        ax_rom = Axis(fig[2, i], 
            ylabel = i == 1 ? L"$x$" : "", 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=i==1 ? true : false,
            yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            xticklabelsize=25, yticklabelsize=25,
        )
        ax_error = Axis(fig[3, i], 
            ylabel = i == 1 ? L"$x$" : "", 
            xlabel = L"$z$",
            yticklabelsvisible=i==1 ? true : false,
            yticksvisible=i==1 ? true : false,
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
    Xtmp = Xtmp[1:nxyz, :] # pressure field data
    Xtmp = unscale(Xtmp, scale["p"], shift["p"])
    Xtmp = uncenter(Xtmp, mean["p"])
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