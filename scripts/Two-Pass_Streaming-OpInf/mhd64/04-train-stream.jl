#=================#
## Load packages ##
#=================#
using LinearAlgebra
using BlockDiagonals
using FileIO
using JLD2
using ProgressMeter
using Revise
using IncrementalSVD
using CUDA
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
baker = load(joinpath(FILEPATH, "data/bases/baker_basis.jld2"))["baker"]
V = baker.Q
rmax = size(V, 2)

# Include the data sourcing module for data access
include(joinpath(FILEPATH, "datasource.jl"))

# Load data source 
ds = DataSource(fn)
nx, ny, nz, n_fields, n_time, n_traj = ds.dims
nxyz = nx * ny * nz
n = n_time * n_traj

# Load the batch OpInf model
op = load(joinpath(FILEPATH, "data/models/opinf_mdl.jld2"))["op"]

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
)

#=========================#
## Generate Reduced Data ##
#=========================#
# Compute the reduced data matrix
Xtmp = V' * X
@info "Generate finite difference data for continuous-time "
# Compute finite difference approximation
Xhat = Vector{Matrix{Float64}}(undef, n_traj)
Xhatdot = Vector{Matrix{Float64}}(undef, n_traj)
# Do it individually for each trajectory
for i in 1:n_traj
    idx_start = (i-1) * n_time + 1
    idx_end = i * n_time
    Xhatdot[i], idx = LnL.time_derivative_approx(
        Xtmp[:, idx_start:idx_end], options)
    Xhat[i] = Xtmp[:, idx_start:idx_end][:, idx]
end
Xhat = reduce(hcat, Xhat)
Xhatdot = reduce(hcat, Xhatdot)


#===========================#
## Train  streaming models ##
#===========================#
Γ = 1e14  # Regularization parameter

# Dict to store streaming errors for each algorithm
stream_errors = Dict(
    :rls    => zeros(n),
    :iqrrls => zeros(n),
    :qrrls  => zeros(n)
)


## Streamify the data based on the selected streamsizes
X_stream = LnL.streamify(Xhat, 1)
Xdot_stream = LnL.streamify(Xhatdot, 1)

## RLS
rls_stream  = LnL.TwoPassStreamingOpInf(
    options=options, n=rmax, algorithm=:RLS, Γs=Γ, use_gpu=true) 

Onorm = norm(op.O, 2)
Ostar = op.O'
@showprogress for i in 1:n
    # The stream of data
    x_i    = X_stream[i]
    xdot_i = Xdot_stream[i]

    # Stream, update, and get data matrix for the state system
    LnL.stream!(rls_stream, x_i, xdot_i, use_gpu=true)   

    # Compute streaming errors
    stream_errors[:rls][i] = norm(Array(rls_stream.cache.O) - Ostar, 2) / Onorm
end
op_rls = LnL.terminate_stream(rls_stream)
rls_stream = nothing
GC.gc()
CUDA.reclaim()

## iQRRLS
iqrrls_stream = LnL.TwoPassStreamingOpInf(
    options=options, n=rmax, algorithm=:iQRRLS, Γs=Γ, 
    qr_method=:givens, use_gpu=false)
@showprogress for i in 1:n
    # The stream of data
    x_i    = X_stream[i]
    xdot_i = Xdot_stream[i]

    # Stream, update, and get data matrix for the state system
    LnL.stream!(iqrrls_stream, x_i, xdot_i)

    # Compute streaming errors
    stream_errors[:iqrrls][i] = norm(Array(iqrrls_stream.cache.O) - Ostar, 2) / Onorm
end
op_iqrrls = LnL.terminate_stream(iqrrls_stream)
iqrrls_stream = nothing
GC.gc()

## QRRLS
qrrls_stream = LnL.TwoPassStreamingOpInf(
    options=options, n=rmax, algorithm=:QRRLS, Γs=Γ, 
    qr_method=:givens, use_gpu=false)
@showprogress for i in 1:n
    # The stream of data
    x_i    = X_stream[i]
    xdot_i = Xdot_stream[i]

    # Stream, update, and get data matrix for the state system
    LnL.stream!(qrrls_stream, x_i, xdot_i)

    # Compute streaming errors
    stream_errors[:qrrls][i] = norm(Array(qrrls_stream.cache.O) - Ostar, 2) / Onorm
end
op_qrrls  = LnL.terminate_stream(qrrls_stream)
qrrls_stream = nothing
GC.gc()

## Save the streaming models
save(joinpath(FILEPATH, "data/models/streaming_opinf_mdl.jld2"),
     "op_rls", op_rls,
     "op_iqrrls", op_iqrrls,
     "op_qrrls", op_qrrls)

## Save the streaming errors
stream_errors_file = joinpath(FILEPATH, "data/results/streaming_errors.jld2")
save(stream_errors_file, "stream_errors", stream_errors)

#================#
## Simulate ROM ##
#================#
include("integrate.jl")

# Integrate a single trajectory 
tmp = Dict(
    :batch  => Vector{Matrix{Float64}}(undef, n_traj),
    :rls    => Vector{Matrix{Float64}}(undef, n_traj),
    :iqrrls => Vector{Matrix{Float64}}(undef, n_traj),
    :qrrls  => Vector{Matrix{Float64}}(undef, n_traj)
)
Xrom = Dict(
    :batch  => Matrix{Float64}(undef, nxyz, n),
    :rls    => Matrix{Float64}(undef, nxyz, n),
    :iqrrls => Matrix{Float64}(undef, nxyz, n),
    :qrrls  => Matrix{Float64}(undef, nxyz, n)
)
for alg in [:batch, :rls, :iqrrls, :qrrls]
    op_ = alg == :rls ? op_rls :
          alg == :iqrrls ? op_iqrrls : 
          op_qrrls
    A = Array(op_.A)
    A2u = Array(op_.A2u)
    A3u = Array(op_.A3u)
    K = Array(op_.K)
    for i in 1:n_traj
        idx_start = (i-1) * n_time + 1
        idx_end = i * n_time
        x0 = V' * X[:, idx_start:idx_end][:,1]
        tmp[alg][i] = rk4_integrate(x0, ds.grid["time"], A, A2u, A3u, K)
    end
    Xrom[alg] = reduce(hcat, tmp[alg])
end
tmp = nothing
GC.gc()

#===========================#
## Plot the sliced density ##
#===========================#
# Original data (unscaled and uncentered)
shift  = load(joinpath(FILEPATH, "data/minmax.jld2"))["shift"]
scale  = load(joinpath(FILEPATH, "data/minmax.jld2"))["scale"]
mean   = load(joinpath(FILEPATH, "data/mean.jld2"))["mean"]

##
using CairoMakie
with_theme(theme_latexfonts()) do 
    alg = "qrrls"
    algsym = Symbol(alg)

    fig = Figure(size=(1200, 900))
    # Pick trajectory
    traj_idx = 2
    # Get midpoint index for z-direction
    x_slice = nx ÷ 2
    y_slice = 1:ny
    z_slice = 1:nz

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
        full_field = ds["rho"][:, :, :, t_idx, traj_idx]
        all_full_data[i] = full_field[x_slice, y_slice, z_slice]
        
        # Get ROM data
        Xrom_traj = Xrom[algsym][:, (traj_idx-1) * n_time + t_idx]
        Xrecon = V * Xrom_traj
        Xrecon = Xrecon[1:nxyz]
        Xrecon = unscale(Xrecon, scale["rho"], shift["rho"])
        Xrecon = uncenter(Xrecon, mean["rho"])
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
            ylabel = i == 1 ? axis_label[2] : "",
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            # xticklabelsvisible=false, xticksvisible=false,
            # yticklabelsvisible=i==1 ? true : false,
            # yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            # xticklabelsize=25, yticklabelsize=25,
            titlesize=30, 
        )
        ax_rom = Axis(fig[2, i], 
            ylabel = i == 1 ? axis_label[2] : "", 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            # xticklabelsvisible=false, xticksvisible=false,
            # yticklabelsvisible=i==1 ? true : false,
            # yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            # xticklabelsize=25, yticklabelsize=25,
        )
        ax_error = Axis(fig[3, i], 
            ylabel = i == 1 ? axis_label[2] : "", 
            xlabel = axis_label[1],
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            # yticklabelsvisible=i==1 ? true : false,
            # yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            # xticklabelsize=25, yticklabelsize=25,
        )

        # Create heatmaps with aligned color ranges
        hm_full = heatmap!(ax_full, horz_span, vert_span, all_full_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max),
            colorscale=log10)
        hm_rom = heatmap!(ax_rom, horz_span, vert_span, all_rom_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max),
            colorscale=log10)
        hm_error = heatmap!(ax_error, horz_span, vert_span, all_error_data[i], 
            colormap = :matter, colorrange = (error_min, error_max),
            colorscale=log10)
    end
    
    # Add colorbars at the end of each row
    Colorbar(fig[1, length(time_indices) + 1], hm_full, label="Full", 
             labelsize=30, ticklabelsize=20)
    Colorbar(fig[2, length(time_indices) + 1], hm_rom, label="ROM", 
             labelsize=30, ticklabelsize=20)
    Colorbar(fig[3, length(time_indices) + 1], hm_error, label="Abs. Error", 
             labelsize=30, ticklabelsize=20)
    
    save(joinpath(FILEPATH, "plots/sliced_density_$(alg).png"), fig)
    display(fig)
end


#===================================#
## Plot the sliced specific volume ##
#===================================#
with_theme(theme_latexfonts()) do 
    alg = "qrrls"
    algsym = Symbol(alg)

    fig = Figure(size=(1200, 900))
    # Pick trajectory
    traj_idx = 1
    # Get midpoint index for z-direction
    x_slice = nx ÷ 2
    y_slice = 1:ny
    z_slice = 1:nz

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
        full_field = ds["z"][:, :, :, t_idx, traj_idx]
        all_full_data[i] = full_field[x_slice, y_slice, z_slice]
        
        # Get ROM data
        Xrom_traj = Xrom[algsym][:, (traj_idx-1) * n_time + t_idx]
        Xrecon = V * Xrom_traj
        Xrecon = Xrecon[1:nxyz]
        Xrecon = unscale(Xrecon, scale["z"], shift["z"])
        Xrecon = uncenter(Xrecon, mean["z"])
        # min_clip = minimum(abs.(Xrecon))
        # Xrecon = max.(Xrecon, min_clip)
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
            ylabel = i == 1 ? axis_label[2] : "",
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            # yticklabelsvisible=i==1 ? true : false,
            # yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            # xticklabelsize=25, yticklabelsize=25,
            titlesize=30, 
        )
        ax_rom = Axis(fig[2, i], 
            ylabel = i == 1 ? axis_label[2] : "", 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            # yticklabelsvisible=i==1 ? true : false,
            # yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            # xticklabelsize=25, yticklabelsize=25,
        )
        ax_error = Axis(fig[3, i], 
            ylabel = i == 1 ? L"$x$" : "", 
            xlabel = axis_label[1],
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            # yticklabelsvisible=i==1 ? true : false,
            # yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            # xticklabelsize=25, yticklabelsize=25,
        )

        # Create heatmaps with aligned color ranges
        hm_full = heatmap!(ax_full, horz_span, vert_span, all_full_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max),
            colorscale=log10)
        hm_rom = heatmap!(ax_rom, horz_span, vert_span, all_rom_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max),
            colorscale=log10)
        hm_error = heatmap!(ax_error, horz_span, vert_span, all_error_data[i], 
            colormap = :matter, colorrange = (error_min, error_max),
            colorscale=log10)
    end
    
    # Add colorbars at the end of each row
    Colorbar(fig[1, length(time_indices) + 1], hm_full, label="Full", 
             labelsize=30, ticklabelsize=20)
    Colorbar(fig[2, length(time_indices) + 1], hm_rom, label="ROM", 
             labelsize=30, ticklabelsize=20)
    Colorbar(fig[3, length(time_indices) + 1], hm_error, label="Abs. Error", 
             labelsize=30, ticklabelsize=20)
    save(joinpath(FILEPATH, "plots/sliced_volume_$(alg).png"), fig)
    display(fig)
end



#============================#
## Plot the sliced momentum ##
#============================#
using CairoMakie
with_theme(theme_latexfonts()) do 
    alg = "qrrls"
    algsym = Symbol(alg)

    fig = Figure(size=(1200, 900))
    # Pick trajectory
    traj_idx = 1
    # Get midpoint index for z-direction
    x_slice = 1:nx
    y_slice = ny ÷ 2
    z_slice = 1:nz

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

    # Select 3 time steps (beginning, middle, end)
    time_indices = Int.([ceil(n_time / 3), ceil(n_time * 2 / 3), n_time])

    # Pre-calculate all data for colorbar scaling
    all_full_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_rom_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_error_data = Vector{Matrix{Float64}}(undef, length(time_indices))

    unscale = (X, scale, shift) -> (scale .* X) .+ shift
    uncenter = (X, Xbar) -> X .+ Xbar

    # Collect all data first
    momentum = "mx"
    if momentum == "mx"
        start_idx = nxyz*2 + 1
        end_idx = nxyz*3
    elseif momentum == "my"
        start_idx = nxyz*3 + 1
        end_idx = nxyz*4
    elseif momentum == "mz"
        start_idx = nxyz*4 + 1
        end_idx = nxyz*5
    end
    for (i, t_idx) in enumerate(time_indices)
        # Get full data
        full_field = ds[momentum][1, :, :, :, t_idx, traj_idx]
        all_full_data[i] = full_field[x_slice, y_slice, z_slice]
        
        # Get ROM data
        Xrom_traj = Xrom[algsym][:, (traj_idx-1) * n_time + t_idx]
        Xrecon = V * Xrom_traj
        Xrecon = Xrecon[start_idx:end_idx]
        Xrecon = unscale(Xrecon, scale[momentum], shift[momentum])
        Xrecon = uncenter(Xrecon, mean[momentum])
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
            ylabel = i == 1 ? axis_label[2] : "",
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            # yticklabelsvisible=i==1 ? true : false,
            # yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            # xticklabelsize=25, yticklabelsize=25,
            titlesize=30, 
        )
        ax_rom = Axis(fig[2, i], 
            ylabel = i == 1 ? axis_label[2] : "", 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            # yticklabelsvisible=i==1 ? true : false,
            # yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            # xticklabelsize=25, yticklabelsize=25,
        )
        ax_error = Axis(fig[3, i], 
            ylabel = i == 1 ? L"$x$" : "", 
            xlabel = axis_label[1],
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            # yticklabelsvisible=i==1 ? true : false,
            # yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            # xticklabelsize=25, yticklabelsize=25,
        )

        # Create heatmaps with aligned color ranges
        hm_full = heatmap!(ax_full, horz_span, vert_span, all_full_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_rom = heatmap!(ax_rom, horz_span, vert_span, all_rom_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_error = heatmap!(ax_error, horz_span, vert_span, all_error_data[i], 
            colormap = :matter, colorrange = (error_min, error_max))
    end
    
    # Add colorbars at the end of each row
    Colorbar(fig[1, length(time_indices) + 1], hm_full, label="Full", 
             labelsize=30, ticklabelsize=20)
    Colorbar(fig[2, length(time_indices) + 1], hm_rom, label="ROM", 
             labelsize=30, ticklabelsize=20)
    Colorbar(fig[3, length(time_indices) + 1], hm_error, label="Abs. Error", 
             labelsize=30, ticklabelsize=20)
    save(joinpath(FILEPATH, "plots/sliced_momentum_$(alg).png"), fig)
    display(fig)
end


#==================================#
## Plot the sliced magnetic field ##
#==================================#
using CairoMakie
with_theme(theme_latexfonts()) do 
    alg = "qrrls"
    algsym = Symbol(alg)

    fig = Figure(size=(1200, 900))
    # Pick trajectory
    traj_idx = 2
    # Get midpoint index for z-direction
    x_slice = 1:nx
    y_slice = ny ÷ 2
    z_slice = 1:nz

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

    # Select 3 time steps (beginning, middle, end)
    time_indices = Int.([ceil(n_time / 3), ceil(n_time * 2 / 3), n_time])

    # Pre-calculate all data for colorbar scaling
    all_full_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_rom_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_error_data = Vector{Matrix{Float64}}(undef, length(time_indices))

    unscale = (X, scale, shift) -> (scale .* X) .+ shift
    uncenter = (X, Xbar) -> X .+ Xbar

    # Collect all data first
    magnetic = "Bx"
    if magnetic == "Bx"
        start_idx = nxyz*5 + 1
        end_idx = nxyz*6
    elseif magnetic == "By"
        start_idx = nxyz*6 + 1
        end_idx = nxyz*7
    elseif magnetic == "Bz"
        start_idx = nxyz*7 + 1
        end_idx = nxyz*8
    end
    for (i, t_idx) in enumerate(time_indices)
        # Get full data
        full_field = ds[magnetic][1, :, :, :, t_idx, traj_idx]
        all_full_data[i] = full_field[x_slice, y_slice, z_slice]
        
        # Get ROM data
        Xrom_traj = Xrom[algsym][:, (traj_idx-1) * n_time + t_idx]
        Xrecon = V * Xrom_traj
        Xrecon = Xrecon[start_idx:end_idx]
        Xrecon = unscale(Xrecon, scale[magnetic], shift[magnetic])
        Xrecon = uncenter(Xrecon, mean[magnetic])
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
            ylabel = i == 1 ? axis_label[2] : "",
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            # yticklabelsvisible=i==1 ? true : false,
            # yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            # xticklabelsize=25, yticklabelsize=25,
            titlesize=30, 
        )
        ax_rom = Axis(fig[2, i], 
            ylabel = i == 1 ? axis_label[2] : "", 
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            # yticklabelsvisible=i==1 ? true : false,
            # yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            # xticklabelsize=25, yticklabelsize=25,
        )
        ax_error = Axis(fig[3, i], 
            ylabel = i == 1 ? L"$x$" : "", 
            xlabel = axis_label[1],
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            # yticklabelsvisible=i==1 ? true : false,
            # yticksvisible=i==1 ? true : false,
            xlabelsize=30, ylabelsize=30, 
            # xticklabelsize=25, yticklabelsize=25,
        )

        # Create heatmaps with aligned color ranges
        hm_full = heatmap!(ax_full, horz_span, vert_span, all_full_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_rom = heatmap!(ax_rom, horz_span, vert_span, all_rom_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_error = heatmap!(ax_error, horz_span, vert_span, all_error_data[i], 
            colormap = :matter, colorrange = (error_min, error_max))
    end
    
    # Add colorbars at the end of each row
    Colorbar(fig[1, length(time_indices) + 1], hm_full, label="Full", 
             labelsize=30, ticklabelsize=20)
    Colorbar(fig[2, length(time_indices) + 1], hm_rom, label="ROM", 
             labelsize=30, ticklabelsize=20)
    Colorbar(fig[3, length(time_indices) + 1], hm_error, label="Abs. Error", 
             labelsize=30, ticklabelsize=20)
    save(joinpath(FILEPATH, "plots/sliced_magnetic_$(alg).png"), fig)
    display(fig)
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
