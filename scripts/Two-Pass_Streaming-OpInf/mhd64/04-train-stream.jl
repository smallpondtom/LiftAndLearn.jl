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
op = load(joinpath(FILEPATH, "data/models/batch_opinf_mdl.jld2"))["op"]

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
# Load the best regularization parameters
betas = load(joinpath(FILEPATH, "data/results", "reg_grid_search.jld2"))["betas"]
options.with_reg = true
options.λ = LnL.TikhonovParameter(
    A=betas.b1, A2=betas.b2, A3=betas.b3, K=betas.b1
)

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
    options=options, n=rmax, algorithm=:RLS, use_gpu=true) 

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
    options=options, n=rmax, algorithm=:iQRRLS,
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
    options=options, n=rmax, algorithm=:QRRLS,
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
    # :rls    => Vector{Matrix{Float64}}(undef, n_traj),
    :iqrrls => Vector{Matrix{Float64}}(undef, n_traj),
    :qrrls  => Vector{Matrix{Float64}}(undef, n_traj)
)
Xrom_train = Dict(
    # :rls    => Matrix{Float64}(undef, rmax, n),
    :iqrrls => Matrix{Float64}(undef, rmax, n),
    :qrrls  => Matrix{Float64}(undef, rmax, n)
)
for alg in [:iqrrls, :qrrls]
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
        tmp[alg][i], _ = rk4_integrate(x0, ds.grid["time"], A, A2u, A3u, K)
    end
    Xrom_train[alg] = reduce(hcat, tmp[alg])
end
tmp = nothing
GC.gc()

## Save states 
save(joinpath(FILEPATH, "data/results/stream_rom_training_states.jld2"), 
     "states", Xrom_train)

#=============================#
## Simulate the testing data ##
#=============================#
include("preprocess.jl")
# Original data (unscaled and uncentered)
shifts  = load(joinpath(FILEPATH, "data/minmax.jld2"))["shift"]
scales  = load(joinpath(FILEPATH, "data/minmax.jld2"))["scale"]
means   = load(joinpath(FILEPATH, "data/mean.jld2"))["mean"]

##
fn_test = joinpath(DATAPATH, "test")
fn_test = readdir(fn_test, join=true)[1]
ds_test = DataSource(fn_test)
Xtest = load(joinpath(FILEPATH, "data/test_data.jld2"))["X"]["all"]
# Integrate a single trajectory 
x0 = V' * preprocess!(Xtest[:,1], vec(means["all"]), vec(shifts["all"]), 
                      vec(scales["all"]))

Xrom_test = Dict(
    # :rls    => Matrix{Float64}(undef, rmax, n),
    :iqrrls => Matrix{Float64}(undef, rmax, n),
    :qrrls  => Matrix{Float64}(undef, rmax, n)
)

for alg in [:iqrrls, :qrrls]
    op_ = alg == :rls ? op_rls :
          alg == :iqrrls ? op_iqrrls : 
          op_qrrls
    A = Array(op_.A)
    A2u = Array(op_.A2u)
    A3u = Array(op_.A3u)
    K = Array(op_.K)
    Xrom_test[alg], _ = rk4_integrate(x0, ds_test.grid["time"], A, A2u, A3u, K)
end

## Save the ROM's training data 
save(joinpath(FILEPATH, "data/results/stream_rom_testing_states.jld2"), 
     "states", Xrom_test)

#=============================#
## Plot the sliced variables ##
#=============================#
using CairoMakie

with_theme(theme_latexfonts()) do 
    alg = "iqrrls"
    algsym = Symbol(alg)

    fig = Figure(size=(1200, 940))
    # Pick trajectory
    traj_idx = 1
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
        Xrom_traj = Xrom[algsym][:, (traj_idx-1) * n_time + t_idx]
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
    Colorbar(fig[2, length(time_indices) + 1], hm_rom, label="ROM", 
             labelsize=30, ticklabelsize=20)
    Colorbar(fig[3, length(time_indices) + 1], hm_error, label="Abs. Error", 
             labelsize=30, ticklabelsize=20)
    
    save(joinpath(FILEPATH, "plots/$(train_or_test)_sliced_$(var)_$(alg).png"), fig)
    display(fig)
end


