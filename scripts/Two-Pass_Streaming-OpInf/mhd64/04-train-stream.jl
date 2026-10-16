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
# using CUDA
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
##
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

## Load states 
Xrom_train = load(joinpath(FILEPATH, "data/results/stream_rom_training_states.jld2"))["states"]

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

## Integrate a single trajectory 
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

## Load test states 
Xrom_test = load(joinpath(FILEPATH, "data/results/stream_rom_testing_states.jld2"))["states"]

##===================================================================#
# Plot the sliced variables (train + test, w/ projection error)     ##
# Layout: 6 columns (3 train + 3 test) x 5 rows                     ##
#   Row 1: Original                                                 ##
#   Row 2: Streaming OpInf                                          ##
#   Row 3: POD Projection                                           ##
#   Row 4: Streaming OpInf Error                                    ##
#   Row 5: POD Projection Error                                     ##
#====================================================================#
using CairoMakie

with_theme(theme_latexfonts()) do
    #========================#
    ## User configuration   ##
    #========================#
    alg = "iqrrls"           # "iqrrls" or "qrrls"
    algsym = Symbol(alg)
    traj_idx = 1             # Training trajectory to plot
    # Slice direction: set exactly one to an Int, others to ranges
    x_slice = nx ÷ 2
    y_slice = 1:ny
    z_slice = 1:nz
    # State: "rho", "z", "mx", "my", "mz", "Bx", "By", "Bz"
    var = "Bz"

    fig = Figure(size=(1800, 950))

    # Open both data sources
    ds_train = DataSource(fn)
    ds_test  = DataSource(fn_test)
    n_time_train = ds_train.dims[5]
    n_time_test  = ds_test.dims[5]

    # Determine slice axes
    if x_slice isa Int
        axis_label = [L"$\omega_2$", L"$\omega_3$"]
        horz_span = ds_train.grid["y"]
        vert_span = ds_train.grid["z"]
    elseif y_slice isa Int
        axis_label = [L"$\omega_1$", L"$\omega_3$"]
        horz_span = ds_train.grid["x"]
        vert_span = ds_train.grid["z"]
    else
        axis_label = [L"$\omega_1$", L"$\omega_2$"]
        horz_span = ds_train.grid["x"]
        vert_span = ds_train.grid["y"]
    end

    # Field index range and momentum/magnetic flags
    is_momentum = false
    is_magnetic = false
    if var == "rho"
        start_idx = 1;            end_idx = nxyz
    elseif var == "z"
        start_idx = nxyz + 1;     end_idx = nxyz * 2
    elseif var == "mx"
        start_idx = nxyz * 2 + 1; end_idx = nxyz * 3; c = 1; is_momentum = true
    elseif var == "my"
        start_idx = nxyz * 3 + 1; end_idx = nxyz * 4; c = 2; is_momentum = true
    elseif var == "mz"
        start_idx = nxyz * 4 + 1; end_idx = nxyz * 5; c = 3; is_momentum = true
    elseif var == "Bx"
        start_idx = nxyz * 5 + 1; end_idx = nxyz * 6; c = 1; is_magnetic = true
    elseif var == "By"
        start_idx = nxyz * 6 + 1; end_idx = nxyz * 7; c = 2; is_magnetic = true
    elseif var == "Bz"
        start_idx = nxyz * 7 + 1; end_idx = nxyz * 8; c = 3; is_magnetic = true
    end

    # 3 training time indices + 3 testing time indices (first test snapshot included)
    train_time_indices = Int.([ceil(n_time_train / 4),
                               ceil(n_time_train / 2),
                               n_time_train - 10])
    test_time_indices  = Int.([1,
                               ceil(n_time_test / 3),
                               n_time_test - 20])

    all_time_data = [
        (train_time_indices[1], "train"),
        (train_time_indices[2], "train"),
        (train_time_indices[3], "train"),
        (test_time_indices[1],  "test"),
        (test_time_indices[2],  "test"),
        (test_time_indices[3],  "test"),
    ]
    n_cols = length(all_time_data)

    unscale  = (X, scale, shift) -> (scale .* X) .+ shift
    uncenter = (X, Xbar)         -> X .+ Xbar

    # Full-state preprocessing parameters (for POD projection of test data)
    means_all  = vec(means["all"])
    shifts_all = vec(shifts["all"])
    scales_all = vec(scales["all"])

    # Pre-allocate
    all_full_data      = Vector{Matrix{Float64}}(undef, n_cols)
    all_rom_data       = Vector{Matrix{Float64}}(undef, n_cols)
    all_pod_data       = Vector{Matrix{Float64}}(undef, n_cols)
    all_error_data     = Vector{Matrix{Float64}}(undef, n_cols)
    all_pod_error_data = Vector{Matrix{Float64}}(undef, n_cols)

    for (col_idx, (t_idx, data_type)) in enumerate(all_time_data)
        if data_type == "train"
            ds_local    = ds_train
            local_traj  = traj_idx
            # Streaming OpInf reduced state (column from training matrix)
            Xrom_col    = Xrom_train[algsym][:, (local_traj - 1) * n_time + t_idx]
            # Preprocessed full state for POD projection (X is already preprocessed)
            x_full_pre  = X[:, (local_traj - 1) * n_time + t_idx]
        else
            ds_local    = ds_test
            local_traj  = 1  # only one trajectory in test set
            Xrom_col    = Xrom_test[algsym][:, t_idx]
            # Preprocess test snapshot for POD projection
            x_full_pre  = preprocess!(copy(Xtest[:, t_idx]),
                                      means_all, shifts_all, scales_all)
        end

        # --- Original full data slice ---
        if is_momentum || is_magnetic
            full_field = ds_local[var][c, :, :, :, t_idx, local_traj]
        else
            full_field = ds_local[var][:, :, :, t_idx, local_traj]
        end
        all_full_data[col_idx] = full_field[x_slice, y_slice, z_slice]

        # --- Streaming OpInf reconstruction ---
        Xrecon = V * Xrom_col
        Xrecon_f = Xrecon[start_idx:end_idx]
        Xrecon_f = unscale(Xrecon_f, scales[var], shifts[var])
        Xrecon_f = uncenter(Xrecon_f, means[var])
        rom_field = reshape(Xrecon_f[1:nxyz], nx, ny, nz)
        all_rom_data[col_idx] = rom_field[x_slice, y_slice, z_slice]

        # --- POD projection (V * V' * x_full_pre) ---
        Xpod   = V * (V' * x_full_pre)
        Xpod_f = Xpod[start_idx:end_idx]
        Xpod_f = unscale(Xpod_f, scales[var], shifts[var])
        Xpod_f = uncenter(Xpod_f, means[var])
        pod_field = reshape(Xpod_f[1:nxyz], nx, ny, nz)
        all_pod_data[col_idx] = pod_field[x_slice, y_slice, z_slice]

        # --- Errors ---
        all_error_data[col_idx]     = abs.(all_full_data[col_idx] - all_rom_data[col_idx])
        all_pod_error_data[col_idx] = abs.(all_full_data[col_idx] - all_pod_data[col_idx])
    end

    # Aligned color ranges
    full_min, full_max = extrema(vcat(all_full_data...))
    rom_min,  rom_max  = extrema(vcat(all_rom_data...))
    pod_min,  pod_max  = extrema(vcat(all_pod_data...))
    err_min,  err_max  = extrema(vcat(all_error_data...))
    perr_min, perr_max = extrema(vcat(all_pod_error_data...))

    common_min = min(full_min, rom_min, pod_min)
    common_max = max(full_max, rom_max, pod_max)
    err_common_min = min(err_min, perr_min)
    err_common_max = max(err_max, perr_max)

    hm_full = nothing
    hm_rom  = nothing
    hm_pod  = nothing
    hm_error = nothing
    hm_pod_error = nothing

    for (col_idx, (t_idx, data_type)) in enumerate(all_time_data)
        ds_local   = data_type == "train" ? ds_train : ds_test
        time_value = ds_local.grid["time"][t_idx]
        col_title  = data_type == "train" ?
            L"Train: $t$=%$(round(time_value, digits=2))" :
            L"Test: $t$=%$(round(time_value, digits=2))"

        ax_full = Axis(fig[1, col_idx],
            title = col_title,
            ylabel = col_idx == 1 ? axis_label[2] : "",
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            xlabelsize=30, ylabelsize=30, titlesize=28,
        )
        ax_rom = Axis(fig[2, col_idx],
            ylabel = col_idx == 1 ? axis_label[2] : "",
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            xlabelsize=30, ylabelsize=30,
        )
        ax_pod = Axis(fig[3, col_idx],
            ylabel = col_idx == 1 ? axis_label[2] : "",
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            xlabelsize=30, ylabelsize=30,
        )
        ax_error = Axis(fig[4, col_idx],
            ylabel = col_idx == 1 ? axis_label[2] : "",
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            xlabelsize=30, ylabelsize=30,
        )
        ax_pod_error = Axis(fig[5, col_idx],
            ylabel = col_idx == 1 ? axis_label[2] : "",
            xlabel = axis_label[1],
            xticklabelsvisible=false, xticksvisible=false,
            yticklabelsvisible=false, yticksvisible=false,
            xlabelsize=30, ylabelsize=30,
        )

        hm_full = heatmap!(ax_full, horz_span, vert_span, all_full_data[col_idx],
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_rom = heatmap!(ax_rom, horz_span, vert_span, all_rom_data[col_idx],
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_pod = heatmap!(ax_pod, horz_span, vert_span, all_pod_data[col_idx],
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_error = heatmap!(ax_error, horz_span, vert_span, all_error_data[col_idx],
            colormap = :matter, colorrange = (err_common_min, err_common_max))
        hm_pod_error = heatmap!(ax_pod_error, horz_span, vert_span, all_pod_error_data[col_idx],
            colormap = :matter, colorrange = (err_common_min, err_common_max))
    end

    Colorbar(fig[1, n_cols + 1], hm_full,
             label="Original", labelsize=26, ticklabelsize=22)
    Colorbar(fig[2, n_cols + 1], hm_rom,
             label="Streaming\nOpInf", labelsize=26, ticklabelsize=22)
    Colorbar(fig[3, n_cols + 1], hm_pod,
             label="Projection", labelsize=26, ticklabelsize=22)
    Colorbar(fig[4, n_cols + 1], hm_error,
             label="Streaming\nOpInf Error", labelsize=26, ticklabelsize=22)
    Colorbar(fig[5, n_cols + 1], hm_pod_error,
             label="Projection\nError", labelsize=26, ticklabelsize=22)

    save(joinpath(FILEPATH, "plots/train_test_sliced_$(var)_$(alg).png"),
         fig, px_per_unit=4.0)
    display(fig)
end


