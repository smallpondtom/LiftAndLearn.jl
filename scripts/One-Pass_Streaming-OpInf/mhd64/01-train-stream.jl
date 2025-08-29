"""
MHD64: Compute Incremental OpInf (iOpInf)
"""

#=================#
## Load Packages ##
#=================#
using FileIO
using JLD2
using LinearAlgebra
using BlockDiagonals
using ProgressMeter
import LiftAndLearn as LnL

#=============#
## Load data ##
#=============#
ROOTPATH = occursin("scripts", pwd()) ?  pwd() : joinpath(pwd(), "scripts")
FILEPATH = joinpath(ROOTPATH, "One-Pass_Streaming-OpInf/mhd64")
DATAPATH = "../../../../DATA/THE_WELL/mhd64"
train_files = readdir(DATAPATH, join=true)
test_files = readdir(joinpath(DATAPATH, "test"), join=true)
fn = train_files[1]
fn_test = test_files[1] 
X = load(joinpath(
    ROOTPATH, 
    "Two-Pass_Streaming-OpInf/mhd64/data/preprocessed_data.jld2"))["X"]["all"]

# Include the data sourcing module for data access
include(joinpath(
    ROOTPATH, 
    "Two-Pass_Streaming-OpInf/mhd64/datasource.jl"))

# Load data source 
ds = DataSource(fn)
nx, ny, nz, n_fields, n_time, n_traj = ds.dims
nxyz = nx * ny * nz
n = n_time * n_traj

#=======================================#
## Some options for operator inference ##
#=======================================#
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
rmax = 50

#====================================#
## Compute One-Pass Streaming-OpInf ##
#====================================#
options.with_reg = true
options.λ = LnL.TikhonovParameter(A=1e12, A2=1e12, A3=1.9307e14, K=1e12)
stream = LnL.OnePassStreamingOpInf(
    X[:,1]; 
    options=options, 
    n=Int(nxyz * n_fields), 
    rank=rmax, 
    finite_diff=true
)
@showprogress for i in 2:n
    LnL.stream!(stream, X[:,i], tol=1e-8)
end
E, Δidx = LnL.finite_diff_matrix(
    options.data.deriv_type, n_time, options.data.Δt
)
E = BlockDiagonal([E, E, E, E, E])
op_stream = LnL.compute_stream_operators(stream, Array(E), (1, n))

## Save the stream object and operators
save(joinpath(FILEPATH, "data/results/onepass_stream.jld2"), 
     "stream", stream, "op_stream", op_stream)


#================#
## Simulate ROM ##
#================#
include(joinpath(ROOTPATH, "Two-Pass_Streaming-OpInf/mhd64/integrate.jl"))
Xrom_train = Vector{Matrix{Float64}}(undef, n_traj)
for i in 1:n_traj
    idx_start = (i-1) * n_time + 1
    idx_end = i * n_time
    x0 = stream.V' * X[:, idx_start:idx_end][:,1]
    Xrom_train[i], _ = rk4_integrate(
        x0, ds.grid["time"], 
        op_stream.A, op_stream.A2u, op_stream.A3u, op_stream.K)
end
Xrom_train = reduce(hcat, Xrom_train)

## Save the ROM's training data 
save(joinpath(FILEPATH, "data/results/rom_training_states.jld2"), 
     "states", Xrom_train)

#=============================#
## Load scaling and shifting ##
#=============================#
include(joinpath(ROOTPATH, "Two-Pass_Streaming-OpInf/mhd64/preprocess.jl"))
means = load(joinpath(ROOTPATH,  "Two-Pass_Streaming-OpInf/mhd64/data/mean.jld2"))["mean"]
scales = load(joinpath(ROOTPATH, "Two-Pass_Streaming-OpInf/mhd64/data/minmax.jld2"))["scale"]
shifts = load(joinpath(ROOTPATH, "Two-Pass_Streaming-OpInf/mhd64/data/minmax.jld2"))["shift"]

#====================================#
## Plot the reconstructed variables ##
#====================================#
using CairoMakie

with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1200, 940))
    # Pick trajectory
    traj_idx = 5
    # Get midpoint index for z-direction
    x_slice = nx ÷ 2
    y_slice = 1:ny
    z_slice = 1:nz
    # Pick state 
    var = "rho"  # "rho", "z", "mx", "my", "mz", "Bx", "By", "Bz"
    # Pick training or testing 
    train_or_test = "train"
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
        Xrecon = stream.V * Xrom_traj
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
    
    # save(joinpath(FILEPATH, "plots/$(train_or_test)_sliced_$(var).png"), fig)
    display(fig)
end