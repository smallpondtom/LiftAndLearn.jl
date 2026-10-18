"""
ABL: Compute the energy spectrum
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using IncrementalSVD
using LinearAlgebra
using ProgressMeter
using SparseArrays
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

#========================#
## Additional functions
#========================#
include(joinpath(FILEPATH, "utils.jl"))

#=============================#
## Load the training dataset
#=============================#
ds = ChannelDataSource(datafile, ["z", "y", "x", "fields", "times"])
Nz, Ny, Nx, n_fields, n = ds.dims
dim_per_field = Nz * Ny * Nx
field_names = ["u", "v", "w", "p"]

#========================================#
## Compute the mean velocity for shifting
#========================================#
SHIFT_MEAN = true
xbar = nothing # preallocate
if SHIFT_MEAN
    mean_file = joinpath(FILEPATH, "data/mean.jld2")
    if isfile(mean_file)
        @info "Loading existing mean from file"
        xbar = load(mean_file, "xbar")
    else
        @info "Starting mean computation with $(Threads.nthreads()) threads"
        @time begin
            # Exclude pressure field for mean computation (only velocity components)
            xbar = compute_mean_parallel_threads(ds, (Nz*Ny*Nx*3,); batch_size=100)
        end
        @info "Mean computation complete"
        save(mean_file, "xbar", xbar)
    end
else
    @info "Skipping mean computation, using zero mean"
    xbar = 0.0
end

ubar = sum(abs, xbar[1:dim_per_field]) / dim_per_field
vbar = sum(abs, xbar[dim_per_field+1:dim_per_field*2]) / dim_per_field
wbar = sum(abs, xbar[dim_per_field*2+1:dim_per_field*3]) / dim_per_field
pbar = sum(abs, xbar[dim_per_field*3+1:dim_per_field*4]) / dim_per_field
scale_factors = [
    1.0, vbar / ubar, wbar / ubar, pbar / ubar
]

#========================================#
## Compute the energy spectrum
#========================================#
# Cov = Dict(fn => zeros(n, n) for fn in field_names)
# singular_values = Dict(fn => zeros(n) for fn in field_names)
# for field in field_names
#     Cov = zeros(n, n)
#     for i in 1:dim_per_field
#         Cov[field] .+= ds[field][i, 1:n]' * ds[field][i, 1:n]
#     end
#     singular_values[field] = eigvals(Cov) .|> sqrt
# end

using CUDA, FLoops
Cov = Dict(fn => zeros(n, n) for fn in field_names)
singular_values = Dict(fn => zeros(n) for fn in field_names)
@time begin
    for field in field_names
        @info "Processing field: $field"
        
        # Load data in chunks to GPU
        CHUNK_SIZE = min(1000, dim_per_field)  # Adjust based on GPU memory
        num_chunks = ceil(Int, dim_per_field / CHUNK_SIZE)
        
        # Initialize covariance matrix on GPU
        Cov_gpu = CUDA.zeros(Float64, n, n)

        # Create a lock for thread-safe GPU operations
        gpu_lock = ReentrantLock()

        @info "Processing $num_chunks chunks with $(Threads.nthreads()) threads"
        
        @floop ThreadedEx() for chunk in 1:num_chunks
            start_idx = (chunk - 1) * CHUNK_SIZE + 1
            end_idx = min(start_idx + CHUNK_SIZE - 1, dim_per_field)
            
            # Load chunk data to GPU
            chunk_data = zeros(Float64, end_idx - start_idx + 1, n)
            range = start_idx:end_idx
            chunk_data[1:length(range), :] = ds[field][range, 1:n]
            
            # Transfer to GPU and compute covariance contribution (synchronized)
            lock(gpu_lock) do
                chunk_data_gpu = cu(chunk_data)
                Cov_gpu .+= chunk_data_gpu' * chunk_data_gpu
                # Force synchronization to ensure computation is complete
                CUDA.synchronize()
            end
        end
        
        # Transfer result back to CPU and compute eigenvalues
        Cov_cpu = Array(Cov_gpu)
        singular_values[field] = eigvals(Cov_cpu) .|> sqrt
    end
end

##
function check_energy_retainment(svals, target=0.999)
    energy = sum(svals.^2)
    energy_ret = cumsum(svals.^2) / energy
    r = findfirst(x -> x > target, energy_ret) + 1
    return energy_ret, r
end

spectrum = copy(singular_values)
target_r = Dict(fn => 0 for fn in field_names)
target_energy = 0.75
for (field, svals) in singular_values
    spectrum[field], target_r[field] = check_energy_retainment(svals, target_energy)
end

#========================================#
## Plot the energy spectrum
#========================================#
using CairoMakie
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1400, 1000))
    ax1 = Axis(
        fig[2, 1], # xlabel=L"singular value index, $i$", 
        ylabel="energy retainment",
        limits=(-50, length(spectrum["pressure"])+10, nothing, nothing),
        # xgridvisible=false, ygridvisible=false,
        topspinevisible=false, rightspinevisible=false,
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25,
        yticks=(0:0.1:1, ["0%", "10%", "20%", "30%", "40%", 
                          "50%", "60%", "70%", "80%", "90%", "100%"])
    )

    colors = Dict(
        key => color for (key, color) in zip(
            field_names, Makie.wong_colors()[1:5]
        )
    )

    # Plot energy retainment for each field
    for (key, svals) in zip(keys(spectrum), [sp, sz, su, sv, sw])
        lines!(ax1, 1:length(svals), spectrum[key], 
               linewidth=4, color=colors[key])
    end

    # Add horizontal line at 99.9% energy retention
    hlines!(ax1, [target_energy], color=:black, linestyle=:dash, linewidth=4)

    ax2 = Axis(
        fig[3, 1], xlabel="singular value index", 
        # xgridvisible=false, ygridvisible=false,
        ylabel="singular value", yscale=log10,
        topspinevisible=false, rightspinevisible=false, ylabelpadding=30,
        limits=(-50, length(spectrum["pressure"])+10, nothing, nothing),
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25,
    )

    for (key, svals) in zip(keys(spectrum), [sp, sz, su, sv, sw])
        lines!(ax2, 1:length(svals), svals, label=key, linewidth=4, 
               color=colors[key])
    end

    # Add vertical lines for target ranks
    for (key, r) in target_r
        if r > 0
            vlines!(ax1, [r], linestyle=:dash, color=colors[key],
                    linewidth=4, label="target rank for $key")
            vlines!(ax2, [r], linestyle=:dash, color=colors[key],
                    linewidth=4, label="target rank for $key")
        end
    end

    # Add legend
    # Legend(fig[:, 2], ax1, framevisible=false, labelsize=25)
    line_elements = [
        [LineElement(color=colors[fn], linewidth=5)]
        for fn in field_names
    ]
    Legend(fig[1, :],
        line_elements,
        [L"$u$", L"$v$", L"$w$", L"$p$"],
        framevisible=false, patchsize=(70, 20),
        labelsize=40, rowgap=10, colgap=50, orientation=:horizontal,
    )
    
    # Display figure
    display(fig)
    save(joinpath(FILEPATH, "plots/energy_spectrum.pdf"), fig, px_per_inch=200)
end