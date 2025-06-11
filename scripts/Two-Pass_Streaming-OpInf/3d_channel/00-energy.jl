"""
3D Channel flow: Compute the energy spectrum
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
dPdx = 0.001722
scale_factors = [sqrt(dPdx), sqrt(dPdx), sqrt(dPdx), dPdx]

#========================================#
## Compute the mean velocity for shifting
#========================================#
SHIFT_MEAN = false
xbar = nothing # preallocate
if SHIFT_MEAN
    mean_file = joinpath(FILEPATH, "data/streaming/mean.jld2")
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

#================================#
## Two-level Block-wise SVD with left_ssm and right_ssm merging
#================================#
@info "Computing complete energy spectrum using multi-process, multi-threaded block SVD..."

using Distributed

# Add worker processes if not already added (one for each velocity component)
velocity_components = ["u", "v", "w"]  # Exclude pressure
if nprocs() < length(velocity_components) + 1
    n_workers_needed = length(velocity_components) - (nprocs() - 1)
    if n_workers_needed > 0
        addprocs(n_workers_needed)
        @info "Added $n_workers_needed worker processes for velocity components"
    end
end

# Load required packages on all workers
@everywhere using LinearAlgebra
@everywhere using HDF5
@everywhere using FileIO
@everywhere using JLD2

# Load utility functions on all workers
@everywhere include(joinpath(@__DIR__, "datasource.jl"))
@everywhere include(joinpath(@__DIR__, "utils.jl"))

# Parameters for complete energy spectrum computation
Nrow = 64      # Number of row blocks per field
Ncol = 25      # Number of column blocks per row block
target_rank = length(ds)    # Maximum possible rank (time dimension)
intermediate_rank = length(ds)  # Rank after column merging
final_rank = length(ds)     # Final rank (complete spectrum)

# Field dimensions
velocity_field_names = velocity_components  # Only u, v, w (exclude pressure)
N = dim_per_field
total_cols = length(ds)

@info "Complete Energy Spectrum SVD parameters:"
@info "  Velocity fields: $velocity_field_names"
@info "  Row blocks per field: $Nrow, Column blocks per row: $Ncol"
@info "  Target rank per block: $target_rank"
@info "  Intermediate rank per row: $intermediate_rank" 
@info "  Final rank: $final_rank"
@info "  Field dimension: $N"
@info "  Time dimension: $total_cols"
@info "  Complete spectrum will have $(min(N, total_cols)) singular values per field"

# Function to compute SVD for a single field using multi-threaded block processing
@everywhere function compute_field_complete_svd(
    field_name::String, field_idx::Int, datafile::String, 
    N::Int, total_cols::Int, Nrow::Int, Ncol::Int,
    target_rank::Int, intermediate_rank::Int, final_rank::Int,
    xbar, dPdx::Float64, velocity_field_names::Vector{String})
    
    @info "Worker $(myid()): Computing complete SVD for field $field_name"
    
    # Create datasource on worker
    ds_worker = ChannelDataSource(datafile, ["z", "y", "x", "fields", "times"])
    
    # Apply appropriate scaling
    scale_factor = field_name != "p" ? sqrt(dPdx) : dPdx
    
    # Storage for row block SVDs
    row_block_svds = Vector{NamedTuple}(undef, Nrow)
    
    # Process each row block
    for row_block in 1:Nrow
        @info "Worker $(myid()): Processing row block $row_block of $Nrow for field $field_name"
        
        # Calculate row range for this block
        row_block_size = ceil(Int, N / Nrow)
        row_start = (row_block - 1) * row_block_size + 1
        row_end = min(row_block * row_block_size, N)
        row_range = row_start:row_end
        
        # Initialize for column merging within this row block
        row_merged_U = Matrix{Float64}(undef, 0, 0)
        row_merged_S = Float64[]
        row_merged_Vt = Matrix{Float64}(undef, 0, 0)
        is_first_col_block = true
        
        # Process column blocks and store results for merging
        col_block_results = Vector{Union{Nothing, NamedTuple}}(undef, Ncol)
        
        Threads.@threads for col_block in 1:Ncol
            try
                @info "Worker $(myid()): Processing column block $col_block of $Ncol for row block $row_block, field $field_name"
                
                # Calculate column range for this block
                col_block_size = ceil(Int, total_cols / Ncol)
                col_start = (col_block - 1) * col_block_size + 1
                col_end = min(col_block * col_block_size, total_cols)
                col_range = col_start:col_end
                
                # Extract block data for this field
                field_data = ds_worker[field_idx, col_range][row_range, :]
                
                # Apply scaling
                field_data .*= scale_factor
                
                # Subtract mean if applicable
                if xbar !== nothing && xbar != 0.0
                    # Calculate global indices for this field
                    field_global_start = (findfirst(==(field_name), velocity_field_names) - 1) * N + 1
                    global_indices = field_global_start .+ (row_range .- 1)
                    mean_block = xbar[global_indices]
                    field_data .-= mean_block
                end
                
                # Compute SVD with rank truncation
                F = svd(field_data)
                k = min(target_rank, length(F.S), size(F.U, 2), size(F.Vt, 1))
                
                # Store result (need U for horizontal merging with left_ssm!)
                col_block_results[col_block] = (U=F.U[:, 1:k], S=F.S[1:k], Vt=F.Vt[1:k, :])
                
                # Clear memory
                field_data = nothing
                F = nothing
                GC.gc()
                
            catch e
                @error "Worker $(myid()): Failed to process row block $row_block, col block $col_block for field $field_name: $e"
                col_block_results[col_block] = nothing
            end
        end
          # Now merge the column blocks using direct SVD of [U1*S1, U2*S2, ..., U_m*S_m]
        # Collect all valid column block results
        valid_col_blocks = filter(x -> x !== nothing, col_block_results)
        
        if !isempty(valid_col_blocks)
            @info "Worker $(myid()): Merging $(length(valid_col_blocks)) column blocks for row block $row_block"
            
            # Create the horizontally concatenated matrix [U1*S1, U2*S2, ..., U_m*S_m]
            col_matrices = []
            for block_svd in valid_col_blocks
                # Compute U*S for this block
                US_block = block_svd.U * Diagonal(block_svd.S)
                push!(col_matrices, US_block)
            end
            
            # Horizontally concatenate all U*S matrices
            merged_matrix = hcat(col_matrices...)
            
            # Take SVD of the merged matrix
            try
                F_merged = svd(merged_matrix)
                k = min(intermediate_rank, length(F_merged.S), size(F_merged.U, 2), size(F_merged.Vt, 1))
                
                # Store the result (we need the V^T for vertical merging later)
                row_merged_S = F_merged.S[1:k]
                row_merged_Vt = F_merged.Vt[1:k, :]
                
                @info "Worker $(myid()): Column merge successful for row block $row_block. Merged: $(length(row_merged_S)) singular values"
                
            catch merge_error
                @error "Worker $(myid()): Column merge failed for row block $row_block, field $field_name: $merge_error"
                # Create empty result
                row_merged_S = Float64[]
                row_merged_Vt = Matrix{Float64}(undef, 0, 0)
            end
        else
            # No valid column blocks
            @warn "Worker $(myid()): No valid column blocks for row block $row_block, field $field_name"
            row_merged_S = Float64[]
            row_merged_Vt = Matrix{Float64}(undef, 0, 0)
        end
        
        # Store the merged result for this row block (drop U to save memory)
        row_block_svds[row_block] = (S=row_merged_S, Vt=row_merged_Vt)
        @info "Worker $(myid()): Completed row block $row_block for field $field_name with $(length(row_merged_S)) singular values"
          # Clear row block data
        row_merged_S = nothing
        row_merged_Vt = nothing
        col_block_results = nothing
        GC.gc()
    end
    
    @info "Worker $(myid()): Completed all row blocks for field $field_name. Now merging row blocks using direct SVD..."
    
    # Now merge all row blocks using direct SVD of [S1*V1^T; S2*V2^T; ...; S_m*V_m^T]
    # Collect all valid row block results
    valid_row_blocks = filter(x -> length(x.S) > 0, row_block_svds)
    
    if !isempty(valid_row_blocks)
        @info "Worker $(myid()): Merging $(length(valid_row_blocks)) row blocks for field $field_name"
        
        # Create the vertically concatenated matrix [S1*V1^T; S2*V2^T; ...; S_m*V_m^T]
        row_matrices = []
        for row_svd in valid_row_blocks
            # Compute S*V^T for this row block
            SVt_block = Diagonal(row_svd.S) * row_svd.Vt
            push!(row_matrices, SVt_block)
        end
        
        # Vertically concatenate all S*V^T matrices
        merged_matrix = vcat(row_matrices...)
        
        # Take SVD of the merged matrix
        try
            F_merged = svd(merged_matrix)
            k = min(final_rank, length(F_merged.S), size(F_merged.U, 2), size(F_merged.Vt, 1))
            
            # Store final result
            final_S = F_merged.S[1:k]
            final_Vt = F_merged.Vt[1:k, :]
            
            @info "Worker $(myid()): Row merge successful for field $field_name. Final: $(length(final_S)) singular values"
            
        catch merge_error
            @error "Worker $(myid()): Row merge failed for field $field_name: $merge_error"
            # Create empty result
            final_S = Float64[]
            final_Vt = Matrix{Float64}(undef, 0, 0)
        end
    else
        # No valid row blocks
        @warn "Worker $(myid()): No valid row blocks for field $field_name"
        final_S = Float64[]
        final_Vt = Matrix{Float64}(undef, 0, 0)
    end
    
    @info "Worker $(myid()): Completed field $field_name with $(length(final_S)) singular values"
    @info "Worker $(myid()): Field $field_name spectral decay - first 10 σ: $(final_S[1:min(10, length(final_S))])"
    
    return (field_name=field_name, S=final_S, Vt=final_Vt)
end

## Main multi-process field-wise SVD computation
@time begin
    # Create tasks for each velocity field (run in separate processes)
    field_tasks = []
    
    for (field_idx, field_name) in enumerate(velocity_field_names)
        # Find the actual field index in the dataset
        actual_field_idx = findfirst(==(field_name), ds.fields)
        
        task = @spawnat :any compute_field_complete_svd(
            field_name, actual_field_idx, datafile, N, total_cols, 
            Nrow, Ncol, target_rank, intermediate_rank, final_rank,
            xbar, dPdx, velocity_field_names
        )
        push!(field_tasks, task)
        @info "Created task for field $field_name (index $actual_field_idx) on worker process"
    end
    
    # Wait for all field tasks to complete and collect results
    @info "Waiting for all field SVD computations to complete..."
    field_results = [fetch(task) for task in field_tasks]
    
    @info "All field SVD computations completed!"
    for result in field_results
        @info "Field $(result.field_name): $(length(result.S)) singular values"
        @info "  First 10 σ: $(result.S[1:min(10, length(result.S))])"
    end
end

## Final global merging of all velocity fields
@info "Merging all velocity field results into global energy spectrum..."

@time begin
    # Now merge all velocity field results vertically using direct SVD
    # Since we're merging different velocity components vertically, we use direct SVD
    
    # Collect all valid field results
    valid_field_results = filter(x -> length(x.S) > 0, field_results)
    
    if !isempty(valid_field_results)
        @info "Merging $(length(valid_field_results)) velocity fields into global energy spectrum..."
        
        # Create the vertically concatenated matrix [S1*V1^T; S2*V2^T; ...; S_m*V_m^T]
        field_matrices = []
        for field_result in valid_field_results
            @info "Adding field $(field_result.field_name) with $(length(field_result.S)) singular values"
            # Compute S*V^T for this field
            SVt_block = Diagonal(field_result.S) * field_result.Vt
            push!(field_matrices, SVt_block)
        end
        
        # Vertically concatenate all S*V^T matrices
        @info "Creating global merged matrix..."
        global_merged_matrix = vcat(field_matrices...)
        @info "Global merged matrix size: $(size(global_merged_matrix))"
        
        # Take SVD of the merged matrix
        try
            @info "Computing SVD of global merged matrix..."
            F_global = svd(global_merged_matrix)
            k = min(final_rank, length(F_global.S), size(F_global.U, 2), size(F_global.Vt, 1))
            
            # Store final global result
            global_merged_S = F_global.S[1:k]
            global_merged_Vt = F_global.Vt[1:k, :]
            
            @info "Global merge successful! Final: $(length(global_merged_S)) singular values"
            
        catch merge_error
            @error "Global merge failed: $merge_error"
            # Create empty result as fallback
            global_merged_S = Float64[]
            global_merged_Vt = Matrix{Float64}(undef, 0, 0)
        end
    else
        @warn "No valid field results to merge"
        global_merged_S = Float64[]
        global_merged_Vt = Matrix{Float64}(undef, 0, 0)
    end
    
    @info "Completed global merging of all velocity fields!"
    @info "Global complete energy spectrum: $(length(global_merged_S)) singular values"
    @info "Global spectral decay - first 20 σ: $(global_merged_S[1:min(20, length(global_merged_S))])"
    @info "Global spectral decay - last 20 σ: $(global_merged_S[max(1, end-19):end])"
end

# Store the complete energy spectrum results
complete_energy_spectrum = global_merged_S[1:min(final_rank, length(global_merged_S))]

## Save the complete energy spectrum for analysis
save(joinpath(FILEPATH, "data/streaming/complete_energy_spectrum.jld2"), 
     "singular_values", complete_energy_spectrum,
     "total_computed", length(global_merged_S),
     "field_results", [(field=r.field_name, n_singular_values=length(r.S), 
                       first_10_sv=r.S[1:min(10, length(r.S))]) for r in field_results],
     "parameters", Dict("velocity_fields" => velocity_field_names,
                       "Nrow" => Nrow, "Ncol" => Ncol, "target_rank" => target_rank, 
                       "intermediate_rank" => intermediate_rank, "final_rank" => final_rank,
                       "field_dimension" => N, "time_dimension" => total_cols))

@info "Complete energy spectrum computation complete."
@info "Computed $(length(global_merged_S)) singular values, saved $(length(complete_energy_spectrum)) for analysis"
@info "Expected theoretical maximum: $(min(N * length(velocity_field_names), total_cols)) singular values"

## Create visualization comparing approaches
using CairoMakie 
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1000, 800))

    # Plot the complete energy spectrum
    ax1 = Axis(fig[1,1], xlabel="Singular Value Index", ylabel="Singular Value", 
            title="Complete Energy Spectrum for 3D Channel Flow",
            titlesize=30, xlabelsize=25, ylabelsize=25,
            xticklabelsize=20, yticklabelsize=20,
        )

    scatterlines!(ax1, 
        1:length(complete_energy_spectrum), 
        cumsum(complete_energy_spectrum.^2) ./ sum(complete_energy_spectrum.^2), 
        markersize=20, linewidth=10, color=:red, 
        label=L"Cumulative Energy Retainment $(u,v,w)$"
    )

    # Add individual field results if available
    if length(field_results) > 0
        colors = [:blue, :green, :orange]
        for (i, result) in enumerate(field_results)
            if i <= length(colors)
                scatterlines!(ax1, 
                    1:length(result.S), 
                    cumsum(result.S .^ 2) ./ sum(result.S .^ 2),
                    markersize=15, linewidth=6, color=colors[i], alpha=0.6, 
                    label=L"Field %$(result.field_name) $$")
            end
        end
    end

    # Add vertical lines 
    hlines!(ax1, [0.75, 0.85, 0.95], color=:black, linestyle=:dash, linewidth=2)
    vlines!(ax1, [100, 200, 500], color=:black, linestyle=:dash, linewidth=2)
    axislegend(ax1, position=:rb, labelsize=25)
    # display(fig)

    ## Save the comparison plot
    save(joinpath(FILEPATH, "plots/complete_energy_spectrum_comparison.png"), fig)
end

# ## Also create a summary plot of key statistics
# fig2 = Figure(size=(1000, 600))
# ax3 = Axis(fig2[1,1], xlabel="Field", ylabel="Number of Computed Singular Values", 
#          title="Energy Spectrum Statistics by Field")

# field_names_plot = [r.field_name for r in field_results]
# n_sv_per_field = [length(r.S) for r in field_results]

# barplot!(ax3, 1:length(field_names_plot), n_sv_per_field, 
#         color=[:blue, :green, :orange][1:length(field_names_plot)])

# ax3.xticks = (1:length(field_names_plot), field_names_plot)

# # Add text annotations
# for (i, n) in enumerate(n_sv_per_field)
#     text!(ax3, i, n + maximum(n_sv_per_field)*0.02, text="$n", align=(:center, :bottom))
# end

# display(fig2)
# save(joinpath(FILEPATH, "plots/energy_spectrum_field_stats.png"), fig2)

# @info "Complete energy spectrum analysis complete"
# @info "Theoretical maximum singular values per field: $(min(N, total_cols))"
# @info "Actual computed singular values: $(length(complete_energy_spectrum))"
# @info "Field-wise breakdown:"
# for result in field_results
#     @info "  $(result.field_name): $(length(result.S)) singular values"
# end