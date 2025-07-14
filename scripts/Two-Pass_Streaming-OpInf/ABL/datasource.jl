using HDF5

# Proxy object to handle field indexing
struct FieldProxy
    ds::Any  # Reference to parent ChannelDataSource
    field_name::String
end

# Improved index implementation for FieldProxy
function Base.getindex(fp::FieldProxy, indices...)
    # Normalize indices: convert integers to ranges
    idx_array = map(indices) do idx
        idx isa Integer ? (idx:idx) : idx
    end
    
    # Ensure we have either 4 indices (x,y,z,time) or 2 index (all times)
    @assert (
        length(idx_array) == 4 || length(idx_array) == 2 || length(idx_array) == 1 
    ) "Expected 4 indices for field access (x,y,z,time), or 1 for each field"
    
    if length(idx_array) == 1
        # If only one index is provided, assume it is for x, y, z, or time
        # i.e., ds["x"][idx] or ds["time"][idx]
        h5f = h5open(fp.ds.hfname, "r") do f
            dset = f[fp.field_name]
            dset[indices...]    
        end 
    elseif length(idx_array) == 2
        # Pattern: ds["u"][idx, time_idx] where idx = a*Nx + b*Ny + c*Nz
        spatial_idx, time_idx = idx_array
        
        # Get dimensions
        Nz, Ny, Nx = fp.ds.dims[1:3]
        
        # Convert linear spatial index to 3D coordinates
        result_data = []
        
        h5open(fp.ds.hfname, "r") do f
            # Get field index
            field_idx = findfirst(==(fp.field_name), fp.ds.fields)
            isnothing(field_idx) && throw(ArgumentError("Field '$(fp.field_name)' not found"))
            
            dset = f["data"]
            
            # For each spatial index, convert to 3D coordinates and extract
            for s_idx in spatial_idx
                # Convert linear index to 3D coordinates (assuming column-major ordering)
                # idx = a + b*Nx + c*Nx*Ny (0-based) -> idx-1 = a-1 + (b-1)*Nx + (c-1)*Nx*Ny (1-based)
                linear_idx = s_idx - 1  # Convert to 0-based
                a = (linear_idx % Nx) + 1
                b = ((linear_idx ÷ Nx) % Ny) + 1
                c = (linear_idx ÷ (Nx * Ny)) + 1
                
                # Extract data for this spatial point across all time indices
                # HDF5 indexing: [z, y, x, field, time]
                spatial_data = dset[c:c, b:b, a:a, field_idx:field_idx, time_idx]
                
                # Reshape to remove singleton dimensions
                spatial_data = reshape(spatial_data, length(time_idx))
                push!(result_data, spatial_data)
            end
        end
        
        # Combine results
        if length(spatial_idx) == 1
            return length(time_idx) == 1 ? result_data[1][1] : result_data[1]
        else
            return hcat(result_data...)
        end
        
    else
        @assert all(maximum.(idx_array) .<= fp.ds.dims[vcat(3:-1:1,5)]) "Indices out of bounds"

        # Get field index
        field_idx = findfirst(==(fp.field_name), fp.ds.fields)
        isnothing(field_idx) && throw(ArgumentError("Field '$(fp.field_name)' not found"))
        
        # Single HDF5 read with correct indices
        h5f = h5open(fp.ds.hfname, "r") do f
            dset = f["data"]
            
            # Create HDF5 index array [z,y,x,field,time]
            h5_idx = [idx_array[3], idx_array[2], idx_array[1], field_idx, idx_array[4]]
            
            # Read data in one operation and permute
            permutedims(dset[h5_idx...], (3, 2, 1, 4))
        end
    end
    
    return h5f
end

struct ChannelDataSource
    hfname::String
    fields::Vector{String}
    dim_order::Vector{String}
    dims::Tuple{Vararg{Int}}
    n_snapshots::Int

    function ChannelDataSource(hfname::String, dim_order::Vector{String})
        # Open file and read metadata
        f = h5open(hfname, "r")
        dset = f["data"]

        fields = String.(read(f["fields"]))  # convert byte array to String array
        shape = size(dset)
        n_snapshots = shape[end]
        close(f)

        new(hfname, fields, dim_order, shape, n_snapshots)
    end
end

# Define the length method
Base.length(ds::ChannelDataSource) = ds.n_snapshots

# Main getindex that returns a FieldProxy for string keys
function Base.getindex(ds::ChannelDataSource, key::String)
    # Check if the field exists
    if !(key in ds.fields) && !(key in filter(x->x != "fields", ds.dim_order))
        throw(ArgumentError("Field '$key' not found. Available fields: $(join(ds.fields, ", "))"))
    end
    
    # Return a proxy that will handle further indexing
    return FieldProxy(ds, key)
end

# Improved ChannelDataSource getindex (including pressure field)
function Base.getindex(ds::ChannelDataSource, index...)
    # Dimensions for convenience
    Nz, Ny, Nx, Nf, Nt = ds.dims
    N = Nz * Ny * Nx
    
    if length(index) == 1 || length(index) == 2
        # Extract and normalize indices
        field_idx = length(index) == 2 ? index[1] : (1:Nf)
        time_idx = length(index) == 2 ? index[2] : index[1]
        
        # Convert to ranges if needed
        field_idx = field_idx isa Integer ? (field_idx:field_idx) : field_idx
        time_idx = time_idx isa Integer ? (time_idx:time_idx) : time_idx
        
        # Preallocate result matrix
        result = zeros(N*length(field_idx), length(time_idx))
        
        # Read data in a single operation when possible
        h5open(ds.hfname, "r") do f
            dset = f["data"]
            
            # Batch process for efficiency
            row_offset = 0
            for (f_idx, field) in enumerate(field_idx)
                # Read all requested times for this field at once
                field_data = dset[:,:,:,field,time_idx]
                
                # Reshape and store in result matrix
                for t_idx in 1:length(time_idx)
                    result[row_offset+1:row_offset+N, t_idx] = reshape(
                        permutedims(field_data[:,:,:,t_idx], (3,2,1)), 
                        :
                    )
                end
                row_offset += N
            end
        end
        
        # Return vector for single snapshots
        return length(time_idx) == 1 ? vec(result) : result
    else
        error("Invalid indexing. Expected 1 or 2 indices, got $(length(index)).")
    end
end


function scale(data::Array{Float64}, dim::Int, factors::Vector{Float64})
    @assert length(factors) == div(size(data, 1), dim) "Number of factors 
        must match number of dimensions"
    for (i, f) in enumerate(factors)
        data[dim*(i-1)+1:dim*i, :] ./= f
    end
    return data
end

function unscale(data::Array{Float64}, dim::Int, factors::Vector{Float64})
    @assert length(factors) == div(size(data, 1), dim) "Number of factors 
        must match number of dimensions"
    for (i, f) in enumerate(factors)
        data[dim*(i-1)+1:dim*i, :] .*= f
    end
    return data
end

function minmax_shift_scale!(X, X_min, X_max)
    shift = X_min / (X_max - X_min)
    scale = X_max - X_min
    X .-= X_min
    X ./= (X_max .- X_min)
    return shift, scale
end


# for _ in 1:100
#     idx = rand(1:dim_per_field)
#     tidx = rand(1:n)
#     name = rand(["u", "v", "w", "p"])
#     field_idx = findfirst(x -> x == name, ["u", "v", "w", "p"])
#     foo = ds[name][idx, tidx]
#     bar = ds[tidx][dim_per_field*(field_idx-1) + idx]
#     @assert foo == bar "Data mismatch for field $name at index $idx and time $tidx"
#     @info "Data check passed for field $name at index $idx and time $tidx"
# end

using HDF5

# Proxy object to handle field indexing
struct FieldProxy
    ds::Any  # Reference to parent ChannelDataSource
    field_name::String
end

# Enhanced FieldProxy with bounds and sampling support
function Base.getindex(fp::FieldProxy, indices...)
    # Normalize indices: convert integers to ranges
    idx_array = map(indices) do idx
        idx isa Integer ? (idx:idx) : idx
    end
    
    # Ensure we have either 4 indices (x,y,z,time) or 2 index (all times)
    @assert (
        length(idx_array) == 4 || length(idx_array) == 2 || length(idx_array) == 1 
    ) "Expected 4 indices for field access (x,y,z,time), or 1 for each field"
    
    if length(idx_array) == 1
        # If only one index is provided, assume it is for x, y, z, or time
        # i.e., ds["x"][idx] or ds["time"][idx]
        h5f = h5open(fp.ds.hfname, "r") do f
            dset = f[fp.field_name]
            # Apply time downsampling if accessing time
            if fp.field_name == "time" && !isnothing(fp.ds.time_indices)
                return dset[fp.ds.time_indices][indices...]
            else
                return dset[indices...]
            end
        end 
    elseif length(idx_array) == 2
        # Pattern: ds["u"][idx, time_idx] where idx = linear spatial index
        spatial_idx, time_idx = idx_array
        
        # Apply bounds and subsampling transformations
        spatial_idx = _apply_spatial_transform(fp.ds, spatial_idx)
        time_idx = _apply_time_transform(fp.ds, time_idx)
        
        # Get transformed dimensions
        Nz, Ny, Nx = fp.ds.dims[1:3]
        
        # Convert linear spatial index to 3D coordinates
        result_data = []
        
        h5open(fp.ds.hfname, "r") do f
            # Get field index
            field_idx = findfirst(==(fp.field_name), fp.ds.fields)
            isnothing(field_idx) && throw(ArgumentError("Field '$(fp.field_name)' not found"))
            
            dset = f["data"]
            
            # For each spatial index, convert to 3D coordinates and extract
            for s_idx in spatial_idx
                # Convert linear index to 3D coordinates (using transformed dimensions)
                linear_idx = s_idx - 1  # Convert to 0-based
                a = (linear_idx % Nx) + 1
                b = ((linear_idx ÷ Nx) % Ny) + 1
                c = (linear_idx ÷ (Nx * Ny)) + 1
                
                # Map back to original grid coordinates
                orig_a = fp.ds.x_indices[a]
                orig_b = fp.ds.y_indices[b]
                orig_c = fp.ds.z_indices[c]
                orig_time = fp.ds.time_indices[time_idx]
                
                # Extract data for this spatial point across all time indices
                # HDF5 indexing: [z, y, x, field, time]
                spatial_data = dset[orig_c:orig_c, orig_b:orig_b, orig_a:orig_a, field_idx:field_idx, orig_time]
                
                # Reshape to remove singleton dimensions
                spatial_data = reshape(spatial_data, length(time_idx))
                push!(result_data, spatial_data)
            end
        end
        
        # Combine results
        if length(spatial_idx) == 1
            return length(time_idx) == 1 ? result_data[1][1] : result_data[1]
        else
            return hcat(result_data...)
        end
        
    else
        # 4D indexing with bounds and sampling support
        x_idx, y_idx, z_idx, t_idx = idx_array
        
        # Apply transformations
        x_idx = _apply_coord_transform(fp.ds.x_indices, x_idx)
        y_idx = _apply_coord_transform(fp.ds.y_indices, y_idx)
        z_idx = _apply_coord_transform(fp.ds.z_indices, z_idx)
        t_idx = _apply_coord_transform(fp.ds.time_indices, t_idx)
        
        # Get field index
        field_idx = findfirst(==(fp.field_name), fp.ds.fields)
        isnothing(field_idx) && throw(ArgumentError("Field '$(fp.field_name)' not found"))
        
        # Single HDF5 read with correct indices
        h5f = h5open(fp.ds.hfname, "r") do f
            dset = f["data"]
            
            # Create HDF5 index array [z,y,x,field,time]
            h5_idx = [z_idx, y_idx, x_idx, field_idx, t_idx]
            
            # Read data in one operation and permute
            permutedims(dset[h5_idx...], (3, 2, 1, 4))
        end
    end
    
    return h5f
end

struct ChannelDataSource
    hfname::String
    fields::Vector{String}
    dim_order::Vector{String}
    dims::Tuple{Vararg{Int}}
    n_snapshots::Int
    
    # New fields for bounds and sampling
    x_indices::Vector{Int}
    y_indices::Vector{Int}
    z_indices::Vector{Int}
    time_indices::Vector{Int}

    function ChannelDataSource(
        hfname::String, 
        dim_order::Vector{String};
        x_bounds::Union{Nothing, Tuple{Int,Int}} = nothing,
        y_bounds::Union{Nothing, Tuple{Int,Int}} = nothing,
        z_bounds::Union{Nothing, Tuple{Int,Int}} = nothing,
        time_bounds::Union{Nothing, Tuple{Int,Int}} = nothing,
        xy_subsample::Int = 1,
        z_subsample::Int = 1,
        time_downsample::Float64 = 1.0
    )
        # Open file and read metadata
        f = h5open(hfname, "r")
        dset = f["data"]

        fields = String.(read(f["fields"]))  # convert byte array to String array
        original_shape = size(dset)
        close(f)

        # Extract original dimensions
        orig_Nz, orig_Ny, orig_Nx, orig_Nf, orig_Nt = original_shape

        # Create index vectors for bounds and subsampling
        x_indices = _create_indices(orig_Nx, x_bounds, xy_subsample)
        y_indices = _create_indices(orig_Ny, y_bounds, xy_subsample)
        z_indices = _create_indices(orig_Nz, z_bounds, z_subsample)
        time_indices = _create_time_indices(orig_Nt, time_bounds, time_downsample)

        # Calculate new dimensions
        new_shape = (length(z_indices), length(y_indices), length(x_indices), orig_Nf, length(time_indices))
        n_snapshots = length(time_indices)

        new(hfname, fields, dim_order, new_shape, n_snapshots, 
            x_indices, y_indices, z_indices, time_indices)
    end
end

# Helper functions for index creation and transformation
function _create_indices(original_size::Int, bounds::Union{Nothing, Tuple{Int,Int}}, subsample::Int)
    # Apply bounds
    if isnothing(bounds)
        range_indices = 1:original_size
    else
        start_idx, end_idx = bounds
        @assert 1 <= start_idx <= end_idx <= original_size "Invalid bounds: ($start_idx, $end_idx) for size $original_size"
        range_indices = start_idx:end_idx
    end
    
    # Apply subsampling
    return collect(range_indices[1:subsample:end])
end

function _create_time_indices(original_size::Int, bounds::Union{Nothing, Tuple{Int,Int}}, downsample::Float64)
    # Apply bounds
    if isnothing(bounds)
        range_indices = 1:original_size
    else
        start_idx, end_idx = bounds
        @assert 1 <= start_idx <= end_idx <= original_size "Invalid time bounds: ($start_idx, $end_idx) for size $original_size"
        range_indices = start_idx:end_idx
    end
    
    # Apply downsampling
    step = max(1, round(Int, 1.0 / downsample))
    return collect(range_indices[1:step:end])
end

function _apply_spatial_transform(ds::ChannelDataSource, spatial_idx)
    # This function handles the mapping from transformed linear indices to original indices
    # For now, we assume the caller provides indices in the transformed space
    return spatial_idx
end

function _apply_time_transform(ds::ChannelDataSource, time_idx)
    # Map from transformed time indices to original time indices
    return [ds.time_indices[i] for i in time_idx if i <= length(ds.time_indices)]
end

function _apply_coord_transform(index_map::Vector{Int}, coord_idx)
    # Map from transformed coordinate indices to original indices
    if coord_idx isa Integer
        return index_map[coord_idx]
    else
        return [index_map[i] for i in coord_idx if i <= length(index_map)]
    end
end

# Define the length method
Base.length(ds::ChannelDataSource) = ds.n_snapshots

# Main getindex that returns a FieldProxy for string keys
function Base.getindex(ds::ChannelDataSource, key::String)
    # Check if the field exists
    if !(key in ds.fields) && !(key in filter(x->x != "fields", ds.dim_order))
        throw(ArgumentError("Field '$key' not found. Available fields: $(join(ds.fields, ", "))"))
    end
    
    # Return a proxy that will handle further indexing
    return FieldProxy(ds, key)
end

# Enhanced ChannelDataSource getindex with bounds and sampling support
function Base.getindex(ds::ChannelDataSource, index...)
    # Dimensions for convenience (these are already transformed dimensions)
    Nz, Ny, Nx, Nf, Nt = ds.dims
    N = Nz * Ny * Nx
    
    if length(index) == 1 || length(index) == 2
        # Extract and normalize indices
        field_idx = length(index) == 2 ? index[1] : (1:Nf)
        time_idx = length(index) == 2 ? index[2] : index[1]
        
        # Convert to ranges if needed
        field_idx = field_idx isa Integer ? (field_idx:field_idx) : field_idx
        time_idx = time_idx isa Integer ? (time_idx:time_idx) : time_idx
        
        # Map time indices to original indices
        orig_time_idx = [ds.time_indices[i] for i in time_idx if i <= length(ds.time_indices)]
        
        # Preallocate result matrix
        result = zeros(N*length(field_idx), length(orig_time_idx))
        
        # Read data efficiently with bounds and sampling
        h5open(ds.hfname, "r") do f
            dset = f["data"]
            
            # Read only the required spatial region
            spatial_data = dset[ds.z_indices, ds.y_indices, ds.x_indices, field_idx, orig_time_idx]
            
            # Reshape and organize data
            row_offset = 0
            for (f_idx, field) in enumerate(field_idx)
                for t_idx in 1:length(orig_time_idx)
                    result[row_offset+1:row_offset+N, t_idx] = reshape(
                        permutedims(spatial_data[:,:,:,f_idx,t_idx], (3,2,1)), 
                        :
                    )
                end
                row_offset += N
            end
        end
        
        # Return vector for single snapshots
        return length(orig_time_idx) == 1 ? vec(result) : result
    else
        error("Invalid indexing. Expected 1 or 2 indices, got $(length(index)).")
    end
end

# ...existing code...
function scale(data::Array{Float64}, dim::Int, factors::Vector{Float64})
    @assert length(factors) == div(size(data, 1), dim) "Number of factors 
        must match number of dimensions"
    for (i, f) in enumerate(factors)
        data[dim*(i-1)+1:dim*i, :] ./= f
    end
    return data
end

function unscale(data::Array{Float64}, dim::Int, factors::Vector{Float64})
    @assert length(factors) == div(size(data, 1), dim) "Number of factors 
        must match number of dimensions"
    for (i, f) in enumerate(factors)
        data[dim*(i-1)+1:dim*i, :] .*= f
    end
    return data
end

function minmax_shift_scale!(X, X_min, X_max)
    shift = X_min / (X_max - X_min)
    scale = X_max - X_min
    X .-= X_min
    X ./= (X_max .- X_min)
    return shift, scale
end