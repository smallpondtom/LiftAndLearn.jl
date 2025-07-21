using HDF5

# Proxy object to handle field indexing
struct FieldProxy
    ds::Any  # Reference to parent ChannelDataSource
    field_name::String
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

    # Subsampling rates
    x_subsample::Int
    y_subsample::Int
    z_subsample::Int

    function ChannelDataSource(
        hfname::String, 
        dim_order::Vector{String};
        x_bounds::Union{Nothing, Tuple{Int,Int}} = nothing,
        y_bounds::Union{Nothing, Tuple{Int,Int}} = nothing,
        z_bounds::Union{Nothing, Tuple{Int,Int}} = nothing,
        time_bounds::Union{Nothing, Tuple{Int,Int}} = nothing,
        x_subsample::Int = 1,
        y_subsample::Int = 1,
        z_subsample::Int = 1,
        time_downsample::Int = 1
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
        x_indices    = _create_indices(orig_Nx, x_bounds, x_subsample)
        y_indices    = _create_indices(orig_Ny, y_bounds, y_subsample)
        z_indices    = _create_indices(orig_Nz, z_bounds, z_subsample)
        time_indices = _create_indices(orig_Nt, time_bounds, time_downsample)

        # Calculate new dimensions
        new_shape = (length(z_indices), length(y_indices), 
                     length(x_indices), orig_Nf, length(time_indices))
        n_snapshots = length(time_indices)

        new(hfname, fields, dim_order, new_shape, n_snapshots, 
            x_indices, y_indices, z_indices, time_indices,
            x_subsample, y_subsample, z_subsample)
    end
end

# Helper functions for index creation and transformation
function _create_indices(original_size::Int, 
                         bounds::Union{Nothing, Tuple{Int,Int}}, subsample::Int)
    # Apply bounds
    if isnothing(bounds)
        range_indices = 1:original_size
    else
        start_idx, end_idx = bounds
        @assert 1 <= start_idx <= end_idx <= original_size "Invalid bounds: " *
            "($start_idx, $end_idx) for size $original_size"
        range_indices = start_idx:end_idx
    end
    
    # Apply subsampling
    return collect(range_indices[1:subsample:end])
end

# Define the length method
Base.length(ds::ChannelDataSource) = ds.n_snapshots

# Main getindex that returns a FieldProxy for string keys
function Base.getindex(ds::ChannelDataSource, key::String)
    # Check if the field exists
    if !(key in ds.fields) && !(key in filter(x->x != "fields", ds.dim_order))
        throw(ArgumentError(
            "Field '$key' not found. Available fields: \
                    $(join(ds.fields, ", "))"
        ))
    end
    
    # Return a proxy that will handle further indexing
    return FieldProxy(ds, key)
end


function Base.getindex(ds::ChannelDataSource, index...)
    # Dimensions for convenience (these are already transformed dimensions)
    Nz, Ny, Nx, Nf, Nt = ds.dims
    N = Nz * Ny * Nx
    
    if length(index) == 1 || length(index) == 2
        # Extract and normalize indices
        field_idx = length(index) == 2 ? index[1] : collect(1:Nf)
        time_idx = length(index) == 2 ? index[2] : index[1]
        
        # Convert to ranges if needed
        field_idx = field_idx isa Integer ? (field_idx:field_idx) : field_idx
        time_idx = time_idx isa Integer ? (time_idx:time_idx) : time_idx
        
        # Map time indices to original indices
        orig_time_idx = [
            ds.time_indices[i] for i in time_idx 
            if i <= length(ds.time_indices)
        ]
        
        # Preallocate result matrix
        result = zeros(N*length(field_idx), length(orig_time_idx))
        
        # Read data efficiently with bounds and sampling
        h5open(ds.hfname, "r") do f
            dset = f["data"]
            
            # Convert index vectors to ranges if they are contiguous, otherwise read in chunks
            z_range = _to_range_or_indices(ds.z_indices)
            y_range = _to_range_or_indices(ds.y_indices) 
            x_range = _to_range_or_indices(ds.x_indices)
            t_range = _to_range_or_indices(orig_time_idx)
            
            # Reshape and organize data
            row_offset = 0
            for (f_idx, field) in enumerate(field_idx)
                if all(r -> r isa AbstractRange, [z_range, y_range, x_range, t_range])
                    # All indices are contiguous ranges - can read directly
                    spatial_data = dset[z_range, y_range, x_range, field, t_range]
                    
                    for t_idx in eachindex(orig_time_idx)
                        result[row_offset+1:row_offset+N, t_idx] = reshape(
                            permutedims(spatial_data[:,:,:,t_idx], (3,2,1)), 
                            :
                        )
                    end
                else
                    # Non-contiguous indices - minimize HDF5 reads
                    if length(orig_time_idx) == 1
                        # Single time step - read once and extract
                        full_data = dset[:, :, :, field, orig_time_idx[1]]
                        spatial_slice = full_data[
                            ds.z_indices, ds.y_indices, ds.x_indices]
                        result[row_offset+1:row_offset+N, 1] = reshape(
                            permutedims(spatial_slice, (3,2,1)), :
                        )
                    else
                        # Multiple time steps - try to read contiguous time blocks
                        for (t_idx, orig_t) in enumerate(orig_time_idx)
                            full_data = dset[:, :, :, field, orig_t]
                            spatial_slice = full_data[
                                ds.z_indices, ds.y_indices, ds.x_indices]
                            result[row_offset+1:row_offset+N, t_idx] = reshape(
                                permutedims(spatial_slice, (3,2,1)), :
                            )
                        end
                    end
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

# Helper function to convert index vectors to ranges when possible
function _to_range_or_indices(indices::Vector{Int})
    if length(indices) <= 1
        return indices[1]:indices[1]
    end
    
    # Check if indices form a contiguous range with step size
    step_size = length(indices) > 1 ? indices[2] - indices[1] : 1
    if all(indices[i] == indices[1] + (i-1)*step_size for i in 1:length(indices))
        # Create a range with the detected step size
        return indices[1]:step_size:indices[end]
    else
        return indices
    end
end


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
                spatial_data = dset[
                    orig_c:orig_c, orig_b:orig_b, 
                    orig_a:orig_a, field_idx:field_idx, orig_time
                ]
                
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
