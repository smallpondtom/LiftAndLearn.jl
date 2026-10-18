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

    elseif length(index) == 5
        # 5D indexing with bounds and sampling support
        x_idx, y_idx, z_idx, f_idx, t_idx = index
        
        # Single HDF5 read with correct indices
        h5f = h5open(ds.hfname, "r") do f
            dset = f["data"]
            
            # Create HDF5 index array [z,y,x,field,time]
            h5_idx = [z_idx, y_idx, x_idx, f_idx, t_idx]
            
            # Read data in one operation and permute
            dset[h5_idx...]
        end
        
        return h5f
    else
        error("Invalid indexing. Expected 1 or 2 indices, got $(length(index)).")
    end
end


# function scale(data::Array{Float64}, dim::Int, factors::Vector{Float64})
#     @assert length(factors) == div(size(data, 1), dim) "Number of factors 
#         must match number of dimensions"
#     for (i, f) in enumerate(factors)
#         data[dim*(i-1)+1:dim*i, :] ./= f
#     end
#     return data
# end

# function unscale(data::Array{Float64}, dim::Int, factors::Vector{Float64})
#     @assert length(factors) == div(size(data, 1), dim) "Number of factors 
#         must match number of dimensions"
#     for (i, f) in enumerate(factors)
#         data[dim*(i-1)+1:dim*i, :] .*= f
#     end
#     return data
# end

# function minmax_shift_scale!(X, X_min, X_max)
#     shift = X_min / (X_max - X_min)
#     scale = X_max - X_min
#     X .-= X_min
#     X ./= (X_max .- X_min)
#     return shift, scale
# end


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


# # Improved ChannelDataSource getindex (ignoring pressure field)
# function Base.getindex(ds::ChannelDataSource, index...)
#     # Dimensions for convenience
#     Nz, Ny, Nx, Nf, Nt = ds.dims
#     N = Nz * Ny * Nx
    
#     # Find and exclude pressure field index
#     p_idx = findfirst(==("p"), ds.fields)
#     velocity_fields = isnothing(p_idx) ? (1:Nf) : [i for i in 1:Nf if i != p_idx]
    
#     if length(index) == 1 || length(index) == 2
#         # Extract and normalize indices
#         field_idx = length(index) == 2 ? index[1] : velocity_fields
#         time_idx = length(index) == 2 ? index[2] : index[1]
        
#         # If specific fields are requested, honor that request
#         # Otherwise use our filtered velocity fields
#         if length(index) == 2 && !(field_idx isa Integer) && field_idx != (1:Nf)
#             # User specified specific fields, keep as is
#         else
#             # Use only velocity fields (filter out pressure)
#             field_idx = field_idx isa Integer ? 
#                         (field_idx <= p_idx ? field_idx : field_idx-1) : 
#                         velocity_fields
#         end
        
#         # Convert to ranges if needed
#         field_idx = field_idx isa Integer ? (field_idx:field_idx) : field_idx
#         time_idx = time_idx isa Integer ? (time_idx:time_idx) : time_idx
        
#         # Preallocate result matrix
#         result = zeros(N*length(field_idx), length(time_idx))
        
#         # Read data in a single operation when possible
#         h5open(ds.hfname, "r") do f
#             dset = f["data"]
            
#             # Batch process for efficiency
#             row_offset = 0
#             for (f_idx, field) in enumerate(field_idx)
#                 # Read all requested times for this field at once
#                 field_data = dset[:,:,:,field,time_idx]
                
#                 # Reshape and store in result matrix
#                 for t_idx in 1:length(time_idx)
#                     result[row_offset+1:row_offset+N, t_idx] = reshape(
#                         permutedims(field_data[:,:,:,t_idx], (3,2,1)), 
#                         :
#                     )
#                 end
#                 row_offset += N
#             end
#         end
        
#         # Return vector for single snapshots
#         return length(time_idx) == 1 ? vec(result) : result
#     else
#         error("Invalid indexing. Expected 1 or 2 indices, got $(length(index)).")
#     end
# end
