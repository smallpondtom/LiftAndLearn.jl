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
    
    # Ensure we have either 4 indices (x,y,z,time) or 1 index (all times)
    @assert (
        length(idx_array) == 4 || length(idx_array) == 1
    ) "Expected 4 indices for field access (x,y,z,time), or 1 for each field"
    
    if length(idx_array) == 1
        # If only one index is provided, assume it is for x, y, z, or time
        h5f = h5open(fp.ds.hfname, "r") do f
            dset = f[fp.field_name]
            dset[indices...]    
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

# Improved ChannelDataSource getindex (ignoring pressure field)
function Base.getindex(ds::ChannelDataSource, index...)
    # Dimensions for convenience
    Nz, Ny, Nx, Nf, Nt = ds.dims
    N = Nz * Ny * Nx
    
    # Find and exclude pressure field index
    p_idx = findfirst(==("p"), ds.fields)
    velocity_fields = isnothing(p_idx) ? (1:Nf) : [i for i in 1:Nf if i != p_idx]
    
    if length(index) == 1 || length(index) == 2
        # Extract and normalize indices
        field_idx = length(index) == 2 ? index[1] : velocity_fields
        time_idx = length(index) == 2 ? index[2] : index[1]
        
        # If specific fields are requested, honor that request
        # Otherwise use our filtered velocity fields
        if length(index) == 2 && !(field_idx isa Integer) && field_idx != (1:Nf)
            # User specified specific fields, keep as is
        else
            # Use only velocity fields (filter out pressure)
            field_idx = field_idx isa Integer ? 
                        (field_idx <= p_idx ? field_idx : field_idx-1) : 
                        velocity_fields
        end
        
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

# # Improved ChannelDataSource getindex (including pressure field)
# function Base.getindex(ds::ChannelDataSource, index...)
#     # Dimensions for convenience
#     Nz, Ny, Nx, Nf, Nt = ds.dims
#     N = Nz * Ny * Nx
    
#     if length(index) == 1 || length(index) == 2
#         # Extract and normalize indices
#         field_idx = length(index) == 2 ? index[1] : (1:Nf)
#         time_idx = length(index) == 2 ? index[2] : index[1]
        
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

# using HDF5

# # Proxy object to handle field indexing
# struct FieldProxy
#     ds::Any  # Reference to parent ChannelDataSource
#     field_name::String
# end

# # Index implementation for FieldProxy
# function Base.getindex(fp::FieldProxy, indices...)
#     # Convert indices to a modifiable array
#     indices = collect(Any, indices)
    
#     # Now modify the array instead of the tuple
#     for (j, ind) in enumerate(indices)
#         if typeof(ind) <: Integer
#             indices[j] = ind:ind
#         end
#     end

#     # Indices should be in the order of 
#     # [x, y, z, time]
#     @assert length(indices) == 4 "Expected 4 indices for field access"
#     @assert all(maximum.(indices) .<= fp.ds.dims[vcat(3:-1:1,5)]) "Indices out of bounds"

#     f = h5open(fp.ds.hfname, "r")
#     dset = f["data"]
    
#     # Get field index based on field name
#     field_idx = findfirst(==(fp.field_name), fp.ds.fields)
#     if isnothing(field_idx)
#         close(f)
#         throw(ArgumentError("Field '$(fp.field_name)' not found in dataset"))
#     end
    
#     # Construct the proper indexing for the HDF5 dataset
#     # Order is typically [z, y, x, field, time]
#     idx = Any[Colon() for _ in 1:length(fp.ds.dims)]
#     idx[4] = field_idx  # Set field dimension
    
#     # Apply any user-provided indices
#     idx[3] = indices[1]  # Set x dimension
#     idx[2] = indices[2]  # Set y dimension
#     idx[1] = indices[3]  # Set z dimension
#     idx[end] = indices[end]  # Set time dimension
    
#     # Read and return the data
#     data = dset[idx...]
#     data = permutedims(data, (3, 2, 1, 4))  # Reorder dimensions to [x, y, z]
#     close(f)
#     return data
# end

# struct ChannelDataSource
#     hfname::String
#     fields::Vector{String}
#     dim_order::Vector{String}
#     dims::Tuple{Vararg{Int}}
#     n_snapshots::Int

#     function ChannelDataSource(hfname::String)
#         # Open file and read metadata
#         f = h5open(hfname, "r")
#         dset = f["data"]

#         fields = String.(read(f["fields"]))  # convert byte array to String array
#         shape = size(dset)
#         n_snapshots = shape[end]
#         close(f)

#         new(hfname, fields, ["z", "y", "x", "fields", "times"], shape, n_snapshots)
#     end
# end

# # Define the length method
# Base.length(ds::ChannelDataSource) = ds.n_snapshots

# # Main getindex that returns a FieldProxy for string keys
# function Base.getindex(ds::ChannelDataSource, key::String)
#     # Check if the field exists
#     if !(key in ds.fields)
#         throw(ArgumentError("Field '$key' not found. Available fields: $(join(ds.fields, ", "))"))
#     end
    
#     # Return a proxy that will handle further indexing
#     return FieldProxy(ds, key)
# end

# # Index as snapshot (column vector)
# function Base.getindex(ds::ChannelDataSource, index...)
#     f = h5open(ds.hfname, "r")
#     dset = f["data"]
    
#     # Dimensions 
#     Nz, Ny, Nx = ds.dims[1:3]
#     N = Nz * Ny * Nx

#     if length(index) == 1
#         idx = index[1]
#         if typeof(idx) <: Integer
#             idx = idx:idx
#             data = Vector{Float64}(undef, N*ds.dims[4])  # preallocate snapshot
#         else
#             data = Matrix{Float64}(undef, N*ds.dims[4], length(idx))  # preallocate snapshot
#         end
#         tmp = 0
#         for i in 1:4
#             for j in 1:length(idx)
#                 t = idx[j]  # Get actual time index
#                 data[tmp+1:tmp+N,j] = reshape(
#                     permutedims(dset[:,:,:,i:i,t:t], (3, 2, 1, 4, 5)),
#                     :, 1
#                 )
#             end
#             tmp += N
#         end
#     elseif length(index) == 2
#         field_index = index[1]
#         tindex = index[2]
        
#         # Convert integer indices to ranges
#         if typeof(field_index) <: Integer
#             field_index = field_index:field_index
#         end
#         if typeof(tindex) <: Integer
#             tindex = tindex:tindex
#             data = Vector{Float64}(undef, N*length(field_index))  # preallocate snapshot
#         else
#             data = Matrix{Float64}(undef, N*length(field_index), length(tindex))  # preallocate snapshot
#         end
        
#         tmp = 0
#         for i in field_index
#             for j in 1:length(tindex)
#                 t = tindex[j]  # Get actual time index
#                 # Read the data for the specific field and time index
#                 data[tmp+1:tmp+N,j] = reshape(
#                     permutedims(dset[:,:,:,i:i,t:t], (3, 2, 1, 4, 5)),
#                     :, 1
#                 )
#             end
#             tmp += N
#         end
#     else
#         error("Invalid number of indices. Expected 1 or 2, got $(length(index)).")
#     end

#     close(f)
#     return data
# end