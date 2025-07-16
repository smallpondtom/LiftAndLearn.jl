# using HDF5

# """
# DataSource
# """
# struct DataSource
#     hfname::String
#     fields::Vector{String}
#     fieldlinks::Dict{String, String}
#     field_order::Vector{String}
#     dims::Tuple{Vararg{Int}}
#     n_time::Int
#     n_trajectories::Int
#     n_snapshots::Int
#     grid::Dict{String, Vector{<:Real}}

#     function DataSource(hfname::String, field_order::Vector{String})
#         # Open file and read metadata
#         h5 = h5open(hfname, "r")

#         grid = Dict{String, Vector{<:Real}}()
#         xspan = read_dataset(h5["dimensions"], "x")
#         yspan = read_dataset(h5["dimensions"], "y")
#         zspan = read_dataset(h5["dimensions"], "z")
#         tspan = read_dataset(h5["dimensions"], "time")
#         grid["x"] = xspan
#         grid["y"] = yspan
#         grid["z"] = zspan
#         grid["time"] = tspan

#         fields = ["p", "z", "u", "v", "w"]
#         fieldlinks = Dict(
#             "p" => "t0_fields/pressure",
#             "z" => "t0_fields/density",  # specific volume is 1/density
#             "u" => "t1_fields/velocity",
#             "v" => "t1_fields/velocity",
#             "w" => "t1_fields/velocity"
#         )
#         velocity_fieldlinks = Dict(
#             1 => "u",
#             2 => "v",
#             3 => "w"
#         )
#         shape = size(h5["t0_fields"]["pressure"])
#         nx = shape[1]
#         ny = shape[2]
#         nz = shape[3]
#         n_fields = length(fields)
#         n_time = shape[end-1]
#         n_trajectories = shape[end]
#         shape = (nz, ny, nx, n_fields, n_time, n_trajectories)
#         n_snapshots = Int(n_time * n_trajectories)
#         close(h5)

#         new(hfname, fields, fieldlinks, dim_order, shape, n_time,
#             n_trajectories, n_snapshots, grid)
#     end
# end

# # Define the length method
# Base.length(ds::DataSource) = ds.n_snapshots 

# # Main getindex that returns a FieldProxy for string keys
# function Base.getindex(ds::DataSource, key::String)
#     # Check if the field exists
#     if !(key in ds.fields) 
#         throw(ArgumentError("Field '$key' not found. Available fields: \
#                                                 $(join(ds.fields, ", "))"))
#     end
    
#     # Return a proxy that will handle further indexing
#     return FieldProxy(ds, key)
# end


# function linear2tensor(idx::Int, dims::Tuple{Int, Int, Int, Int})
#     nx, ny, nz, nf = dims
#     # Convert to 0-based index for easier calculation
#     idx = linear_idx - 1
#     # Calculate dimensions
#     spatial_size = nx * ny * nz
#     # Extract field index
#     f = (idx ÷ spatial_size) + 1
#     @assert f <= nf "Field index out of bounds: $f > $nf"
#     # Extract spatial index within the field
#     spatial_idx = idx % spatial_size
#     # Convert linear index to 3D tensor coordinates
#     a = (spatial_idx - 1) % nz + 1
#     b = ((spatial_idx - 1) ÷ nz) % ny + 1
#     c = (spatial_idx - 1) ÷ (nz * ny) + 1
#     return (a, b, c, f)
# end


# # Improved DataSource getindex (including pressure field)
# function Base.getindex(ds::DataSource, index...)
#     # Dimensions for convenience
#     nx, ny, nz, nf, nt, ntraj = ds.dims
#     n = nz * ny * nx
    
#     if length(index) == 1
#         # Extract and normalize indices
#         time_idx = index[1]
        
#         # Convert to ranges if needed
#         time_idx = time_idx isa Integer ? (time_idx:time_idx) : time_idx
        
#         # Preallocate result matrix
#         result = zeros(n*nf, length(time_idx))
        
#         # Read data in a single operation when possible
#         h5open(ds.hfname, "r") do h5
#             # Batch process for efficiency
#             row_offset = 0
#             for field in ds.fields
#                 # Read all requested times for this field at once
#                 field_data = h5[ds.fieldlinks[field]]
#                 if field == "z"
#                     field_data = 1 ./ field_data  # Invert density for specific volume
#                 end
#                 # Reshape and store in result matrix
#                 for t_idx in 1:length(time_idx)
#                     if field == "u" || field == "v" || field == "w"
#                         # For velocity fields, we need to reshape to 3D
#                         n = nz * ny * nx
#                         result[row_offset+1:row_offset+n, t_idx] = reshape(
#                             field_data[velocity_fieldlinks[field],:,:,:,t_idx,:], n
#                         )
#                     else
#                         # For scalar fields (pressure, density), reshape directly
#                         result[row_offset+1:row_offset+n, t_idx] = reshape(
#                             field_data[:,:,:,t_idx,:], :
#                         )
#                     end
#                 end
#                 row_offset += n
#             end
#         end
#         # Return vector for single snapshots
#         return length(time_idx) == 1 ? vec(result) : result

#     elseif length(index) == 2
#         spatial_idx, time_idx = index
#         spatial_idx = spatial_idx isa Integer ? (spatial_idx:spatial_idx) : spatial_idx
#         time_idx = time_idx isa Integer ? (time_idx:time_idx) : time_idx
#         result = zeros(length(spatial_idx), length(time_idx))
#         h5open(ds.hfname, "r") do h5
#             for s_idx in spatial_idx
#                 # Convert linear index to 3D coordinates
#                 a, b, c, f = linear2tensor(s_idx, (nx, ny, nz, nf))
                
#                 # Extract data for this spatial point across all time indices
#                 field_data = h5[ds.fieldlinks[ds.fields[f]]]
#                 if ds.fields[f] == "z"
#                     field_data = 1 ./ field_data  # Invert density for specific volume
#                 end
#                 if ds.fields[f] in ["u", "v", "w"]
#                     # For velocity fields, we need to reshape to 3D
#                     spatial_data = field_data[velocity_fieldlinks[field], 
#                                               a:a, b:b, c:c, time_idx, :]
#                 else
#                     # For scalar fields (pressure, density), reshape directly
#                     spatial_data = field_data[a:a, b:b, c:c, time_idx, :]
#                 end
                
#                 # Reshape to remove singleton dimensions
#                 spatial_data = reshape(spatial_data, length(time_idx))
#                 result[s_idx - (spatial_idx[1] - 1), :] = spatial_data
#             end
#         end
#         return result
#     elseif length(index) == 3
#         spatial_idx, time_idx, traj_idx = index
#         spatial_idx = spatial_idx isa Integer ? (spatial_idx:spatial_idx) : spatial_idx
#         time_idx = time_idx isa Integer ? (time_idx:time_idx) : time_idx
#         traj_idx = traj_idx isa Integer ? (traj_idx:traj_idx) : traj_idx
        
#         result = zeros(length(spatial_idx), length(time_idx)*length(traj_idx))
        
#         h5open(ds.hfname, "r") do h5
#             for s_idx in spatial_idx
#                 # Convert linear index to 3D coordinates
#                 a, b, c, f = linear2tensor(s_idx, (nx, ny, nz, nf))
                
#                 # Extract data for this spatial point and field across all time indices
#                 field_data = h5[ds.fieldlinks[ds.fields[f]]]
#                 if ds.fields[f] == "z"
#                     field_data = 1 ./ field_data  # Invert density for specific volume
#                 end
#                 if ds.fields[f] in ["u", "v", "w"]
#                     # For velocity fields, we need to reshape to 3D
#                     spatial_data = field_data[velocity_fieldlinks[f], 
#                                               a:a, b:b, c:c, time_idx, traj_idx]
#                 else
#                     # For scalar fields (pressure, density), reshape directly
#                     spatial_data = field_data[a:a, b:b, c:c, time_idx, traj_idx]
#                 end
                
#                 # Reshape to remove singleton dimensions
#                 spatial_data = reshape(spatial_data, length(time_idx)*length(traj_idx))
#                 result[s_idx - (spatial_idx[1] - 1), :] = spatial_data
#             end
#         end
#         return result
#     elseif length(index) == 4
#         spatial_idx, time_idx, traj_idx, field_idx = index
#         h5open(ds.hfname, "r") do h5
#             # Convert linear index to 3D coordinates
#             a, b, c, _ = linear2tensor(s_idx, (nx, ny, nz, nf))
                
#             # Extract data for this spatial point and field across all time indices
#             field_data = h5[ds.fieldlinks[ds.fields[field_idx]]]
#             if ds.fields[field_idx] == "z"
#                 field_data = 1 ./ field_data  # Invert density for specific volume
#             end
#             if ds.fields[field_idx] in ["u", "v", "w"]
#                 # For velocity fields, we need to reshape to 3D
#                 spatial_data = field_data[velocity_fieldlinks[field_idx], 
#                                           a:a, b:b, c:c, time_idx, traj_idx]
#             else
#                 # For scalar fields (pressure, density), reshape directly
#                 spatial_data = field_data[a, b, c, time_idx, traj_idx]
#             end
#         end        
#         return spatial_data
#     else
#         error("Invalid indexing. Expected 1 or 2 indices, got $(length(index)).")
#     end
# end

# # Proxy object to handle field indexing
# struct FieldProxy
#     ds::Any  # Reference to parent DataSource
#     field_name::String
# end

# ## Adjust this function
# function Base.getindex(fp::FieldProxy, indices...)
#     # Normalize indices: convert integers to ranges
#     idx_array = map(indices) do idx
#         idx isa Integer ? (idx:idx) : idx
#     end
    
#     # Ensure we have either 4 indices (x,y,z,time) or 2 index (all times)
#     @assert (
#         length(idx_array) == 5 || length(idx_array) == 2 || length(idx_array) == 1 
#     ) "Expected 4 indices for field access (x,y,z,time), or 1 for each field"
    
#     if length(idx_array) == 1
#         # If only one index is provided, assume it is for x, y, z, or time
#         # i.e., ds["x"][idx] or ds["time"][idx]
#         h5f = h5open(fp.ds.hfname, "r") do f
#             dset = f[fp.field_name]
#             dset[indices...]    
#         end 
#     elseif length(idx_array) == 2
#         # Pattern: ds["u"][idx, time_idx] where idx = a*Nx + b*Ny + c*Nz
#         spatial_idx, time_idx = idx_array
        
#         # Get dimensions
#         Nz, Ny, Nx = fp.ds.dims[1:3]
        
#         # Convert linear spatial index to 3D coordinates
#         result_data = []
        
#         h5open(fp.ds.hfname, "r") do f
#             # Get field index
#             field_idx = findfirst(==(fp.field_name), fp.ds.fields)
#             isnothing(field_idx) && throw(ArgumentError("Field '$(fp.field_name)' not found"))
            
#             dset = f["data"]
            
#             # For each spatial index, convert to 3D coordinates and extract
#             for s_idx in spatial_idx
#                 # Convert linear index to 3D coordinates (assuming column-major ordering)
#                 # idx = a + b*Nx + c*Nx*Ny (0-based) -> idx-1 = a-1 + (b-1)*Nx + (c-1)*Nx*Ny (1-based)
#                 linear_idx = s_idx - 1  # Convert to 0-based
#                 a = (linear_idx % Nx) + 1
#                 b = ((linear_idx ÷ Nx) % Ny) + 1
#                 c = (linear_idx ÷ (Nx * Ny)) + 1
                
#                 # Extract data for this spatial point across all time indices
#                 # HDF5 indexing: [z, y, x, field, time]
#                 spatial_data = dset[c:c, b:b, a:a, field_idx:field_idx, time_idx]
                
#                 # Reshape to remove singleton dimensions
#                 spatial_data = reshape(spatial_data, length(time_idx))
#                 push!(result_data, spatial_data)
#             end
#         end
        
#         # Combine results
#         if length(spatial_idx) == 1
#             return length(time_idx) == 1 ? result_data[1][1] : result_data[1]
#         else
#             return hcat(result_data...)
#         end
        
#     else
#         @assert all(maximum.(idx_array) .<= fp.ds.dims[vcat(3:-1:1,5)]) "Indices out of bounds"

#         # Get field index
#         field_idx = findfirst(==(fp.field_name), fp.ds.fields)
#         isnothing(field_idx) && throw(ArgumentError("Field '$(fp.field_name)' not found"))
        
#         # Single HDF5 read with correct indices
#         h5f = h5open(fp.ds.hfname, "r") do f
#             dset = f["data"]
            
#             # Create HDF5 index array [z,y,x,field,time]
#             h5_idx = [idx_array[3], idx_array[2], idx_array[1], field_idx, idx_array[4]]
            
#             # Read data in one operation and permute
#             permutedims(dset[h5_idx...], (3, 2, 1, 4))
#         end
#     end
    
#     return h5f
# end

using HDF5

# Map velocity component names to index
VelComp = Dict("u" => 1, "v" => 2, "w" => 3)

"""
Lightweight descriptor for an HDF5-based supernova dataset.
Supports lazy slicing of individual fields via FieldProxy,
as well as full all-fields snapshots via DataSource indexing.
"""
struct DataSource
    hfname::String
    grid::Dict{String, Vector{Float64}}
    fields::Vector{String}
    links::Dict{String, String}
    dims::NTuple{6, Int}  # (nx, ny, nz, nfields, nt, ntraj)
end

"""
Constructor: reads grid vectors, defines fields, infers dimensions.
"""
function DataSource(hfname::String)
    h5 = h5open(hfname, "r")
    grid = Dict(
        "x" => read(h5["dimensions/x"]),
        "y" => read(h5["dimensions/y"]),
        "z" => read(h5["dimensions/z"]),
        "time" => read(h5["dimensions/time"])
    )
    fields = ["p", "z", "u", "v", "w"]
    links = Dict(
        "p" => "t0_fields/pressure",
        "z" => "t0_fields/density",
        "u" => "t1_fields/velocity",
        "v" => "t1_fields/velocity",
        "w" => "t1_fields/velocity"
    )
    pshape = size(h5[links["p"]])  # (nx, ny, nz, nt, ntraj)
    nx, ny, nz, nt, ntraj = pshape
    dims = (nx, ny, nz, length(fields), nt, ntraj)
    close(h5)
    return DataSource(hfname, grid, fields, links, dims)
end

# total snapshots
Base.length(ds::DataSource) = ds.dims[5] * ds.dims[6]

# Proxy type for individual field slicing
struct FieldProxy
    ds::DataSource
    name::String
end
# allow ds["p"] → FieldProxy
Base.getindex(ds::DataSource, fld::String) =
    fld in ds.fields ? FieldProxy(ds, fld) :
    throw(ArgumentError("Unknown field '$fld'; choose from: $(join(ds.fields, ", "))"))

"""
Convert 1-based linear index → (ix, iy, iz).
"""
function linear_to_sub(idx::Int, nx::Int, ny::Int, nz::Int)
    lin = idx - 1
    ix   = (lin ÷ (ny * nz)) + 1
    rem1 = lin % (ny * nz)
    iy   = (rem1 ÷ nz) + 1
    iz   = (rem1 % nz) + 1
    return ix, iy, iz
end

"""
Drop any singleton dimensions from an array.
Calls Base.dropdims for multi-arg dispatch.
"""
function dropdims(arr)
    dims = findall(x->x==1, size(arr))
    isempty(dims) ? arr : dropdims(arr, dims...)
end

"""
Optimized FieldProxy indexing:
  fp[spatial, time] or fp[spatial, time, traj]
Supports Int or vector-like for indices, vectorized hyperslab.
"""
function Base.getindex(fp::FieldProxy, spatial, time, traj=1)
    ds = fp.ds
    nx, ny, nz = ds.dims[1:3]
    # normalize indices to vectors
    sidx = collect(spatial)
    tidx = collect(time)
    rtrj = collect(traj)
    link = ds.links[fp.name]
    isvel = haskey(VelComp, fp.name)

    return h5open(ds.hfname, "r") do h5
        dset = h5[link]
        # determine contiguous hyperslab ranges
        tmin, tmax = minimum(tidx), maximum(tidx)
        trmin, trmax = minimum(rtrj), maximum(rtrj)
        time_range = tmin:tmax
        traj_range = trmin:trmax
        # read block
        block = if isvel
            c = VelComp[fp.name]
            dset[c, :, :, :, time_range, traj_range]
        else
            data = dset[:, :, :, time_range, traj_range]
            fp.name == "z" ? 1.0 ./ data : data
        end
        # flatten spatial dims
        nxyz = nx * ny * nz
        resh = reshape(block, nxyz, length(time_range), length(traj_range))
        # locate positions in block
        tind = tidx .- tmin .+ 1
        rjnd = rtrj .- trmin .+ 1
        # gather
        out = resh[sidx, tind, rjnd]
        # drop singletons
        dropdims(out)
    end
end

function Base.getindex(fp::FieldProxy, index...)
    # If no trajectory is specified, default to the first one
    return h5open(fp.ds.hfname, "r") do h5
        dset = h5[fp.ds.links[fp.name]]
        dset[index...]
    end
end

"""
DataSource indexing:
  ds[time]
  ds[spatial, time]
  ds[spatial, time, traj]
  ds[ix, iy, iz, time, traj]
  ds[ix, iy, iz, time, traj, field]
"""
# (1) time-only
function Base.getindex(ds::DataSource, time::Union{Int, AbstractVector{Int}})
    allsp = 1:(ds.dims[1]*ds.dims[2]*ds.dims[3])
    return ds[allsp, time, 1:ds.dims[6]]
end
# (2) spatial + time
Base.getindex(ds::DataSource, spatial, time) = ds[spatial, time, 1:ds.dims[6]]
# (3) spatial + time + traj
function Base.getindex(ds::DataSource, spatial, time, traj)
    nx, ny, nz, nf, nt, ntraj = ds.dims
    sidx = collect(spatial)
    tidx = collect(time)
    rtrj = collect(traj)
    nsp  = length(sidx)
    ncol = length(tidx)*length(rtrj)
    out  = zeros(Float64, nsp*nf, ncol)
    for (i,fld) in enumerate(ds.fields)
        fp = FieldProxy(ds,fld)
        blk = fp[sidx, tidx, rtrj] # nsp×#time×#traj
        B = reshape(blk, nsp, ncol)
        rows = ((i-1)*nsp+1):(i*nsp)
        out[rows,:] .= B
    end
    out
end
# (4) ix,iy,iz + time + traj
function Base.getindex(ds::DataSource, ix, iy, iz, time, traj)
    sid = (ix-1)*ds.dims[2]*ds.dims[3] + (iy-1)*ds.dims[3] + iz
    ds[sid, time, traj]
end
# (5) ix,iy,iz + time + traj + field
function Base.getindex(ds::DataSource, ix::Int, iy::Int, iz::Int, time, traj, field)
    fname = field isa Int ? ds.fields[field] : field
    FieldProxy(ds,fname)[(ix-1)*ds.dims[2]*ds.dims[3]+(iy-1)*ds.dims[3]+iz, time, traj]
end
