using HDF5

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
    MomComp::Dict{String, Int} 
    MagComp::Dict{String, Int}
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
    # Note "m" is the momentum = rho * {u, v, w}
    fields = ["rho", "z", "mx", "my", "mz", "Bx", "By", "Bz"]  
    links = Dict(
        "rho" => "t0_fields/density",
        "z"   => "t0_fields/density",
        "mx"  => "t1_fields/velocity",
        "my"  => "t1_fields/velocity",
        "mz"  => "t1_fields/velocity",
        "Bx"  => "t1_fields/magnetic_field",
        "By"  => "t1_fields/magnetic_field",
        "Bz"  => "t1_fields/magnetic_field"
    )
    pshape = size(h5[links["rho"]])  # (nx, ny, nz, nt, ntraj)
    nx, ny, nz, nt, ntraj = pshape
    dims = (nx, ny, nz, length(fields), nt, ntraj)

    # Map velocity component names to index
    MomComp = Dict("mx" => 1, "my" => 2, "mz" => 3)
    MagComp = Dict("Bx" => 1, "By" => 2, "Bz" => 3)

    close(h5)
    return DataSource(hfname, grid, fields, links, dims, MomComp, MagComp)
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
    isvel = haskey(ds.MomComp, fp.name)
    ismag = haskey(ds.MagComp, fp.name)

    return h5open(ds.hfname, "r") do h5
        dset = h5[link]
        # determine contiguous hyperslab ranges
        tmin, tmax = minimum(tidx), maximum(tidx)
        trmin, trmax = minimum(rtrj), maximum(rtrj)
        time_range = tmin:tmax
        traj_range = trmin:trmax
        # read block
        block = if isvel
            c = ds.MomComp[fp.name]
            rho = h5[ds.links["rho"]][:, :, :, time_range, traj_range]
            # Return momentum
            dset[c, :, :, :, time_range, traj_range] .* rho
        elseif ismag
            c = ds.MagComp[fp.name]
            dset[c, :, :, :, time_range, traj_range]
        else
            data = dset[:, :, :, time_range, traj_range]
            fp.name == "z" ? 1.0 ./ data : data
        end
        # flatten spatial dims
        nxyz = nx * ny * nz
        resh = reshape(block, nxyz, length(time_range) * length(traj_range))
        # # locate positions in block
        # tind = tidx .- tmin .+ 1
        # rjnd = rtrj .- trmin .+ 1
        # # gather
        # out = resh[sidx, tind, rjnd]
        # # drop singletons
        # dropdims(out)
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
function Base.getindex(ds::DataSource, time::Union{Int,AbstractVector{Int}})
    # unpack dims
    nt    = ds.dims[5]                   # snapshots per trajectory
    ntraj = ds.dims[6]                   # number of trajectories
    nxyz  = ds.dims[1] * ds.dims[2] * ds.dims[3]  # total spatial points

    # turn into a Vector of “global” indices and make 0-based
    tg = collect(time)
    t0 = tg .- 1

    # bounds check
    total = nt * ntraj
    if any(t0 .< 0) || any(t0 .>= total)
        throw(BoundsError(ds, time))
    end

    # for each global index, compute time‐within‐traj and traj        
    tidx = (t0 .% nt) .+ 1
    trj  = (t0 .÷ nt) .+ 1

    # figure out the minimal contiguous hyperslab we need to read
    tmin, tmax = minimum(tidx), maximum(tidx)
    trmin, trmax = minimum(trj),  maximum(trj)

    # read that block for *all* spatial points
    # this calls your (3)-arg getindex under the hood
    block = ds[1:nxyz, tmin:tmax, trmin:trmax]

    # now pick out exactly the columns in the order requested
    # each trajectory‐chunk has (tmax−tmin+1) columns
    ncolT = tmax - tmin + 1
    cols = ((tidx .- tmin) .+ 1) .+ ((trj .- trmin) .* ncolT)

    # select them and drop the 2nd dim if it's a singleton
    return dropdims(block[:, cols])
end

# (2) spatial + time
Base.getindex(ds::DataSource, spatial, time) = ds[time][spatial, :]


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
    return out
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


