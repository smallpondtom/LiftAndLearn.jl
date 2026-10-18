using Statistics 

function get_wind_speed_profile(u, v)
    u_mean = dropdims(mean(u, dims=(1,2)), dims=(1,2))
    v_mean = dropdims(mean(v, dims=(1,2)), dims=(1,2))
    wprof = sqrt.(u_mean.^2 + v_mean.^2)
    wang = atand.(v_mean, u_mean)
    return wprof, wang
end

function get_qois(ds::ChannelDataSource, tidx::Int)
    nz, ny, nx = ds.dims[1:3]
    z = ds["z"][:]
    u = @views dropdims(ds["u"][1:nx, 1:ny, 1:nz, tidx], dims=4)
    v = @views dropdims(ds["v"][1:nx, 1:ny, 1:nz, tidx], dims=4)
    wprof, wang = get_wind_speed_profile(u, v)
    return (
        wprof = wprof,
        wang = wang
    )
end

function get_qois_rom(Xhat::Matrix{T}, V::Matrix{T}, z::Vector{T}, 
                      means::Vector{T}, shifts::Vector{T}, scales::Vector{T},
                      dims::Tuple, tidx::Int) where {T<:Real}
    xhat = Xhat[:, tidx]
    x = V * xhat 
    x = unprocess!(x, means, shifts, scales)
    nz, ny, nx = dims[1:3]
    nxyz = nx * ny * nz
    u_ = @view x[1:nxyz]
    v_ = @view x[nxyz+1:2*nxyz]
    u = reshape(u_, nx, ny, nz)
    v = reshape(v_, nx, ny, nz)
    wprof, wang = get_wind_speed_profile(u, v)
    return (
        wprof = wprof,
        wang = wang
    )
end

