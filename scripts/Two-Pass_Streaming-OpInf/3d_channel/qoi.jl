using Statistics 

function get_z_profile(u)
    return dropdims(mean(u, dims=(1,2)), dims=(1,2))
end

function get_utau(u, v, z, zidx=3)
    uh = sqrt.(u[:, :, zidx].^2 + v[:, :, zidx].^2)
    ubar = mean(uh)
    return solve_utau(ubar, z[zidx] + z[2] / 2)
end

function loglaw(u_tau, zf)
    kappa = 0.384
    B = 4.27
    nu = 8e-6
    return u_tau * (log.(zf * u_tau / nu) / kappa + B)
end

function lowlaw_deriv(u_tau, zf)
    kappa = 0.384
    B = 4.27
    nu = 8e-6
    return (1 + log.(zf * u_tau / nu)) / kappa + B
end

function solve_utau(u, zf)
    prev = 0.0
    curr = 0.0414
    iters = 0
    while abs(curr - prev) > 1e-12 && iters < 25
        prev = curr 
        curr = prev - (loglaw(prev, zf) - u) / lowlaw_deriv(prev, zf)
        iters += 1
    end
    if iters == 25
        @warn "Error: convergence not reached in utau computation."
    end
    return curr
end

function get_qois(ds::ChannelDataSource, tidx::Int)
    nz, ny, nx = ds.dims[1:3]
    z = ds["z"][:]
    u = @views dropdims(ds["u"][1:nx, 1:ny, 1:nz, tidx], dims=4)
    v = @views dropdims(ds["v"][1:nx, 1:ny, 1:nz, tidx], dims=4)
    return (
        zprof = get_z_profile(u),
        utau = get_utau(u, v, z)
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
    return (
        zprof = get_z_profile(u),
        utau = get_utau(u, v, z)
    )
end