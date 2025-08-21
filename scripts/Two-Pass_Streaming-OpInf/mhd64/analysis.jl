using FFTW
using Statistics

# ---------- helpers ----------
@inline function kvec_fftw(n::Int, dx::Real)
    # FFTW output order for complex fft along a dimension:
    # 0, 1, 2, ..., floor((n-1)/2), -ceil((n-1)/2), ..., -1
    npos = fld(n-1, 2)
    nneg = cld(n-1, 2)
    idx  = vcat(0:npos, -nneg:-1)
    return (2π / (n*dx)) .* idx
end

@inline function kz_rfft(n::Int, dz::Real)
    # nonnegative frequencies only, for rfft along the last dim
    return (2π / (n*dz)) .* collect(0:fld(n,2))
end

@inline function rfft_weight_lastdim(n::Int)
    # Weight to account for the omitted negative-kz half in rfft.
    # modes with kz>0 (and kz≠Nyquist) represent conjugate pairs -> weight 2
    nzr = fld(n,2) + 1
    w = ones(Float64, nzr)
    if iseven(n)
        if nzr >= 3
            w[2:end-1] .= 2.0   # exclude kz=0 and kz=Nyquist
        end
    else
        if nzr >= 2
            w[2:end]   .= 2.0   # exclude only kz=0
        end
    end
    return w
end

# bin edges: :log (default) or :linear
function make_bins(kvals::AbstractVector{<:Real}, n_bins::Int; mode::Symbol=:log)
    kmin = minimum(kvals[kvals .> 0])
    kmax = maximum(kvals)
    if mode === :log
        edges = exp.(range(log(kmin), log(kmax), length=n_bins+1))
    elseif mode === :linear
        edges = range(kmin, kmax; length=n_bins+1) |> collect
    else
        error("bins mode must be :log or :linear")
    end
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    Δk = diff(edges)
    return centers, edges, Δk
end

# ---------- core: isotropic spectrum for a 3-component real vector field ----------
"""
    energy_spectrum_isotropic_rfft(u, dx, dy, dz; n_bins, bins=:log,
                                   subtract_mean=true, weight_lastdim=true)

Compute isotropic 1D energy spectrum for a real 3D **vector** field `u = (ux,uy,uz)`,
each component size `(nx,ny,nz)` with spacings `(dx,dy,dz)`.

**Normalization (turbulence standard):**
Returns `(k, E(k))` such that `sum(E .* Δk) ≈ 0.5 * mean(|u|^2)`.

Notes:
- Uses `rfft` (half-spectrum along z) and correct conjugate-pair weighting.
- Means of each component are removed if `subtract_mean=true`.
"""
function energy_spectrum_isotropic_rfft(u::NTuple{3,AbstractArray{<:Real,3}},
                                        dx::Real, dy::Real, dz::Real;
                                        n_bins::Int = min(60, size(u[1],1) ÷ 2),
                                        bins::Symbol = :log,
                                        subtract_mean::Bool = true,
                                        weight_lastdim::Bool = true)

    ux, uy, uz = u
    @assert size(ux) == size(uy) == size(uz) "All components must have same size"
    nx, ny, nz = size(ux)
    Ntot = nx * ny * nz

    # subtract means (avoid huge DC)
    if subtract_mean
        ux = ux .- mean(ux);  uy = uy .- mean(uy);  uz = uz .- mean(uz)
    end

    # rfft on the full 3D cube (FFTW reduces the last dim)
    Ux = rfft(ux) ./ Ntot
    Uy = rfft(uy) ./ Ntot
    Uz = rfft(uz) ./ Ntot

    # modal energy density per mode (sum over components)
    P = abs2.(Ux) .+ abs2.(Uy) .+ abs2.(Uz)   # |ûx|^2 + |ûy|^2 + |ûz|^2

    # account for omitted negative kz half
    if weight_lastdim
        wz = rfft_weight_lastdim(nz)
        P .*= reshape(wz, length(wz), 1, 1)
    end

    # k-grid (kx, ky full; kz nonnegative only)
    kx = kvec_fftw(nx, dx);  ky = kvec_fftw(ny, dy);  kz = kz_rfft(nz, dz)
    kx2 = reshape(kx.^2, nx, 1, 1)
    ky2 = reshape(ky.^2, 1, ny, 1)
    kz2 = reshape(kz.^2, 1, 1, length(kz))
    kmag = sqrt.(kx2 .+ ky2 .+ kz2)

    # flatten, exclude DC (k=0)
    kf = vec(kmag); Pf = vec(P)
    mask = kf .> 0
    kf = kf[mask]; Pf = Pf[mask]

    # bins
    centers, edges, Δk = make_bins(kf, n_bins; mode=bins)

    # shell-sum and convert to E(k) = (1/2) * (shell_sum / Δk)
    sums = zeros(Float64, n_bins)
    @inbounds for i in eachindex(kf)
        b = searchsortedfirst(edges, kf[i]) - 1
        if 1 ≤ b ≤ n_bins
            sums[b] += Pf[i]
        end
    end

    E = 0.5 .* (sums ./ Δk)
    return centers, E
end

# ---------- MHD wrapper: kinetic & magnetic spectra ----------
"""
    mhd_energy_spectra(u, B, dx, dy, dz; ρ0=1.0, μ0=1.0,
                       magnetic_units=:alfven, n_bins, bins)

Compute isotropic 1D **kinetic** and **magnetic** energy spectra for MHD.

Inputs:
- `u = (ux,uy,uz)` velocity components.
- `B = (Bx,By,Bz)` magnetic field.
- spacings `dx,dy,dz`.

Keyword options:
- `ρ0` (reference density) and `μ0` (permeability, SI).
- `magnetic_units`:
    - `:alfven` (default): convert `B` → `b = B/√(μ0 ρ0)` (energy per mass).
      Returns `E_b` with ∫E_b dk = 0.5 ⟨|b|^2⟩.
    - `:si_volume`: energy per **volume**; returns `E_b` with
      ∫E_b dk = ⟨|B|^2/(2μ0)⟩.
- `n_bins`, `bins = :log | :linear`.

Return:
`k, E_u, E_b`.
"""
function mhd_energy_spectra(u::NTuple{3,AbstractArray{<:Real,3}},
                            B::NTuple{3,AbstractArray{<:Real,3}},
                            dx::Real, dy::Real, dz::Real;
                            ρ0::Real = 1.0, μ0::Real = 1.0,
                            magnetic_units::Symbol = :alfven,
                            n_bins::Int = min(60, size(u[1],1) ÷ 2),
                            bins::Symbol = :log)

    # kinetic spectrum (per mass): ∫E_u dk = 0.5⟨|u|^2⟩
    k, E_u = energy_spectrum_isotropic_rfft(u, dx, dy, dz;
                                            n_bins=n_bins, bins=bins)

    # magnetic spectrum
    if magnetic_units === :alfven
        # convert B to Alfvén velocity b = B / sqrt(μ0 ρ0)
        s = 1 / sqrt(μ0 * ρ0)
        b = (B[1] .* s, B[2] .* s, B[3] .* s)
        _, E_b = energy_spectrum_isotropic_rfft(b, dx, dy, dz;
                                                n_bins=n_bins, bins=bins)
    elseif magnetic_units === :si_volume
        # start with per-mass form using b, then scale by ρ0 to get per-volume,
        # and multiply by (μ0 ρ0) to undo the b-conversion (net factor 1/μ0).
        # Easier: compute directly from B and scale the final spectrum by 1/(2μ0):
        # We want ∫E_b dk = ⟨|B|^2/(2μ0)⟩.
        # energy_spectrum_isotropic_rfft returns shell_sum/Δk for the "field" squared/2.
        # If we feed it B and then scale by 1/μ0, we get the right integral:
        _, E_b_raw = energy_spectrum_isotropic_rfft(B, dx, dy, dz;
                                                    n_bins=n_bins, bins=bins)
        E_b = (1/μ0) .* E_b_raw
    else
        error("magnetic_units must be :alfven or :si_volume")
    end

    return k, E_u, E_b
end
