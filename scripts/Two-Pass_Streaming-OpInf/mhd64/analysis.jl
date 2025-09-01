using FFTW
using Statistics
using QuadGK
using Interpolations


# ---------- helpers ----------
@inline function kvec_fftw(n::Int, dx::Real)
    # FFTW frequency ordering for fft: 0, 1, …, floor((n-1)/2), -ceil((n-1)/2), …, -1
    npos = fld(n-1, 2)
    nneg = cld(n-1, 2)
    idx  = vcat(0:npos, -nneg:-1)
    return (2π / (n*dx)) .* idx
end

# bin edges: :log (default) or :linear
function make_bins(kvals::AbstractVector{<:Real}, n_bins::Int; mode::Symbol=:log)
    kpos = kvals[kvals .> 0]
    kmin = minimum(kpos)
    kmax = maximum(kpos)
    edges = mode === :log  ?  exp.(range(log(kmin), log(kmax), length=n_bins+1)) :
            mode === :linear ?  collect(range(kmin, kmax; length=n_bins+1)) :
            error("bins mode must be :log or :linear")
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    Δk = diff(edges)
    return centers, edges, Δk
end

# ---------- core: isotropic spectrum for a 3-component real vector field ----------
"""
    energy_spectrum_isotropic_fft(u, dx, dy, dz;
                                  n_bins=min(60, size(u[1],1) ÷ 2),
                                  bins=:log,
                                  subtract_mean=true,
                                  kmax_rule=:none,   # :none or :two_thirds
                                  return_counts=false,
                                  check_energy=false)

Compute isotropic 1D energy spectrum for a **real 3D vector field**
`u = (ux,uy,uz)` of size `(nx,ny,nz)` on spacings `(dx,dy,dz)` using **full fft**.

Normalization (turbulence standard):
`sum(E .* Δk) ≈ 0.5 * mean(|u|^2)`.

Options:
- `kmax_rule=:two_thirds` applies the 2/3 de-alias cutoff.
- `return_counts=true` also returns per-bin mode counts.
- `check_energy=true` asserts the normalization.
"""
function energy_spectrum_isotropic_fft(u::NTuple{3,AbstractArray{<:Real,3}},
                                       dx::Real, dy::Real, dz::Real;
                                       n_bins::Int = min(60, size(u[1],1) ÷ 2),
                                       bins::Symbol = :log,
                                       subtract_mean::Bool = true,
                                       kmax_rule::Symbol = :none,
                                       return_counts::Bool = false,
                                       check_energy::Bool = false)

    ux, uy, uz = u
    @assert size(ux) == size(uy) == size(uz) "All components must have same size"
    nx, ny, nz = size(ux)
    Ntot = nx * ny * nz

    # subtract means (avoid huge DC)
    if subtract_mean
        ux = ux .- mean(ux);  uy = uy .- mean(uy);  uz = uz .- mean(uz)
    end

    # full complex FFT (FFTW forward is unnormalized; we divide by N to match Parseval on means)
    Ux = fft(ux) ./ Ntot
    Uy = fft(uy) ./ Ntot
    Uz = fft(uz) ./ Ntot

    # modal energy density per mode (sum over components)
    P = abs2.(Ux) .+ abs2.(Uy) .+ abs2.(Uz)   # |ûx|^2 + |ûy|^2 + |ûz|^2

    # k-grid (full, all signs)
    kx = kvec_fftw(nx, dx);  ky = kvec_fftw(ny, dy);  kz = kvec_fftw(nz, dz)
    kx2 = reshape(kx.^2, nx, 1, 1)
    ky2 = reshape(ky.^2, 1, ny, 1)
    kz2 = reshape(kz.^2, 1, 1, nz)
    kmag = sqrt.(kx2 .+ ky2 .+ kz2)

    # flatten, exclude DC, optionally apply 2/3 rule
    kf = vec(kmag); Pf = vec(P)
    mask = kf .> 0
    if kmax_rule === :two_thirds
        kny = min(π/dx, π/dy, π/dz)
        mask .&= kf .<= (2/3)*kny
    elseif kmax_rule === :none
        # nothing more
    else
        error("kmax_rule must be :none or :two_thirds")
    end
    kf = kf[mask]; Pf = Pf[mask]

    # bins
    centers, edges, Δk = make_bins(kf, n_bins; mode=bins)

    # shell sum → E(k) = 1/2 * (shell_sum / Δk)
    sums   = zeros(Float64, n_bins)
    counts = zeros(Int, n_bins)
    @inbounds for i in eachindex(kf)
        b = searchsortedfirst(edges, kf[i]) - 1
        if 1 ≤ b ≤ n_bins
            sums[b]   += Pf[i]
            counts[b] += 1
        end
    end
    E = 0.5 .* (sums ./ Δk)

    if check_energy
        Eu_total = sum(E .* Δk)
        target   = 0.5 * mean(ux.^2 .+ uy.^2 .+ uz.^2)
        @assert isapprox(Eu_total, target; rtol=1e-6) "Energy check failed: ∫E dk = $Eu_total vs 0.5⟨|u|²⟩ = $target"
    end

    return return_counts ? (centers, E, counts, Δk) : (centers, E)
end

# ---------- MHD wrapper: kinetic & magnetic spectra ----------
"""
    mhd_energy_spectra_fft(u, B, dx, dy, dz; ρ0=1.0, μ0=1.0,
                           magnetic_units=:alfven, n_bins, bins, kmax_rule)

Kinetic (per mass) and magnetic spectra. For `magnetic_units=:alfven`,
we convert `B`→`b = B/√(μ0ρ0)` so that ∫E_b dk = 0.5⟨|b|²⟩.
For `:si_volume`, we return per-volume with ∫E_b dk = ⟨|B|²/(2μ0)⟩.
"""
function mhd_energy_spectra(u::NTuple{3,AbstractArray{<:Real,3}},
                                B::NTuple{3,AbstractArray{<:Real,3}},
                                dx::Real, dy::Real, dz::Real;
                                ρ0::Real = 1.0, μ0::Real = 1.0,
                                magnetic_units::Symbol = :alfven,
                                n_bins::Int = min(60, size(u[1],1) ÷ 2),
                                bins::Symbol = :log,
                                kmax_rule::Symbol = :none)

    k, E_u = energy_spectrum_isotropic_fft(u, dx, dy, dz;
                                           n_bins=n_bins, bins=bins,
                                           kmax_rule=kmax_rule)

    if magnetic_units === :alfven
        s = 1 / sqrt(μ0 * ρ0)
        b = (B[1] .* s, B[2] .* s, B[3] .* s)
        _, E_b = energy_spectrum_isotropic_fft(b, dx, dy, dz;
                                               n_bins=n_bins, bins=bins,
                                               kmax_rule=kmax_rule)
    elseif magnetic_units === :si_volume
        # Compute on B, then scale by 1/μ0 so that ∫E_b dk = ⟨|B|²/(2μ0)⟩
        _, E_b_raw = energy_spectrum_isotropic_fft(B, dx, dy, dz;
                                                   n_bins=n_bins, bins=bins,
                                                   kmax_rule=kmax_rule)
        E_b = (1/μ0) .* E_b_raw
    else
        error("magnetic_units must be :alfven or :si_volume")
    end

    return k, E_u, E_b
end

"""
Compute Legendre polynomial P_l(x)
"""
function legendre_polynomial(l::Int, x::Float64)
    if l == 0
        return 1.0
    elseif l == 1
        return x
    else
        P_prev, P_curr = 1.0, x
        for n in 2:l
            P_next = ((2*n - 1) * x * P_curr - (n - 1) * P_prev) / n
            P_prev, P_curr = P_curr, P_next
        end
        return P_curr
    end
end

"""
Create interpolated function from NPCFs.jl angular data
"""
function create_angular_interpolator(mu_grid, zeta_values)
    # Handle edge cases and ensure proper interpolation
    valid_indices = isfinite.(zeta_values) .&& (.!isnan.(zeta_values))
    
    if sum(valid_indices) < 2
        # Not enough points for interpolation, return constant function
        mean_val = mean(zeta_values[valid_indices])
        return μ -> isnan(mean_val) ? 0.0 : mean_val
    end
    
    # Create interpolation object
    interp = linear_interpolation(
        mu_grid[valid_indices], zeta_values[valid_indices], 
        extrapolation_bc=Line())
    
    return μ -> interp(μ)
end

"""
Convert NPCFs.jl angular basis to μ values for integration
Assumes NPCFs.jl uses cosine of angle between r1 and r2
"""
function get_angular_grid(n_angular::Int)
    # NPCFs.jl typically uses Gauss-Legendre quadrature points
    # For simplicity, use uniform grid - adjust based on actual 
    # NPCFs.jl implementation
    return collect(range(-1.0, 1.0, length=n_angular))
end

"""
Project NPCFs.jl 3PCF output onto Legendre polynomial basis using QuadGK
"""
function project_to_legendre(npcf_output, max_l=5; rtol=1e-6)
    nbins = size(npcf_output, 1)
    n_angular = size(npcf_output, 3)
    
    # Get angular grid points (μ = cos θ values)
    mu_grid = get_angular_grid(n_angular)
    
    # Initialize Legendre coefficient matrices
    zeta_l = Dict{Int, Matrix{Float64}}()
    for l in 0:max_l
        zeta_l[l] = zeros(nbins, nbins)
    end
    
    # Project each (r1, r2) pair onto Legendre basis
    for i in 1:nbins, j in i:nbins
        # Extract 3PCF values for this (r1, r2) pair across angular bins
        zeta_angular = npcf_output[i, j, :]
        
        # Create interpolated function for this (r1, r2) pair
        zeta_func = create_angular_interpolator(mu_grid, zeta_angular)
        
        # Project onto each Legendre multipole using QuadGK
        for l in 0:max_l
            # Define integrand: ζ(r1, r2, μ) * P_l(μ)
            integrand(mu) = zeta_func(mu) * legendre_polynomial(l, mu)
            
            try
                # Numerical integration using QuadGK
                result, error = quadgk(integrand, -1.0, 1.0, rtol=rtol)
                
                # Apply normalization factor (2l+1)/2
                coefficient = (2*l + 1) / 2.0 * result
                zeta_l[l][i, j] = coefficient
                zeta_l[l][j, i] = coefficient  # Symmetry
                
            catch e
                @warn "Integration failed for l=$l, bins ($i,$j): $e"
                zeta_l[l][i, j] = 0.0
                zeta_l[l][j, i] = 0.0
            end
        end
    end
    
    return zeta_l
end

# Function to normalize and symmetrize the Legendre coefficient matrices
function process_legendre_matrix(zeta_l_dict, ell)
    # Extract the ell-th multipole coefficient matrix
    matrix = zeta_l_dict[ell]
    
    # Normalize by standard deviation (excluding zeros)
    non_zero_vals = matrix[matrix .!= 0]
    if length(non_zero_vals) > 0
        std_val = std(non_zero_vals)
        if std_val > 0
            matrix = matrix ./ std_val
        end
    end
    
    # Matrix should already be symmetric from projection, but ensure it
    n = size(matrix, 1)
    for i in 1:n
        for j in 1:i-1
            matrix[i, j] = matrix[j, i]
        end
    end
    
    # Rotate matrix so [0,0] is at bottom-left
    matrix = reverse(matrix, dims=1)
    
    return matrix
end

## Function to normalize and symmetrize the 3PCF matrices
function process_3pcf_matrix(matrix_3d, ell_idx)
    # Extract the ell-th multipole (3rd dimension index)
    matrix = matrix_3d[:, :, ell_idx]
    
    # Normalize by standard deviation (excluding zeros)
    non_zero_vals = matrix[matrix .!= 0]
    if length(non_zero_vals) > 0
        std_val = std(non_zero_vals)
        if std_val > 0
            matrix = matrix ./ std_val
        end
    end
    
    # Make symmetric by copying upper triangle to lower triangle
    n = size(matrix, 1)
    for i in 1:n
        for j in 1:i-1
            matrix[i, j] = matrix[j, i]
        end
    end
    
    # Rotate matrix so [0,0] is at bottom-left
    # This means we need to flip vertically
    matrix = reverse(matrix, dims=1)
    
    return matrix
end
