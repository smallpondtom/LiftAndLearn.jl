"""
    Kuramoto-Sivashinsky equation PDE model
"""
module KuramotoSivashinsky

using DocStringExtensions
using FFTW
using LinearAlgebra
using SparseArrays

import ..LiftAndLearn: AbstractModel, vech, ⊘, Operators, elimat

export KuramotoSivashinskyModel


"""
$(TYPEDEF)

Kuramoto-Sivashinsky equation PDE model
    
```math
\\frac{\\partial u}{\\partial t} = -\\mu\\frac{\\partial^4 u}{\\partial x^4} - \\frac{\\partial^2 u}{\\partial x^2} - u\\frac{\\partial u}{\\partial x}
```

where ``u`` is the state variable and ``\\mu`` is the viscosity coefficient.

## Fields
- `spatial_domain::Tuple{Real,Real}`: spatial domain
- `time_domain::Tuple{Real,Real}`: temporal domain
- `param_domain::Tuple{Real,Real}`: parameter domain
- `Δx::Real`: spatial grid size
- `Δt::Real`: temporal step size
- `BC::Symbol`: boundary condition
- `IC::Array{Float64}`: initial condition
- `xspan::Vector{Float64}`: spatial grid points
- `tspan::Vector{Float64}`: temporal points
- `diffusion_coeffs::Union{Real,AbstractArray{<:Real}}`: parameter vector
- `fourier_modes::Vector{Float64}`: Fourier modes
- `spatial_dim::Int64`: spatial dimension
- `time_dim::Int64`: temporal dimension
- `param_dim::Int64`: parameter dimension
- `conservation_type::Symbol`: conservation type
- `model_type::Symbol`: model type
- `finite_diff_model::Function`: finite difference model
- `pseudo_spectral_model::Function`: pseudo spectral model
- `elementwise_pseudo_spectral_model::Function`: element-wise pseudo spectral model
- `spectral_galerkin_model::Function`: spectral Galerkin model
- `integrate_model::Function`: integrator
- `jacobian::Function`: Jacobian matrix
"""
mutable struct KuramotoSivashinskyModel <: AbstractModel
    # Domains
    spatial_domain::Tuple{Real,Real}  # spatial domain
    time_domain::Tuple{Real,Real}  # temporal domain
    param_domain::Tuple{Real,Real}  # parameter domain

    # Discritization grid
    Δx::Real  # spatial grid size
    Δt::Real  # temporal step size

    # Boundary condition
    BC::Symbol  # boundary condition

    # Initial conditino
    IC::Array{Float64}  # initial condition

    # grid points
    xspan::Vector{Float64}  # spatial grid points
    tspan::Vector{Float64}  # temporal points
    diffusion_coeffs::Union{Real,AbstractArray{<:Real}}  # parameter vector

    # Fourier
    fourier_modes::Vector{Float64}  # Fourier modes

    # Dimensions
    spatial_dim::Int64  # spatial dimension
    time_dim::Int64  # temporal dimension
    param_dim::Int64  # parameter dimension

    # Convervation type
    conservation_type::Symbol

    # Model type
    model_type::Symbol

    finite_diff_model::Function
    pseudo_spectral_model::Function
    elementwise_pseudo_spectral_model::Function
    spectral_galerkin_model::Function
    integrate_model::Function
    jacobian::Function
end



"""
$(SIGNATURES)

Constructor for the Kuramoto-Sivashinsky equation model.
"""
function KuramotoSivashinskyModel(;spatial_domain::Tuple{Real,Real}, time_domain::Tuple{Real,Real}, Δx::Real, Δt::Real, 
                       diffusion_coeffs::Union{Real,AbstractArray{<:Real}}, BC::Symbol=:periodic,
                       conservation_type::Symbol=:NC, model_type::Symbol=:FD)
    # Discritization grid info
    @assert BC ∈ (:periodic, :dirichlet, :neumann, :mixed, :robin, :cauchy, :flux) "Invalid boundary condition"
    if BC == :periodic
        xspan = collect(spatial_domain[1]:Δx:spatial_domain[2]-Δx)
    elseif BC ∈ (:dirichlet, :neumann, :mixed, :robin, :cauchy) 
        xspan = collect(spatial_domain[1]:Δx:spatial_domain[2])[2:end-1]
    end
    tspan = collect(time_domain[1]:Δt:time_domain[2])
    spatial_dim = length(xspan)
    time_dim = length(tspan)

    # Initial condition
    IC = zeros(spatial_dim)

    # Parameter dimensions or number of parameters 
    param_dim = length(diffusion_coeffs)
    param_domain = extrema(diffusion_coeffs)

    # Fourier modes
    fourier_modes = collect(-spatial_dim/2:1.0:spatial_dim/2-1)

    @assert conservation_type ∈ (:EP, :NC, :C) "Invalid conservation type"
    @assert model_type ∈ (:FD, :PS, :EWPS, :SG) "Invalid model type"
    integrate_model = integrate_finite_diff_model

    KuramotoSivashinskyModel(
        spatial_domain, time_domain, param_domain,
        Δx, Δt, BC, IC, xspan, tspan, diffusion_coeffs, fourier_modes,
        spatial_dim, time_dim, param_dim, conservation_type, model_type,
        finite_diff_model, pseudo_spectral_model, elementwise_pseudo_spectral_model, 
        spectral_galerkin_model, integrate_model, jacobian
    )
end



function finite_diff_model(model::KuramotoSivashinskyModel, μ::Real)
    if model.BC == :periodic
        if model.conservation_type == :NC
            return finite_diff_periodic_nonconservative_model(model.spatial_dim, model.Δx, μ)
        elseif model.conservation_type == :C
            return finite_diff_periodic_conservative_model(model.spatial_dim, model.Δx, μ)
        elseif model.conservation_type == :EP
            return finite_diff_periodic_energy_preserving_model(model.spatial_dim, model.Δx, μ)
        else
            error("Conservation type not implemented")
        end
    else
        error("Boundary condition not implemented")
    end
end


function finite_diff_periodic_nonconservative_model(N::Real, Δx::Real, μ::Real)
    # Create A matrix
    ζ = 2/Δx^2 - 6*μ/Δx^4
    η = 4*μ/Δx^4 - 1/Δx^2
    ϵ = -μ/Δx^4

    A = spdiagm(
        0 => ζ * ones(N), 
        1 => η * ones(N - 1), -1 => η * ones(N - 1),
        2 => ϵ * ones(N - 2), -2 => ϵ * ones(N - 2)
    )
    # For the periodicity for the first and final few indices
    A[1, end-1:end] = [ϵ, η]
    A[2, end] = ϵ
    A[end-1, 1] = ϵ
    A[end, 1:2] = [η, ϵ]
    
    # Create F matrix
    S = Int(N * (N + 1) / 2)
    if N >= 3
        Fval = repeat([1.0, -1.0], outer=N - 2)
        row_i = repeat(2:(N-1), inner=2)
        seq = Int.([2 + (N + 1) * (x - 1) - x * (x - 1) / 2 for x in 1:(N-1)])
        col_i = vcat(seq[1], repeat(seq[2:end-1], inner=2), seq[end])
        F = sparse(row_i, col_i, Fval, N, S) / 2 / Δx

        # For the periodicity for the first and final indices
        F[1, 2] = - 1 / 2 / Δx
        F[1, N] = 1 / 2 / Δx
        F[N, N] = - 1 / 2 / Δx
        F[N, end-1] = 1 / 2 / Δx 
    else
        F = zeros(N, S)
    end
    return A, sparse(F)
end



function finite_diff_periodic_conservative_model(N::Real, Δx::Real, μ::Real)
    # Create A matrix
    ζ = 2/Δx^2 - 6*μ/Δx^4
    η = 4*μ/Δx^4 - 1/Δx^2
    ϵ = -μ/Δx^4

    A = spdiagm(
        0 => ζ * ones(N), 
        1 => η * ones(N - 1), -1 => η * ones(N - 1),
        2 => ϵ * ones(N - 2), -2 => ϵ * ones(N - 2)
    )
    # For the periodicity for the first and final few indices
    A[1, end-1:end] = [ϵ, η]
    A[2, end] = ϵ
    A[end-1, 1] = ϵ
    A[end, 1:2] = [η, ϵ]
    
    # Create F matrix
    S = Int(N * (N + 1) / 2)
    if N >= 3
        ii = repeat(2:(N-1), inner=2)
        m = 2:N-1
        mm = Int.([N*(N+1)/2 - (N-m).*(N-m+1)/2 - (N-m) - (N-(m-2)) for m in 2:N-1])  # this is where the x_{i-1}^2 term is
        mp = Int.([N*(N+1)/2 - (N-m).*(N-m+1)/2 - (N-m) + (N-(m-1)) for m in 2:N-1])  # this is where the x_{i+1}^2 term is
        jj = reshape([mp'; mm'],2*N-4);
        vv = reshape([-ones(1,N-2); ones(1,N-2)],2*N-4)/(4*Δx);
        F = sparse(ii,jj,vv,N,S)

        # Boundary conditions (Periodic)
        F[1,N+1] = -1/4/Δx
        F[1,end] = 1/4/Δx
        F[N,end-2] = 1/4/Δx
        F[N,1] = -1/4/Δx
    else
        F = zeros(N, S)
    end
    return A, sparse(F)
end


function finite_diff_periodic_energy_preserving_model(N::Real, Δx::Real, μ::Real)
    # Create A matrix
    ζ = 2/Δx^2 - 6*μ/Δx^4
    η = 4*μ/Δx^4 - 1/Δx^2
    ϵ = -μ/Δx^4

    A = spdiagm(
        0 => ζ * ones(N), 
        1 => η * ones(N - 1), -1 => η * ones(N - 1),
        2 => ϵ * ones(N - 2), -2 => ϵ * ones(N - 2)
    )
    # For the periodicity for the first and final few indices
    A[1, end-1:end] = [ϵ, η]
    A[2, end] = ϵ
    A[end-1, 1] = ϵ
    A[end, 1:2] = [η, ϵ]
    
    # Create F matrix
    S = Int(N * (N + 1) / 2)
    if N >= 3
        ii = repeat(2:(N-1), inner=4)
        m = 2:N-1
        mi = Int.([N*(N+1)/2 - (N-m)*(N-m+1)/2 - (N-m) for m in 2:N-1])               # this is where the xi^2 term is
        mm = Int.([N*(N+1)/2 - (N-m).*(N-m+1)/2 - (N-m) - (N-(m-2)) for m in 2:N-1])  # this is where the x_{i-1}^2 term is
        mp = Int.([N*(N+1)/2 - (N-m).*(N-m+1)/2 - (N-m) + (N-(m-1)) for m in 2:N-1])  # this is where the x_{i+1}^2 term is
        jp = mi .+ 1  # this is the index of the x_{i+1}*x_i term
        jm = mm .+ 1  # this is the index of the x_{i-1}*x_i term
        jj = reshape([mp'; mm'; jp'; jm'],4*N-8);
        vv = reshape([-ones(1,N-2); ones(1,N-2); -ones(1,N-2); ones(1,N-2)],4*N-8)/(6*Δx);
        F = sparse(ii,jj,vv,N,S)

        # Boundary conditions (Periodic)
        F[1,2] = -1/6/Δx
        F[1,N+1] = -1/6/Δx
        F[1,N] = 1/6/Δx
        F[1,end] = 1/6/Δx
        F[N,end-1] = 1/6/Δx
        F[N,end-2] = 1/6/Δx
        F[N,1] = -1/6/Δx
        F[N,N] = -1/6/Δx
    else
        F = zeros(N, S)
    end
    return A, sparse(F)
end



"""
    pseudo_spectral_model(model, μ) → A, F

Generate A, F matrices for the Kuramoto-Sivashinsky equation using the Pseudo-Spectral/Fast Fourier Transform method.

## Arguments
- `model`: Kuramoto-Sivashinsky equation model
- `μ`: parameter value

## Returns
- `A`: A matrix
- `F`: F matrix  (take out 1.0im)
"""
function pseudo_spectral_model(model::KuramotoSivashinskyModel, μ::Float64)
    L = model.spatial_domain[2] - model.spatial_domain[1]

    # Create A matrix
    A = spdiagm(
        0 => [(2 * π * k / L)^2 - μ*(2 * π * k / L)^4 for k in model.fourier_modes]
    )
    
    # Create F matix
    F = spdiagm(
        0 => [-π * k / L for k in model.fourier_modes]
    )
    model.model_type = :PS
    model.integrate_model = integrate_pseudo_spectral_model
    return A, F
end


"""
    elementwise_pseudo_spectral_model(model, μ) → A, F    

Generate A, F matrices for the Kuramoto-Sivashinsky equation using the Fast Fourier Transform method (element-wise).

## Arguments
- `model`: Kuramoto-Sivashinsky equation model
- `μ`: parameter value

## Returns
- `A`: A matrix
- `F`: F matrix
"""
function elementwise_pseudo_spectral_model(model::KuramotoSivashinskyModel, μ::Float64)
    L = model.spatial_domain[2] - model.spatial_domain[1]

    # Create A matrix
    A = [(2 * π * k / L)^2 - μ*(2 * π * k / L)^4 for k in model.fourier_modes]
    # Create F matix
    F = [-π * 1.0im * k / L for k in model.fourier_modes]
    model.model_type = :EWPS
    model.integrate_model = integrate_elementwise_pseudo_spectral_model
    return A, F
end


"""
    spectral_galerkin_model(model, μ) → A, F
    
Generate A, F matrices for the Kuramoto-Sivashinsky equation using the Spectral-Galerkin method.

## Arguments
- `model`: Kuramoto-Sivashinsky equation model
- `μ`: parameter value

## Returns
- `A`: A matrix
- `F`: F matrix
"""
function spectral_galerkin_model(model::KuramotoSivashinskyModel, μ::Float64)
    N = model.spatial_dim
    L = model.spatial_domain[2] - model.spatial_domain[1]

    # Create A matrix
    A = spdiagm(
        0 => [(2 * π * k / L)^2 - μ*(2 * π * k / L)^4 for k in model.fourier_modes]
    )

    # Create F matrix
    # WARNING: The 1.0im is taken out from F
    F = spzeros(N, Int(N * (N + 1) / 2))
    for k in model.fourier_modes
        foo = zeros(N, N)
        for p in model.fourier_modes, q in model.fourier_modes
            if  p + q == k
                pshift = Int(p + N/2 + 1)
                qshift = Int(q + N/2 + 1)
                if pshift > qshift
                    foo[pshift, qshift] += -π * k / L
                else 
                    foo[qshift, pshift] += -π * k / L
                end
            end
        end
        F[Int(k+N/2+1), :] =  vech(foo)
    end

    # # INFO: Another way to creat F matrix inspired by the convolution operator
    # F = spzeros(N, Int(N * (N + 1) / 2))
    # D = Dict{Integer, AbstractVector}()
    # for i in 0:-1:-N+1
    #     D[i] = abs(i-1)*ones(N) .+ 1im*collect(1:N)
    # end
    # ConvIdxMat = spdiagm(2*N-1, N, D...)
    # ConvIdxMat = ConvIdxMat[Int(N/2)+1:Int(N+N/2), :]
    # F = spzeros(N, Int(N*(N+1)/2))

    # for (i,k) in enumerate(model.k)
    #     foo = spzeros(N, N)
    #     for j in 1:N
    #         idx = ConvIdxMat[i, j]
    #         if idx != 0.0
    #             p = idx |> real |> Int
    #             q = idx |> imag |> Int
    #             if p > q
    #                 foo[p, q] += -π * k / L
    #             else
    #                 foo[q, p] += -π * k / L
    #             end
    #         end
    #     end
    #     F[i, :] = vech(foo)
    # end

    model.model_type = :SG
    model.integrate_model = integrate_spectral_galerkin_model
    return A, F
end


"""
$(SIGNATURES)

Integrator using Crank-Nicholson Adams-Bashforth method for (FD)

## Arguments
- `A`: A matrix
- `F`: F matrix
- `tdata`: temporal points
- `IC`: initial condition
- `const_stepsize`: whether to use a constant time step size
- `u2_lm1`: u2 at j-2

## Returns
- `u`: state matrix
"""
function integrate_finite_diff_model(A, F, tdata, IC; const_stepsize=true, u2_lm1=nothing)
    Xdim = length(IC)
    Tdim = length(tdata)
    u = zeros(Xdim, Tdim)
    u[:, 1] = IC
    # u2_lm1 = Vector{Float64}()  # u2 at j-2 placeholder

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(1.0I(Xdim) - Δt/2 * A) \ 1.0I(Xdim) # |> sparse
        IpdtA = (1.0I(Xdim) + Δt/2 * A)

        @inbounds for j in 2:Tdim
            # u2 = vech(u[:, j-1] * u[:, j-1]')
            u2 = u[:, j-1] ⊘ u[:, j-1]
            if j == 2 && isnothing(u2_lm1)
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * Δt)
            else
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * 3*Δt/2 - F * u2_lm1 * Δt/2)
            end
            u2_lm1 = u2
        end
    else
        @inbounds for j in 2:Tdim
            Δt = tdata[j] - tdata[j-1]
            # u2 = vech(u[:, j-1] * u[:, j-1]')
            u2 = u[:, j-1] ⊘ u[:, j-1]
            if j == 2 && isnothing(u2_lm1)
                u[:, j] = (1.0I(Xdim) - Δt/2 * A) \ ((1.0I(Xdim) + Δt/2 * A) * u[:, j-1] + F * u2 * Δt)
            else
                u[:, j] = (1.0I(Xdim) - Δt/2 * A) \ ((1.0I(Xdim) + Δt/2 * A) * u[:, j-1] + F * u2 * 3*Δt/2 - F * u2_lm1 * Δt/2)
            end
            u2_lm1 = u2
        end
    end
    return u
end


"""
$(SIGNATURES)

Integrator using Crank-Nicholson Adams-Bashforth method for (FD). 
This is a dispatch function for `integrate_FD(A, F, tdata, IC; const_stepsize=true, u2_lm1=nothing)`.
Using the operator struct `ops` instead of `A` and `F`.

## Arguments
- `ops`: Operators
- `tdata`: temporal points
- `IC`: initial condition
- `params`: keyword arguments
    - `const_stepsize`: whether to use a constant time step size

## Returns
- `u`: state matrix
"""
function integrate_finite_diff_model(ops, tdata, IC; params...)
    # Unpack the parameters
    const_stepsize = get(params, :const_stepsize, true)

    A = ops.A
    F = ops.F

    Xdim = length(IC)
    Tdim = length(tdata)
    u = zeros(Xdim, Tdim)
    u[:, 1] = IC
    u2_lm1 = Vector{Float64}()  # u2 at j-2 placeholder

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(1.0I(Xdim) - Δt/2 * A) \ 1.0I(Xdim) # |> sparse
        IpdtA = (1.0I(Xdim) + Δt/2 * A)

        for j in 2:Tdim
            u2 = vech(u[:, j-1] * u[:, j-1]')
            if j == 2 
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * Δt)
            else
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * 3*Δt/2 - F * u2_lm1 * Δt/2)
            end
            u2_lm1 = u2
        end
    else
        for j in 2:Tdim
            Δt = tdata[j] - tdata[j-1]
            u2 = vech(u[:, j-1] * u[:, j-1]')
            if j == 2
                u[:, j] = (1.0I(Xdim) - Δt/2 * A) \ ((1.0I(Xdim) + Δt/2 * A) * u[:, j-1] + F * u2 * Δt)
            else
                u[:, j] = (1.0I(Xdim) - Δt/2 * A) \ ((1.0I(Xdim) + Δt/2 * A) * u[:, j-1] + F * u2 * 3*Δt/2 - F * u2_lm1 * Δt/2)
            end
            u2_lm1 = u2
        end
    end
    return u
end



"""
$(SIGNATURES)

Integrator using Crank-Nicholson Adams-Bashforth method for (FFT)

## Arguments
- `A`: A matrix
- `F`: F matrix
- `tdata`: temporal points
- `IC`: initial condition

## Returns
- `u`: state matrix
- `uhat`: state matrix in the Fourier space
"""
function integrate_pseudo_spectral_model(A, F, tdata, IC)
    Xdim = length(IC)
    Tdim = length(tdata)
    foo = zeros(ComplexF64, Xdim)

    # Plan the fourier transform and inverse fourier transform
    pfft = plan_fft(IC)
    pifft = plan_ifft(foo)

    u = zeros(Xdim, Tdim)  # state in the physical space
    u[:, 1] = IC
    uhat = zeros(ComplexF64, Xdim, Tdim)  # state in the Fourier space
    uhat[:, 1] = fftshift(pfft * u[:, 1]) / Xdim
    uhat2_lm1 = Vector{ComplexF64}()  # uhat2 at j-2 placeholder

    for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        uhat2 = fftshift(pfft * (u[:, j-1].^2)) / Xdim

        # WARNING: The 1.0im is taken out from F
        if j == 2
            uhat[:, j] = (1.0I(Xdim) - Δt/2 * A) \ ((1.0I(Xdim) + Δt/2 * A) * uhat[:, j-1] + 1.0im * F * uhat2 * Δt)
        else
            uhat[:, j] = (1.0I(Xdim) - Δt/2 * A) \ ((1.0I(Xdim) + Δt/2 * A) * uhat[:, j-1] + 1.0im * F * uhat2 * 3*Δt/2 - 1.0im * F * uhat2_lm1 * Δt/2)
        end

        # Get the state in the physical space
        u[:, j] = real.(Xdim * pifft * (ifftshift(uhat[:, j])))

        # Clean
        uhat[:, j] = fftshift(pfft * u[:, j]) / Xdim
        uhat2_lm1 = uhat2
    end
    return u, uhat
end


"""
$(SIGNATURES)

Integrator using Crank-Nicholson Adams-Bashforth method for (FFT) (element-wise)

## Arguments
- `A`: A matrix
- `F`: F matrix
- `tdata`: temporal points
- `IC`: initial condition

## Returns
- `u`: state matrix
- `uhat`: state matrix in the Fourier space
"""
function integrate_elementwise_pseudo_spectral_model(A, F, tdata, IC)
    Xdim = length(IC)
    Tdim = length(tdata)
    foo = zeros(ComplexF64, Xdim)

    # Plan the fourier transform and inverse fourier transform
    pfft = plan_fft(IC)
    pifft = plan_ifft(foo)

    u = zeros(Xdim, Tdim)  # state in the physical space
    u[:, 1] = IC
    uhat = zeros(ComplexF64, Xdim, Tdim)  # state in the Fourier space
    uhat[:, 1] = fftshift(pfft * u[:, 1]) / Xdim
    uhat2_lm1 = Vector{ComplexF64}()  # uhat2 at j-2 placeholder

    for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        uhat2 = fftshift(pfft * (u[:, j-1].^2)) / Xdim

        if j == 2
            uhat[:, j] = (1.0 ./ (1.0 .- Δt/2 * A)) .* ((1.0 .+ Δt/2 * A) .* uhat[:, j-1] + F .* uhat2 * Δt)
        else
            uhat[:, j] = (1.0 ./ (1.0 .- Δt/2 * A)) .* ((1.0 .+ Δt/2 * A) .* uhat[:, j-1] + F .* uhat2 * 3*Δt/2 - F .* uhat2_lm1 * Δt/2)
        end

        # Get the state in the physical space
        u[:, j] = real.(Xdim * pifft * (ifftshift(uhat[:, j])))

        # Clean
        uhat[:, j] = fftshift(pfft * u[:, j]) / Xdim
        uhat2_lm1 = uhat2
    end
    return u, uhat
end


"""
$(SIGNATURES)

Integrator for model produced with Spectral-Galerkin method.

## Arguments
- `A`: A matrix
- `F`: F matrix
- `tdata`: temporal points
- `IC`: initial condition

## Returns
- `u`: state matrix
- `uhat`: state matrix in the Fourier space
"""
function integrate_spectral_galerkin_model(A, F, tdata, IC)
    Xdim = length(IC)
    Tdim = length(tdata)
    foo = zeros(ComplexF64, Xdim)

    # Plan the fourier transform and inverse fourier transform
    pfft = plan_fft(IC)
    pifft = plan_ifft(foo)

    u = zeros(Xdim, Tdim)  # state in the physical space
    u[:, 1] = IC
    uhat = zeros(ComplexF64, Xdim, Tdim)  # state in the Fourier space
    uhat[:, 1] = fftshift(pfft * u[:, 1]) / Xdim
    uhat2_lm1 = Vector{ComplexF64}()  # uhat2 at j-2 placeholder

    for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        # uhat2 = vech(uhat[:, j-1] * transpose(uhat[:, j-1]))
        uhat2 = uhat[:, j-1] ⊘ uhat[:, j-1]

        # WARNING: The 1.0im is taken out from F
        if j == 2
            uhat[:, j] = (1.0I(Xdim) - Δt/2 * A) \ ((1.0I(Xdim) + Δt/2 * A) * uhat[:, j-1] + 1.0im * F * uhat2 * Δt)
        else
            uhat[:, j] = (1.0I(Xdim) - Δt/2 * A) \ ((1.0I(Xdim) + Δt/2 * A) * uhat[:, j-1] + 1.0im * F * uhat2 * 3*Δt/2 - 1.0im * F * uhat2_lm1 * Δt/2)
        end

        # Get the state in the physical space
        u[:, j] = real.(Xdim * pifft * (ifftshift(uhat[:, j])))

        # Clean
        uhat[:, j] = fftshift(pfft * u[:, j]) / Xdim
        uhat2_lm1 = uhat2
    end
    return u, uhat
end


"""
$(SIGNATURES)

Generate Jacobian matrix

## Arguments
- `A`: A matrix
- `H`: H matrix
- `x`: state

## Returns
- `J`: Jacobian matrix
"""
function jacobian(ops::Operators, x::AbstractVector{T}) where {T}
    n = length(x)
    # return ops.A + ops.H * kron(1.0I(n), x) + ops.H * kron(x, 1.0I(n))
    return ops.A + ops.F * elimat(n) * ( kron(1.0I(n), x) + kron(x, 1.0I(n)) )
end

end