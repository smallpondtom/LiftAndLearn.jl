"""
    Viscous Burgers' equation model
"""
module Burgers

using DocStringExtensions
using LinearAlgebra
using SparseArrays

import ..LiftAndLearn: AbstractModel, vech, ⊘, Operators

export BurgersModel


"""
$(TYPEDEF)

Viscous Burgers' equation model

```math
\\frac{\\partial u}{\\partial t} = \\mu\\frac{\\partial^2 u}{\\partial x^2} - u\\frac{\\partial u}{\\partial x}
```

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
- `spatial_dim::Int64`: spatial dimension
- `time_dim::Int64`: temporal dimension
- `diffusion_coeffs::Union{Real,AbstractArray{<:Real}}`: parameter vector
- `param_dim::Int64`: parameter dimension
- `conservation_type::Symbol`: conservation type
- `finite_diff_model::Function`: model using Finite Difference
- `integrate_model::Function`: model integration
"""
mutable struct BurgersModel <: AbstractModel
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

    # Dimensions
    spatial_dim::Int64  # spatial dimension
    time_dim::Int64  # temporal dimension
    param_dim::Int64  # parameter dimension

    # Convervation type
    conservation_type::Symbol

    finite_diff_model::Function
    integrate_model::Function
end


"""
$(SIGNATURES)

Constructor for the Burgers' equation model.
"""
function BurgersModel(;spatial_domain::Tuple{Real,Real}, time_domain::Tuple{Real,Real}, Δx::Real, Δt::Real, 
                       diffusion_coeffs::Union{Real,AbstractArray{<:Real}}, BC::Symbol=:dirichlet,
                       conservation_type::Symbol=:NC)
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

    @assert conservation_type ∈ (:EP, :NC, :C) "Invalid conservation type"

    BurgersModel(
        spatial_domain, time_domain, param_domain,
        Δx, Δt, BC, IC, xspan, tspan, diffusion_coeffs,
        spatial_dim, time_dim, param_dim, conservation_type,
        finite_diff_model, integrate_model
    )
end


"""
$(SIGNATURES)

Finite Difference Model for Burgers equation

## Arguments
- `model::BurgersModel`: Burgers model
- `μ::Real`: diffusion coefficient

## Returns
- operators
"""
function finite_diff_model(model::BurgersModel, μ::Real)
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
    elseif model.BC == :dirichlet
        return finite_diff_dirichlet_model(model.spatial_dim, model.Δx, model.Δt, μ)
    else
        error("Boundary condition not implemented")
    end
end



"""
    finite_diff_dirichlet_model(N::Real, Δx::Real, μ::Real) → A, B, F

Generate A, B, F matrices for the Burgers' equation for Dirichlet boundary condition.
This is by default the non-conservative form.
"""
function finite_diff_dirichlet_model(N::Real, Δx::Real, Δt::Real, μ::Float64)
    # Create A matrix
    A = spdiagm(0 => (-2) * ones(N), 1 => ones(N - 1), -1 => ones(N - 1)) * μ / Δx^2
    A[1, 1:2] = [-1/Δt, 0]
    A[end, end-1:end] = [0, -1/Δt]

    # Create F matrix
    S = Int(N * (N + 1) / 2)
    if N >= 3
        Fval = repeat([1.0, -1.0], outer=N - 2)
        row_i = repeat(2:(N-1), inner=2)
        seq = Int.([2 + (N + 1) * (x - 1) - x * (x - 1) / 2 for x in 1:(N-1)])
        col_i = vcat(seq[1], repeat(seq[2:end-1], inner=2), seq[end])
        F = sparse(row_i, col_i, Fval, N, S) / 2 / Δx
    else
        F = zeros(N, S)
    end

    # Create B matrix
    B = [1; zeros(N - 2, 1); -1] ./ Δt

    return A, B, F
end


"""
    finite_diff_periodic_nonconservative_model(N::Real, Δx::Real, μ::Real) → A, F

Generate A, F matrices for the Burgers' equation for periodic boundary condition (Non-conservative).
"""
function finite_diff_periodic_nonconservative_model(N::Real, Δx::Real, μ::Real)
    # Create A matrix
    A = spdiagm(0 => (-2) * ones(N), 1 => ones(N - 1), -1 => ones(N - 1)) * μ / Δx^2
    A[1, end] = μ / Δx^2  # periodic boundary condition
    A[end, 1] = μ / Δx^2  

    # Create F matrix
    S = Int(N * (N + 1) / 2)
    if N >= 3
        Fval = repeat([1.0, -1.0], outer=N - 2)
        row_i = repeat(2:(N-1), inner=2)
        seq = Int.([2 + (N + 1) * (x - 1) - x * (x - 1) / 2 for x in 1:(N-1)])
        col_i = vcat(seq[1], repeat(seq[2:end-1], inner=2), seq[end])
        F = sparse(row_i, col_i, Fval, N, S) / 2 / Δx

        F[1, 2] = - 1 / 2 / Δx
        F[1, N] = 1 / 2 / Δx
        F[N, N] = - 1 / 2 / Δx
        F[N, end-1] = 1 / 2 / Δx 
    else
        F = zeros(N, S)
    end

    return A, F
end


"""
    finite_diff_periodic_conservative_model(N::Real, Δx::Real, μ::Real) → A, F

Generate A, F matrices for the Burgers' equation for periodic boundary condition (Conservative form).
"""
function finite_diff_periodic_conservative_model(N::Real, Δx::Real, μ::Float64)
    # Create A matrix
    A = spdiagm(0 => (-2) * ones(N), 1 => ones(N - 1), -1 => ones(N - 1)) * μ / Δx^2
    A[1, end] = μ / Δx^2  # periodic boundary condition
    A[end, 1] = μ / Δx^2  

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

    return A, F
end


"""
    finite_diff_periodic_energy_preserving_model(N::Real, Δx::Real, μ::Float64) → A, F

Generate A, F matrices for the Burgers' equation for periodic boundary condition (Energy preserving form).
"""
function finite_diff_periodic_energy_preserving_model(N::Real, Δx::Real, μ::Float64)
    # Create A matrix
    A = spdiagm(0 => (-2) * ones(N), 1 => ones(N - 1), -1 => ones(N - 1)) * μ / Δx^2
    A[1, N] = μ / Δx^2  # periodic boundary condition
    A[N, 1] = μ / Δx^2  # periodic boundary condition

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

    return A, F
end



"""
    integrate_model(A, B, F, U, tdata, IC) → states

Semi-Implicit Euler scheme
"""
function integrate_model(A, B, F, U, tdata, IC)
    Xdim = length(IC)
    Tdim = length(tdata)
    state = zeros(Xdim, Tdim)
    state[:, 1] = IC

    for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        # state2 = vech(state[:, j-1] * state[:, j-1]')
        state2 = state[:, j-1] ⊘ state[:, j-1]
        state[:, j] = (1.0I(Xdim) - Δt * A) \ (state[:, j-1] + F * state2 * Δt + B * U[j-1] * Δt)
    end
    return state
end


"""
    integrate_model(A, F, tdata, IC) → states

Semi-Implicit Euler scheme without control (dispatch)
"""
function integrate_model(A, F, tdata, IC)
    Xdim = length(IC)
    Tdim = length(tdata)
    state = zeros(Xdim, Tdim)
    state[:, 1] = IC

    for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        # state2 = vech(state[:, j-1] * state[:, j-1]')
        state2 = state[:, j-1] ⊘ state[:, j-1]
        state[:, j] = (1.0I(Xdim) - Δt * A) \ (state[:, j-1] + F * state2 * Δt)
    end
    return state
end


"""
    integrate_model(ops, tdata, IC) → states

Semi-Implicit Euler scheme without control (dispatch)
"""
function integrate_model(ops, tdata, IC)
    A = ops.A 
    F = ops.F
    Xdim = length(IC)
    Tdim = length(tdata)
    state = zeros(Xdim, Tdim)
    state[:, 1] = IC

    for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        # state2 = vech(state[:, j-1] * state[:, j-1]')
        state2 = state[:, j-1] ⊘ state[:, j-1]
        state[:, j] = (1.0I(Xdim) - Δt * A) \ (state[:, j-1] + F * state2 * Δt)
    end
    return state
end


end