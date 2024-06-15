"""
    1 Dimensional Heat Equation Model
"""
module Heat1D

using DocStringExtensions
using LinearAlgebra

import ..LiftAndLearn: AbstractModel

export Heat1DModel


"""
$(TYPEDEF)

1 Dimensional Heat Equation Model

```math
\\frac{\\partial u}{\\partial t} = \\mu\\frac{\\partial^2 u}{\\partial x^2}
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
- `finite_diff_model::Function`: model using Finite Difference
"""
mutable struct Heat1DModel <: AbstractModel
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

    finite_diff_model::Function
end


"""
$(SIGNATURES)

Constructor 1 Dimensional Heat Equation Model

## Arguments
- `spatial_domain::Tuple{Real,Real}`: spatial domain
- `time_domain::Tuple{Real,Real}`: temporal domain
- `Δx::Real`: spatial grid size
- `Δt::Real`: temporal step size
- `diffusion_coeffs::Union{Real,AbstractArray{<:Real}}`: parameter vector
- `BC::Symbol=:dirichlet`: boundary condition

## Returns
- `Heat1DModel`: 1D heat equation model
"""
function Heat1DModel(;spatial_domain::Tuple{Real,Real}, time_domain::Tuple{Real,Real}, Δx::Real, Δt::Real, 
                 diffusion_coeffs::Union{Real,AbstractArray{<:Real}}, BC::Symbol=:dirichlet)
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

    Heat1DModel(
        spatial_domain, time_domain, param_domain,
        Δx, Δt, BC, IC, xspan, tspan, diffusion_coeffs,
        spatial_dim, time_dim, param_dim, finite_diff_model
    )
end


"""
$(SIGNATURES)

Finite Difference Model for 1D Heat Equation

## Arguments
- `model::Heat1DModel`: 1D heat equation model
- `μ::Real`: diffusion coefficient

## Returns
- operators
"""
function finite_diff_model(model::Heat1DModel, μ::Real)
    if model.BC == :periodic
        return finite_diff_periodic_model(model.spatial_dim, model.Δx, μ)
    elseif model.BC == :dirichlet
        return finite_diff_dirichlet_model(model.spatial_dim, model.Δx, μ)
    else
        error("Boundary condition not implemented")
    end
end


function finite_diff_periodic_model(N::Real, Δx::Real, μ::Real)
    # Create A matrix
    A = spdiagm(0 => (-2) * ones(N), 1 => ones(N - 1), -1 => ones(N - 1)) * μ / Δx^2
    A[1, end] = 1 / Δx^2  # periodic boundary condition
    A[end, 1] = 1 / Δx^2  
    return A
end


function finite_diff_dirichlet_model(N::Real, Δx::Real, μ::Real)
    A = diagm(0 => (-2)*ones(N), 1 => ones(N-1), -1 => ones(N-1)) * μ / Δx^2
    B = [1; zeros(N-2,1); 1] * μ / Δx^2   #! Fixed this to generalize input u
    return A, B
end

end