"""
    2 Dimensional Heat Equation Model
"""
module Heat2D

using DocStringExtensions
using LinearAlgebra

import ..LiftAndLearn: AbstractModel

export heat1d


"""
$(TYPEDEF)

2 Dimensional Heat Equation Model

```math
\\frac{\\partial u}{\\partial t} = \\mu\\frac{\\partial^2 u}{\\partial x^2}
```

## Fields
"""
mutable struct heat2d <: AbstractModel
    # Domains
    spatial_domain::Tuple{Tuple{Real,Real}, Tuple{Real,Real}}  # spatial domain (x, y)
    time_domain::Tuple{Real,Real}  # temporal domain
    param_domain::Tuple{Real,Real}  # parameter domain

    # Grids
    Δx::Real  # spatial grid size (x-axis)
    Δy::Real  # spatial grid size (y-axis)
    Δt::Real  # temporal step size

    # Dimensions
    spatial_dim::Tuple{Int64,Int64}  # spatial dimension x and y
    time_dim::Int64  # temporal dimension
    param_dim::Int64  # parameter dimension

    # Boundary and Initial Conditions
    BC::Tuple{Symbol,Symbol}  # boundary condition
    IC::Array{Float64}  # initial condition

    # Parameters
    diffusion_coeffs::Union{Real, Vector{<:Real}} # diffusion coefficients

    # Data
    xspan::Vector{Float64}  # spatial grid points (x-axis)
    yspan::Vector{Float64}  # spatial grid points (y-axis)
    tspan::Vector{Float64}  # temporal points
    param_span::Vector{Float64}  # parameter vector

    # Functions
    finite_diff_model::Function
    integrate_model::Function
end


"""
$(SIGNATURES)

2 Dimensional Heat Equation Model

## Arguments

## Returns
- `heat2d`: 2D heat equation model
"""
function heat2d(;spatial_domain::Tuple{Tuple{Real,Real},Tuple{Real,Real}}, time_domain::Tuple{Real,Real}, 
                 Δx::Real, Δt::Real, diffusion_coeffs::Union{Real, Vector{<:Real}}, BC::Symbol)
    # Discritization grid info
    @assert BC ∈ (:periodic, :dirichlet, :neumann, :mixed, :robin, :cauchy, :flux) "Invalid boundary condition"
    # x-axis
    if BC[1] == :periodic
        xspan = collect(spatial_domain[1][1]:Δx:spatial_domain[1][2]-Δx)
    elseif BC[1] ∈ (:dirichlet, :neumann, :mixed, :robin, :cauchy) 
        xspan = collect(spatial_domain[1][1]:Δx:spatial_domain[1][2])[2:end-1]
    end
    # y-axis
    if BC[2] == :periodic
        yspan = collect(spatial_domain[2][1]:Δy:spatial_domain[2][2]-Δy)
    elseif BC[2] ∈ (:dirichlet, :neumann, :mixed, :robin, :cauchy) 
        yspan = collect(spatial_domain[2][1]:Δy:spatial_domain[2][2])[2:end-1]
    end
    tspan = collect(time_domain[1]:Δt:time_domain[2])
    spatial_dim = (length(xspan), length(yspan))
    time_dim = length(tspan)

    # Initial condition
    IC = zeros(sum(spatial_dim))

    # Parameter dimensions or number of parameters 
    param_dim = length(diffusion_coeffs)
    param_domain = extrema(diffusion_coeffs)

    heat2d(spatial_domain, time_domain, param_domain, Δx, Δy, Δt,
           spatial_dim, time_dim, param_dim, BC, IC, diffusion_coeffs,
           xspan, tspan, diffusion_coeffs,
           generateABmatrix, integrateFD!)
end


"""
$(SIGNATURES)

Generate A and B matrices for the 1D heat equation.

## Arguments
- `N::Int64`: number of spatial grid points
- `μ::Float64`: viscosity coefficients
- `Δx::Float64`: spatial grid size

## Returns
- `A::Matrix{Float64}`: A matrix
- `B::Matrix{Float64}`: B matrix
"""
function generateABmatrix(N, μ, Δx)
    A = diagm(0 => (-2)*ones(N), 1 => ones(N-1), -1 => ones(N-1)) * μ / Δx^2
    B = [1; zeros(N-2,1); 1] * μ / Δx^2   #! Fixed this to generalize input u
    return A, B
end

end
