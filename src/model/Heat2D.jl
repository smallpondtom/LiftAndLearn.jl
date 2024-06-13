"""
    2 Dimensional Heat Equation Model
"""
module Heat2D

using DocStringExtensions
using LinearAlgebra
using SparseArrays

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
    spatial_domain::Tuple{Tuple{<:Real,<:Real}, Tuple{<:Real,<:Real}}  # spatial domain (x, y)
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
    diffusion_coeffs::Union{Vector{<:Real},Real} # diffusion coefficients

    # Data
    xspan::Vector{Float64}  # spatial grid points (x-axis)
    yspan::Vector{Float64}  # spatial grid points (y-axis)
    tspan::Vector{Float64}  # temporal points

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
                 Δx::Real, Δy::Real, Δt::Real, diffusion_coeffs::Union{Vector{<:Real},Real}, BC::Tuple{Symbol,Symbol})
    # Discritization grid info
    possible_BC = (:periodic, :dirichlet, :neumann, :mixed, :robin, :cauchy, :flux)
    @assert all([BC[i] ∈ possible_BC for i in eachindex(BC)]) "Invalid boundary condition"
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
    IC = zeros(prod(spatial_dim))

    # Parameter dimensions or number of parameters 
    param_dim = length(diffusion_coeffs)
    param_domain = extrema(diffusion_coeffs)

    heat2d(spatial_domain, time_domain, param_domain, Δx, Δy, Δt,
           spatial_dim, time_dim, param_dim, BC, IC, diffusion_coeffs,
           xspan, yspan, tspan, 
           finite_diff_model, integrate_model)
end


function finite_diff_dirichlet_model(model::heat2d, μ::Real)
    Nx, Ny = model.spatial_dim
    Δx, Δy = model.Δx, model.Δy

    # A matrix
    Ax = spdiagm(0 => (-2)*ones(Nx), 1 => ones(Nx-1), -1 => ones(Nx-1)) * μ / Δx^2
    Ay = spdiagm(0 => (-2)*ones(Ny), 1 => ones(Ny-1), -1 => ones(Ny-1)) * μ / Δy^2
    A = kron(Ay, I(Nx)) + kron(I(Ny), Ax)

    # B matrix (different inputs for each boundary)
    Bx = spzeros(Nx*Ny,2)
    Bx[1:Ny,1] .= μ / Δx^2
    Bx[end-Ny+1:end,2] .= μ / Δx^2
    By = spzeros(Nx*Ny,2)
    idx = [Ny*(n-1)+1 for n in 1:Nx]
    By[idx,1] .= μ / Δy^2
    idx = [Ny*n for n in 1:Nx]
    By[idx,2] .= μ / Δy^2
    B = hcat(Bx, By)

    return A, B
end


"""
$(SIGNATURES)

Generate A and B matrices for the 2D heat equation.

## Arguments
- `model::heat2d`: 2D heat equation model
- `μ::Real`: diffusion coefficient

## Returns
- `A::Matrix{Float64}`: A matrix
- `B::Matrix{Float64}`: B matrix
"""
function finite_diff_model(model::heat2d, μ::Real)
    if all(model.BC .== :dirichlet)
        return finite_diff_dirichlet_model(model, μ)
    else
        error("Not implemented")
    end
end



"""
$(SIGNATURES)

Integrate the 2D heat equation model.

## Arguments
- `A::Matrix{Float64}`: A matrix
- `B::Matrix{Float64}`: B matrix
- `U::Vector{Float64}`: input vector
- `tdata::Vector{Float64}`: time points
- `IC::Vector{Float64}`: initial condition

## Returns
- `state::Matrix{Float64}`: state matrix
"""
function integrate_model(A, B, U, tdata, IC)
    Xdim = length(IC)
    Tdim = length(tdata)
    state = Matrix{Float64}(undef, Xdim, Tdim)
    state[:,1] = IC
    @inbounds for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        state[:,j] = (I - Δt * A) \ (state[:,j-1] + B * U[:,j-1] * Δt)
    end
    return state
end

end
