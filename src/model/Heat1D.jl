"""
    1 Dimensional Heat Equation Model
"""
module Heat1D

using DocStringExtensions
using LinearAlgebra

export heat1d

# Import abstract type Abstract_Models from LiftAndLearn
import ..LiftAndLearn: Abstract_Models

"""
$(TYPEDEF)

1 Dimensional Heat Equation Model

```math
\\frac{\\partial u}{\\partial t} = \\mu\\frac{\\partial^2 u}{\\partial x^2}
```

## Fields
- `Omega::Vector{Float64}`: spatial domain
- `T::Vector{Float64}`: temporal domain
- `D::Vector{Float64}`: parameter domain
- `Δx::Float64`: spatial grid size
- `Δt::Float64`: temporal step size
- `Ubc::Matrix{Float64}`: boundary condition (input)
- `IC::Matrix{Float64}`: initial condition
- `x::Vector{Float64}`: spatial grid points
- `t::Vector{Float64}`: temporal points
- `μs::Vector{Float64}`: parameter vector
- `Xdim::Int64`: spatial dimension
- `Tdim::Int64`: temporal dimension
- `Pdim::Int64`: parameter dimension
- `generateABmatrix::Function`: function to generate A and B matrices
"""
mutable struct heat1d <: Abstract_Models
    Omega::Vector{Float64}  # spatial domain
    T::Vector{Float64}  # temporal domain
    D::Vector{Float64}  # parameter domain
    Δx::Float64  # spatial grid size
    Δt::Float64  # temporal step size
    Ubc::Matrix{Float64}  # boundary condition (input)
    IC::Matrix{Float64}  # initial condition
    x::Vector{Float64}  # spatial grid points
    t::Vector{Float64}  # temporal points
    μs::Vector{Float64}  # parameter vector
    Xdim::Int64  # spatial dimension
    Tdim::Int64  # temporal dimension
    Pdim::Int64  # parameter dimension

    generateABmatrix::Function
end


"""
$(SIGNATURES)

1 Dimensional Heat Equation Model

## Arguments
- `Omega::Vector{Float64}`: spatial domain
- `T::Vector{Float64}`: temporal domain
- `D::Vector{Float64}`: parameter domain
- `Δx::Float64`: spatial grid size  
- `Δt::Float64`: temporal step size
- `Pdim::Int64`: parameter dimension

## Returns
- `heat1d`: 1D heat equation model
"""
function heat1d(Omega, T, D, Δx, Δt, Pdim)
    x = (Omega[1]:Δx:Omega[2])[2:end-1]
    t = T[1]:Δt:T[2]
    μs = range(D[1], D[2], Pdim)
    Xdim = length(x)
    Tdim = length(t)
    Ubc = ones(Tdim,1)
    IC = zeros(Xdim,1)

    heat1d(Omega, T, D, Δx, Δt, Ubc, IC, x, t, μs, Xdim, Tdim, Pdim, generateABmatrix)
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