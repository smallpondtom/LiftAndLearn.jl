"""
    Chafee-Infante equation PDE model
"""
module ChafeeInfante

using DocStringExtensions
using LinearAlgebra
using SparseArrays

import ..LiftAndLearn: AbstractModel, ⊘, makeCubicOp

export ChafeeInfanteModel


"""
$(TYPEDEF)

    
```math
\\frac{\\partial u}{\\partial t} =  \\frac{\\partial^2 u}{\\partial x^2} + \\lambda (u - u^3)
```

where ``u`` is the state variable, ``\\lambda`` is the coefficient of nonlinearity.

## Fields
- `spatial_domain::Tuple{Real,Real}`: spatial domain
- `time_domain::Tuple{Real,Real}`: temporal domain
- `param_domain::Tuple{Real,Real}`: parameter domain (diffusion coeff)
- `Δx::Real`: spatial grid size
- `Δt::Real`: temporal step size
- `xspan::Vector{<:Real}`: spatial grid points
- `tspan::Vector{<:Real}`: temporal points
- `spatial_dim::Int`: spatial dimension
- `time_dim::Int`: temporal dimension
- `diffusion_coeffs::Union{Vector{<:Real},Real}`: diffusion coefficient
- `param_dim::Int`: parameter dimension
- `IC::AbstractArray{<:Real}`: initial condition
- `BC::Symbol`: boundary condition
- `finite_diff_model::Function`: model using Finite Difference
- `integrate_model::Function`: integrator using Crank-Nicholson (linear) Explicit (nonlinear) method
"""
mutable struct ChafeeInfanteModel <: AbstractModel
    # Domains
    spatial_domain::Tuple{Real,Real}  # spatial domain
    time_domain::Tuple{Real,Real}  # temporal domain
    param_domain::Tuple{Real,Real}  # parameter domain (diffusion coeff)

    # Discritization grid
    Δx::Real  # spatial grid size
    Δt::Real  # temporal step size
    xspan::Vector{<:Real}  # spatial grid points
    tspan::Vector{<:Real}  # temporal points
    spatial_dim::Int  # spatial dimension
    time_dim::Int  # temporal dimension

    # Parameters
    diffusion_coeffs::Union{AbstractArray{<:Real},Real}  # diffusion coefficient
    param_dim::Int  # parameter dimension

    # Initial condition
    IC::AbstractArray{<:Real}  # initial condition

    # Boundary condition
    BC::Symbol  # boundary condition

    # Functions
    finite_diff_model::Function  # model using Finite Difference
    integrate_model::Function # integrator using Crank-Nicholson (linear) Explicit (nonlinear) method
end


function ChafeeInfanteModel(;spatial_domain::Tuple{Real,Real}, time_domain::Tuple{Real,Real}, Δx::Real, Δt::Real, 
                    diffusion_coeffs::Union{AbstractArray{<:Real},Real}, BC::Symbol=:periodic)
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

    ChafeeInfanteModel(
        spatial_domain, time_domain, param_domain,
        Δx, Δt, xspan, tspan, spatial_dim, time_dim,
        diffusion_coeffs, param_dim, IC, BC,
        finite_diff_model, integrate_model
    )
end


"""
    finite_diff_model(model::chafeeinfante, μ::Real)

Create the matrices A (linear operator) and E (cubic operator) for the Chafee-Infante model.

## Arguments
- `model::ChafeeInfanteModel`: Chafee-Infante model
- `μ::Real`: diffusion coefficient
"""
function finite_diff_model(model::ChafeeInfanteModel, μ::Real)
    if model.BC == :periodic
        return finite_diff_periodic_model(model.spatial_dim, model.Δx, μ)
    elseif model.BC == :mixed
        return finite_diff_mixed_model(model.spatial_dim, model.Δx, μ)
    end
end


"""
    finite_diff_periodic_model(model::chafeeinfante, μ::Real)

Create the matrices A (linear operator) and E (cubic operator) for the Chafee-Infante model.

## Arguments
- `N::Real`: spatial dimension
- `Δx::Real`: spatial grid size
- `μ::Real`: diffusion coefficient

## Returns
- `A::SparseMatrixCSC{Float64,Int}`: linear operator
- `E::SparseMatrixCSC{Float64,Int}`: cubic operator
"""
function finite_diff_periodic_model(N::Real, Δx::Real, μ::Real)
    # Create A matrix
    A = spdiagm(0 => (μ-2/Δx^2) * ones(N), 1 => (1/Δx^2) * ones(N - 1), -1 => (1/Δx^2) * ones(N - 1))
    A[1, end] = 1 / Δx^2  # periodic boundary condition
    A[end, 1] = 1 / Δx^2  

    # Create E matrix
    indices = [(i,i,i,i) for i in 1:N]
    values = [-μ for _ in 1:N]
    E = makeCubicOp(N, indices, values, which_cubic_term='E')
    return A, E
end


"""
    finite_diff_mixed_model(model::chafeeinfante, μ::Real)

Create the matrices A (linear operator), B (input operator), and E (cubic operator) for Chafee-Infante 
model using the mixed boundary condition. If the spatial domain is [0,1], then we assume u(0,t) to be 
homogeneous dirichlet boundary condition and u(1,t) to be Neumann boundary condition of some function h(t).

## Arguments
- `N::Real`: spatial dimension
- `Δx::Real`: spatial grid size
- `μ::Real`: diffusion coefficient

## Returns
- `A::SparseMatrixCSC{Float64,Int}`: linear operator
- `B::SparseMatrixCSC{Float64,Int}`: input operator
- `E::SparseMatrixCSC{Float64,Int}`: cubic operator
"""
function finite_diff_mixed_model(N::Real, Δx::Real, μ::Real)
    # Create A matrix
    A = spdiagm(0 => (μ-2/Δx^2) * ones(N), 1 => (1/Δx^2) * ones(N - 1), -1 => (1/Δx^2) * ones(N - 1))
    A[end,end] = μ - 1/Δx^2  # influence of Neumann boundary condition

    # Create E matrix
    indices = [(i,i,i,i) for i in 1:N]
    values = [-μ for _ in 1:N]
    E = makeCubicOp(N, indices, values, which_cubic_term='E')

    # Create B matrix
    B = spzeros(N,2)
    B[1,1] = 1 / Δx^2  # from Dirichlet boundary condition
    B[end,2] = 1 / Δx  # from Neumann boundary condition

    return A, B, E
end


"""
    integrate_model(A::AbstractArray{<:Real}, E::AbstractArray{<:Real}, tspan::AbstractArray{<:Real}, 
                    IC::AbstractArray{<:Real}; const_stepsize::Bool=false)
    
Integrate the Chafee-Infante model using the Crank-Nicholson (linear) Explicit (nonlinear) method.

## Arguments
- `A::AbstractArray{<:Real}`: linear operator
- `E::AbstractArray{<:Real}`: cubic operator
- `IC::AbstractArray{<:Real}`: initial condition
- `tspan::AbstractArray{<:Real}`: time span
- `const_stepsize::Bool=false`: constant time step size

## Returns
- `state::AbstractArray{<:Real}`: solution
"""
function integrate_model(A::AbstractArray{<:Real}, E::AbstractArray{<:Real}, tspan::AbstractArray{<:Real}, 
                         IC::AbstractArray{<:Real}; const_stepsize::Bool=false)
    Xdim = length(IC)
    Tdim = length(tspan)
    state = zeros(Xdim, Tdim)
    state[:, 1] = IC

    if const_stepsize
        Δt = tspan[2] - tspan[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:Tdim
            state3 = ⊘(state[:, j-1], state[:, j-1], state[:, j-1])
            state[:, j] = ImdtA_inv * (IpdtA * state[:, j-1] + E * state3 * Δt)
        end
    else
        for j in 2:Tdim
            Δt = tspan[j] - tspan[j-1]
            state3 = ⊘(state[:, j-1], state[:, j-1], state[:, j-1])
            state[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * state[:, j-1] + E * state3 * Δt)
        end
    end
    return state
end


"""
    integrate_model(A::AbstractArray{T}, B::AbstractArray{T}, E::AbstractArray{T}, U::AbstractArray{T}, 
                    tspan::AbstractArray{T}, IC::AbstractArray{T}; const_stepsize::Bool=false) where T<:Real

Integrate the Chafee-Infante model using the Crank-Nicholson (linear) Adam-Bashforth (nonlinear) method.

## Arguments
- `A::AbstractArray{T}`: linear operator
- `B::AbstractArray{T}`: input operator
- `E::AbstractArray{T}`: cubic operator
- `U::AbstractArray{T}`: input
- `tspan::AbstractArray{T}`: time span
- `IC::AbstractArray{T}`: initial condition
- `const_stepsize::Bool=false`: constant time step size

## Returns
- `state::AbstractArray{T}`: solution
"""
function integrate_model(A::AbstractArray{T}, B::AbstractArray{T}, E::AbstractArray{T}, U::AbstractArray{T}, 
                         tspan::AbstractArray{T}, IC::AbstractArray{T}; const_stepsize::Bool=false) where T<:Real
    Xdim = length(IC)
    Tdim = length(tspan)
    state = zeros(Xdim, Tdim)
    state[:, 1] = IC
    state3_jm1 = 0  # preallocate state3_{j-1}

    if const_stepsize
        Δt = tspan[2] - tspan[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:Tdim
            state3 = ⊘(state[:, j-1], state[:, j-1], state[:, j-1])
            if j == 2 
                state[:, j] = ImdtA_inv * (IpdtA * state[:, j-1] + E * state3 * Δt + B * U[:,j-1] * Δt)
            else
                state[:, j] = ImdtA_inv * (IpdtA * state[:, j-1] + E * state3 * 3*Δt/2 - E * state3_jm1 * Δt/2 + B * U[:,j-1] * Δt)
            end
            state3_jm1 = state3
        end
    else
        for j in 2:Tdim
            Δt = tspan[j] - tspan[j-1]
            state3 = ⊘(state[:, j-1], state[:, j-1], state[:, j-1])
            if j == 2 
                state[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * state[:, j-1] + E * state3 * Δt + B * U[:,j-1] * Δt)
            else
                state[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * state[:, j-1] + E * state3 * 3*Δt/2 - E * state3_jm1 * Δt/2 + B * U[:,j-1] * Δt)
            end
            state3_jm1 = state3
        end
    end
    return state
end


end # ChafeeInfante module
