"""
    Gardner equation model
"""
module Gardner

using DocStringExtensions
using LinearAlgebra
using SparseArrays

import ..LiftAndLearn: AbstractModel, vech, ⊘, Operators, makeCubicOp

export GardnerModel


"""
$(TYPEDEF)

Gardner equation model

```math
\\frac{\\partial u}{\\partial t} = -\\alpha\\frac{\\partial^2 u}{\\partial x^3} + \beta u\\frac{\\partial u}{\\partial x} + \\gamma u^2\\frac{\\partial u}{\\partial x}
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
- `params::Union{Real,AbstractArray{<:Real}}`: parameter vector
- `param_dim::Int64`: parameter dimension
- `finite_diff_model::Function`: model using Finite Difference
- `integrate_model::Function`: model integration
"""
mutable struct GardnerModel <: AbstractModel
    # Domains
    spatial_domain::Tuple{Real,Real}  # spatial domain
    time_domain::Tuple{Real,Real}  # temporal domain
    param_domain::Dict{Symbol,Tuple{Real,Real}}  # parameter domain

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
    params::Dict{Symbol,Union{Real,AbstractArray{<:Real}}} # parameters

    # Dimensions
    spatial_dim::Int64  # spatial dimension
    time_dim::Int64  # temporal dimension
    param_dim::Dict{Symbol,Int64} # parameter dimension

    finite_diff_model::Function
    integrate_model::Function
end


"""
$(SIGNATURES)

Constructor for the Gardner equation model.
"""
function GardnerModel(;spatial_domain::Tuple{Real,Real}, time_domain::Tuple{Real,Real}, Δx::Real, Δt::Real, 
                       params::Dict{Symbol,<:Union{Real,AbstractArray{<:Real}}}, BC::Symbol=:dirichlet)
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
    param_dim = Dict([k => length(v) for (k, v) in params])
    param_domain = Dict([k => extrema(v) for (k,v) in params])

    GardnerModel(
        spatial_domain, time_domain, param_domain,
        Δx, Δt, BC, IC, xspan, tspan, params,
        spatial_dim, time_dim, param_dim,
        finite_diff_model, integrate_model
    )
end


"""
$(SIGNATURES)

Finite Difference Model for Gardner equation

## Arguments
- `model::GardnerModel`: Gardner model
- `params::Real`: params including a, b, c

## Returns
- operators
"""
function finite_diff_model(model::GardnerModel, params::Dict)
    if model.BC == :periodic
        return finite_diff_periodic_model(model.spatial_dim, model.Δx, params)
    else
        error("Boundary condition not implemented")
    end
end



"""
    finite_diff_periodic_model(N::Real, Δx::Real, params::Dict) → A, F, E

Generate A, F, E matrices for the Gardner equation for periodic boundary condition (Non-conservative).
"""
function finite_diff_periodic_model(N::Real, Δx::Real, params::Dict)
    # Create A matrix
    α = params[:a]
    β = params[:b]
    γ = params[:c]

    A = spdiagm(
        2 => 0.5 * ones(N - 2),
        1 => -1 * ones(N - 1),
        0 => zeros(N),
        -1 => 1 * ones(N - 1), 
        -2 => -0.5 * ones(N - 2)
    ) * (-α) / Δx^3
    A[1, end-1] = -0.5 * (-α) / Δx^3  # periodic boundary condition
    A[1, end] = (-α) / Δx^3 
    A[2, end] = -0.5 * (-α) / Δx^3
    A[end-1, 1] = 0.5 * (-α) / Δx^3
    A[end, 1] = -(-α) / Δx^3
    A[end, 2] = 0.5 * (-α) / Δx^3

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
    F *= β

    # Create E matrix
    indices = NTuple{4,<:Int}[]
    for i in 2:N-1
        push!(indices, (i, i, i+1, i))
        push!(indices, (i-1, i, i, i))
    end
    push!(indices, (1,1,2,1))
    push!(indices, (N,1,1,1))
    push!(indices, (N-1,N,N,N))
    push!(indices, (N,N,1,N))
    values = γ / 2 / Δx * ones(length(indices))
    E = makeCubicOp(N, indices, values, which_cubic_term='E')

    return A, F, E
end


"""
    integrate_model(A, B, F, E, U, tdata, IC) → states

Semi-Implicit Euler scheme
"""
function integrate_model(A, B, F, E, U, tdata, IC)
    Xdim = length(IC)
    Tdim = length(tdata)
    state = zeros(Xdim, Tdim)
    state[:, 1] = IC

    for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        # state2 = vech(state[:, j-1] * state[:, j-1]')
        state2 = state[:, j-1] ⊘ state[:, j-1]
        state3 = ⊘(state[:, j-1], state[:, j-1], state[:, j-1])
        state[:, j] = (1.0I(Xdim) - Δt * A) \ (state[:, j-1] + F * state2 * Δt + E * state3 * Δt + B * U[j-1] * Δt)
    end
    return state
end


"""
    integrate_model(A, F, E, tdata, IC) → states

Semi-Implicit Euler scheme without control (dispatch)
"""
function integrate_model(A, F, E, tdata, IC)
    Xdim = length(IC)
    Tdim = length(tdata)
    state = zeros(Xdim, Tdim)
    state[:, 1] = IC

    for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        # state2 = vech(state[:, j-1] * state[:, j-1]')
        state2 = state[:, j-1] ⊘ state[:, j-1]
        state3 = ⊘(state[:, j-1], state[:, j-1], state[:, j-1])
        state[:, j] = (1.0I(Xdim) - Δt * A) \ (state[:, j-1] + F * state2 * Δt + E * state3 * Δt)
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
    E = ops.E
    Xdim = length(IC)
    Tdim = length(tdata)
    state = zeros(Xdim, Tdim)
    state[:, 1] = IC

    for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        # state2 = vech(state[:, j-1] * state[:, j-1]')
        state2 = state[:, j-1] ⊘ state[:, j-1]
        state3 = ⊘(state[:, j-1], state[:, j-1], state[:, j-1])
        state[:, j] = (1.0I(Xdim) - Δt * A) \ (state[:, j-1] + F * state2 * Δt + E * state3 * Δt)
    end
    return state
end


end
