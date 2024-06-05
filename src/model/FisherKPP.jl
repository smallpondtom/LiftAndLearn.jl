"""
    Fisher Kolmogorov-Petrovsky-Piskunov equation PDE model
"""
module FisherKPP

using DocStringExtensions
using LinearAlgebra
using SparseArrays

import ..LiftAndLearn: AbstractModel, vech, ⊘, operators, elimat

export fisherkpp


"""
$(TYPEDEF)

Fisher Kolmogorov-Petrovsky-Piskunov equation (Fisher-KPP) model is a reaction-diffusion equation or
logistic diffusion process in population dynamics. The model is given by the following PDE: 
    
```math
\\frac{\\partial u}{\\partial t} =  D\\frac{\\partial^2 u}{\\partial x^2} + ru(1-u)
```

where ``u`` is the state variable, ``D`` is the diffusion coefficient, and ``r`` is the growth rate.

## Fields
"""
mutable struct fisherkpp <: AbstractModel
    # Domains
    spatial_domain::Tuple{<:Real}  # spatial domain
    time_domain::Tuple{<:Real}  # temporal domain
    diffusion_coeff_domain::Tuple{<:Real}  # parameter domain (diffusion coeff)
    growth_rate_domain::Tuple{<:Real}  # parameter domain (growth rate)

    # Discritization grid
    num_of_space_gp::Real  # number of spatial grid points
    Δx::Real  # spatial grid size
    Δt::Real  # temporal step size
    xspan::Vector{<:Real}  # spatial grid points
    tspan::Vector{<:Real}  # temporal points
    spatial_dim::Int  # spatial dimension
    time_dim::Int  # temporal dimension

    # Parameters
    diffusion_coeffs::Union{Vector{<:Real},Real}  # diffusion coefficient
    growth_rates::Union{Vector{<:Real},Real}  # growth rate
    param_dim::Dict{:Symbol,<:Int}  # parameter dimension

    # Initial condition
    IC::AbstractArray{<:Real}  # initial condition

    # Functions
    finite_diff_model::Function  # model using Finite Difference
    integrate_model::Function # integrator using Crank-Nicholson (linear) Explicit (nonlinear) method
end


function fisherkpp(;spatial_domain, time_domain, num_of_space_gp, Δt, diffusion_coeffs, growth_rates)
    # Discritization grid info
    Δx = (spatial_domain[2] - spatial_domain[1]) / num_of_space_gp  # spatial grid size
    xspan = collect(spatial_domain[1]:Δx:spatial_domain[2]-Δx)  # assuming a periodic boundary condition
    tspan = collect(time_domain[1]:Δt:time_domain[2])
    spatial_dim = length(xspan)
    time_dim = length(tspan)

    # Parameter dimensions or number of parameters 
    param_dim = Dict(:diffusion_coeff => length(diffusion_coeffs), :growth_rate => length(growth_rates))
    diffusion_coeff_domain = extrema(diffusion_coeffs)
    growth_rate_domain = extrema(growth_rates)

    # Initial condition
    IC = zeros(spatial_dim)

    fisherkpp(
        spatial_domain, time_domain, diffusion_coeff_domain, growth_rate_domain,
        num_of_space_gp, Δx, Δt, xspan, tspan, spatial_dim, time_dim,
        diffusion_coeffs, growth_rates, param_dim, IC,
        finite_diff_model, integrate_model
    )
end


"""
    finite_diff_model(model::fisherkpp, D::Real, r::Real)

Create the matrices A (linear operator) and F (quadratic operator) for the Fisher-KPP model.

## Arguments
- `model::fisherkpp`: Fisher-KPP model
- `D::Real`: diffusion coefficients
- `r::Real`: growth rates

## Returns
- `A::SparseMatrixCSC{Float64,Int}`: linear operator
- `F::SparseMatrixCSC{Float64,Int}`: quadratic operator
"""
function finite_diff_model(model::fisherkpp, D::Real, r::Real)
    N = model.spatial_dim
    Δx = model.Δx

    # Create A matrix
    A = spdiagm(0 => (r-2*D/Δx^2) * ones(N), 1 => (D/Δx^2) * ones(N - 1), -1 => (D/Δx^2) * ones(N - 1))
    A[1, end] = D / Δx^2  # periodic boundary condition
    A[end, 1] = D / Δx^2  

    # Create F matrix
    S = Int(N * (N + 1) / 2)
    ii = 1:N  # row index
    jj = Int.([N*(N+1)/2 - (N-m)*(N-m+1)/2 - (N-m) for m in 1:N])  # col index where the xi^2 term is
    vv = ones(N);
    F = sparse(ii,jj,vv,N,S)

    return A, F
end


"""
    integrate_model(A::AbstractArray{<:Real}, F::AbstractArray{<:Real}, IC::AbstractArray{<:Real}, 
                    tspan::AbstractArray{<:Real}; const_stepsize::Bool=false)
    
Integrate the Fisher-KPP model using the Crank-Nicholson (linear) Explicit (nonlinear) method.

## Arguments
- `A::AbstractArray{<:Real}`: linear operator
- `F::AbstractArray{<:Real}`: quadratic operator
- `IC::AbstractArray{<:Real}`: initial condition
- `tspan::AbstractArray{<:Real}`: time span
- `const_stepsize::Bool=false`: constant time step size

## Returns
- `u::AbstractArray{<:Real}`: solution
"""
function integrate_model(A::AbstractArray{<:Real}, F::AbstractArray{<:Real}, IC::AbstractArray{<:Real}, 
                         tspan::AbstractArray{<:Real}; const_stepsize::Bool=false)
    Xdim = length(IC)
    Tdim = length(tspan)
    u = zeros(Xdim, Tdim)
    u[:, 1] = IC

    if const_stepsize
        Δt = tspan[2] - tspan[1]  # assuming a constant time step size
        ImdtA_inv = (I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:Tdim
            u2 = u[:, j-1] ⊘ u[:, j-1]
            u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * Δt)
        end
    else
        for j in 2:Tdim
            Δt = tspan[j] - tspan[j-1]
            u2 = u[:, j-1] ⊘ u[:, j-1]
            u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + F * u2 * Δt)
        end
    end
    return u
end


# function integrate_model(A::AbstractArray{<:Real}, F::AbstractArray{<:Real}, IC::AbstractArray{<:Real}, 
#                          tspan::AbstractArray{<:Real}; const_stepsize::Bool=false)
#     Xdim = length(IC)
#     Tdim = length(tspan)
#     u = zeros(Xdim, Tdim)
#     u[:, 1] = IC
#     u2_jm1 = 0  # preallocate u2_{j-1}

#     if const_stepsize
#         Δt = tspan[2] - tspan[1]  # assuming a constant time step size
#         ImdtA_inv = (I - Δt/2 * A) \ I
#         IpdtA = (I + Δt/2 * A)
#         for j in 2:Tdim
#             u2 = u[:, j-1] ⊘ u[:, j-1]
#             if j == 2 
#                 u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * Δt)
#             else
#                 u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * 3*Δt/2 - F * u2_jm1 * Δt/2)
#             end
#             u2_jm1 = u2
#         end
#     else
#         for j in 2:Tdim
#             Δt = tspan[j] - tspan[j-1]
#             u2 = u[:, j-1] ⊘ u[:, j-1]
#             if j == 2 
#                 u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + F * u2 * Δt)
#             else
#                 u[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * u[:, j-1] + F * u2 * 3*Δt/2 - F * u2_jm1 * Δt/2)
#             end
#             u2_jm1 = u2
#         end
#     end
#     return u
# end