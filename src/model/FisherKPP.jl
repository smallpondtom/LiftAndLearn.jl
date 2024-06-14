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
    spatial_domain::Tuple{Real,Real}  # spatial domain
    time_domain::Tuple{Real,Real}  # temporal domain
    diffusion_coeff_domain::Tuple{Real,Real}  # parameter domain (diffusion coeff)
    growth_rate_domain::Tuple{Real,Real}  # parameter domain (growth rate)

    # Discritization grid
    Δx::Real  # spatial grid size
    Δt::Real  # temporal step size
    xspan::Vector{<:Real}  # spatial grid points
    tspan::Vector{<:Real}  # temporal points
    spatial_dim::Int  # spatial dimension
    time_dim::Int  # temporal dimension

    # Parameters
    diffusion_coeffs::Union{AbstractArray{<:Real},Real}  # diffusion coefficient
    growth_rates::Union{AbstractArray{<:Real},Real}  # growth rate
    param_dim::Dict{Symbol,<:Int}  # parameter dimension

    # Initial condition
    IC::AbstractArray{<:Real}  # initial condition

    # Boundary condition
    BC::Symbol  # boundary condition

    # Functions
    finite_diff_model::Function  # model using Finite Difference
    integrate_model::Function # integrator using Crank-Nicholson (linear) Explicit (nonlinear) method
end


function fisherkpp(;spatial_domain::Tuple{Real,Real}, time_domain::Tuple{Real,Real}, Δx::Real, Δt::Real, 
                    diffusion_coeffs::Union{AbstractArray{<:Real},Real}, growth_rates::Union{AbstractArray{<:Real},Real}, 
                    BC::Symbol=:periodic)
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
    param_dim = Dict(:diffusion_coeff => length(diffusion_coeffs), :growth_rate => length(growth_rates))
    diffusion_coeff_domain = extrema(diffusion_coeffs)
    growth_rate_domain = extrema(growth_rates)

    fisherkpp(
        spatial_domain, time_domain, diffusion_coeff_domain, growth_rate_domain,
        Δx, Δt, xspan, tspan, spatial_dim, time_dim,
        diffusion_coeffs, growth_rates, param_dim, IC, BC,
        finite_diff_model, integrate_model
    )
end


"""
    finite_diff_model(model::fisherkpp, D::Real, r::Real)

Create the matrices A (linear operator) and F (quadratic operator) for the Fisher-KPP model. For different
boundary conditions, the matrices are created differently.

## Arguments
- `model::fisherkpp`: Fisher-KPP model
- `D::Real`: diffusion coefficients
- `r::Real`: growth rates

"""
function finite_diff_model(model::fisherkpp, D::Real, r::Real)
    if model.BC == :periodic
        return finite_diff_periodic_model(model, D, r)
    elseif model.BC == :mixed
        return finite_diff_mixed_model(model, D, r)
    end
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
function finite_diff_periodic_model(model::fisherkpp, D::Real, r::Real)
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
    finite_diff_mixed_model(model::fisherkpp, D::Real, r::Real)

Create the matrices A (linear operator), B (input operator), and F (quadratic operator) for the Fisher-KPP 
model using the mixed boundary condition. If the spatial domain is [0,1], then we assume u(0,t) to be 
homogeneous dirichlet boundary condition and u(1,t) to be Neumann boundary condition of some function h(t).

## Arguments
- `model::fisherkpp`: Fisher-KPP model
- `D::Real`: diffusion coefficients
- `r::Real`: growth rates

## Returns
- `A::SparseMatrixCSC{Float64,Int}`: linear operator
- `B::SparseMatrixCSC{Float64,Int}`: input operator
- `F::SparseMatrixCSC{Float64,Int}`: quadratic operator
"""
function finite_diff_mixed_model(model::fisherkpp, D::Real, r::Real)
    N = model.spatial_dim
    Δx = model.Δx

    # Create A matrix
    A = spdiagm(0 => (r-2*D/Δx^2) * ones(N), 1 => (D/Δx^2) * ones(N - 1), -1 => (D/Δx^2) * ones(N - 1))
    A[end,end] = r - D/Δx^2  # influence of Neumann boundary condition

    # Create F matrix
    S = Int(N * (N + 1) / 2)
    ii = 1:N  # row index
    jj = Int.([N*(N+1)/2 - (N-m)*(N-m+1)/2 - (N-m) for m in 1:N])  # col index where the xi^2 term is
    vv = ones(N);
    F = sparse(ii,jj,vv,N,S)

    # Create B matrix
    B = spzeros(N,2)
    B[1,1] = D / Δx^2  # from Dirichlet boundary condition
    B[end,2] = D / Δx  # from Neumann boundary condition

    return A, B, F
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
function integrate_model(A::AbstractArray{<:Real}, F::AbstractArray{<:Real}, tspan::AbstractArray{<:Real}, 
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
            state2 = state[:, j-1] ⊘ state[:, j-1]
            state[:, j] = ImdtA_inv * (IpdtA * state[:, j-1] + F * state2 * Δt)
        end
    else
        for j in 2:Tdim
            Δt = tspan[j] - tspan[j-1]
            state2 = state[:, j-1] ⊘ state[:, j-1]
            state[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * state[:, j-1] + F * state2 * Δt)
        end
    end
    return state
end


"""
    integrate_model(A::AbstractArray{T}, B::AbstractArray{T}, F::AbstractArray{T}, U::AbstractArray{T}, 
                    tspan::AbstractArray{T}, IC::AbstractArray{T}; const_stepsize::Bool=false) where T<:Real

Integrate the Fisher-KPP model using the Crank-Nicholson (linear) Adam-Bashforth (nonlinear) method.

## Arguments
- `A::AbstractArray{T}`: linear operator
- `B::AbstractArray{T}`: input operator
- `F::AbstractArray{T}`: quadratic operator
- `U::AbstractArray{T}`: input
- `tspan::AbstractArray{T}`: time span
- `IC::AbstractArray{T}`: initial condition
- `const_stepsize::Bool=false`: constant time step size

## Returns
- `state::AbstractArray{T}`: solution
"""
function integrate_model(A::AbstractArray{T}, B::AbstractArray{T}, F::AbstractArray{T}, U::AbstractArray{T}, 
                         tspan::AbstractArray{T}, IC::AbstractArray{T}; const_stepsize::Bool=false) where T<:Real
    Xdim = length(IC)
    Tdim = length(tspan)
    state = zeros(Xdim, Tdim)
    state[:, 1] = IC
    state2_jm1 = 0  # preallocate state2_{j-1}

    # Make input matrix a tall matrix
    U = reshape(U, (Tdim - 1, 2))

    if const_stepsize
        Δt = tspan[2] - tspan[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(I - Δt/2 * A) \ I
        IpdtA = (I + Δt/2 * A)
        for j in 2:Tdim
            state2 = state[:, j-1] ⊘ state[:, j-1]
            if j == 2 
                state[:, j] = ImdtA_inv * (IpdtA * state[:, j-1] + F * state2 * Δt + B * U[j-1,:] * Δt)
            else
                state[:, j] = ImdtA_inv * (IpdtA * state[:, j-1] + F * state2 * 3*Δt/2 - F * state2_jm1 * Δt/2 + B * U[j-1,:] * Δt)
            end
            state2_jm1 = state2
        end
    else
        for j in 2:Tdim
            Δt = tspan[j] - tspan[j-1]
            state2 = state[:, j-1] ⊘ state[:, j-1]
            if j == 2 
                state[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * state[:, j-1] + F * state2 * Δt + B * U[j-1,:] * Δt)
            else
                state[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * state[:, j-1] + F * state2 * 3*Δt/2 - F * state2_jm1 * Δt/2 + B * U[j-1,:] * Δt)
            end
            state2_jm1 = state2
        end
    end
    return state
end


# function integrate_model(A::AbstractArray{T}, B::AbstractArray{T}, F::AbstractArray{T}, U::AbstractArray{T}, 
#                          tspan::AbstractArray{T}, IC::AbstractArray{T}; const_stepsize::Bool=false) where T<:Real
#     Xdim = length(IC)
#     Tdim = length(tspan)
#     state = zeros(Xdim, Tdim)
#     state[:, 1] = IC

#     # Make input matrix a tall matrix
#     U = reshape(U, (Tdim - 1, 2))

#     if const_stepsize
#         Δt = tspan[2] - tspan[1]  # assuming a constant time step size
#         ImdtA_inv = Matrix(I - Δt/2 * A) \ I
#         IpdtA = (I + Δt/2 * A)
#         for j in 2:Tdim
#             state2 = state[:, j-1] ⊘ state[:, j-1]
#             state[:, j] = ImdtA_inv * (IpdtA * state[:, j-1] + F * state2 * Δt + B * U[j-1,:] * Δt)
#         end
#     else
#         for j in 2:Tdim
#             Δt = tspan[j] - tspan[j-1]
#             state2 = state[:, j-1] ⊘ state[:, j-1]
#             state[:, j] = (I - Δt/2 * A) \ ((I + Δt/2 * A) * state[:, j-1] + F * state2 * Δt + B * U[j-1,:] * Δt)
#         end
#     end
#     return state
# end

end # FisherKPP module