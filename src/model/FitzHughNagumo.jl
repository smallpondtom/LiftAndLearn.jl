"""
    FitzHugh-Nagumo PDE model
"""
module FitzHughNagumo

using DocStringExtensions
using LinearAlgebra
using SparseArrays

import ..LiftAndLearn: AbstractModel

export FitzHughNagumoModel


"""
$(TYPEDEF)

FitzHugh-Nagumo PDE model
    
```math
\\begin{aligned}
\\frac{\\partial u}{\\partial t} &=  \\epsilon^2\\frac{\\partial^2 u}{\\partial x^2} + u(u-0.1)(1-u) - v + g \\\\
\\frac{\\partial v}{\\partial t} &= hu + \\gamma v + g
\\end{aligned}
```

where ``u`` and ``v`` are the state variables, ``g`` is the control input, and ``h``, ``\\gamma``, and ``\\epsilon`` are the parameters.
Specifically, for this problem we assume the control input to begin
```math
g(t) = \\alpha t^3 \\exp(-\\beta t)
```
where ``\\alpha`` and ``\\beta`` are the parameters that are going to be varied for training.

## Fields
- `spatial_domain::Tuple{Real,Real}`: spatial domain
- `time_domain::Tuple{Real,Real}`: temporal domain
- `alpha_input_param_domain::Tuple{Real,Real}`: parameter domain
- `beta_input_param_domain::Tuple{Real,Real}`: parameter domain
- `Δx::Real`: spatial grid size
- `Δt::Real`: temporal step size
- `BC::Symbol`: boundary condition
- `IC::Array{Float64}`: initial condition
- `IC_lift::Array{Float64}`: initial condition
- `xspan::Vector{Float64}`: spatial grid points
- `tspan::Vector{Float64}`: temporal points
- `alpha_input_params::Union{Real,AbstractArray{<:Real}}`: input parameter vector
- `beta_input_params::Union{Real,AbstractArray{<:Real}}`: input parameter vector
- `spatial_dim::Int64`: spatial dimension
- `time_dim::Int64`: temporal dimension
- `param_dim::Dict{Symbol,<:Int}`: parameter dimension
- `full_order_model::Function`: full order model
- `lifted_finite_diff_model::Function`: lifted finite difference model
"""
mutable struct FitzHughNagumoModel <: AbstractModel
    # Domains
    spatial_domain::Tuple{Real,Real}  # spatial domain
    time_domain::Tuple{Real,Real}  # temporal domain
    alpha_input_param_domain::Tuple{Real,Real}  # parameter domain
    beta_input_param_domain::Tuple{Real,Real}  # parameter domain

    # Discritization grid
    Δx::Real  # spatial grid size
    Δt::Real  # temporal step size

    # Boundary condition
    BC::Symbol  # boundary condition

    # Initial conditino
    IC::Array{Float64}  # initial condition
    IC_lift::Array{Float64}  # initial condition

    # grid points
    xspan::Vector{Float64}  # spatial grid points
    tspan::Vector{Float64}  # temporal points
    alpha_input_params::Union{Real,AbstractArray{<:Real}}  # input parameter vector
    beta_input_params::Union{Real,AbstractArray{<:Real}}  # input parameter vector

    # Dimensions
    spatial_dim::Int64  # spatial dimension
    time_dim::Int64  # temporal dimension
    param_dim::Dict{Symbol,<:Int}  # parameter dimension

    full_order_model::Function
    lifted_finite_diff_model::Function
end


"""
    FitzHughNagumoModel(;spatial_domain::Tuple{Real,Real}, time_domain::Tuple{Real,Real}, Δx::Real, Δt::Real, 
                    alpha_input_params::Union{AbstractArray{<:Real},Real}, beta_input_params::Union{AbstractArray{<:Real},Real}, 
                    BC::Symbol=:neumann) → FitzHughNagumoModel

Constructor FitzHugh-Nagumo PDE model
"""
function FitzHughNagumoModel(;spatial_domain::Tuple{Real,Real}, time_domain::Tuple{Real,Real}, Δx::Real, Δt::Real, 
                    alpha_input_params::Union{AbstractArray{<:Real},Real}, beta_input_params::Union{AbstractArray{<:Real},Real}, 
                    BC::Symbol=:neumann)
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
    IC = zeros(spatial_dim*2)
    IC_lift = zeros(spatial_dim*3)

    # Parameter dimensions or number of parameters 
    param_dim = Dict(:alpha => length(alpha_input_params), :beta => length(beta_input_params))
    alpha_input_param_domain = extrema(alpha_input_params)
    beta_input_param_domain = extrema(beta_input_params)

    FitzHughNagumoModel(
        spatial_domain, time_domain, 
        alpha_input_param_domain, beta_input_param_domain,
        Δx, Δt, BC, IC, IC_lift, 
        xspan, tspan, alpha_input_params, beta_input_params,
        spatial_dim, time_dim, param_dim,
        full_order_model, lifted_finite_diff_model
    )
end


"""
    full_order_model(k, l) → A, B, C, K, f

Create the full order operators with the nonlinear operator expressed as f(x). 

## Arguments
- `k::Int64`: number of spatial grid points
- `l::Float64`: spatial domain length

## Returns
- `A::SparseMatrixCSC{Float64,Int64}`: A matrix
- `B::SparseMatrixCSC{Float64,Int64}`: B matrix
- `C::SparseMatrixCSC{Float64,Int64}`: C matrix
- `K::SparseMatrixCSC{Float64,Int64}`: K matrix
- `f::Function`: nonlinear operator
"""
function full_order_model(k, l)
    h = l / (k - 1)
    Alift = spzeros(3 * k, 3 * k)
    Blift = spzeros(3 * k, 2)
    gamma = 0.015
    R = 0.5
    c = 2
    g = 0.05

    E = sparse(1.0I, 3 * k, 3 * k)
    for i in 1:k
        E[i, i] = gamma
        E[i+2*k, i+2*k] = gamma
    end

    # Left boundary
    Alift[1, 1] = -gamma^2 / h^2 - 0.1
    Alift[1, 2] = gamma^2 / h^2
    Alift[1, k+1] = -1
    # Alift[1,2*k+1] = 1.1

    Alift[k+1, 1] = R
    Alift[k+1, k+1] = -c

    Alift[2*k+1, 1] = 2 * g
    Alift[2*k+1, 2*k+1] = -2 * gamma^2 / h^2 - 0.2

    # Right boundary 
    Alift[k, k-1] = gamma^2 / h^2
    Alift[k, k] = -gamma^2 / h^2 - 0.1
    Alift[k, 2*k] = -1
    # Alift[k,3*k] = 1.1

    Alift[2*k, k] = R
    Alift[2*k, 2*k] = -c

    Alift[3*k, 3*k] = -2 * gamma^2 / h^2 - 0.2
    Alift[3*k, k] = 2 * g

    # Inner points
    for i in 2:k-1
        Alift[i, i-1] = gamma^2 / h^2
        Alift[i, i] = -2 * gamma^2 / h^2 - 0.1
        Alift[i, i+1] = gamma^2 / h^2
        Alift[i, i+k] = -1

        Alift[i+k, i] = R
        Alift[i+k, i+k] = -c

        Alift[i+2*k, i] = 2 * g
        Alift[i+2*k, 1+2*k] = -4 * gamma^2 / h^2 - 0.2
    end

    # B matrix
    Blift[1, 1] = gamma^2 / h
    # Blift[:,2] = g
    # NOTE: The second column of the input matrix B corresponds to the constant
    # terms of the FHN PDE.
    Blift[1:2*k, 2] .= g

    Atmp = E \ Alift
    Btmp = E \ Blift

    # A and B matrix
    A = Atmp[1:2*k, 1:2*k]
    B = Btmp[1:2*k, 1]
    K = Btmp[1:2*k, 2]

    # C matrix
    Clift = spzeros(2, 3 * k)
    Clift[1, 1] = 1
    Clift[2, 1+k] = 1
    C = Clift[:, 1:2*k]

    f = (x,u) -> [-x[1:k, :] .^ 3 + 1.1 * x[1:k, :] .^ 2; spzeros(k, size(x, 2))] / gamma

    return A, B, C, K, f
end


"""
    lifted_finite_diff_model(k, l) → A, B, C, H, N, K

Generate the full order operators used for the intrusive model operators

## Arguments
- `k::Int64`: number of spatial grid points
- `l::Float64`: spatial domain length

## Returns
- `A::SparseMatrixCSC{Float64,Int64}`: A matrix
- `B::SparseMatrixCSC{Float64,Int64}`: B matrix
- `C::SparseMatrixCSC{Float64,Int64}`: C matrix
- `H::SparseMatrixCSC{Float64,Int64}`: H matrix
- `N::SparseMatrixCSC{Float64,Int64}`: N matrix
- `K::SparseMatrixCSC{Float64,Int64}`: K matrix

"""
function lifted_finite_diff_model(k, l)
    h = l / (k - 1)
    E = sparse(1.0I, 3 * k, 3 * k)
    A = spzeros(3 * k, 3 * k)
    H = spzeros(3 * k, 9 * k^2)
    N = spzeros(3 * k, 3 * k)
    B = spzeros(3 * k, 2)

    gamma = 0.015
    R = 0.5
    c = 2
    g = 0.05

    for i in 1:k
        E[i, i] = gamma
        E[i+2*k, i+2*k] = gamma
    end

    # Left boundary
    A[1, 1] = -gamma^2 / h^2 - 0.1
    A[1, 2] = gamma^2 / h^2
    A[1, k+1] = -1

    A[k+1, 1] = R
    A[k+1, k+1] = -c

    A[2*k+1, 1] = 2 * g
    A[2*k+1, 2*k+1] = -2 * gamma^2 / h^2 - 0.2

    # Right Boundary
    A[k, k-1] = gamma^2 / h^2
    A[k, k] = -gamma^2 / h^2 - 0.1
    A[k, 2*k] = -1

    A[2*k, k] = R
    A[2*k, 2*k] = -c

    A[3*k, 3*k] = -2 * gamma^2 / h^2 - 0.2
    A[3*k, k] = 2 * g

    # inner points
    for i = 2:k-1
        A[i, i-1] = gamma^2 / h^2
        A[i, i] = -2 * gamma^2 / h^2 - 0.1
        A[i, i+1] = gamma^2 / h^2
        A[i, i+k] = -1

        A[i+k, i] = R
        A[i+k, i+k] = -c

        A[i+2*k, i] = 2 * g
        A[i+2*k, i+2*k] = -4 * gamma^2 / h^2 - 0.2
    end

    # left boundary
    H[1, 1] = 1.1
    H[1, 2*k+1] = -0.5
    H[1, 2*k*3*k+1] = -0.5

    H[2*k+1, 2] = gamma^2 / h^2
    H[2*k+1, 3*k+1] = gamma^2 / h^2
    H[2*k+1, 2*k+1] = 1.1
    H[2*k+1, 2*k*3*k+1] = 1.1
    H[2*k+1, 2*k*3*k+2*k+1] = -2
    H[2*k+1, k+1] = -1
    H[2*k+1, k*3*k+1] = -1

    # right boundary
    H[k, 3*k*(k-1)+k] = 1.1

    H[k, (k-1)*3*k+3*k] = -0.5
    H[k, (3*k-1)*3*k+k] = -0.5

    H[3*k, (k-2)*3*k+k] = gamma^2 / h^2
    H[3*k, (k-1)*3*k+k-1] = gamma^2 / h^2
    H[3*k, (k-1)*3*k+3*k] = 1.1
    H[3*k, (3*k-1)*3*k+k] = 1.1
    H[3*k, 9*k^2] = -2
    H[3*k, (k-1)*3*k+2*k] = -1
    H[3*k, (2*k-1)*3*k+k] = -1

    # inner points
    for i = 2:k-1
        H[i, (i-1)*3*k+i] = 1.1
        H[i, (i-1)*3*k+2*k+i] = -0.5
        H[i, (i+2*k-1)*3*k+i] = -0.5

        H[i+2*k, (i-1)*3*k+i+1] = gamma^2 / h^2
        H[i+2*k, i*3*k+i] = gamma^2 / h^2
        H[i+2*k, (i-1)*3*k+i-1] = gamma^2 / h^2
        H[i+2*k, (i-2)*3*k+i] = gamma^2 / h^2
        H[i+2*k, (i-1)*3*k+i+2*k] = 1.1
        H[i+2*k, (i-1+2*k)*3*k+i] = 1.1
        H[i+2*k, (i+2*k-1)*3*k+i+2*k] = -2
        H[i+2*k, (i-1)*3*k+k+i] = -1
        H[i+2*k, (i+k-1)*3*k+i] = -1
    end

    N[2*k+1, 1] = 2 * gamma^2 / h

    B[1, 1] = gamma^2 / h
    # B[:,2] = g
    B[1:2*k, 2] .= g

    C = spzeros(2, 3 * k)
    C[1, 1] = 1
    C[2, 1+k] = 1

    A = E \ A
    tmp = E \ B
    B = tmp[:,1]
    K = tmp[:,2]
    H = E \ H
    N = E \ N

    # NOTE: (BELOW) making the sparse H matrix symmetric
    # n = 3 * k
    # Htensor = ndSparse(H, n)
    # H2 = reshape(permutedims(Htensor, [2, 1, 3]), (n, n^2))
    # H3 = reshape(permutedims(Htensor, [3, 1, 2]), (n, n^2))
    # symH2 = 0.5 * (H2 + H3)
    # symHtensor2 = reshape(symH2, (n, n, n))
    # symHtensor = permutedims(symHtensor2, [2 1 3])
    # symH = reshape(symHtensor, (n, n^2))
    # H = sparse(symH)

    return A, B, C, H, N, K
end

end