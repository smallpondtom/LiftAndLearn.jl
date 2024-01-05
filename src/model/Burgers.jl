"""
    Viscous Burgers' equation model
"""
module Burgers

using DocStringExtensions
using LinearAlgebra
using SparseArrays

export burgers

# Import abstract type Abstract_Models from LiftAndLearn
import ..LiftAndLearn: Abstract_Models, operators

"""
$(TYPEDEF)

Viscous Burgers' equation model

```math
\\frac{\\partial u}{\\partial t} = \\mu\\frac{\\partial^2 u}{\\partial x^2} - u\\frac{\\partial u}{\\partial x}
```

## Fields
- `Omega::Vector{Float64}`: spatial domain
- `T::Vector{Float64}`: temporal domain
- `D::Vector{Float64}`: parameter domain
- `Δx::Float64`: spatial grid size
- `Δt::Float64`: temporal step size
- `IC::VecOrMat{Float64}`: initial condition
- `x::Vector{Float64}`: spatial grid points
- `t::Vector{Float64}`: temporal points
- `μs::Union{Vector{Float64},Float64}`: parameter vector
- `Xdim::Int64`: spatial dimension
- `Tdim::Int64`: temporal dimension
- `Pdim::Int64`: parameter dimension
- `BC::String`: boundary condition
- `generateABFmatrix::Function`: function to generate A, B, F matrices
- `generateMatrix_NC_periodic::Function`: function to generate A, F matrices for the non-energy preserving Burgers' equation. (Non-conservative Periodic boundary condition)
- `generateMatrix_C_periodic::Function`: function to generate A, F matrices for the non-energy preserving Burgers' equation. (conservative periodic boundary condition)
- `generateEPmatrix::Function`: function to generate A, F matrices for the Burgers' equation. (Energy-preserving form)
- `semiImplicitEuler::Function`: function to integrate the system using semi-implicit Euler scheme
"""
mutable struct burgers <: Abstract_Models
    Omega::Vector{Float64}  # spatial domain
    T::Vector{Float64}  # temporal domain
    D::Vector{Float64}  # parameter domain
    Δx::Float64  # spatial grid size
    Δt::Float64  # temporal step size
    IC::VecOrMat{Float64}  # initial condition
    x::Vector{Float64}  # spatial grid points
    t::Vector{Float64}  # temporal points
    μs::Union{Vector{Float64},Float64}  # parameter vector
    Xdim::Int64  # spatial dimension
    Tdim::Int64  # temporal dimension
    Pdim::Int64  # parameter dimension
    BC::String   # boundary condition

    generateABFmatrix::Function
    generateMatrix_NC_periodic::Function
    generateMatrix_C_periodic::Function
    generateEPmatrix::Function
    semiImplicitEuler::Function
end


"""
    burgers(Omega, T, D, Δx, Δt, Pdim, BC) → burgers

Constructor for the Burgers' equation model.
"""
function burgers(Omega, T, D, Δx, Δt, Pdim, BC)
    if BC == "dirichlet"
        x = collect(Omega[1]:Δx:Omega[2])  # include boundary conditions
    elseif BC == "periodic"
        x = collect(Omega[1]:Δx:Omega[2] - Δx)  # exclude boundary conditions
    end
    t = collect(T[1]:Δt:T[2])
    μs = Pdim == 1 ? D[1] : collect(range(D[1], D[2], Pdim))
    Xdim = length(x)
    Tdim = length(t)
    IC = zeros(Xdim, 1)

    burgers(
        Omega, T, D, Δx, Δt, IC, x, t, μs, Xdim, Tdim, Pdim, BC,
        generateABFmatrix, generateMatrix_NC_periodic, generateMatrix_C_periodic,
        generateEPmatrix, semiImplicitEuler
    )
end


"""
    generateABFmatrix(model, μ) → A, B, F

Generate A, B, F matrices for the Burgers' equation.

## Arguments
- `model`: Burgers' equation model
- `μ`: parameter value

## Returns
- `A`: A matrix
- `B`: B matrix
- `F`: F matrix
"""
function generateABFmatrix(model::burgers, μ::Float64)
    N = model.Xdim
    Δx = model.Δx
    Δt = model.Δt

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
    generateMatrix_NC_periodic(model, μ) → A, F

Generate A, F matrices for the non-energy preserving Burgers' equation. (Non-conservative Periodic boundary condition)

## Arguments
- `model`: Burgers' equation model
- `μ`: parameter value

## Returns
- `A`: A matrix
- `F`: F matrix
"""
function generateMatrix_NC_periodic(model::burgers, μ::Float64)
    N = model.Xdim
    Δx = model.Δx

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
    generateMatrix_C_periodic(model, μ) → A, F

Generate A, F matrices for the non-energy preserving Burgers' equation. (conservative periodic boundary condition)

## Arguments
- `model`: Burgers' equation model
- `μ`: parameter value

## Returns
- `A`: A matrix
- `F`: F matrix
"""
function generateMatrix_C_periodic(model::burgers, μ::Float64)
    N = model.Xdim
    Δx = model.Δx

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
    generateEPmatrix(model, μ) → A, F

Generate A, F matrices for the Burgers' equation. (Energy-preserving form)

## Arguments
- `model`: Burgers' equation model
- `μ`: parameter value

## Returns
- `A`: A matrix
- `F`: F matrix
"""
function generateEPmatrix(model::burgers, μ::Float64)
    N = model.Xdim
    Δx = model.Δx

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
    vech(A) → v

Half-vectorization operation

## Arguments
- `A`: matrix to half-vectorize

## Returns
- `v`: half-vectorized form
"""
function vech(A::AbstractMatrix{T}) where {T}
    m = LinearAlgebra.checksquare(A)
    v = Vector{T}(undef, (m * (m + 1)) >> 1)
    k = 0
    for j = 1:m, i = j:m
        @inbounds v[k+=1] = A[i, j]
    end
    return v
end


"""
    semiImplicitEuler(A, B, F, U, tdata, IC) → states

Semi-Implicit Euler scheme

## Arguments
- `A`: linear state operator
- `B`: linear input operator
- `F`: quadratic state operator
- `U`: input data
- `tdata`: time data
- `IC`: initial condtions

## Returns
- `states`: integrated states
"""
function semiImplicitEuler(A, B, F, U, tdata, IC)
    Xdim = length(IC)
    Tdim = length(tdata)
    state = zeros(Xdim, Tdim)
    state[:, 1] = IC

    for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        state2 = vech(state[:, j-1] * state[:, j-1]')
        state[:, j] = (1.0I(Xdim) - Δt * A) \ (state[:, j-1] + F * state2 * Δt + B * U[j-1] * Δt)
    end
    return state
end


"""
    semiImplicitEuler(A, F, tdata, IC) → states

Semi-Implicit Euler scheme without control (dispatch)

## Arguments
- `A`: linear state operator
- `F`: quadratic state operator
- `tdata`: time data
- `IC`: initial condtions

## Returns
- `states`: integrated states
"""
function semiImplicitEuler(A, F, tdata, IC)
    Xdim = length(IC)
    Tdim = length(tdata)
    state = zeros(Xdim, Tdim)
    state[:, 1] = IC

    for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        state2 = vech(state[:, j-1] * state[:, j-1]')
        state[:, j] = (1.0I(Xdim) - Δt * A) \ (state[:, j-1] + F * state2 * Δt)
    end
    return state
end


"""
    semiImplicitEuler(ops, tdata, IC) → states

Semi-Implicit Euler scheme without control (dispatch)

## Arguments
- `ops`: operators
- `tdata`: time data
- `IC`: initial condtions

## Returns
- `states`: integrated states
"""
function semiImplicitEuler(ops, tdata, IC)
    A = ops.A 
    F = ops.F
    Xdim = length(IC)
    Tdim = length(tdata)
    state = zeros(Xdim, Tdim)
    state[:, 1] = IC

    for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        state2 = vech(state[:, j-1] * state[:, j-1]')
        state[:, j] = (1.0I(Xdim) - Δt * A) \ (state[:, j-1] + F * state2 * Δt)
    end
    return state
end


end