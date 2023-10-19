using LinearAlgebra
using SparseArrays

abstract type Abstract_Models end

mutable struct Heat1D <: Abstract_Models
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


function Heat1D(Omega, T, D, Δx, Δt, Pdim)
    x = (Omega[1]:Δx:Omega[2])[2:end-1]
    t = T[1]:Δt:T[2]
    μs = range(D[1], D[2], Pdim)
    Xdim = length(x)
    Tdim = length(t)
    Ubc = ones(Tdim,1)
    IC = zeros(Xdim,1)

    Heat1D(Omega, T, D, Δx, Δt, Ubc, IC, x, t, μs, Xdim, Tdim, Pdim, generateABmatrix)
end


function generateABmatrix(N, μ, Δx)
    A = diagm(0 => (-2)*ones(N), 1 => ones(N-1), -1 => ones(N-1)) * μ / Δx^2
    B = [1; zeros(N-2,1); 1] * μ / Δx^2   #! Fixed this to generalize input u
    return A, B
end
