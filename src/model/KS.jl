using LinearAlgebra
using SparseArrays


mutable struct KS
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

    model_FFT::Function  # model using Fast Fourier Transform
    model_FT::Function  # model using Fourier Transform
    model_FD::Function  # model using Finite Difference
    integrator::Function  # integrator using Crank-Nicholson Adams-Bashforth method
end

function KS(Omega, T, D, Δx, Δt, Pdim)
    x = collect(Omega[1]:Δx:Omega[2]-Δx)  # assuming a periodic boundary condition
    t = collect(T[1]:Δt:T[2])
    μs = Pdim == 1 ? D[1] : collect(range(D[1], D[2], Pdim))
    Xdim = length(x)
    Tdim = length(t)
    IC = zeros(Xdim, 1)

    KS(
        Omega, T, D, Δx, Δt, IC, x, t, μs, Xdim, Tdim, Pdim,
        model_FFT, model_FT, model_FD, integrator
    )
end


"""
    Generate A, F matrices for the Kuramoto-Sivashinsky equation using the Finite Difference method.

    # Arguments
    - `model`: Kuramoto-Sivashinsky equation model
    - `μ`: parameter value

    # Return
    - `A`: A matrix
    - `F`: F matrix
"""
function model_FD(model::KS, μ::Float64)
    N = model.Xdim
    Δx = model.Δx
    Δt = model.Δt

    # Create A matrix
    ζ = 2/Δx^2 - 6*μ/Δx^4
    η = 4*μ/Δx^4 - 1/Δx^2
    ϵ = -μ/Δx^4

    A = diagm(
        0 => ζ * ones(N), 
        1 => η * ones(N - 1), -1 => η * ones(N - 1),
        2 => ϵ * ones(N - 2), -2 => ϵ * ones(N - 2)
    )
    # For the periodicity for the first and final few indices
    A[1, end-1:end] = [ϵ, η]
    A[2, end] = ϵ
    A[end-1, 1] = ϵ
    A[end, 1:2] = [η, ϵ]
    
    # Create F matrix
    S = Int(N * (N + 1) / 2)
    if N >= 3
        Fval = repeat([1.0, -1.0], outer=N - 2)
        row_i = repeat(2:(N-1), inner=2)
        seq = Int.([2 + (N + 1) * (x - 1) - x * (x - 1) / 2 for x in 1:(N-1)])
        col_i = vcat(seq[1], repeat(seq[2:end-1], inner=2), seq[end])
        F = sparse(row_i, col_i, Fval, N, S) / 2 / Δx

        # For the periodicity for the first and final indices
        F[1, N] = 1 / 2 / Δx
        F[N, N] = -1 / 2 / Δx
    else
        F = zeros(N, S)
    end

    return A, F
end
