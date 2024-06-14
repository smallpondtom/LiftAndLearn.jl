using LiftAndLearn
using Test

const LnL = LiftAndLearn

# 1D Heat equation setup
Nx = 2^7; dt = 1e-3
heat1d = LnL.Heat1DModel(
    spatial_domain=(0.0, 1.0), time_domain=(0.0, 1.0), Δx=1/Nx, Δt=dt, 
    diffusion_coeffs=range(0.1, 10, 10)
)
Ubc = ones(heat1d.time_dim)

@testset "Forward Euler" begin
    Us = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
    for (i, μ) in enumerate(heat1d.diffusion_coeffs)
        A, B = heat1d.finite_diff_model(heat1d, μ)
        U = LnL.forwardEuler(A,B,Ubc,heat1d.tspan,heat1d.IC)
        Us[i] = U
    end
    @test any(isnan.(Us[2]))
end

@testset "Crank Nicolson" begin
    Uf = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
    r = 10  # order of the reduced form

    for (i, μ) in enumerate(heat1d.diffusion_coeffs)
        A, B = heat1d.finite_diff_model(heat1d, μ)
        Uf[i] = LnL.crankNicolson(A,B,Ubc,heat1d.tspan,heat1d.IC)
    end
    @test !any(isnan.(Uf[2]))
end

@testset "Backward Euler" begin
    Ub = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
    r = 10  # order of the reduced form

    for (i, μ) in enumerate(heat1d.diffusion_coeffs)
        A, B = heat1d.finite_diff_model(heat1d, μ)
        Ub[i] = LnL.backwardEuler(A,B,Ubc,heat1d.tspan,heat1d.IC)
    end
    @test !any(isnan.(Ub[2]))
end
