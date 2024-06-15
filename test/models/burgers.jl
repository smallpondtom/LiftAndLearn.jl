using LiftAndLearn
using Test

LnL = LiftAndLearn

@testset "Burgers models" begin
    # First order Burger's equation setup
    Ω = (0.0, 1.0)
    Nx = 2^7; dt = 1e-4
    burger = LnL.BurgersModel(
        spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
        diffusion_coeffs=0.1, BC=:periodic, conservation_type=:NC
    )

    n = burger.spatial_dim
    μ = burger.diffusion_coeffs[1]
    A, F = burger.finite_diff_model(burger, μ)
    @test size(A) == (n, n)
    @test size(F) == (n, Int(n*(n+1)/2))

    burger.conservation_type = :C
    A, F = burger.finite_diff_model(burger, μ)
    @test size(A) == (n, n)
    @test size(F) == (n, Int(n*(n+1)/2))

    burger.conservation_type = :EP
    A, F = burger.finite_diff_model(burger, μ)
    @test size(A) == (n, n)
    @test size(F) == (n, Int(n*(n+1)/2))
end


