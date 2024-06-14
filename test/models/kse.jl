using LiftAndLearn
using Test

LnL = LiftAndLearn

@testset "ks conservative/non-conservative models" begin
    Ω = (0.0, 22.0); dt = 1e-2; L = Ω[2] - Ω[1]; N = 2^8
    KSE = LnL.KuramotoSivashinskyModel(
        spatial_domain=Ω, time_domain=(0.0, 100.0), diffusion_coeffs=1.0,
        Δx=(Ω[2] - 1/N)/N, Δt=dt, conservation_type=:NC
    )

    n = KSE.spatial_dim
    μ = KSE.diffusion_coeffs
    A, F = KSE.finite_diff_model(KSE, μ)
    @test size(A) == (n, n)
    @test size(F) == (n, Int(n*(n+1)/2))

    KSE.conservation_type = :C
    A, F = KSE.finite_diff_model(KSE, μ)
    @test size(A) == (n, n)
    @test size(F) == (n, Int(n*(n+1)/2))
end


@testset "ks pseudo-spectral and spectral-galerkin models" begin
    Ω = (0.0, 22.0); dt = 1e-2; L = Ω[2] - Ω[1]; N = 2^8
    KSE = LnL.KuramotoSivashinskyModel(
        spatial_domain=Ω, time_domain=(0.0, 100.0), diffusion_coeffs=1.0,
        Δx=(Ω[2] - 1/N)/N, Δt=dt, conservation_type=:NC
    )

    DS = 100
    L = KSE.spatial_domain[2] - KSE.spatial_domain[1]

    # Initial condition
    a = 1.0
    b = 0.1
    u0 = a*cos.((2*π*KSE.xspan)/L) + b*cos.((4*π*KSE.xspan)/L) # initial condition version 1


    A, F = KSE.pseudo_spectral_model(KSE, KSE.diffusion_coeffs)
    u_PS, _ = KSE.integrate_model(A, F, KSE.tspan, u0)
    @test size(u_PS) == (KSE.spatial_dim, KSE.time_dim)

    A, F = KSE.elementwise_pseudo_spectral_model(KSE, KSE.diffusion_coeffs)
    u_EWPS, _ = KSE.integrate_model(A, F, KSE.tspan, u0)
    @test size(u_EWPS) == (KSE.spatial_dim, KSE.time_dim)

    A, F = KSE.spectral_galerkin_model(KSE, KSE.diffusion_coeffs)
    u_SG, _ = KSE.integrate_model(A, F, KSE.tspan, u0)
    @test size(u_SG) == (KSE.spatial_dim, KSE.time_dim)
end