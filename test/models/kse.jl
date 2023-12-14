using LiftAndLearn
using Test

LnL = LiftAndLearn

@testset "ks conservative/non-conservative models" begin
    KSE = LnL.ks(
        [0.0, 22.0], [0.0, 100.0], [1.0, 1.0],
        256, 0.01, 1, "nc"
    )

    n = KSE.Xdim
    μ = KSE.μs[1]
    A, F = KSE.model_FD(KSE, μ)
    @test size(A) == (n, n)
    @test size(F) == (n, Int(n*(n+1)/2))

    KSE = LnL.ks(
        [0.0, 22.0], [0.0, 100.0], [1.0, 1.0],
        256, 0.01, 1, "c"
    )
    n = KSE.Xdim
    μ = KSE.μs[1]
    A, F = KSE.model_FD(KSE, μ)
    @test size(A) == (n, n)
    @test size(F) == (n, Int(n*(n+1)/2))
end


@testset "ks pseudo-spectral and spectral-galerkin models" begin
    # Settings for the KS equation
    KSE = LnL.ks(
        [0.0, 100.0], [0.0, 300.0], [1.0, 1.0],
        256, 0.01, 1, "ep"
    )

    DS = 100
    L = KSE.Omega[2]

    # Initial condition
    a = 1.0
    b = 0.1
    u0 = a*cos.((2*π*KSE.x)/L) + b*cos.((4*π*KSE.x)/L) # initial condition version 1


    A, F = KSE.model_PS(KSE, KSE.μs[1])
    u_PS, _ = KSE.integrate_PS(A, F, KSE.t, u0)
    @test size(u_PS) == (KSE.Xdim, KSE.Tdim)

    A, F = KSE.model_SG(KSE, KSE.μs[1])
    u_SG, _ = KSE.integrate_SG(A, F, KSE.t, u0)
    @test size(u_SG) == (KSE.Xdim, KSE.Tdim)
end