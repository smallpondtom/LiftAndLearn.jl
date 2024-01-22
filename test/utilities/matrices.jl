using LiftAndLearn 
using Test
using Random

const LnL = LiftAndLearn

@testset "Duplication Matrix" begin
    n = 2
    D = LnL.dupmat(n)
    @test all(D .== [1 0 0; 0 1 0; 0 1 0; 0 0 1])
end

@testset "Elimination Matrix" begin
    n = 2
    L = LnL.elimat(n)
    @test all(L .== [1 0 0 0; 0 1 0 0; 0 0 0 1])
end

@testset "Symmetric Elimination Matrix" begin
    n = 2
    L = LnL.elimat(n) * LnL.nommat(n)
    @test all(L .== [1 0 0 0; 0 0.5 0.5 0; 0 0 0 1])
end

@testset "Commutation matrix" begin
    n = 2
    K = LnL.commat(n)
    @test all(K .== [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1])
end


@testset "vech" begin
    A = [1 2; 3 4]
    a = [1; 3; 4]
    @test all(a .== LnL.vech(A))
end


@testset "inverse vectorization" begin
    A = [1 2; 3 4]
    a = vec(A)
    @test all(A .== LnL.invec(a, 2, 2))
end


@testset "Quadratic matrix conversion" begin    # Settings for the KS equation
    KSE = LnL.ks(
        [0.0, 22.0], [0.0, 100.0], [1.0, 1.0],
        32, 0.01, 1, "ep"
    )
    _, F = KSE.model_SG(KSE, KSE.μs[1])
    n = size(F, 1)

    @test all(F .== LnL.H2F(LnL.F2H(F)))
    @test all(F .== LnL.H2F(LnL.F2Hs(F)))

    H = LnL.F2H(F)
    Q = LnL.H2Q(H)
    Hnew = Matrix(LnL.Q2H(Q))
    @test all(H .== Hnew)
end


@testset "Insert and Extract" begin
    H = [
        0.0  1.0  0.0  0.0  1.0  0.0  0.0  0.0  1.0
        0.0  0.0  1.0  0.0  1.0  0.0  0.0  0.0  1.0
        1.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0
    ]
    H2 = LnL.insert2H(H, 4)
    H2test = [
        0.0  1.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0
        0.0  0.0  1.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0
        1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
    ]
    @test all(H2 .== H2test)
    @test all(H .== LnL.extractH(H2, 3))

    F = LnL.H2F(H)
    F2 = LnL.insert2F(F, 4)
    F2test = [
        0.0  1.0  0.0  0.0  1.0  0.0  0.0  1.0  0.0  0.0
        0.0  0.0  1.0  0.0  1.0  0.0  0.0  1.0  0.0  0.0
        1.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
    ]
    @test all(F2 .== F2test)
    @test all(F .== LnL.extractF(F2, 3))
end