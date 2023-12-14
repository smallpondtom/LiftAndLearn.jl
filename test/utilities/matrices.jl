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


@testset "Quadratic matrix conversion" begin
    A = rand(3,6)
    ct = 0
    for i in A
        A[ct+=1] = i >= 0.5 ? 1 : 0
    end
    H = [
        0.0  1.0  0.0  0.0  1.0  0.0  0.0  0.0  1.0
        0.0  0.0  1.0  0.0  1.0  0.0  0.0  0.0  1.0
        1.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0
    ]
    Hs = [
        0.0  0.5  0.0  0.5  1.0  0.0  0.0  0.0  1.0
        0.0  0.0  0.5  0.0  1.0  0.0  0.5  0.0  1.0
        1.0  0.0  0.0  0.0  1.0  0.5  0.0  0.5  0.0
    ]
    @test all(H .== LnL.F2H(A))
    @test all(Hs .== LnL.F2Hs(A))
    @test all(A .== LnL.H2F(H))

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