using LiftAndLearn 
using Test

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