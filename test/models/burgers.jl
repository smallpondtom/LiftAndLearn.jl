using LiftAndLearn
using Test

LnL = LiftAndLearn

@testset "Burgers models" begin
    # First order Burger's equation setup
    burger = LnL.burgers(
        [0.0, 1.0], [0.0, 1.0], [0.1, 1.0],
        2^(-7), 1e-4, 1, "periodic"
    )
    n = burger.Xdim
    μ = burger.μs[1]
    A, F = burger.generateMatrix_NC_periodic(burger, μ)
    @test size(A) == (n, n)
    @test size(F) == (n, Int(n*(n+1)/2))

    A, F = burger.generateMatrix_C_periodic(burger, μ)
    @test size(A) == (n, n)
    @test size(F) == (n, Int(n*(n+1)/2))
end


