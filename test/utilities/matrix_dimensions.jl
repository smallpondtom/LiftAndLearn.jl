@testset "tall to fat matrix" begin
    A = [1 2 3; 4 5 6]
    @test all(A .== LnL.tall2fat(A))
end

@testset "fat to tall matrix" begin
    A = [1 2 3; 4 5 6]
    @test all(A' .== LnL.fat2tall(A))
end