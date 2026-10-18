@testset "tall to fat matrix" begin
    A = [1 2 3; 4 5 6]
    @test all(A .== LnL.tall2fat(A))
end

@testset "fat to tall matrix" begin
    A = [1 2 3; 4 5 6]
    @test all(A' .== LnL.fat2tall(A))
end

@testset "checksize" begin
    @test LnL.checksize([1.0, 2.0, 3.0]) == (3, 1)
    @test LnL.checksize([1.0]) == (1, 1)
    @test LnL.checksize(zeros(2, 5)) == (2, 5)
    @test LnL.checksize(zeros(5, 2)) == (5, 2)
    @test LnL.checksize(zeros(3, 3)) == (3, 3)
    @test LnL.checksize(zeros(2, 5)') == (5, 2)
end

@testset "matrix orientation edge cases" begin
    tall = reshape(collect(1.0:6.0), 3, 2)
    fat = reshape(collect(1.0:6.0), 2, 3)
    sq = [1.0 2.0; 3.0 4.0]
    v = [1.0, 2.0, 3.0]

    # tall2fat: returned matrix has rows ≤ columns
    @test LnL.tall2fat(fat) === fat
    @test LnL.tall2fat(sq) === sq
    @test LnL.tall2fat(tall) == tall'
    @test size(LnL.tall2fat(tall)) == (2, 3)
    @test LnL.tall2fat(v) == v'                 # vector → 1×n row
    @test size(LnL.tall2fat(v)) == (1, 3)
    @test LnL.tall2fat([5.0]) == [5.0]          # length-1 vector unchanged

    # fat2tall: returned matrix has rows ≥ columns
    @test LnL.fat2tall(tall) === tall
    @test LnL.fat2tall(sq) === sq
    @test LnL.fat2tall(fat) == fat'
    @test size(LnL.fat2tall(fat)) == (3, 2)
    @test LnL.fat2tall(v) === v                 # vectors are already "tall"
    @test LnL.fat2tall(v') == v                 # 1×n row → n×1
    @test LnL.fat2tall(v') === v             # adjoint of adjoint is the parent vector

    # round trips
    @test LnL.fat2tall(LnL.tall2fat(tall)) == tall
    @test LnL.tall2fat(LnL.fat2tall(fat)) == fat
end
