@testset "Spline interpolation" begin
    # Example usage
    z = [1.0, 2.0, 3.0]
    matrices = [
        [1.0 2.0; 3.0 4.0],
        [1.5 2.5; 3.5 4.5],
        [2.0 3.0; 4.0 5.0]
    ]
    ẑ = 2.5

    # Interpolate
    m1 = LnL.interpolate_matrix_elements(z, matrices, ẑ; order=3)
    m2 = LnL.interpolate_matrix_elements(z, matrices, ẑ; order=4)
    m3 = LnL.interpolate_matrix_elements(z, matrices, ẑ; order=5)

    @test size(m1) == (2, 2)
    @test size(m2) == (2, 2)
    @test size(m3) == (2, 2)
end
