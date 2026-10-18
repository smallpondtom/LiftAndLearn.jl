# Smoke tests for the (non-regularized) least-squares solver used by OpInf.
@testset "standard_least_squares" begin
    # Deterministic, well-conditioned tall data matrix (m × n) and known solution (n × p)
    m, n, p = 40, 5, 3
    D = [sin(0.3i + 0.7j) + (i == j ? 2.0 : 0.0) for i in 1:m, j in 1:n]
    O = [cos(0.5i - 0.2j) for i in 1:n, j in 1:p]
    Rt = D * O

    @testset "default (direct LinearSolve.jl batch)" begin
        Oh = LnL.standard_least_squares(D, Rt)
        @test size(Oh) == (n, p)
        @test Oh ≈ O rtol=1e-8
    end

    @testset "backslash" begin
        Oh = LnL.standard_least_squares(D, Rt; use_backslash=true)
        @test Oh isa Matrix{Float64}
        @test Oh ≈ O rtol=1e-8
    end

    @testset "normal equations (m > 3n)" begin
        Oh = LnL.standard_least_squares(D, Rt; use_normal_equations=true)
        @test size(Oh) == (n, p)
        @test Oh ≈ O rtol=1e-6   # iterative solver, absolute tolerance 1e-12
    end

    @testset "QR with batching (p > chunk_size)" begin
        Oh = LnL.standard_least_squares(D, Rt; chunk_size=1)
        @test Oh ≈ O rtol=1e-8
        Oh = LnL.standard_least_squares(D, Rt; chunk_size=2)
        @test Oh ≈ O rtol=1e-8
    end

    @testset "estimate_memory flag" begin
        Oh = LnL.standard_least_squares(D, Rt; estimate_memory=true)
        @test Oh ≈ O rtol=1e-8
    end

    @testset "type promotion convenience method" begin
        Oh = LnL.standard_least_squares(D, Float32.(Rt))
        @test eltype(Oh) == Float64
        @test Oh ≈ O rtol=1e-5
    end

    @testset "Adjoint data matrix (as produced by opinf)" begin
        Dadj = copy(D')'
        @test Dadj isa Adjoint
        @test LnL.standard_least_squares(Dadj, Rt) ≈ O rtol=1e-8
        @test LnL.standard_least_squares(Dadj, Rt; chunk_size=1) ≈ O rtol=1e-8
        @test LnL.standard_least_squares(Dadj, Rt; use_normal_equations=true) ≈ O rtol=1e-6
        @test LnL.standard_least_squares(Dadj, Rt; use_backslash=true) ≈ O rtol=1e-8
    end

    @testset "single right-hand side column" begin
        Oh = LnL.standard_least_squares(D, Rt[:, 1:1])
        @test Oh ≈ O[:, 1:1] rtol=1e-8
    end

    @testset "dimension mismatch" begin
        @test_throws DimensionMismatch LnL.standard_least_squares(D, Rt[1:end-1, :])
    end

    @testset "estimate_memory_usage" begin
        mem = LnL.estimate_memory_usage(D, Rt)
        base = sizeof(Float64) * (m * n + m * p + n * p)
        @test mem.base == base
        @test mem.qr_method == base + sizeof(Float64) * m * n
        @test mem.normal_equations == base + sizeof(Float64) * n * n
    end

    @testset "LeastSquaresSolver constructor" begin
        s = LnL.LeastSquaresSolver()
        @test s isa LnL.LeastSquaresSolver{Float64}
        @test s.chunk_size == 1000
        @test s.tolerance == 1e-12
        @test !s.use_gpu && !s.use_normal_equations
        @test isnothing(s.algorithm)
    end
end
