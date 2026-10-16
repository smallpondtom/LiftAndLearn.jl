# Smoke tests for the Tikhonov-regularized least-squares solvers.
@testset "Tikhonov regularization" begin
    m, n, p = 40, 5, 3
    D = [sin(0.3i + 0.7j) + (i == j ? 2.0 : 0.0) for i in 1:m, j in 1:n]
    O = [cos(0.5i - 0.2j) for i in 1:n, j in 1:p]
    Rt = D * O

    # Reference solution of (D'D + Γ) x = D'Rt for a non-uniform diagonal Γ
    γ = [0.3, 0.1, 0.5, 0.2, 0.4]
    Γ = Matrix(Diagonal(γ))
    Γsp = LnL.SparseArrays.spdiagm(0 => γ)
    ref = (D' * D + Γ) \ (D' * Rt)

    @testset "zero / tiny Γ recovers least squares" begin
        @test LnL.tikhonov(Rt, D, zeros(n, n); use_backslash=true) ≈ O rtol=1e-8
        @test LnL.tikhonov(Rt, D, 1e-14 * Matrix(1.0I, n, n)) ≈ O rtol=1e-6
        @test LnL.tikhonov(Rt, D, 1e-14 * Matrix(1.0I, n, n); use_normal_form=true, use_backslash=true) ≈ O rtol=1e-8
    end

    @testset "solver paths agree with the normal-form reference" begin
        for G in (Γ, Γsp)
            # default: matrix-free iterative solver
            @test LnL.tikhonov(Rt, D, G) ≈ ref rtol=1e-6
            # augmented system with backslash
            @test LnL.tikhonov(Rt, D, G; use_backslash=true) ≈ ref rtol=1e-8
            # normal form (m > 2n, p < 100), iterative and backslash
            @test LnL.tikhonov(Rt, D, G; use_normal_form=true) ≈ ref rtol=1e-6
            @test LnL.tikhonov(Rt, D, G; use_normal_form=true, use_backslash=true) ≈ ref rtol=1e-8
            # SVD truncation flag (well-conditioned => not selected, iterative fallback)
            @test LnL.tikhonov(Rt, D, G; use_svd_truncation=true) ≈ ref rtol=1e-6
        end
        # estimate_memory flag only logs
        @test LnL.tikhonov(Rt, D, Γ; use_backslash=true, estimate_memory=true) ≈ ref rtol=1e-8
        # mixed element types / Adjoint inputs go through the promotion method
        @test LnL.tikhonov(Float32.(Rt), D, Γ; use_backslash=true) ≈ ref rtol=1e-5
        @test LnL.tikhonov(Rt, copy(D')', Γ; use_backslash=true) ≈ ref rtol=1e-8
    end

    @testset "SVD truncation path" begin
        # A zero entry makes the (entry-wise) condition estimate infinite => SVD path selected
        D0 = copy(D)
        D0[1, 1] = 0.0
        ref0 = (D0' * D0 + Γ) \ (D0' * Rt)
        @test LnL.tikhonov(Rt, D0, Γ; use_svd_truncation=true) ≈ ref0 rtol=1e-8
        # scalar Γ = γI uses the SVD of D directly
        ref0s = (D0' * D0 + 0.3I) \ (D0' * Rt)
        @test LnL.tikhonov(Rt, D0, 0.3 * Matrix(1.0I, n, n); use_svd_truncation=true) ≈ ref0s rtol=1e-8

        s = LnL.TikhonovSolver(; use_svd_truncation=true)
        @test LnL.solve_svd_truncation(D, Rt, Γ, s) ≈ ref rtol=1e-8
        # rank-deficient, unregularized: truncated pseudoinverse == minimum-norm LS solution
        Dz = copy(D)
        Dz[:, end] = Dz[:, 1]
        @test LnL.solve_svd_truncation(Dz, Rt, zeros(n, n), s) ≈ pinv(Dz) * Rt rtol=1e-6
    end

    @testset "large Γ shrinks the solution" begin
        big = 1e6 * Matrix(1.0I, n, n)
        Ob = LnL.tikhonov(Rt, D, big; use_backslash=true)
        @test norm(Ob) < 1e-3 * norm(O)
        Om = LnL.tikhonov(Rt, D, Matrix(1.0I, n, n); use_backslash=true)
        @test norm(Ob) < norm(Om) < norm(O)
    end

    @testset "dimension checks" begin
        @test_throws DimensionMismatch LnL.tikhonov(Rt[1:end-1, :], D, Γ)
        @test_throws DimensionMismatch LnL.tikhonov(Rt, D, zeros(n + 1, n + 1))
    end

    @testset "tikhonov_matrix!" begin
        λ = LnL.TikhonovParameter(A=1.0, B=2.0, A2=3.0, N=4.0, K=5.0)
        dims = [2, 1, 3, 2, 1]
        syms = [:A, :B, :A2u, :N, :K]
        g = zeros(sum(dims))
        LnL.tikhonov_matrix!(g, dims, syms, λ)
        @test g == [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0]

        # redundant symbol and sparse storage (as in leastsquares_solve)
        g2 = LnL.SparseArrays.spzeros(2 + 4)
        LnL.tikhonov_matrix!(g2, [2, 4], [:A, :A2], LnL.TikhonovParameter(A=0.5, A2=7.0))
        @test Vector(g2) == [0.5, 0.5, 7.0, 7.0, 7.0, 7.0]

        # cubic operator and per-column linear parameters
        g3 = zeros(2 + 4)
        LnL.tikhonov_matrix!(g3, [2, 4], [:A, :A3u], LnL.TikhonovParameter(A=Real[0.1, 0.2], A3=9.0))
        @test g3 == [0.1, 0.2, 9.0, 9.0, 9.0, 9.0]
    end

    @testset "helpers" begin
        @test LnL.is_scalar_regularization(2.0 * Matrix(1.0I, 3, 3)) == (true, 2.0)
        @test LnL.is_scalar_regularization(Γ) == (false, 0.0)
        @test LnL.is_scalar_regularization([1.0 0.1; 0.1 1.0])[1] == false

        U = LnL.compute_regularization_matrix(Γ, :cholesky)
        @test U' * U ≈ Γ
        @test LnL.compute_regularization_matrix(Γsp, :cholesky)' * LnL.compute_regularization_matrix(Γsp, :cholesky) ≈ Γ
        @test LnL.compute_regularization_matrix(zeros(2, 2)) == zeros(2, 2)  # not posdef => sqrt.(Γ)

        S = LnL.TikhonovSolver()
        @test S isa LnL.TikhonovSolver{Float64}
        @test !S.use_backslash
        @test !LnL.TikhonovSolver(; use_gpu=true, use_backslash=true).use_backslash  # GPU disables backslash

        @test LnL.select_optimal_method(D, Rt, Γ, LnL.TikhonovSolver(; use_backslash=true))[2] == :augmented
        @test LnL.select_optimal_method(D, Rt, Γ, LnL.TikhonovSolver(; use_normal_form=true))[2] == :normal
        @test LnL.select_optimal_method(D, Rt, Γ, LnL.TikhonovSolver())[2] == :auto

        @test LnL.solve_least_squares_batch(D, Rt, 2) ≈ O rtol=1e-8
    end

    @testset "tikhonov_regularization_path" begin
        sols = LnL.tikhonov_regularization_path(Rt, D, [0.0, 0.1, 10.0])
        @test length(sols) == 3
        @test sols[1] ≈ O rtol=1e-8
        @test sols[2] ≈ (D' * D + 0.1I) \ (D' * Rt)
        @test norm(sols[3]) < norm(sols[2]) < norm(sols[1])
    end

    @testset "estimate_tikhonov_memory" begin
        T = sizeof(Float64)
        base = T * (m * n + m * p + n * n)
        @test LnL.estimate_tikhonov_memory(D, Rt, Γ) == base
        @test LnL.estimate_tikhonov_memory(D, Rt, Γ, :augmented) == base + T * (m + n) * (n + p)
        @test LnL.estimate_tikhonov_memory(D, Rt, Γ, :normal) == base + T * (n * n + n * p)
        @test LnL.estimate_tikhonov_memory(D, Rt, Γ, :svd) == base + T * 3 * n * n
    end

    @testset "optimize_regularization_parameter" begin
        b = Rt[:, 1] .+ 1e-3 .* cos.(1:m)   # slightly perturbed right-hand side
        γopt, scores = LnL.optimize_regularization_parameter(D, b, Matrix(1.0I, n, n); n_params=10)
        @test length(scores) == 10
        @test 1e-8 <= γopt <= 1e2
        @test all(isfinite, scores)
        @test scores[argmin(scores)] == minimum(scores)
        @test_throws ArgumentError LnL.optimize_regularization_parameter(D, b, Matrix(1.0I, n, n); method=:lcurve)
    end
end
