# Smoke tests for the finite-difference time-derivative approximations.
@testset "time derivative approximation" begin
    schemes = ["FE", "BE", "SI", "FE4", "BE4", "CTD4", "FBCT4"]
    expected_idx(s, K) = Dict(
        "FE" => 1:K-1, "BE" => 2:K, "SI" => 2:K,
        "FE4" => 1:K-4, "BE4" => 5:K, "CTD4" => 3:K-2, "FBCT4" => 1:K,
    )[s]

    # Smooth deterministic signal: rows = state, columns = time
    Δt = 1e-3
    t = collect(0.0:Δt:0.2)
    K = length(t)
    X = [sin.(t)'; cos.(2 .* t)'; (t .^ 2)']
    dX = [cos.(t)'; -2 .* sin.(2 .* t)'; (2 .* t)']

    @testset "accuracy and indices ($s)" for s in schemes
        dXdt, idx = LnL.time_derivative_approx(X, s, Δt)
        @test idx == expected_idx(s, K)
        @test size(dXdt) == (3, length(idx))
        tol = s in ("FE", "BE", "SI") ? 5e-3 : 1e-8
        @test maximum(abs.(dXdt .- dX[:, idx])) < tol
    end

    @testset "4th-order schemes are exact for quartic polynomials" begin
        Δt2 = 0.1
        t2 = collect(0.0:Δt2:1.0)
        P = [(t2 .^ 4)'; (1 .+ t2 .- 3 .* t2 .^ 3)']
        dP = [(4 .* t2 .^ 3)'; (1 .- 9 .* t2 .^ 2)']
        for s in ("FE4", "BE4", "CTD4", "FBCT4")
            dPdt, idx = LnL.time_derivative_approx(P, s, Δt2)
            @test dPdt ≈ dP[:, idx] atol=1e-10
        end
        # 1st-order schemes are exact for linear signals
        L = [(2 .* t2 .+ 1)'; (-t2)']
        for s in ("FE", "BE", "SI")
            dLdt, _ = LnL.time_derivative_approx(L, s, Δt2)
            @test dLdt ≈ repeat([2.0, -1.0], 1, size(dLdt, 2)) atol=1e-10
        end
    end

    @testset "options-based method matches explicit method ($s)" for s in schemes
        opts = LnL.LSOpInfOption(data=LnL.DataStructure(Δt=Δt, deriv_type=s))
        d1, i1 = LnL.time_derivative_approx(X, opts)
        d2, i2 = LnL.time_derivative_approx(X, s, Δt)
        @test i1 == i2
        @test d1 == d2
    end

    @testset "finite_diff_matrix consistency ($s)" for s in schemes
        E, idx = LnL.finite_diff_matrix(s, K, Δt)
        dXdt, idx2 = LnL.time_derivative_approx(X, s, Δt)
        @test E isa LnL.SparseArrays.SparseMatrixCSC
        @test idx == idx2
        @test size(E) == (K, length(idx))
        @test X * E ≈ dXdt rtol=1e-10 atol=1e-8
    end

    @testset "invalid inputs" begin
        @test_throws ErrorException LnL.time_derivative_approx(X, "RK4", Δt)
        opts = LnL.LSOpInfOption(data=LnL.DataStructure(Δt=Δt, deriv_type="XYZ"))
        @test_throws ErrorException LnL.time_derivative_approx(X, opts)
        @test_throws ErrorException LnL.finite_diff_matrix("RK4", K, Δt)
        @test_throws ErrorException LnL.finite_diff_matrix("FE", 1, Δt)
        @test_throws ErrorException LnL.finite_diff_matrix("CTD4", 4, Δt)
        @test_throws ErrorException LnL.fwd4(X[:, 1:4], Δt, true)
    end

    @testset "stencil helpers" begin
        xs = X[:, 1:5]
        xv = [X[:, j] for j in 1:5]
        @test LnL.fwd4(xs, Δt, true) ≈ LnL.fwd4(xv, Δt, true)
        @test LnL.fwd4(xs, Δt, false) ≈ LnL.fwd4(xv, Δt, false)
        @test LnL.bwd4(xs, Δt, true) ≈ LnL.bwd4(xv, Δt, true)
        @test LnL.bwd4(xs, Δt, false) ≈ LnL.bwd4(xv, Δt, false)
        @test LnL.ctd4(xs, Δt) ≈ LnL.ctd4(xv, Δt)
        @test LnL.fwd4(xs, Δt, true) ≈ dX[:, 1] atol=1e-8
        @test LnL.fwd4(xs, Δt, false) ≈ dX[:, 2] atol=1e-8
        @test LnL.ctd4(xs, Δt) ≈ dX[:, 3] atol=1e-8
        @test LnL.bwd4(xs, Δt, false) ≈ dX[:, 4] atol=1e-8
        @test LnL.bwd4(xs, Δt, true) ≈ dX[:, 5] atol=1e-8
    end
end
