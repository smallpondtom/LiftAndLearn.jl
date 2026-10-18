# Regression smoke tests for code paths that used to crash (default inputs / initial guesses).
@testset "default-input and initial-guess code paths" begin
    X = [cos(0.37k + i) for i in 1:2, k in 1:40]
    Vn = Matrix(1.0I, 2, 2)

    @testset "epopinf without Xdot and without U" begin
        opts = LnL.EPHECOpInfOption(
            system=LnL.SystemStructure(state=[1, 2]),
            data=LnL.DataStructure(Δt=0.1, deriv_type="FE"),
        )
        op = LnL.epopinf(X, Vn, opts)
        @test size(op.A) == (2, 2)
        @test size(op.A2u) == (2, 3)
    end

    @testset "epopinf with initial guess" begin
        opts = LnL.EPHECOpInfOption(
            system=LnL.SystemStructure(state=[1, 2]),
            optim=LnL.OptimizationSetting(initial_guess=true),
        )
        IG = LnL.Operators(A=-1.0 * Matrix(1.0I, 2, 2), A2u=zeros(2, 3))
        op = LnL.epopinf(X, Vn, opts; Xdot=-X, IG=IG)
        @test size(op.A2u) == (2, 3)

        opts.optim.nonredundant_operators = false
        IG2 = LnL.Operators(A=-1.0 * Matrix(1.0I, 2, 2), A2=zeros(2, 4))
        op = LnL.epopinf(X, Vn, opts; Xdot=-X, IG=IG2)
        @test size(op.A2) == (2, 4)
    end

    @testset "Lift & Learn opinf without reprojection and without U/Y" begin
        lifter = LnL.lifting(2, 3, [x -> x[1] .^ 2])
        W = lifter.map([X[1:1, :], X[2:2, :]])
        opts = LnL.LSOpInfOption(
            system=LnL.SystemStructure(state=[1, 2]),
            vars=LnL.VariableStructure(N=2, N_lift=3),
            data=LnL.DataStructure(Δt=0.1, deriv_type="FE"),
        )
        op = LnL.opinf(W, Matrix(1.0I, 3, 3), lifter, LnL.Operators(A=-1.0 * Matrix(1.0I, 2, 2)), opts)
        @test size(op.A) == (3, 3)
        @test size(op.A2u) == (3, 6)
    end
end
