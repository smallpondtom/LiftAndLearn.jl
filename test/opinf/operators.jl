# Smoke tests for Operators, get_data_matrix, unpack_operators! and opinf (incl. Tikhonov path).
@testset "Operators structure" begin
    op0 = LnL.Operators()
    @test op0.A == 0 && op0.B == 0 && op0.A2u == 0 && op0.N == 0 && op0.K == 0
    @test all(values(op0.dims) .== 0)

    A = [-2.0 0.0; 0.0 -1.0]
    A2u = [0.0 1.0 0.0; 0.0 1.0 0.0]
    B = reshape([0.0, -0.01], 2, 1)
    op = LnL.Operators(A=A, A2u=A2u, B=B)
    @test op.dims[:A] == 2
    @test op.dims[:B] == 1
    @test op.dims[:A2u] == 3
    @test op.dims[:A2] == 0

    # default nonlinear function: A2u * (x ⊘ x)
    LnL._reconstruct_function!(op)
    x = [2.0, 3.0]
    @test op.f(x, [0.0]) ≈ A2u * [x[1]^2, x[1] * x[2], x[2]^2]
end

@testset "get_data_matrix" begin
    K = 12
    Xhat = [sin(0.3k + i) for i in 1:2, k in 1:K]
    Ut = reshape([cos(0.2k) for k in 1:K], K, 1)
    opts = LnL.LSOpInfOption(
        system=LnL.SystemStructure(state=[1, 2], control=1, coupled_input=1, constant=1),
    )
    D, dims, syms = LnL.get_data_matrix(Xhat, Matrix(Xhat'), Ut, opts; verbose=true)
    @test dims == [2, 1, 3, 2, 1]
    @test syms == [:A, :B, :A2u, :N, :K]
    @test size(D) == (K, 9)
    @test D[:, 1:2] == Xhat'
    @test D[:, 3] == Ut[:, 1]
    @test D[:, 4:6] ≈ hcat(Xhat[1, :] .^ 2, Xhat[1, :] .* Xhat[2, :], Xhat[2, :] .^ 2)
    @test D[:, 7:8] ≈ Xhat' .* Ut[:, 1]
    @test D[:, 9] == ones(K)

    # verbose=false returns only D; 3-argument method is equivalent
    @test LnL.get_data_matrix(Xhat, Matrix(Xhat'), Ut, opts; verbose=false) == D
    @test LnL.get_data_matrix(Xhat, Ut, opts; verbose=false) == D

    # redundant quadratic operator
    opts.optim.nonredundant_operators = false
    D2, dims2, syms2 = LnL.get_data_matrix(Xhat, Matrix(Xhat'), Ut, opts; verbose=true)
    @test dims2 == [2, 1, 4, 2, 1]
    @test syms2 == [:A, :B, :A2, :N, :K]
    @test size(D2) == (K, 10)
end

@testset "unpack_operators!" begin
    n = 2
    A = [1.0 2.0; 3.0 4.0]
    B = reshape([5.0, 6.0], 2, 1)
    A2u = [7.0 8.0 9.0; 10.0 11.0 12.0]
    K = reshape([13.0, 14.0], 2, 1)
    O = hcat(A, B, A2u, K)

    op = LnL.Operators()
    LnL.unpack_operators!(op, O, [2, 1, 3, 1], [:A, :B, :A2u, :K])
    @test op.A == A
    @test op.B == B
    @test op.A2u == A2u
    @test op.K == K

    # bilinear operator with two inputs -> n × n × m array
    N1 = [1.0 0.0; 0.0 2.0]
    N2 = [0.0 3.0; 4.0 0.0]
    op = LnL.Operators()
    LnL.unpack_operators!(op, hcat(A, N1, N2), [2, 4], [:A, :N])
    @test size(op.N) == (2, 2, 2)
    @test op.N[:, :, 1] == N1
    @test op.N[:, :, 2] == N2

    # warning if :B is not right after :A
    op = LnL.Operators()
    @test_logs (:warn, r"should come right after") LnL.unpack_operators!(op, hcat(A, A2u, B), [2, 3, 1], [:A, :A2u, :B])
    @test op.B == B

    # output operator fitted from data (7-argument method)
    Kt = 20
    Xhat_t = [sin(0.4k + i) for k in 1:Kt, i in 1:n]
    C = [0.5 -0.25]
    Yt = Xhat_t * C'
    opts = LnL.LSOpInfOption(system=LnL.SystemStructure(state=1, output=1))
    op = LnL.Operators()
    LnL.unpack_operators!(op, A, Yt, Xhat_t, [2], [:A], opts)
    @test op.A == A
    @test op.C ≈ C

    # regularized output fit shrinks C
    opts_reg = LnL.LSOpInfOption(system=LnL.SystemStructure(state=1, output=1),
                                 with_reg=true, λ=LnL.TikhonovParameter(C=10.0))
    op = LnL.Operators()
    LnL.unpack_operators!(op, A, Yt, Xhat_t, [2], [:A], opts_reg)
    @test norm(op.C) < norm(C)
end

@testset "opinf on a tiny linear system" begin
    A = [-1.0 0.5; -0.5 -1.0]
    Kt = 30
    X = [cos(0.37k + i) * exp(-0.01k * i) for i in 1:2, k in 1:Kt]
    Xdot = A * X
    Vn = Matrix(1.0I, 2, 2)
    sys = LnL.SystemStructure(state=1)

    @testset "least squares (no regularization)" begin
        opts = LnL.LSOpInfOption(system=sys)
        op = LnL.opinf(X, Vn, opts; Xdot=Xdot)
        @test op.A ≈ A rtol=1e-8
        @test op.O ≈ A rtol=1e-8
        # reduced-data method
        op = LnL.opinf(X, opts; Xhatdot=Xdot)
        @test op.A ≈ A rtol=1e-8
        @test_throws ErrorException LnL.opinf(X, opts)   # Xhatdot is required
        # finite-difference derivative data (no Xdot): just check it runs
        op = LnL.opinf(X, Vn, LnL.LSOpInfOption(system=sys, data=LnL.DataStructure(Δt=0.1, deriv_type="FE")))
        @test size(op.A) == (2, 2)
    end

    @testset "Tikhonov path ($label)" for (label, kw) in [
            ("augmented backslash", (;)),
            ("iterative", (use_backslash=false,)),
            ("normal form", (use_normal_equations=true, use_backslash=false)),
            ("normal form backslash", (use_normal_equations=true,)),
        ]
        opts = LnL.LSOpInfOption(; system=sys, with_reg=true, λ=LnL.TikhonovParameter(A=1e-10), kw...)
        op = LnL.opinf(X, Vn, opts; Xdot=Xdot)
        @test op.A ≈ A rtol=1e-6

        opts_big = LnL.LSOpInfOption(; system=sys, with_reg=true, λ=LnL.TikhonovParameter(A=1e3), kw...)
        op_big = LnL.opinf(X, Vn, opts_big; Xdot=Xdot)
        @test norm(op_big.A) < norm(A)
    end

    @testset "LS solver options forwarded from LSOpInfOption" begin
        for kw in [(use_backslash=false,), (use_backslash=false, use_normal_equations=true),
                   (use_backslash=false, chunk_size=1)]
            op = LnL.opinf(X, Vn, LnL.LSOpInfOption(; system=sys, kw...); Xdot=Xdot)
            @test op.A ≈ A rtol=1e-6
        end
    end
end

@testset "reproject" begin
    A = [-1.0 0.5; -0.5 -1.0]
    A2u = [0.0 1.0 0.0; 0.0 1.0 0.0]
    full_op = LnL.Operators(A=A, A2u=A2u)
    X = [cos(0.37k + i) for i in 1:2, k in 1:10]
    Ut = zeros(10, 1)
    opts = LnL.LSOpInfOption(system=LnL.SystemStructure(state=[1, 2]))
    Rt = LnL.reproject(X, Matrix(1.0I, 2, 2), Ut, full_op, opts)
    @test size(Rt) == (10, 2)
    @test Rt ≈ (A * X .+ A2u * vcat(X[1:1, :] .^ 2, X[1:1, :] .* X[2:2, :], X[2:2, :] .^ 2))'

    # A user-supplied nonlinear function in `Operators.f` is used (and not overwritten)
    fcustom = (x, u) -> [0.0, -sin(x[1])]
    custom = LnL.Operators(A=A, f=fcustom)
    Rt2 = LnL.reproject(X, Matrix(1.0I, 2, 2), Ut, custom, opts)
    @test Rt2 ≈ (A * X .+ vcat(zeros(1, 10), -sin.(X[1:1, :])))'
    @test custom.f === fcustom
end
