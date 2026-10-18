# Reduced linear system  ẋ = A x + B u  with exact derivative data
r, m, K = 3, 1, 60
A = [-1.0 0.2 0.0; -0.2 -0.5 0.1; 0.0 -0.1 -0.8]
B = reshape([1.0, 0.5, -0.3], r, 1)
Δt = 0.05
ts = (0:K-1) .* Δt
U = reshape(sin.(2 .* ts) .+ cos.(5 .* ts), m, K)
Xhat = zeros(r, K)
Xhat[:, 1] = [1.0, -0.5, 0.25]
for k in 1:K-1
    Xhat[:, k+1] = Xhat[:, k] + Δt * (A * Xhat[:, k] + B * U[:, k])
end
Xdot = A * Xhat + B * U

options = LnL.LSOpInfOption(system=LnL.SystemStructure(state=[1], control=1))
op_batch = LnL.opinf(Xhat, options; U=U, Xhatdot=Xdot)

Xs = LnL.streamify(Xhat, 1)
Rs = LnL.streamify(Xdot, 1)
Us = LnL.streamify(U, 1)

@testset "Two-pass constructor" begin
    s = LnL.TwoPassStreamingOpInf(options=options, n=r, m=m, algorithm=:RLS, Γs=1e-8)
    @test s isa LnL.RLSOpInf
    @test s isa LnL.TwoPassStreamingOpInf
    @test s.dims[:d] == r + m
    @test LnL.TwoPassStreamingOpInf(options=options, n=r, m=m, algorithm=:QRRLS, Γs=1e-8) isa LnL.QRRLSOpInf
    @test LnL.TwoPassStreamingOpInf(options=options, n=r, m=m, algorithm=:iQRRLS, Γs=1e-8) isa LnL.iQRRLSOpInf

    # with output: returns a (state, output) tuple
    so = LnL.TwoPassStreamingOpInf(options=options, n=r, m=m, l=2, algorithm=:RLS, Γs=1e-8, Γo=1e-8)
    @test so isa Tuple && length(so) == 2

    # invalid inputs
    @test_throws ErrorException LnL.TwoPassStreamingOpInf(options=options, n=r, m=m, algorithm=:foo, Γs=1e-8)
    @test_throws ArgumentError LnL.TwoPassStreamingOpInf(options=options, n=r, m=m, algorithm=:QRRLS)
    @test_throws ArgumentError LnL.TwoPassStreamingOpInf(options=options, n=r, m=m, algorithm=:iQRRLS)
    @test_throws ArgumentError LnL.TwoPassStreamingOpInf(options=options, n=r, m=m, l=1, algorithm=:iQRRLS, Γs=1e-8)
end

@testset "Two-pass $alg ($qr_method) recovers linear operators" for alg in (:RLS, :QRRLS, :iQRRLS),
                                                                 qr_method in (:qr, :givens)
    alg == :RLS && qr_method == :givens && continue
    s = LnL.TwoPassStreamingOpInf(options=options, n=r, m=m, algorithm=alg, Γs=1e-10,
                                  qr_method=qr_method)
    D = LnL.stream_all!(s, Xs, Rs; U=Us)
    @test size(D) == (1, r + m)
    op = LnL.terminate_stream(s)
    @test op.A ≈ A atol=1e-7
    @test op.B ≈ B atol=1e-7
    # operator matrix has the same orientation as batch OpInf
    @test size(op.O) == size(op_batch.O)
    @test op.O ≈ op_batch.O atol=1e-7
end

@testset "Two-pass stream! matches stream_all!" begin
    s1 = LnL.TwoPassStreamingOpInf(options=options, n=r, m=m, algorithm=:iQRRLS, Γs=1e-10)
    s2 = LnL.TwoPassStreamingOpInf(options=options, n=r, m=m, algorithm=:iQRRLS, Γs=1e-10)
    for i in 1:K
        LnL.stream!(s1, Xs[i], Rs[i]; U=Us[i])
    end
    LnL.stream_all!(s2, Xs, Rs; U=Us)
    @test LnL.terminate_stream(s1).O ≈ LnL.terminate_stream(s2).O
end

@testset "Two-pass RLS options" begin
    # Tikhonov matrix built from options.λ
    opt_reg = LnL.LSOpInfOption(system=LnL.SystemStructure(state=[1], control=1), with_reg=true,
                                λ=LnL.TikhonovParameter(A=1e-8, B=1e-8))
    s = LnL.TwoPassStreamingOpInf(options=opt_reg, n=r, m=m, algorithm=:RLS)
    LnL.stream_all!(s, Xs, Rs; U=Us)
    op = LnL.terminate_stream(s)
    @test op.A ≈ A atol=1e-5

    # block (rank-5) updates without regularization: first block initializes the solver
    Xb, Rb, Ub = LnL.streamify(Xhat, 5), LnL.streamify(Xdot, 5), LnL.streamify(U, 5)
    sb = LnL.TwoPassStreamingOpInf(options=options, n=r, m=m, algorithm=:RLS, rank=5)
    LnL.stream_all!(sb, Xb, Rb; U=Ub)
    opb = LnL.terminate_stream(sb)
    @test opb.A ≈ A atol=1e-7
    @test opb.B ≈ B atol=1e-7

    # block (rank-5) updates with regularization
    sb2 = LnL.TwoPassStreamingOpInf(options=options, n=r, m=m, algorithm=:RLS, rank=5, Γs=1e-10)
    LnL.stream_all!(sb2, Xb, Rb; U=Ub)
    @test LnL.terminate_stream(sb2).A ≈ A atol=1e-7

    # streams must match the block size
    sb3 = LnL.TwoPassStreamingOpInf(options=options, n=r, m=m, algorithm=:RLS, rank=5, Γs=1e-8)
    @test_throws DimensionMismatch LnL.stream!(sb3, Xhat[:, 1:3], Xdot[:, 1:3]; U=U[:, 1:3])

    # a constant noise covariance does not change the least-squares solution
    sq = LnL.TwoPassStreamingOpInf(options=options, n=r, m=m, algorithm=:RLS, Γs=1e-10)
    LnL.stream_all!(sq, Xs, Rs; U=Us, Q=2.0)
    @test LnL.terminate_stream(sq).A ≈ A atol=1e-7

    # forgetting factor λ < 1 still fits noise-free data
    sf = LnL.TwoPassStreamingOpInf(options=options, n=r, m=m, algorithm=:RLS, Γs=1e-10, λ=0.99)
    LnL.stream_all!(sf, Xs, Rs; U=Us)
    @test LnL.terminate_stream(sf).A ≈ A atol=1e-6
end

@testset "Two-pass quadratic model without input" begin
    opt_q = LnL.LSOpInfOption(system=LnL.SystemStructure(state=[1, 2]))
    H = 0.01 .* reshape(collect(1.0:18.0), r, 6) ./ 18
    Xq2 = LnL.UniqueKronecker.unique_kron_snapshot_matrix(Xhat, 2)
    Xqdot = A * Xhat + H * Xq2
    for alg in (:RLS, :iQRRLS, :QRRLS)
        s = LnL.TwoPassStreamingOpInf(options=opt_q, n=r, algorithm=alg, Γs=1e-12)
        LnL.stream_all!(s, LnL.streamify(Xhat, 1), LnL.streamify(Xqdot, 1))
        op = LnL.terminate_stream(s)
        @test op.A ≈ A atol=1e-6
        @test op.A2u ≈ H atol=1e-5
    end
end

@testset "Two-pass $alg with output" for alg in (:RLS, :QRRLS, :iQRRLS)
    C = [1.0 0.0 2.0; 0.5 -1.0 0.0]
    Y = C * Xhat
    s, so = LnL.TwoPassStreamingOpInf(options=options, n=r, m=m, l=2, algorithm=alg,
                                      Γs=1e-10, Γo=1e-10)
    LnL.stream_all!(s, Xs, Rs; U=Us)
    LnL.stream_output_all!(so, Xs, LnL.streamify(Y, 1))
    op = LnL.terminate_stream(s, so)
    @test op.A ≈ A atol=1e-7
    @test size(op.C) == size(C)
    @test op.C ≈ C atol=1e-7
end
