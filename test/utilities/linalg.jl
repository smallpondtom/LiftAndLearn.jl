## Smoke tests for givens.jl, qrf.jl, and reorthogonalize.jl

# deterministic, well-conditioned test matrix
_testmat(m, n) = [sin(i * j) + (i == j) for i in 1:m, j in 1:n]

@testset "Givens rotation" begin
    for (a, b) in ((3.0, 4.0), (-3.0, 4.0), (4.0, -3.0), (-1e-3, -2.0),
                   (2.0, 0.0), (-2.0, 0.0), (0.0, 5.0), (0.0, -5.0), (0.0, 0.0))
        c, s, r = LnL.givens_rotation(a, b)
        @test c^2 + s^2 ≈ 1
        @test r ≥ 0
        @test r ≈ hypot(a, b)
        @test [c -s; s c] * [a, b] ≈ [r, 0.0] atol=1e-14
    end
    c, s, r = LnL.givens_rotation(1.0f0, 2.0f0)
    @test r isa Float32
end

@testset "QR via Givens rotations" begin
    for (m, n) in ((4, 4), (6, 3), (3, 5))
        A0 = _testmat(m, n)
        A = copy(A0)
        @test LnL.qr_givens!(A) === nothing
        # A is overwritten with the upper-triangular/trapezoidal R
        @test norm(tril(A, -1)) < 1e-12
        R = triu(A)
        @test R' * R ≈ A0' * A0
        # agrees with LAPACK QR up to row signs
        Rref = qr(A0).R
        k = min(m, n)
        @test abs.(R[1:k, :]) ≈ abs.(Rref[1:k, :])
    end
end

@testset "qrf!" begin
    A0 = _testmat(6, 3)

    # two-argument method: P ← Q, R ← R
    P = copy(A0)
    R = fill(NaN, 3, 3)
    out = LnL.qrf!(P, R)
    @test out === R
    @test istriu(R)
    @test P' * P ≈ I(3)
    @test P * R ≈ A0

    # R larger than b×b is zero-filled outside the leading block
    P = copy(A0)
    Rbig = ones(4, 4)
    LnL.qrf!(P, Rbig)
    @test all(iszero, Rbig[4, :]) && all(iszero, Rbig[:, 4])
    @test P * Rbig[1:3, 1:3] ≈ A0

    # one-argument method: P ← Q, Q returned
    P = copy(A0)
    Q = LnL.qrf!(P)
    @test Q === P
    @test size(Q) == (6, 3)
    @test Q' * Q ≈ I(3)
    @test Q * (Q' * A0) ≈ A0   # same column space as A0

    # square input
    P = _testmat(4, 4)
    Q = LnL.qrf!(P)
    @test Q' * Q ≈ I(4)

    # fat input is rejected
    @test_throws DimensionMismatch LnL.qrf!(_testmat(2, 3))
    @test_throws DimensionMismatch LnL.qrf!(_testmat(2, 3), zeros(3, 3))
end

@testset "reorthogonalize! (Euclidean)" begin
    tol = 1e-10
    Q = Matrix(qr(_testmat(8, 4)).Q)
    @test norm(Q' * Q - I) < 1e-13

    # already orthonormal: left untouched
    V = copy(Q)
    @test LnL.reorthogonalize!(V, tol) === nothing
    @test V == Q

    # perturbed: orthonormality is restored and the span is kept
    Vp = Q .+ 1e-6 .* _testmat(8, 4)
    Vp[:, end] .+= 1e-6 .* Vp[:, 1]   # make sure the first/last columns are not orthogonal
    @test abs(dot(Vp[:, end], Vp[:, 1])) > tol
    @test norm(Vp' * Vp - I) > 1e-7
    V = copy(Vp)
    LnL.reorthogonalize!(V, tol)
    @test norm(V' * V - I) < 1e-12
    @test norm(Vp - V * (V' * Vp)) < 1e-12
    @test norm(V - Q) < 1e-4

    # single column is just re-normalized
    v = reshape([3.0, 4.0], 2, 1)
    LnL.reorthogonalize!(v, tol)
    @test v ≈ reshape([0.6, 0.8], 2, 1)
end

@testset "reorthogonalize! (weighted)" begin
    tol = 1e-10
    B = _testmat(8, 8)
    M = B' * B + I          # symmetric positive definite weight
    Q = Matrix(qr(_testmat(8, 4)).Q)
    W = cholesky(Symmetric(M)).U \ Q   # M-orthonormal: W' M W = I
    @test norm(W' * M * W - I) < 1e-12

    # already M-orthonormal: left untouched
    V = copy(W)
    @test LnL.reorthogonalize!(V, M, tol) === nothing
    @test V == W

    # perturbed: M-orthonormality is restored and the span is kept
    Vp = W .+ 1e-6 .* _testmat(8, 4)
    Vp[:, end] .+= 1e-6 .* Vp[:, 1]
    @test abs(dot(Vp[:, end], M * Vp[:, 1])) > tol
    V = copy(Vp)
    LnL.reorthogonalize!(V, M, tol)
    @test norm(V' * M * V - I) < 1e-12
    @test norm(Vp - V * (V' * M * Vp)) < 1e-10
end
