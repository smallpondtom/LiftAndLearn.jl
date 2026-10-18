# Well-conditioned full-order data: n = 5 states, K = 200 snapshots built from
# sinusoids with distinct frequencies, plus a scalar input.
n, K, Δt = 5, 200, 0.01
ts = (0:K-1) .* Δt
M = [2.0 0.3 0.0 0.1 0.0;
     0.0 1.5 0.2 0.0 0.1;
     0.1 0.0 1.2 0.3 0.0;
     0.0 0.2 0.0 1.0 0.2;
     0.3 0.0 0.1 0.0 0.8]
S = vcat(ones(1, K), sin.(2π .* ts)', cos.(3π .* ts)', sin.(5π .* ts)', cos.(7π .* ts)')
X = M * S
U = reshape(sin.(4π .* ts), 1, K)
Xdot = M * vcat(zeros(1, K), (2π .* cos.(2π .* ts))', (-3π .* sin.(3π .* ts))',
                (5π .* cos.(5π .* ts))', (-7π .* sin.(7π .* ts))')

options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(state=[1], control=1),
    data=LnL.DataStructure(Δt=Δt, deriv_type="FE"),
)

@testset "One-pass constructor" begin
    s = LnL.OnePassStreamingOpInf(X[:, 1]; options=options, n_state=n, n_input=1, rank=3)
    @test s isa LnL.OnePassStreamingOpInf1
    @test s.num_of_snapshots == 1
    s2 = LnL.OnePassStreamingOpInf(X[:, 1]; xdot_init=Xdot[:, 1], options=options, n_state=n,
                                   n_input=1, rank=3, finite_diff=false)
    @test s2 isa LnL.OnePassStreamingOpInf2
    sk = LnL.OnePassStreamingOpInf(; options=options, n_state=n, n_input=1, n_snapshots=K,
                                   rank=3, isvd_method=:sketchy)
    @test sk isa LnL.OnePassStreamingOpInfSketchy
    @test (sk.k, sk.s) == (13, 27)   # defaults k = 4r+1, s = 2k+1

    # Baker iSVD needs a nonzero initial snapshot; unknown methods are rejected
    @test_throws AssertionError LnL.OnePassStreamingOpInf(; options=options, n_state=n, rank=3)
    @test_throws AssertionError LnL.OnePassStreamingOpInf(X[:, 1]; options=options, n_state=n,
                                                         rank=3, isvd_method=:foo)
end

@testset "sparse_sign_matrix" begin
    Ξ = LnL.sparse_sign_matrix(6, 10, 3)
    @test size(Ξ) == (6, 10)
    @test all(count(!iszero, Ξ[:, j]) == 3 for j in 1:10)
    @test all(abs.(LnL.SparseArrays.nonzeros(Ξ)) .== 1)
    @test_throws ErrorException LnL.sparse_sign_matrix(6, 10, 7)
end

@testset "One-pass Baker iSVD (finite differences)" begin
    F = svd(X)
    for rank in (3, n)
        s = LnL.OnePassStreamingOpInf(X[:, 1]; options=options, n_state=n, n_input=1, rank=rank)
        for k in 2:K
            LnL.stream!(s, X[:, k])
        end
        @test s.num_of_snapshots == K
        @test size(s.V) == (n, rank)
        @test size(s.W) == (K, rank)
        # The iSVD accumulates rounding error in V over the 200 updates (worst when the data
        # are exactly rank n, as here); the amount is BLAS/CPU dependent (~1e-11 to ~2e-10).
        @test norm(s.V' * s.V - I) < 1e-8
        if rank == n
            # no truncation: the iSVD is exact
            @test s.Σ ≈ F.S rtol=1e-8
            @test norm(X - s.V * Diagonal(s.Σ) * s.W') < 1e-8 * norm(X)
        else
            # truncated single-pass iSVD only approximates the dominant singular triplets
            @test s.Σ ≈ F.S[1:rank] rtol=0.1
        end
    end

    # Full rank: the streamed model equals batch OpInf with the same basis
    s = LnL.OnePassStreamingOpInf(X[:, 1]; options=options, n_state=n, n_input=1, rank=n)
    for k in 2:K
        LnL.stream!(s, X[:, k])
    end
    E, idx = LnL.finite_diff_matrix("FE", K, Δt)
    op = LnL.compute_stream_operators(s, E, idx; U=Matrix(U'))
    op_tuple = LnL.compute_stream_operators(s, E, (idx[1], idx[end]); U=Matrix(U'))
    op_batch = LnL.opinf(X, s.V, options; U=U)
    @test op.A ≈ op_batch.A rtol=1e-6
    @test op.B ≈ op_batch.B rtol=1e-6
    @test op_tuple.A ≈ op.A
    @test size(op.O) == (n, n + 1)

    # reduced rank via keyword
    op3 = LnL.compute_stream_operators(s, E, idx; U=Matrix(U'), rank=3)
    @test size(op3.A) == (3, 3)
    @test size(op3.B) == (3, 1)
end

@testset "One-pass Baker iSVD (streamed derivatives)" begin
    s = LnL.OnePassStreamingOpInf(X[:, 1]; xdot_init=Xdot[:, 1], options=options, n_state=n,
                                  n_input=1, rank=n, finite_diff=false)
    for k in 2:K
        LnL.stream!(s, X[:, k], Xdot[:, k])
    end
    @test norm(s.V' * s.V - I) < 1e-8   # see the BLAS-dependent rounding note above
    @test s.Σ ≈ svdvals(X) rtol=1e-8
    @test s.S ≈ svdvals(Xdot) rtol=1e-8
    op = LnL.compute_stream_operators(s; U=Matrix(U'))
    op_batch = LnL.opinf(X, s.V, options; U=U, Xdot=Xdot)
    @test op.A ≈ op_batch.A atol=1e-8
    @test op.B ≈ op_batch.B atol=1e-8
end

@testset "One-pass sketchy SVD" begin
    E, idx = LnL.finite_diff_matrix("FE", K, Δt)

    # snapshot-by-snapshot
    s = LnL.OnePassStreamingOpInf(; options=options, n_state=n, n_input=1, n_snapshots=K,
                                  rank=n, isvd_method=:sketchy)
    for k in 1:K
        LnL.stream!(s, X[:, k])
    end
    @test s.num_of_snapshots == K
    LnL.compute_svd_sketchy!(s; rank=n)
    @test norm(s.V' * s.V - I) < 1e-10
    @test s.Σ ≈ svdvals(X) rtol=1e-8
    op = LnL.compute_stream_operators(s, E, idx; U=Matrix(U'), rank=n)
    op_tuple = LnL.compute_stream_operators(s, E, (idx[1], idx[end]); U=Matrix(U'), rank=n)
    op_batch = LnL.opinf(X, s.V, options; U=U)
    @test op.A ≈ op_batch.A rtol=1e-6
    @test op_tuple.A ≈ op.A

    # block streaming gives the same sketches up to round-off
    sb = LnL.OnePassStreamingOpInf(; options=options, n_state=n, n_input=1, n_snapshots=K,
                                   rank=n, isvd_method=:sketchy)
    for blk in LnL.streamify(X, 20)
        LnL.stream!(sb, blk)
    end
    @test sb.num_of_snapshots == K
    LnL.compute_svd_sketchy!(sb; rank=n)
    @test sb.Σ ≈ svdvals(X) rtol=1e-8

    # purge! frees the sketches but keeps the SVD factors
    LnL.purge!(s)
    @test isempty(s.Xcore) && isempty(s.Xrange) && isempty(s.Ξ)
    @test size(s.V) == (n, n)
end
