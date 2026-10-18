@testset "streamify (uniform size)" begin
    X = reshape(collect(1.0:24.0), 3, 8)   # fat: columns are snapshots

    S = LnL.streamify(X, 3)
    @test length(S) == 3
    @test size.(S) == [(3, 3), (3, 3), (3, 2)]
    @test reduce(hcat, S) == X

    # tall matrices are split along the rows
    St = LnL.streamify(Matrix(X'), 3)
    @test size.(St) == [(3, 3), (3, 3), (2, 3)]
    @test reduce(vcat, St) == X'

    # one snapshot per stream
    S1 = LnL.streamify(X, 1)
    @test length(S1) == 8
    @test all(S1[k] == X[:, k:k] for k in 1:8)

    # vectors are split along their length
    Sv = LnL.streamify(collect(1.0:10.0), 4)
    @test size.(Sv) == [(4, 1), (4, 1), (2, 1)]
end

@testset "streamify (varying sizes)" begin
    X = reshape(collect(1.0:24.0), 3, 8)
    S = LnL.streamify(X, [2, 3, 3])
    @test size.(S) == [(3, 2), (3, 3), (3, 3)]
    @test reduce(hcat, S) == X

    St = LnL.streamify(Matrix(X'), [5, 3])
    @test size.(St) == [(5, 3), (3, 3)]
    @test reduce(vcat, St) == X'
end
