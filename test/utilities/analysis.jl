@testset "Relative state error" begin
    Xf = [1.0 2.0; 3.0 4.0]
    X = [0.5 1.0; 2.0 0.0]
    Vr = [1.0 0.0; 0.0 1.0]
    expected_SE = norm(Xf - Vr * X, 2) / norm(Xf, 2)
    computed_SE = LnL.rel_state_error(Xf, X, Vr)
    @test isapprox(computed_SE, expected_SE, atol=1e-5)
end

@testset "Relative output error" begin
    Yf = [1.0 2.0; 3.0 4.0]
    Y = [0.5 1.0; 2.0 0.0]
    expected_OE = norm(Yf - Y, 2) / norm(Yf, 2)
    computed_OE = LnL.rel_output_error(Yf, Y)
    @test isapprox(computed_OE, expected_OE, atol=1e-5)
end

@testset "Projection error" begin
    Xf = [1.0, 2.0, 3.0]
    Vr = [0.5 0.5 0.5; 0.5 0.5 0.5; 0.5 0.5 0.5]
    PE = LnL.proj_error(Xf, Vr)
    @test isapprox(PE, 1.2174328963613794, atol=1e-8)
end

@testset "Compute all errors" begin
    Xf = [1.0 2.0; 3.0 4.0]
    Yf = [1.0 2.0]
    Xint = [0.9 1.8; 2.7 3.6]
    Yint = [0.9 1.8]
    Xinf = [1.1 2.1; 3.1 4.1]
    Yinf = [1.1 2.1]
    Vr = [0.5 0.5; 0.5 0.5]
    
    PE, ISE, IOE, OSE, OOE = LnL.compute_all_errors(Xf, Yf, Xint, Yint, Xinf, Yinf, Vr)
    
    @test PE >= 0
    @test ISE >= 0
    @test IOE >= 0
    @test OSE >= 0
    @test OOE >= 0
end

@testest "EP constraint residual" begin
    n = 3
    X = rand(n, n^2)
    redundant = true
    with_moment = true
    ϵX, mmt = LnL.ep_constraint_residual(X, n, redundant; with_moment=with_moment)
    @test abs(ϵX) > 0
    @test abs(mmt) > 0

    X = rand(n, n*(n+1)÷2)
    redundant = false
    with_moment = true
    ϵX, mmt = LnL.ep_constraint_residual(X, n, redundant; with_moment=with_moment)
    @test abs(ϵX) > 0
    @test abs(mmt) > 0
end

@testest "EP constraint residual" begin
    n = 3
    X = rand(n, n^2)
    data = rand(n, 5)
    redundant = true
    viol = LnL.ep_constraint_violation(data, X, redundant)
    @test abs(viol) > 0

    X = rand(n, n*(n+1)÷2)
    redundant = false
    viol = LnL.ep_constraint_violation(data, X, redundant)
    @test abs(viol) > 0

    X = rand(n, n^2)
    flag = LnL.isenergypreserving(X)
    @test flag == false
end