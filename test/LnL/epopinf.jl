@testset "EP-OpInf test" begin
    system_ = LnL.SystemStructure(
        state=[1,2],
    )
    vars_ = LnL.VariableStructure(
        N=1,
    )
    data_ = LnL.DataStructure(
        Δt=1e-4,
        DS=100,
    )
    optim_ = LnL.OptimizationSetting(
        verbose=true,
        initial_guess=false,
        max_iter=1000,
        reproject=false,
        SIGE=false,
    )

    X = Array[]
    U = Array[]
    Xdot = Array[]
    tspan = (0.0, 10.0)
    h = 0.01
    tdim = Int(ceil((tspan[2] - tspan[1]) / h))
    for _ in 1:20
        u = zeros(1, tdim+1)
        _, foo, bar = rk4(quad2d, tspan, rand(2), h, u=u)
        push!(X, foo)
        push!(U, u)
        push!(Xdot, bar)
    end
    X = reduce(hcat, X)  
    U = reduce(hcat, U)
    Xdot = reduce(hcat, Xdot)

    full_op = LnL.Operators(
        A=[-2 0; 0 -1], A2u=[0 1 0; 0 1 0]
    )

    options = LnL.EPHECOpInfOption(
        system=system_,
        vars=vars_,
        data=data_,
        optim=optim_,
    )
    op1 = LnL.epopinf(X, 1.0I(2), options; Xdot=Xdot)

    @test size(op1.A) == (2,2)
    @test size(op1.A2u) == (2,3)

    options.optim.nonredundant_operators = false
    op1 = LnL.epopinf(X, 1.0I(2), options; Xdot=Xdot)

    @test size(op1.A) == (2,2)
    @test size(op1.A2) == (2,4)

    #####

    optim_.nonredundant_operators = true
    options = LnL.EPSICOpInfOption(
        system=system_,
        vars=vars_,
        data=data_,
        optim=optim_,
        ϵ=1e-3,
    )
    op2 = LnL.epopinf(X, 1.0I(2), options; Xdot=Xdot)

    @test size(op2.A) == (2,2)
    @test size(op2.A2u) == (2,3)

    options.optim.nonredundant_operators = false
    op2 = LnL.epopinf(X, 1.0I(2), options; Xdot=Xdot)

    @test size(op2.A) == (2,2)
    @test size(op2.A2) == (2,4)

    #####

    optim_.nonredundant_operators = true
    options = LnL.EPPOpInfOption(
        system=system_,
        vars=vars_,
        data=data_,
        optim=optim_,
        α=1e8,
    )
    op3 = LnL.epopinf(X, 1.0I(2), options; Xdot=Xdot)

    @test size(op3.A) == (2,2)
    @test size(op3.A2u) == (2,3)

    options.optim.nonredundant_operators = false
    op3 = LnL.epopinf(X, 1.0I(2), options; Xdot=Xdot)

    @test size(op3.A) == (2,2)
    @test size(op3.A2) == (2,4)

    #####

    optim_.SIGE = true
    optim_.nonredundant_operators = true

    options = LnL.EPHECOpInfOption(
        system=system_,
        vars=vars_,
        data=data_,
        optim=optim_,
    )
    op1 = LnL.epopinf(X, 1.0I(2), options; Xdot=Xdot, IG=full_op)

    @test size(op1.A) == (2,2)
    @test size(op1.A2u) == (2,3)

    options = LnL.EPSICOpInfOption(
        system=system_,
        vars=vars_,
        data=data_,
        optim=optim_,
        ϵ=1e-3,
    )
    op2 = LnL.epopinf(X, 1.0I(2), options; Xdot=Xdot, IG=full_op)

    @test size(op2.A) == (2,2)
    @test size(op2.A2u) == (2,3)

    options = LnL.EPPOpInfOption(
        system=system_,
        vars=vars_,
        data=data_,
        optim=optim_,
        α=1e8,
    )
    op3 = LnL.epopinf(X, 1.0I(2), options; Xdot=Xdot, IG=full_op)

    @test size(op3.A) == (2,2)
    @test size(op3.A2u) == (2,3)
end

