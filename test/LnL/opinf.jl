@testset "Standard OpInf" begin
    options = LnL.LSOpInfOption(
        system=LnL.SystemStructure(
            state=[1,2],
            control=1,
            output=1,
        ),
        vars=LnL.VariableStructure(
            N=1,
        ),
        data=LnL.DataStructure(
            Δt=0.01,
            deriv_type="BE"
        ),
        optim=LnL.OptimizationSetting(
            verbose=true,
        ),
    )

    X = Array[]
    U = Array[]
    Xdot = Array[]
    tspan = (0.0, 10.0)
    h = 0.01
    tdim = Int(ceil((tspan[2] - tspan[1]) / h))
    for _ in 1:20
        u = zeros(1, tdim+1)
        u[rand(1:tdim, 5)] .= 0.2
        _, foo, bar = rk4(quad2d, tspan, rand(2), h, u=u)
        push!(X, foo)
        push!(U, u)
        push!(Xdot, bar)
    end
    X = reduce(hcat, X)  
    U = reduce(hcat, U)
    Xdot = reduce(hcat, Xdot)
    Y = 0.5 * X[2, :]

    op1 = LnL.opinf(X, 1.0I(2), options; U=U, Y=Y, Xdot=Xdot)

    full_op = LnL.Operators(
        A=[-2 0; 0 -1], A2u=[0 1 0; 0 1 0], B=reshape([0; -0.01],2,1), C=reshape([0, 0.5],1,2)
    )
    op2 = LnL.opinf(X, 1.0I(2), full_op, options; U=U, Y=Y)

    @test op1.A ≈ full_op.A
    @test op1.A2u ≈ full_op.A2u
    @test op1.B ≈ full_op.B
    @test op1.C ≈ full_op.C

    @test op2.A ≈ full_op.A
    @test op2.A2u ≈ full_op.A2u
    @test op2.B ≈ full_op.B
    @test op2.C ≈ full_op.C

    op3 = LnL.opinf(X, 1.0I(2), options; U=U, Y=Y)
    @test size(op3.A) == (2, 2)

    options.data.deriv_type = "SI"
    op4 = LnL.opinf(X, 1.0I(2), options; U=U, Y=Y)
    @test size(op4.A) == (2, 2)

    options.data.deriv_type = "FE"
    op5 = LnL.opinf(X, 1.0I(2), options; U=U, Y=Y)
    @test size(op5.A) == (2, 2)
end