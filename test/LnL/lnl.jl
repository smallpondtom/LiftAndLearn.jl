@testset "Lift And Learn test" begin
    options = LnL.LSOpInfOption(
        system=LnL.SystemStructure(
            state=[1, 2],
            control=1,
            output=1,
        ),
        vars=LnL.VariableStructure(
            N=2,
            N_lift=4,
        ),
        data=LnL.DataStructure(
            Δt=1e-4,
            DS=100,
        ),
        optim=LnL.OptimizationSetting(
            verbose=true,
            nonredundant_operators=true,
            reproject=true,
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
        u[rand(1:tdim, 5)] .= 0.1
        _, foo, bar = rk4(nonlinear_pendulum, tspan, rand(2), h, u=u)
        push!(X, foo)
        push!(U, u)
        push!(Xdot, bar)
    end
    X = reduce(hcat, X)  
    U = reduce(hcat, U)
    Xdot = reduce(hcat, Xdot)
    Y = -0.1 * X[1, :] + 0.5 * X[2, :]

    Xsep = [X[1:1, :], X[2:2, :]]
    lifter = LnL.lifting(2, 4, [x -> sin.(x[1]), x -> cos.(x[1])])
    Xlift = lifter.map(Xsep)

    full_op = begin
        A = zeros(2,2)
        A[1,2] = 1.0

        f = (x,u) -> [0; -sin(x[1])]

        LnL.Operators(
            A=A, f=f, B=reshape([0.01; -0.02],2,1), C=reshape([-0.1, 0.5],1,2)
        )
    end

    op1 = LnL.opinf(Xlift, 1.0I(4), lifter, full_op, options; U=U, Y=Y)

    @test op1.A[1:2,1:2] ≈ full_op.A
    @test op1.B[1:2] ≈ full_op.B
    @test reshape(op1.C[1:2],1,2) ≈ full_op.C

    options.optim.reproject = false
    op3 = LnL.opinf(Xlift, 1.0I(4), lifter, full_op, options; U=U, Y=Y)
    @test size(op3.A) == (4, 4)
end