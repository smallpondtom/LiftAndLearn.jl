using Kronecker
using LinearAlgebra
using Test
import Random: rand, rand!
import DifferentialEquations: solve, ODEProblem, RK4
# import HSL_jll


## My modules
using LiftAndLearn
const LnL = LiftAndLearn
const LFI = LyapInf

@testset "Test lifted LyapInf intrusive" begin
    A = zeros(4,4)
    A[1,2] = 1
    A[2,2] = -0.5
    A[2,3] = -1
    H = LnL.makeQuadOp(4, [(2,4,3), (2,3,4)], [1, -1])
    F = LnL.H2F(H)

    function nonlinear_pendulum!(xdot, x, p, t)
        xdot .= A * x + H * (x ⊗ x)
    end

    # Generate the data
    num_ic = 3  # number of initial conditions
    tf = 10.0
    dt = 0.001
    tspan = 0.0:dt:tf
    DS = 100  # down-sampling

    # Lifting
    lifter = LnL.lifting(2, 4, [x -> sin.(x[1]), x -> cos.(x[1])])

    X = []
    Xdot = []
    for i in 1:num_ic
        x0 = lifter.map(π*rand(2) .- π/2)
        prob = ODEProblem(nonlinear_pendulum!, x0, (0, tf))
        sol = solve(prob, RK4(); dt=dt, adaptive=false)
        data = sol[1:4,:]
        ddata = sol(tspan, Val{1})[1:4,:]

        push!(X, data[:,1:DS:end])
        push!(Xdot, ddata[:,1:DS:end])
    end
    X = reduce(hcat, X)
    Xdot = reduce(hcat, Xdot)

    ## Compute the Lyapunov Function using the intrusive method
    lyapinf_options = LFI.Int_LyapInf_options(
        optimizer="Ipopt",
        verbose=true,
        optimize_PandQ="P",
        extra_iter=1,
        opt_max_iter=10,
    )
    op = LnL.operators(A=A, H=H, F=F)
    P, Q, cost, ∇cost = LFI.Int_LyapInf(op, X, lyapinf_options)
    ρ_min, ρ_max = LFI.DoA(P[1:2,1:2])

    @test isposdef(P) || ∇cost <= 0.1
    V = (x) -> x' * P * x
    Vdot = (x) -> x' * P * A * x + x' * P * F * (x ⊘ x)

    c_star, c_all, x_sample = LFI.doa_sampling(
        V,
        Vdot,
        1000, 2, (-8,8); Nl=4, lifter=lifter,
        method="memoryless", history=true
    )
    @test 0 <= c_star < 1

    c_star, c_all, x_sample = LFI.doa_sampling(
        V,
        Vdot,
        1000, 2, (-8,8); Nl=4, lifter=lifter,
        method="memory", history=true
    )
    @test 0 <= c_star < 1

    c_star, c_all, x_sample = LFI.doa_sampling(
        V,
        Vdot,
        1000, 2, [(-8,8),(-8,8)]; Nl=4, lifter=lifter,
        method="enhanced", history=true
    )
    @test 0 <= c_star < 1
end


@testset "Intrusive LyapInf with all methods no lifting" begin
    A = [-2.0 0.0; 0.0 -1.0]
    H = LnL.makeQuadOp(2, [(1,2,1), (1,2,2)], [1.0, 1.0])
    F = LnL.H2F(H)

    function E1!(xdot, x, p, t)
        xdot .= A * x + H * (x ⊗ x)
    end

    # Generate the data
    num_ic = 3  # number of initial conditions
    tf = 10.0
    dt = 0.001
    tspan = 0.0:dt:tf
    DS = 100  # down-sampling

    X = []
    Xdot = []
    for i in 1:num_ic
        x0 = 3*rand(2) .- 1.5
        prob = ODEProblem(E1!, x0, (0, tf))
        sol = solve(prob, RK4(); dt=dt, adaptive=false)
        data = sol[1:2,:]
        ddata = sol(tspan, Val{1})[1:2,:]

        push!(X, data[:,1:DS:end])
        push!(Xdot, ddata[:,1:DS:end])
    end
    X = reduce(hcat, X)
    Xdot = reduce(hcat, Xdot)

    ## Compute the Lyapunov Function using the intrusive method
    lyapinf_options = LFI.Int_LyapInf_options(
        optimizer="Ipopt",
        # ipopt_linear_solver="ma57",
        verbose=true,
        optimize_PandQ="P",
        extra_iter=1,
        opt_max_iter=10,
        # HSL_lib_path=HSL_jll.libhsl_path,
        δJ=5e-1,
    )
    Pi = [1 0; 0 1]
    op = LnL.operators(A=A, H=H, F=F)
    P, Q, cost, ∇cost = LFI.Int_LyapInf(op, X, lyapinf_options; Pi=Pi)

    @test isposdef(P) || ∇cost <= 0.1
    

    ρ_min, ρ_max = LFI.DoA(P)
    ρ_est = LFI.est_stability_rad(A, H, P)


    V = (x) -> x' * P * x
    Vdot = (x) -> x' * P * A * x + x' * P * F * (x ⊘ x)
    c_star, _, _ = LFI.doa_sampling(
        V,
        Vdot,
        500, 2, (-5,5);
        method="memoryless", history=true
    )
    @test 0 <= c_star < 1
    c_star, _, _ = LFI.doa_sampling(
        V,
        Vdot,
        500, 2, (-5,5);
        method="memory", history=true
    )
    @test 0 <= c_star < 1
    c_star, _, _ = LFI.doa_sampling(
        V,
        Vdot,
        500, 2, [(-5,5),(-5,5)];
        method="enhanced", history=true
    )
    @test 0 <= c_star < 1

    lyapinf_options.optimize_PandQ = "both"
    P, Q, cost, ∇cost = LFI.Int_LyapInf(op, X, lyapinf_options; Pi=Pi)
    @test isposdef(P) || ∇cost <= 0.1
    lyapinf_options.optimize_PandQ = "together"
    P, Q, cost, ∇cost = LFI.Int_LyapInf(op, X, lyapinf_options; Pi=Pi)
    @test isposdef(P) || ∇cost <= 0.1

    lyapinf_options.optimize_PandQ = "P"
    lyapinf_options.optimizer = "SCS"
    P, Q, cost, ∇cost = LFI.Int_LyapInf(op, X, lyapinf_options; Pi=Pi)
    @test true
end


@testset "Non-Intrusive LyapInf with all methods" begin
    A = [-2.0 0.0; 0.0 -1.0]
    H = LnL.makeQuadOp(2, [(1,2,1), (1,2,2)], [1.0, 1.0])
    F = LnL.H2F(H)

    function E1!(xdot, x, p, t)
        xdot .= A * x + H * (x ⊗ x)
    end

    # Generate the data
    num_ic = 3  # number of initial conditions
    tf = 10.0
    dt = 0.001
    tspan = 0.0:dt:tf
    DS = 100  # down-sampling

    X = []
    Xdot = []
    for i in 1:num_ic
        x0 = 3*rand(2) .- 1.5
        prob = ODEProblem(E1!, x0, (0, tf))
        sol = solve(prob, RK4(); dt=dt, adaptive=false)
        data = sol[1:2,:]
        ddata = sol(tspan, Val{1})[1:2,:]

        push!(X, data[:,1:DS:end])
        push!(Xdot, ddata[:,1:DS:end])
    end
    X = reduce(hcat, X)
    Xdot = reduce(hcat, Xdot)

    ## Compute the Lyapunov Function using the intrusive method
    lyapinf_options = LFI.NonInt_LyapInf_options(
        optimizer="Ipopt",
        verbose=true,
        optimize_PandQ="P",
        extra_iter=1,
        max_iter=10,
        δJ=5e-1,
    )
    Pi = [1 0; 0 1]
    P, Q, cost, ∇cost = LFI.NonInt_LyapInf(X, Xdot, lyapinf_options; Pi=Pi)

    @test isposdef(P) || ∇cost <= 0.1

    lyapinf_options.optimize_PandQ = "both"
    P, Q, cost, ∇cost = LFI.NonInt_LyapInf(X, Xdot, lyapinf_options; Pi=Pi)
    @test isposdef(P) || ∇cost <= 0.1
    lyapinf_options.optimize_PandQ = "together"
    P, Q, cost, ∇cost = LFI.NonInt_LyapInf(X, Xdot, lyapinf_options; Pi=Pi)
    @test isposdef(P) || ∇cost <= 0.1

    lyapinf_options.optimize_PandQ = "P"
    lyapinf_options.optimizer = "SCS"
    P, Q, cost, ∇cost = LFI.NonInt_LyapInf(X, Xdot, lyapinf_options; Pi=Pi)
    @test true
end