## Packages
using Kronecker
using LinearAlgebra
using ProgressMeter
import Random: rand, rand!
import DifferentialEquations: solve, ODEProblem, RK4
import MatrixEquations: lyapc
import HSL_jll

# My modules
using LiftAndLearn
const LnL = LiftAndLearn
const LFI = LyapInf

## 
function nonlinear_pendulum_example()
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
    num_ic = 5  # number of initial conditions
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
        data = sol[:,:]
        ddata = sol(tspan, Val{1})[:,:]

        push!(X, data[:,1:DS:end])
        push!(Xdot, ddata[:,1:DS:end])
    end
    X = reduce(hcat, X)
    Xdot = reduce(hcat, Xdot)

    ## Compute the Lyapunov Function using the intrusive method
    lyapinf_options = LFI.Int_LyapInf_options(
        extra_iter=3,
        optimizer="Ipopt",
        ipopt_linear_solver="ma86",
        verbose=true,
        optimize_PandQ="P",
        HSL_lib_path=HSL_jll.libhsl_path,
    )
    # Pi = lyapc([0 1; -1 -0.5]', 1.0I(2))
    # Pi = [Pi zeros(2,2); zeros(2,4)]
    op = LnL.operators(A=A, H=H, F=F)
    P, Q, cost, ∇cost = LFI.Int_LyapInf(op, X, lyapinf_options)
    ρ_min, ρ_max = LFI.DoA(P[1:2,1:2])
    return P, Q, cost, ∇cost, ρ_min, ρ_max, A, F, lifter
end


function E1_example()
    A = [-2.0 0.0; 0.0 -1.0]
    H = LnL.makeQuadOp(2, [(1,2,1), (1,2,2)], [1.0, 1.0])
    F = LnL.H2F(H)

    function E1!(xdot, x, p, t)
        xdot .= A * x + H * (x ⊗ x)
    end

    # Generate the data
    num_ic = 5  # number of initial conditions
    tf = 10.0
    dt = 0.001
    tspan = 0.0:dt:tf
    DS = 100  # down-sampling

    X = []
    Xdot = []
    for i in 1:num_ic
        x0 = 4*rand(2) .- 2
        prob = ODEProblem(E1!, x0, (0, tf))
        sol = solve(prob, RK4(); dt=dt, adaptive=false)
        data = sol[:,:]
        ddata = sol(tspan, Val{1})[:,:]

        push!(X, data[:,1:DS:end])
        push!(Xdot, ddata[:,1:DS:end])
    end
    X = reduce(hcat, X)
    Xdot = reduce(hcat, Xdot)

    ## Compute the Lyapunov Function using the intrusive method
    lyapinf_options = LFI.Int_LyapInf_options(
        extra_iter=3,
        optimizer="Ipopt",
        ipopt_linear_solver="ma86",
        verbose=true,
        optimize_PandQ="P",
        HSL_lib_path=HSL_jll.libhsl_path,
    )
    Pi = [1 0; 0 1]
    op = LnL.operators(A=A, H=H, F=F)
    P, Q, cost, ∇cost = LFI.Int_LyapInf(op, X, lyapinf_options; Pi=Pi)
    ρ_min, ρ_max = LFI.DoA(P)
    ρ_est = LFI.est_stability_rad(A, H, P)
    return P, Q, cost, ∇cost, ρ_min, ρ_max, ρ_est, A, F
end


function E2_example()
    A = zeros(3,3)
    A[1,2] = -1.0
    A[2,1] = 1.0
    A[2,2] = -1.0
    H = LnL.makeQuadOp(3, [(2,3,2), (1,2,3)], [1.0, -2.0])
    F = LnL.H2F(H)

    function E1!(xdot, x, p, t)
        xdot .= A * x + H * (x ⊗ x)
    end

    # Generate the data
    num_ic = 5  # number of initial conditions
    tf = 10.0
    dt = 0.001
    tspan = 0.0:dt:tf
    DS = 100  # down-sampling

    # Lifting
    lifter = LnL.lifting(2, 3, [x -> x[1]^2])

    X = []
    Xdot = []
    for i in 1:num_ic
        x0 = lifter.map(2 * rand(2) .- 1)
        prob = ODEProblem(E1!, x0, (0, tf))
        sol = solve(prob, RK4(); dt=dt, adaptive=false)
        data = sol[:,:]
        ddata = sol(tspan, Val{1})[:,:]

        push!(X, data[:,1:DS:end])
        push!(Xdot, ddata[:,1:DS:end])
    end
    X = reduce(hcat, X)
    Xdot = reduce(hcat, Xdot)

    ## Compute the Lyapunov Function using the intrusive method
    lyapinf_options = LFI.Int_LyapInf_options(
        extra_iter=3,
        optimizer="Ipopt",
        ipopt_linear_solver="ma86",
        verbose=true,
        optimize_PandQ="P",
        HSL_lib_path=HSL_jll.libhsl_path,
    )
    op = LnL.operators(A=A, H=H, F=F)
    P, Q, cost, ∇cost = LFI.Int_LyapInf(op, X, lyapinf_options)
    ρ_min, ρ_max = LFI.DoA(P)
    return P, Q, cost, ∇cost, ρ_min, ρ_max, A, F, lifter
end


function E3_example()
    A = zeros(4,4)
    A[1,1] = -1.0
    A[2,2] = -1.0
    A[3,3] = -1.0
    H = LnL.makeQuadOp(4, [(2,4,1), (1,2,2), (3,3,4)], [1.0, 1.0, -2.0])
    F = LnL.H2F(H)

    function E1!(xdot, x, p, t)
        xdot .= A * x + H * (x ⊗ x)
    end

    # Generate the data
    num_ic = 5  # number of initial conditions
    tf = 10.0
    dt = 0.001
    tspan = 0.0:dt:tf
    DS = 100  # down-sampling

    # Lifting
    lifter = LnL.lifting(3, 4, [x -> x[3]^2])

    X = []
    Xdot = []
    for i in 1:num_ic
        foo1, foo2, foo3 = 4.0*rand(3) .- 2.0
        x0 = lifter.map([foo1, foo2, foo3])
        prob = ODEProblem(E1!, x0, (0, tf))
        sol = solve(prob, RK4(); dt=dt, adaptive=false)
        data = sol[:,:]
        ddata = sol(tspan, Val{1})[:,:]

        push!(X, data[:,1:DS:end])
        push!(Xdot, ddata[:,1:DS:end])
    end
    X = reduce(hcat, X)
    Xdot = reduce(hcat, Xdot)

    ## Compute the Lyapunov Function using the intrusive method
    lyapinf_options = LFI.Int_LyapInf_options(
        extra_iter=3,
        optimizer="Ipopt",
        ipopt_linear_solver="ma86",
        verbose=true,
        optimize_PandQ="P",
        HSL_lib_path=HSL_jll.libhsl_path,
    )
    op = LnL.operators(A=A, H=H, F=F)
    P, Q, cost, ∇cost = LFI.Int_LyapInf(op, X, lyapinf_options)
    ρ_min, ρ_max = LFI.DoA(P[1:3,1:3])
    return P, Q, cost, ∇cost, ρ_min, ρ_max, A, F, lifter
end


function E5_example()
    A = zeros(4,4)
    A[1,2] = 1.0
    A[2,2] = -2.0
    A[2,3] = -1.0
    H = LnL.makeQuadOp(4, [(3,4,2), (2,4,3), (2,3,4)], [0.81, 1.0, -1.0])
    F = LnL.H2F(H)

    function E1!(xdot, x, p, t)
        xdot .= A * x + H * (x ⊗ x)
    end

    # Generate the data
    num_ic = 5  # number of initial conditions
    tf = 10.0
    dt = 0.001
    tspan = 0.0:dt:tf
    DS = 100  # down-sampling

    # Lifting
    lifter = LnL.lifting(2, 4, [x -> sin.(x[1]), x -> cos.(x[1])])

    X = []
    Xdot = []
    for i in 1:num_ic
        foo1, foo2 = 2π/3*rand(3) .- π/3
        x0 = lifter.map([foo1, foo2])
        prob = ODEProblem(E1!, x0, (0, tf))
        sol = solve(prob, RK4(); dt=dt, adaptive=false)
        data = sol[:,:]
        ddata = sol(tspan, Val{1})[:,:]

        push!(X, data[:,1:DS:end])
        push!(Xdot, ddata[:,1:DS:end])
    end
    X = reduce(hcat, X)
    Xdot = reduce(hcat, Xdot)

    ## Compute the Lyapunov Function using the intrusive method
    lyapinf_options = LFI.Int_LyapInf_options(
        extra_iter=3,
        optimizer="Ipopt",
        ipopt_linear_solver="ma86",
        verbose=true,
        optimize_PandQ="P",
        HSL_lib_path=HSL_jll.libhsl_path,
    )
    op = LnL.operators(A=A, H=H, F=F)
    P, Q, cost, ∇cost = LFI.Int_LyapInf(op, X, lyapinf_options)
    ρ_min, ρ_max = LFI.DoA(P[1:2,1:2])
    return P, Q, cost, ∇cost, ρ_min, ρ_max, A, F, lifter
end

##
P, Q, cost, ∇cost, ρ_min, ρ_max, A, F, lifter = nonlinear_pendulum_example()
c_star = LFI.doa_sampling(
    (x) -> x' * P * x, 
    (x) -> x' * P * A * x + x' * P * F * (x ⊘ x), 
    500, 2, (-5,5), Nl=4, lifter=lifter
)


##
P, Q, cost, ∇cost, ρ_min, ρ_max, ρ_est, A, F = E1_example()
c_star = LFI.doa_sampling(
    (x) -> x' * P * x, 
    (x) -> x' * P * A * x + x' * P * F * (x ⊘ x), 
    500, 2, (-5,5)
)

##
P, Q, cost, ∇cost, ρ_min, ρ_max, A, F, lifter = E2_example()
c_star = LFI.doa_sampling(
    (x) -> x' * P * x, 
    (x) -> x' * P * A * x + x' * P * F * (x ⊘ x), 
    500, 2, (-5,5), Nl=3, lifter=lifter
)

##
P, Q, cost, ∇cost, ρ_min, ρ_max, A, F, lifter = E3_example()
c_star = LFI.doa_sampling(
    (x) -> x' * P * x, 
    (x) -> x' * P * A * x + x' * P * F * (x ⊘ x), 
    500, 3, (-5,5), Nl=4, lifter=lifter
)

##
P, Q, cost, ∇cost, ρ_min, ρ_max, A, F, lifter = E5_example()
c_star = LFI.doa_sampling(
    (x) -> x' * P * x, 
    (x) -> x' * P * A * x + x' * P * F * (x ⊘ x), 
    500, 2, (-5,5), Nl=4, lifter=lifter
)