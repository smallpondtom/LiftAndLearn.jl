using LiftAndLearn
using LinearAlgebra
using Random
using Test

const LnL = LiftAndLearn
const CG = LiftAndLearn.ChaosGizmo


function lorenz_jacobian(ops::LnL.Operators, x::AbstractArray)
    n = size(x,1)
    return ops.A + ops.H * kron(I(n),x) + ops.H*kron(x,I(n))
end

function RK4(J, Q, dt)
    # RK4 steps
    k1 = J * Q
    k2 = J * (Q + 0.5*dt*k1)
    k3 = J * (Q + 0.5*dt*k2)
    k4 = J * (Q + dt*k3)
    # Update the perturbation state
    Qnew = Q + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return Qnew
end

function lorenz_integrator(ops::LnL.Operators, tspan::AbstractArray, IC::Array; params...)
    K = length(tspan)
    N = size(IC,1)
    f = let A = ops.A, H = ops.H, F = ops.F
        if ops.H == 0
            (x, t) -> A*x + F*LnL.vech(x*x')
        else
            (x, t) -> A*x + H*kron(x,x)
        end
    end
    xk = zeros(N,K)
    xk[:,1] = IC

    for k in 2:K
        timestep = tspan[k] - tspan[k-1]
        k1 = f(xk[:,k-1], tspan[k-1])
        k2 = f(xk[:,k-1] + 0.5 * timestep * k1, tspan[k-1] + 0.5 * timestep)
        k3 = f(xk[:,k-1] + 0.5 * timestep * k2, tspan[k-1] + 0.5 * timestep)
        k4 = f(xk[:,k-1] + timestep * k3, tspan[k-1] + timestep)
        xk[:,k] = xk[:,k-1] + (timestep / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    end
    return xk
end

function lorenz9(u, t, p)
    σ, r, b1, b2, b3, b4, b5, b6 = p
    du = zeros(9)
    du[1] = -σ*b1*u[1] - u[2]*u[4] + b4*u[4]^2 + b3*u[3]*u[5] - σ*b2*u[7]
    du[2] = -σ*u[2] + u[1]*u[4] - u[2]*u[5] + u[4]*u[5] - σ*u[9]/2
    du[3] = -σ*b1*u[3] + u[2]*u[4] - b4*u[2]^2 - b3*u[1]*u[5] + σ*b2*u[8]
    du[4] = -σ*u[4] - u[2]*u[3] - u[2]*u[5] + u[4]*u[5] + σ*u[9]/2
    du[5] = -σ*b5*u[5] + u[2]^2/2 - u[4]^2/2
    du[6] = -b6*u[6] + u[2]*u[9] - u[4]*u[9]
    du[7] = -b1*u[7] - r*u[1] + 2*u[5]*u[8] - u[4]*u[9]
    du[8] = -b1*u[8] + r*u[3] - 2*u[5]*u[7] + u[2]*u[9]
    du[9] = -u[9] - r*u[2] + r*u[4] - 2*u[2]*u[6] + 2*u[4]*u[6] + u[4]*u[7] - u[2]*u[8]
    return du
end

function lorenz9_integrator(ops::LnL.Operators, tspan::AbstractArray, IC::Array; params...)
    sigma = get(params, :sigma, 0)
    r = get(params, :r, 0)
    b1 = get(params, :b1, 0)
    b2 = get(params, :b2, 0)
    b3 = get(params, :b3, 0)
    b4 = get(params, :b4, 0)
    b5 = get(params, :b5, 0)
    b6 = get(params, :b6, 0)
    p = [sigma, r, b1, b2, b3, b4, b5, b6]

    K = length(tspan)
    N = size(IC,1)
    f = lorenz9
    xk = zeros(N,K)
    xk[:,1] = IC

    for k in 2:K
        timestep = tspan[k] - tspan[k-1]
        k1 = f(xk[:,k-1], tspan[k-1], p)
        k2 = f(xk[:,k-1] + 0.5 * timestep * k1, tspan[k-1] + 0.5 * timestep, p)
        k3 = f(xk[:,k-1] + 0.5 * timestep * k2, tspan[k-1] + 0.5 * timestep, p)
        k4 = f(xk[:,k-1] + timestep * k3, tspan[k-1] + timestep, p)
        xk[:,k] = xk[:,k-1] + (timestep / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    end
    return xk
end


@testset "LE Full model" begin
    # Lorenz system Definitions
    sigma = 10.
    rho = 28.
    beta = 8. / 3.
    x0 = [1.5, -1.5, 20.]
    t0 = 0.
    Δt = 1e-2

    A = [
        -sigma sigma 0.;
        rho -1. 0.;
        0. 0. -beta
    ]

    H = zeros(3,9)
    H[2,3] = -0.5
    H[2,7] = -0.5
    H[3,2] = 0.5
    H[3,4] = 0.5

    lorenz_ops = LnL.Operators(A=A, H=H)


    N = 10000
    τ = 100
    m = 3

    # Algorithm 1: without using Jacobian
    options = CG.LE_options(N=N, τ=τ, τ0=0.0, Δt=Δt, m=m, T=Δt, ϵ=1e-8, verbose=true, history=true)
    λ, λall = CG.lyapunovExponent(lorenz_ops, lorenz_integrator, x0, options)
    dky = CG.kaplanYorkeDim(λ; sorted=false)
    @test dky ≈ 2.06 atol=2e-2

    # Algorithm 2: using Jacobian
    options = CG.LE_options(N=N, τ=τ, τ0=0.0, Δt=Δt, m=m, T=Δt, ϵ=1e-8, verbose=false, history=true)
    λ, λall = CG.lyapunovExponentJacobian(lorenz_ops, lorenz_integrator, lorenz_jacobian, x0, options)
    dky = CG.kaplanYorkeDim(λ; sorted=true)
    @test dky ≈ 2.06 atol=2e-2

    # Test all perturbation integrators
    # Euler: Use a smaller timestep of 1e-3 to avoid divergence
    options = CG.LE_options(N=N, τ=τ, τ0=0.0, Δt=1e-3, m=m, T=1e-3, ϵ=1e-8, verbose=false, history=true, pert_integrator=CG.EULER)
    λ, λall = CG.lyapunovExponentJacobian(lorenz_ops, lorenz_integrator, lorenz_jacobian, x0, options)
    dky = CG.kaplanYorkeDim(λ; sorted=false)
    @test dky ≈ 2.06 atol=2e-2
    # RK2
    options = CG.LE_options(N=N, τ=τ, τ0=0.0, Δt=Δt, m=m, T=Δt, ϵ=1e-8, verbose=false, history=true, pert_integrator=CG.RK2)
    λ, λall = CG.lyapunovExponentJacobian(lorenz_ops, lorenz_integrator, lorenz_jacobian, x0, options)
    dky = CG.kaplanYorkeDim(λ; sorted=false)
    @test dky ≈ 2.06 atol=2e-2
    # SSPRK3
    options = CG.LE_options(N=N, τ=τ, τ0=0.0, Δt=Δt, m=m, T=Δt, ϵ=1e-8, verbose=false, history=true, pert_integrator=CG.SSPRK3)
    λ, λall = CG.lyapunovExponentJacobian(lorenz_ops, lorenz_integrator, lorenz_jacobian, x0, options)
    dky = CG.kaplanYorkeDim(λ; sorted=false)
    @test dky ≈ 2.06 atol=2e-2
    # RALSTON4
    options = CG.LE_options(N=N, τ=τ, τ0=0.0, Δt=Δt, m=m, T=Δt, ϵ=1e-8, verbose=false, history=true, pert_integrator=CG.RALSTON4)
    λ, λall = CG.lyapunovExponentJacobian(lorenz_ops, lorenz_integrator, lorenz_jacobian, x0, options)
    dky = CG.kaplanYorkeDim(λ; sorted=false)
    @test dky ≈ 2.06 atol=2e-2
end


@testset  "Reduced model Lyapunov Exponent with 9D Lorenz" begin
    # Define the 9-dimensional Lorenz system
    a = 0.5  # wave number in the horizontal direction
    b1 = 4*(1+a^2) / (1+2*a^2)
    b2 = (1+2*a^2) / (2*(1+a^2))
    b3 = 2*(1-a^2) / (1+a^2)
    b4 = a^2 / (1+a^2)
    b5 = 8*a^2 / (1+2*a^2)
    b6 = 4 / (1+2*a^2)

    sigma = 0.5  # Prandtl number
    r = 15.10  # reduced Rayleigh number

    n = 9  # dimension of the system
    # Linear operator
    A = zeros(n,n)
    A[1,1] = -sigma * b1
    A[1,7] = -sigma * b2
    A[2,2] = -sigma
    A[2,9] = -sigma / 2
    A[3,3] = -sigma * b1
    A[3,8] = sigma * b2
    A[4,4] = -sigma
    A[4,9] = sigma / 2
    A[5,5] = -sigma * b5
    A[6,6] = -b6
    A[7,1] = -r
    A[7,7] = -b1
    A[8,3] = r
    A[8,8] = -b1
    A[9,2] = -r
    A[9,4] = r
    A[9,9] = -1

    # Quadratic operator
    indices = [
        (2,4,1), (4,4,1), (3,5,1),
        (1,4,2), (2,5,2), (4,5,2),
        (2,4,3), (2,2,3), (1,5,3),
        (2,3,4), (2,5,4), (4,5,4),
        (2,2,5), (4,4,5),
        (2,9,6), (4,9,6),
        (5,8,7), (4,9,7),
        (5,7,8), (2,9,8),
        (2,6,9), (4,6,9), (4,7,9), (2,8,9)
    ]
    values = [
        -1, b4, b3,
        1, -1, 1,
        1, -b4, -b3,
        -1, -1, 1,
        0.5, -0.5,
        1, -1,
        2, -1,
        -2, 1,
        -2, 2, 1, -1
    ]
    H = LnL.makeQuadOp(n, indices, values; which_quad_term="H")
    F = LnL.makeQuadOp(n, indices, values; which_quad_term="F")
    lorenz9_ops = LnL.Operators(A=A, H=H, F=F)

    x0 = [0.01, 0, 0.01, 0.0, 0.0, 0.0, 0, 0, 0.01] 
    options = CG.LE_options(N=1e4, τ=1e2, τ0=0.0, Δt=1e-2, m=n, T=1e-2, verbose=false, history=true)
    λ, _ = CG.lyapunovExponentJacobian(lorenz9_ops, lorenz_integrator, lorenz_jacobian, x0, options)
    dky = CG.kaplanYorkeDim(λ; sorted=false)

    λ2, _ = CG.lyapunovExponent(lorenz9_ops, lorenz9_integrator, x0, options; 
                                    sigma=sigma, r=r, b1=b1, b2=b2, b3=b3, b4=b4, b5=b5, b6=b6)
    dky2 = CG.kaplanYorkeDim(λ2; sorted=false)

    @test dky ≈ dky2 atol=5e-1

    # Lyapunov exponent of POD reduced model
    data1 = lorenz_integrator(lorenz9_ops, 0:1e-2:1e3, 2*rand(9).-1)
    data2 = lorenz_integrator(lorenz9_ops, 0:1e-2:1e3, 2*rand(9).-1)
    data3 = lorenz_integrator(lorenz9_ops, 0:1e-2:1e3, 2*rand(9).-1)
    data4 = lorenz_integrator(lorenz9_ops, 0:1e-2:1e3, 2*rand(9).-1)
    data5 = lorenz_integrator(lorenz9_ops, 0:1e-2:1e3, 2*rand(9).-1)
    data = hcat(data1, data2, data3, data4, data5)
    r = 5
    rmax = 8
    Vr = svd(data).U[:,1:rmax]   # choose rmax columns
    rom_option = LnL.LSOpInfOption(
        system=LnL.SystemStructure(is_lin=true, is_quad=true),
    )
    oprom = LnL.pod(lorenz9_ops, Vr, rom_option)

    options = CG.LE_options(N=1e4, τ=1e2, τ0=0.0, Δt=1e-2, m=r, T=1e-2, verbose=false, history=true)
    λr, _ = CG.lyapunovExponentJacobian(oprom, lorenz_integrator, lorenz_jacobian, Vr' * x0, options)
    dkyr = CG.kaplanYorkeDim(λr; sorted=false)

    # r < rmax
    λr2, _ = CG.lyapunovExponent(oprom, lorenz_integrator, Vr[:,1:r], x0, options)
    dkyr2 = CG.kaplanYorkeDim(λr2; sorted=false)

    # r = rmax
    λr3, _ = CG.lyapunovExponent(oprom, lorenz_integrator, Vr, x0, options)
    dkyr3 = CG.kaplanYorkeDim(λr3; sorted=false)
    @test dkyr ≈ dkyr3 atol=2.0
end