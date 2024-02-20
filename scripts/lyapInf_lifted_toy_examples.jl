using Kronecker
using LinearAlgebra
using CairoMakie
import Random: rand, rand!
import DifferentialEquations: solve, ODEProblem, RK4
import HSL_jll

## My modules
using LiftAndLearn
const LnL = LiftAndLearn
const LFI = LyapInf

##
function nonlinear_pendulum_example(; method="P", type="I")
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
    num_ic = 20  # number of initial conditions
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

    if type == "I"
        ## Compute the Lyapunov Function using the intrusive method
        lyapinf_options = LFI.Int_LyapInf_options(
            extra_iter=3,
            optimizer="Ipopt",
            ipopt_linear_solver="ma86",
            verbose=true,
            optimize_PandQ=method,
            HSL_lib_path=HSL_jll.libhsl_path,
        )
        # Pi = lyapc([0 1; -1 -0.5]', 1.0I(2))
        # Pi = [Pi zeros(2,2); zeros(2,4)]
        op = LnL.operators(A=A, H=H, F=F)
        P, Q, cost, ∇cost = LFI.Int_LyapInf(op, X, lyapinf_options)
    elseif type == "NI"
        ## Compute the Lyapunov Function using the non-intrusive method
        lyapinf_options = LFI.NonInt_LyapInf_options(
            extra_iter=3,
            optimizer="Ipopt",
            ipopt_linear_solver="ma86",
            verbose=true,
            optimize_PandQ=method,
            HSL_lib_path=HSL_jll.libhsl_path,
        )
        P, Q, cost, ∇cost = LFI.NonInt_LyapInf(X, Xdot, lyapinf_options)
    else
        error("Invalid type")
    end
    ρ_min, ρ_max = LFI.DoA(P[1:2,1:2])
    return P, Q, cost, ∇cost, ρ_min, ρ_max, A, F, lifter
end


## Plot DoA
function plot_doa_results(A, F, c_all, c_star, x_sample, P, Vdot, lifter, xrange, yrange;
        lift=false, heatmap_lb=-100)
    fig1 = Figure(fontsize=20)
    ax1 = Axis(fig1[1,1],
        title="Level Convergence",
        ylabel=L"c_*",
        xlabel="Sample Number",
        xticks=0:100:length(c_all),
    )
    lines!(ax1, 1:length(c_all), c_all)

    fig2 = Figure(size=(800,800), fontsize=20)
    ax2 = Axis(fig2[1,1],
        title="Domain of Attraction Estimate",
        xlabel=L"x_1",
        ylabel=L"x_2",
        xlabelsize=25,
        ylabelsize=25,
        aspect=DataAspect()
    )
    # Heatmap to show area where Vdot <= 0
    xi, xf = xrange[1], xrange[2]
    yi, yf = yrange[1], yrange[2]
    xpoints = xi:0.01:xf
    ypoints = yi:0.01:yf

    if lift
        data = [Vdot(vec(lifter.map([x,y]))) for x=xpoints, y=ypoints]
    else
        data = [Vdot([x,y]) for x=xpoints, y=ypoints]
    end
    hm = heatmap!(ax2, xpoints, ypoints, data, colorrange=(heatmap_lb,0), alpha=0.7, highclip=:transparent)
    Colorbar(fig2[:, end+1], hm, label=L"\dot{V}(x) \leq 0")
    # Scatter plot of Monte Carlo samples
    scatter!(ax2, x_sample[1,:], x_sample[2,:], color=:red, alpha=0.6)
    # Plot the Lyapunov function ellipse
    α = range(0, 2π, length=1000)
    Λ, V = eigen(P)
    θ = acos(dot(V[:,1], [1,0]) / (norm(V[:,1])))
    xα = (α) -> sqrt(c_star/Λ[1])*cos(θ)*cos.(α) .+ sqrt(c_star/Λ[2])*sin(θ)*sin.(α)
    yα = (α) -> -sqrt(c_star/Λ[1])*sin(θ)*cos.(α) .+ sqrt(c_star/Λ[2])*cos(θ)*sin.(α)
    lines!(ax2, xα(α), yα(α), label="", color=:black, linewidth=2)
    # Plot the Minimal DoA with single radius
    xβ = (β) -> sqrt(c_star/maximum(Λ))*cos.(β)
    yβ = (β) -> sqrt(c_star/maximum(Λ))*sin.(β)
    lines!(ax2, xβ(α), yβ(α), label="", color=:blue, linestyle=:dash, linewidth=3)
    # Vector field of state trajectories
    f(x) = Point2f( (A*vec(lifter.map(x)) + F*(vec(lifter.map(x)) ⊘ vec(lifter.map(x))))[1:2] )
    streamplot!(ax2, f, xi..xf, yi..yf, arrow_size=10, colormap=:gray1)

    return fig1, fig2
end

# ## Example 0 #########################################################################
P, Q, cost, ∇cost, ρ_min, ρ_max, A, F, lifter = nonlinear_pendulum_example()
## Reproduce Paper Result
Ptest = [2.25 0.5; 0.5 2.0]
V = (x) -> x' * Ptest * x
Vdot = (x) -> 4.5*x[1]*x[2] - x[2]^2 - (x[1] + 4*x[2])*sin(x[1])
c_star, c_all, x_sample = LFI.doa_sampling(
    V,
    Vdot,
    500, 2, (-5,5); 
    method="memory", history=true
)
## Plot
fig1, fig2 = plot_doa_results(A, F, c_all, c_star, x_sample, Ptest, Vdot, lifter, (-5,5), (-5,5))
##
fig1
##
fig2

## LyapInf
P, Q, cost, ∇cost, ρ_min, ρ_max, A, F, lifter = nonlinear_pendulum_example()
##
V = (x) -> x' * P * x
Vdot = (x) -> x' * P * A * x + x' * P * F * (x ⊘ x)
c_star, c_all, x_sample = LFI.doa_sampling(
    V,
    Vdot,
    1000, 2, (-8,8); Nl=4, lifter=lifter,
    method="memory", history=true
)
## Plot
fig1, fig2 = plot_doa_results(A, F, c_all, 1, x_sample, P[1:2,1:2], Vdot, lifter, (-8,8), (-8,8); lift=true, heatmap_lb=-0.5)
##
fig1
##
fig2