## Packages
using Kronecker
using LinearAlgebra
using CairoMakie
import Random: rand, rand!
import DifferentialEquations: solve, ODEProblem, RK4
# import MatrixEquations: lyapc
import HSL_jll

## My modules
using LiftAndLearn
const LnL = LiftAndLearn
const LFI = LyapInf

## 
function E1_example(; method="P", type="I")
    A = [-2.0 0.0; 0.0 -1.0]
    H = LnL.makeQuadOp(2, [(1,2,1), (1,2,2)], [1.0, 1.0])
    F = LnL.H2F(H)

    function E1!(xdot, x, p, t)
        xdot .= A * x + H * (x ⊗ x)
    end

    # Generate the data
    num_ic = 1  # number of initial conditions
    tf = 10.0
    dt = 0.001
    tspan = 0.0:dt:tf
    DS = 100  # down-sampling

    X = []
    Xdot = []
    for i in 1:num_ic
        x0 = 3.6*rand(2) .- 1.8
        prob = ODEProblem(E1!, x0, (0, tf))
        sol = solve(prob, RK4(); dt=dt, adaptive=false)
        data = sol[1:2,:]
        ddata = sol(tspan, Val{1})[1:2,:]

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
        Pi = [1 0; 0 1]
        op = LnL.operators(A=A, H=H, F=F)
        P, Q, cost, ∇cost = LFI.Int_LyapInf(op, X, lyapinf_options; Pi=Pi)
    elseif type == "NI"
        ## Compute the Lyapunov Function using the intrusive method
        lyapinf_options = LFI.NonInt_LyapInf_options(
            extra_iter=3,
            optimizer="Ipopt",
            ipopt_linear_solver="ma86",
            verbose=true,
            optimize_PandQ=method,
            HSL_lib_path=HSL_jll.libhsl_path,
        )
        Pi = [1 0; 0 1]
        P, Q, cost, ∇cost = LFI.NonInt_LyapInf(X, Xdot, lyapinf_options; Pi=Pi)
    else 
        error("Invalid type")
    end
    ρ_min, ρ_max = LFI.DoA(P)
    ρ_est = LFI.skp_stability_rad(A, H, Q)
    return P, Q, cost, ∇cost, ρ_min, ρ_max, ρ_est, A, F
end



function E6_example(; method="P", type="I")
    A = zeros(2,2)
    A[1,2] = 1.0
    A[2,1] = -1.0
    A[2,2] = -0.5
    H = LnL.makeQuadOp(2, [(1,2,2), (2,2,2)], [-0.5, -0.1])
    F = LnL.H2F(H)

    function E1!(xdot, x, p, t)
        xdot .= A * x + H * (x ⊗ x)
    end

    # Generate the data
    num_ic = 100  # number of initial conditions
    tf = 5.0
    dt = 0.001
    tspan = 0.0:dt:tf
    DS = 1  # down-sampling

    X = []
    Xdot = []
    for i in 1:num_ic
        x0 = 3.0 * rand(2) .- 1.5
        prob = ODEProblem(E1!, x0, (0, tf))
        sol = solve(prob, RK4(); dt=dt, adaptive=false)
        data = sol[1:2,:]
        ddata = sol(tspan, Val{1})[1:2,:]

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
    ρ_min, ρ_max = LFI.DoA(P)
    ρ_est = LFI.skp_stability_rad(A, H, Q)
    return P, Q, cost, ∇cost, ρ_min, ρ_max, ρ_est, A, F
end


## Plotting function
function plot_doa_results(A, F, c_all, c_star, x_sample, P, Vdot, xrange, yrange;
         heatmap_lb=-100, meshsize=1e-3)
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
    xpoints = xi:meshsize:xf
    ypoints = yi:meshsize:yf

    data = [Vdot([x,y]) for x=xpoints, y=ypoints]
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
    f(x) = Point2f( (A*x + F*(x ⊘ x)) )
    streamplot!(ax2, f, xi..xf, yi..yf, arrow_size=10, colormap=:gray1)

    return fig1, fig2
end

 
function plot_doa_comparison_results(A, F, c_star1, c_star2, P1, P2, Vdot1, Vdot2, xrange, yrange, ρest; 
        heatmap_lb=-1000, meshsize=1e-3)
    fig = Figure(size=(800,800), fontsize=20)
    ax2 = Axis(fig[1,1],
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
    xpoints = xi:meshsize:xf
    ypoints = yi:meshsize:yf

    data = [Vdot1([x,y]) for x=xpoints, y=ypoints]
    hm1 = heatmap!(ax2, xpoints, ypoints, data, colorrange=(heatmap_lb,0), 
        colormap=:blues, alpha=0.3, highclip=:transparent)

    data = [Vdot2([x,y]) for x=xpoints, y=ypoints]
    hm2 = heatmap!(ax2, xpoints, ypoints, data, colorrange=(heatmap_lb,0), 
        colormap=:greens, alpha=0.3, highclip=:transparent)

    # Plot the Lyapunov function ellipse of intrusive
    α = range(0, 2π, length=1000)
    Λ, V = eigen(P1)
    θ = acos(dot(V[:,1], [1,0]) / (norm(V[:,1])))
    xα = (α) -> sqrt(c_star1/Λ[1])*cos(θ)*cos.(α) .+ sqrt(c_star1/Λ[2])*sin(θ)*sin.(α)
    yα = (α) -> -sqrt(c_star1/Λ[1])*sin(θ)*cos.(α) .+ sqrt(c_star1/Λ[2])*cos(θ)*sin.(α)
    lines!(ax2, xα(α), yα(α), label="Intrusive", color=:blue, linewidth=2)
    # # Plot the Minimal DoA with single radius
    # xβ = (β) -> sqrt(c_star1/maximum(Λ))*cos.(β)
    # yβ = (β) -> sqrt(c_star1/maximum(Λ))*sin.(β)
    # lines!(ax2, xβ(α), yβ(α), label="", color=:black, linestyle=:dash, linewidth=3)

    # Plot the Lyapunov function ellipse of non-intrusive
    Λ, V = eigen(P2)
    θ = acos(dot(V[:,1], [1,0]) / (norm(V[:,1])))
    xα = (α) -> sqrt(c_star2/Λ[1])*cos(θ)*cos.(α) .+ sqrt(c_star2/Λ[2])*sin(θ)*sin.(α)
    yα = (α) -> -sqrt(c_star2/Λ[1])*sin(θ)*cos.(α) .+ sqrt(c_star2/Λ[2])*cos(θ)*sin.(α)
    lines!(ax2, xα(α), yα(α), label="Non-Intrusive", color=:red, linewidth=2)
    # # Plot the Minimal DoA with single radius
    # xβ = (β) -> sqrt(c_star2/maximum(Λ))*cos.(β)
    # yβ = (β) -> sqrt(c_star2/maximum(Λ))*sin.(β)
    # lines!(ax2, xβ(α), yβ(α), label="", color=:red, linestyle=:dash, linewidth=3)

    # Plot the estimated stability radius
    xγ = (γ) -> ρest*cos.(γ)
    yγ = (γ) -> ρest*sin.(γ)
    lines!(ax2, xγ(α), yγ(α), label="SKP", color=:yellow, linewidth=3)

    # Vector field of state trajectories
    f(x) = Point2f( (A*x + F*(x ⊘ x)) )
    streamplot!(ax2, f, xi..xf, yi..yf, arrow_size=10, colormap=:bone)

    axislegend(position = :lb)

    return fig
end


## Example 1 #########################################################################
P1, Q, cost, ∇cost, ρ_min, ρ_max, ρ_est, A, F = E1_example(method="together", type="I")
##
V1 = (x) -> x' * P1 * x
Vdot1 = (x) -> x' * P1 * A * x + x' * P1 * F * (x ⊘ x)
c_star1, c_all, x_sample = LFI.doa_sampling(
    V1,
    Vdot1,
    1000, 2, (-5,5);
    method="memory", history=true, n_strata=8, uniform_state_space=true
)
## Plot Only for Intrusive
fig1, fig2 = plot_doa_results(A, F, c_all, c_star1, x_sample, P1[1:2,1:2], Vdot1, (-5,5), (-5,5);
                                 heatmap_lb=-5, meshsize=1e-2)
##
fig1
##
fig2

##
P2, Q, cost, ∇cost, ρ_min, ρ_max, ρ_est, A, F = E1_example(method="P", type="NI")
##
V2 = (x) -> x' * P2 * x
Vdot2 = (x) -> x' * P2 * A * x + x' * P2 * F * (x ⊘ x)
c_star2, c_all, x_sample = LFI.doa_sampling(
    V2,
    Vdot2,
    1000, 2, (-5,5);
    method="memory", history=true, n_strata=8, uniform_state_space=true
)
## Plot Only for Non-Intrusive
fig3, fig4 = plot_doa_results(A, F, c_all, c_star2, x_sample, P2[1:2,1:2], Vdot2, (-5,5), (-5,5);
                                 heatmap_lb=-5, meshsize=1e-2)
##
fig3
##
fig4

## Plot the comparison
fig5 = plot_doa_comparison_results(A, F, c_star1, c_star2, P1, P2, Vdot1, Vdot2, (-5,5), (-5,5), ρ_est; 
        heatmap_lb=-5, meshsize=1e-2)
fig5


## Example 6 #########################################################################
P1, Q, cost, ∇cost, ρ_min, ρ_max, ρ_est, A, F = E6_example(type="I")
##
V1 = (x) -> x' * P1 * x
Vdot1 = (x) -> x' * P1 * A * x + x' * P1 * F * (x ⊘ x)
c_star1, c_all, x_sample = LFI.doa_sampling(
    V1,
    Vdot1,
    1000, 2, (-2,2);
    method="memory", history=true
)
## Plot
fig6, fig7 = plot_doa_results(A, F, c_all, c_star1, x_sample, P1[1:2,1:2], Vdot1, (-2,2), (-2,2);
                                 heatmap_lb=-1e-1, meshsize=1e-2)
##
fig6
##
fig7

##
P2, Q, cost, ∇cost, ρ_min, ρ_max, ρ_est, A, F = E6_example(type="NI")
##
V2 = (x) -> x' * P2 * x
Vdot2 = (x) -> x' * P2 * A * x + x' * P2 * F * (x ⊘ x)
c_star2, c_all, x_sample = LFI.doa_sampling(
    V2,
    Vdot2,
    1000, 2, (-2,2);
    method="memory", history=true
)
## Plot
fig8, fig9 = plot_doa_results(A, F, c_all, c_star2, x_sample, P2[1:2,1:2], Vdot2, (-2,2), (-2,2);
                                 heatmap_lb=-1e-1, meshsize=1e-2)
##
fig8
##
fig9

##
fig10 = plot_doa_comparison_results(A, F, c_star1, c_star2, P1, P2, Vdot1, Vdot2, (-2,2), (-2,2), ρ_est; 
        meshsize=1e-2)


## ########################
# function E2_example()
#     A = zeros(3,3)
#     A[1,2] = -1.0
#     A[2,1] = 1.0
#     A[2,2] = -1.0
#     H = LnL.makeQuadOp(3, [(2,3,2), (1,2,3)], [1.0, -2.0])
#     F = LnL.H2F(H)

#     function E1!(xdot, x, p, t)
#         xdot .= A * x + H * (x ⊗ x)
#     end

#     # Generate the data
#     num_ic = 5  # number of initial conditions
#     tf = 10.0
#     dt = 0.001
#     tspan = 0.0:dt:tf
#     DS = 100  # down-sampling

#     # Lifting
#     lifter = LnL.lifting(2, 3, [x -> x[1]^2])

#     X = []
#     Xdot = []
#     for i in 1:num_ic
#         x0 = lifter.map(2 * rand(2) .- 1)
#         prob = ODEProblem(E1!, x0, (0, tf))
#         sol = solve(prob, RK4(); dt=dt, adaptive=false)
#         data = sol[1:3,:]
#         ddata = sol(tspan, Val{1})[1:3,:]

#         push!(X, data[:,1:DS:end])
#         push!(Xdot, ddata[:,1:DS:end])
#     end
#     X = reduce(hcat, X)
#     Xdot = reduce(hcat, Xdot)

#     ## Compute the Lyapunov Function using the intrusive method
#     lyapinf_options = LFI.Int_LyapInf_options(
#         extra_iter=3,
#         optimizer="Ipopt",
#         ipopt_linear_solver="ma86",
#         verbose=true,
#         optimize_PandQ="P",
#         HSL_lib_path=HSL_jll.libhsl_path,
#     )
#     op = LnL.operators(A=A, H=H, F=F)
#     P, Q, cost, ∇cost = LFI.Int_LyapInf(op, X, lyapinf_options)
#     ρ_min, ρ_max = LFI.DoA(P)
#     return P, Q, cost, ∇cost, ρ_min, ρ_max, A, F, lifter
# end


# function E3_example()
#     A = zeros(4,4)
#     A[1,1] = -1.0
#     A[2,2] = -1.0
#     A[3,3] = -1.0
#     H = LnL.makeQuadOp(4, [(2,4,1), (1,2,2), (3,3,4)], [1.0, 1.0, -2.0])
#     F = LnL.H2F(H)

#     function E1!(xdot, x, p, t)
#         xdot .= A * x + H * (x ⊗ x)
#     end

#     # Generate the data
#     num_ic = 5  # number of initial conditions
#     tf = 10.0
#     dt = 0.001
#     tspan = 0.0:dt:tf
#     DS = 100  # down-sampling

#     # Lifting
#     lifter = LnL.lifting(3, 4, [x -> x[3]^2])

#     X = []
#     Xdot = []
#     for i in 1:num_ic
#         foo1, foo2, foo3 = 4.0*rand(3) .- 2.0
#         x0 = lifter.map([foo1, foo2, foo3])
#         prob = ODEProblem(E1!, x0, (0, tf))
#         sol = solve(prob, RK4(); dt=dt, adaptive=false)
#         data = sol[1:4,:]
#         ddata = sol(tspan, Val{1})[1:4,:]

#         push!(X, data[:,1:DS:end])
#         push!(Xdot, ddata[:,1:DS:end])
#     end
#     X = reduce(hcat, X)
#     Xdot = reduce(hcat, Xdot)

#     ## Compute the Lyapunov Function using the intrusive method
#     lyapinf_options = LFI.Int_LyapInf_options(
#         extra_iter=3,
#         optimizer="Ipopt",
#         ipopt_linear_solver="ma86",
#         verbose=true,
#         optimize_PandQ="P",
#         HSL_lib_path=HSL_jll.libhsl_path,
#     )
#     op = LnL.operators(A=A, H=H, F=F)
#     P, Q, cost, ∇cost = LFI.Int_LyapInf(op, X, lyapinf_options)
#     ρ_min, ρ_max = LFI.DoA(P[1:3,1:3])
#     return P, Q, cost, ∇cost, ρ_min, ρ_max, A, F, lifter
# end


# function E5_example()
#     A = zeros(4,4)
#     A[1,2] = 1.0
#     A[2,2] = -2.0
#     A[2,3] = -1.0
#     H = LnL.makeQuadOp(4, [(3,4,2), (2,4,3), (2,3,4)], [0.81, 1.0, -1.0])
#     F = LnL.H2F(H)

#     function E1!(xdot, x, p, t)
#         xdot .= A * x + H * (x ⊗ x)
#     end

#     # Generate the data
#     num_ic = 5  # number of initial conditions
#     tf = 10.0
#     dt = 0.001
#     tspan = 0.0:dt:tf
#     DS = 100  # down-sampling

#     # Lifting
#     lifter = LnL.lifting(2, 4, [x -> sin.(x[1]), x -> cos.(x[1])])

#     X = []
#     Xdot = []
#     for i in 1:num_ic
#         foo1, foo2 = 2π/3*rand(3) .- π/3
#         x0 = lifter.map([foo1, foo2])
#         prob = ODEProblem(E1!, x0, (0, tf))
#         sol = solve(prob, RK4(); dt=dt, adaptive=false)
#         data = sol[1:4,:]
#         ddata = sol(tspan, Val{1})[1:4,:]

#         push!(X, data[:,1:DS:end])
#         push!(Xdot, ddata[:,1:DS:end])
#     end
#     X = reduce(hcat, X)
#     Xdot = reduce(hcat, Xdot)

#     ## Compute the Lyapunov Function using the intrusive method
#     lyapinf_options = LFI.Int_LyapInf_options(
#         extra_iter=3,
#         optimizer="Ipopt",
#         ipopt_linear_solver="ma86",
#         verbose=true,
#         optimize_PandQ="P",
#         HSL_lib_path=HSL_jll.libhsl_path,
#     )
#     op = LnL.operators(A=A, H=H, F=F)
#     P, Q, cost, ∇cost = LFI.Int_LyapInf(op, X, lyapinf_options)
#     ρ_min, ρ_max = LFI.DoA(P[1:2,1:2])
#     return P, Q, cost, ∇cost, ρ_min, ρ_max, A, F, lifter
# end



# ## Example 2
# P, Q, cost, ∇cost, ρ_min, ρ_max, A, F, lifter = E2_example()
# ##
# c_star = LFI.doa_sampling(
#     (x) -> x' * P * x, 
#     (x) -> x' * P * A * x + x' * P * F * (x ⊘ x), 
#     500, 2, (-5,5), Nl=3, lifter=lifter
# )

# ## Example 3
# P, Q, cost, ∇cost, ρ_min, ρ_max, A, F, lifter = E3_example()
# ##
# c_star = LFI.doa_sampling(
#     (x) -> x' * P * x, 
#     (x) -> x' * P * A * x + x' * P * F * (x ⊘ x), 
#     500, 3, (-5,5), Nl=4, lifter=lifter
# )

# ## Example 5
# P, Q, cost, ∇cost, ρ_min, ρ_max, A, F, lifter = E5_example()
# ##
# c_star = LFI.doa_sampling(
#     (x) -> x' * P * x, 
#     (x) -> x' * P * A * x + x' * P * F * (x ⊘ x), 
#     500, 2, (-5,5), Nl=4, lifter=lifter
# )