##############
## Packages
##############
using Kronecker
using LinearAlgebra
using CairoMakie
import Random: rand, rand!
import DifferentialEquations: solve, ODEProblem, RK4
import HSL_jll


################
## My modules
################
using LiftAndLearn
const LnL = LiftAndLearn
const LFI = LyapInf

##########
## CONSTS
##########
SAMPLE = true

################################
## Functions for the examples
################################
function lin_quad_model!(xdot, x, p, t)
    A, F = p[1], p[2]
    xdot .= A * x + F * (x ⊘ x)
end

function lin_cubic_model!(xdot, x, p, t)
    A, E = p[1], p[2]
    xdot .= A * x + E * ⊘(x,x,x)
end

function integrate_model(ops, x0, ti, tf, dt, type)
    if type == "Q"
        prob = ODEProblem(lin_quad_model!, x0, (ti, tf), ops)
    elseif type == "C"
        prob = ODEProblem(lin_cubic_model!, x0, (ti, tf), ops)
    else
        error("Invalid type")
    end
    sol = solve(prob, RK4(); dt=dt, adaptive=false)
    data = sol[1:2,:]
    ddata = sol(ti:dt:tf, Val{1})[1:2,:]
    return data, ddata, Symbol(sol.retcode)
end

# Lotka-Volterra Predator-Prey (LVPP) Example
function lvpp_example(; method="P", type="I", x0_bnds=(-4.0, 4.0), optimizer="SCS")
    A = [-2.0 0.0; 0.0 -1.0]
    H = LnL.makeQuadOp(2, [(1,2,1), (1,2,2)], [1.0, 1.0])
    F = LnL.H2F(H)

    # Generate the data
    num_ic = 20  # number of initial conditions
    tf = 10.0
    dt = 0.001
    DS = 100  # down-sampling

    X = []
    Xdot = []
    lb, ub = x0_bnds
    ct = 0
    while ct < num_ic
        x0 = (ub - lb)*rand(2) .+ lb
        data, ddata, retcode = integrate_model([A, F], x0,0.0,tf,dt,"Q")
        if retcode in (:Unstable, :Terminated, :Failure)
            continue
        elseif retcode == :Success
            ct += 1
        else
            error("Invalid retcode: $retcode")
        end
        push!(X, data[:,1:DS:end])
        push!(Xdot, ddata[:,1:DS:end])
    end
    X = reduce(hcat, X)
    Xdot = reduce(hcat, Xdot)

    if type == "I"
        ## Compute the Lyapunov Function using the intrusive method
        lyapinf_options = LFI.Int_LyapInf_options(
            extra_iter=3,
            optimizer=optimizer,
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
            optimizer=optimizer,
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
    ρ_est = LFI.skp_stability_rad(P, A, H; dims=(1,2))
    return P, Q, cost, ∇cost, ρ_min, ρ_max, ρ_est, A, F, H
end


# Van der Pol Oscillator (VPO) Example
function vpo_example(; method="P", type="I", optimizer="SCS", x0_bnds=(-4.0, 4.0), μ=1.0)
    A = zeros(2,2)
    A[1,2] = -1.0
    A[2,1] = 1.0
    A[2,2] = -μ
    G = zeros(2,8)
    G[2,2] = μ/3
    G[2,3] = μ/3
    G[2,5] = μ/3
    E = LnL.G2E(G)

    # Generate the data
    num_ic = 20  # number of initial conditions
    tf = 5.0
    dt = 0.001
    DS = 100  # down-sampling

    X = []
    Xdot = []
    lb, ub = x0_bnds
    ct = 0
    while ct < num_ic
        x0 = (ub - lb) * rand(2) .+ lb

        # rest = (x1,x2) -> (1.5*x1^2 - x1*x2 + x2^2) / (1 - 0.2362*x1^2 + 0.31747*x1*x2 - 0.1091*x2^2)
        # if rest(x0...) > 5
        #     continue
        # end

        data, ddata, retcode = integrate_model([A,E],x0,0.0,tf,dt,"C")
        if retcode in (:Unstable, :Terminated, :Failure)
            continue
        elseif retcode == :Success
            ct += 1
        else
            error("Invalid retcode: $retcode")
        end
        push!(X, data[:,1:DS:end])
        push!(Xdot, ddata[:,1:DS:end])
    end
    X = reduce(hcat, X)
    println(size(X))
    Xdot = reduce(hcat, Xdot)

    if type == "I"
        ## Compute the Lyapunov Function using the intrusive method
        lyapinf_options = LFI.Int_LyapInf_options(
            extra_iter=3,
            optimizer=optimizer,
            ipopt_linear_solver="ma86",
            verbose=true,
            optimize_PandQ=method,
            HSL_lib_path=HSL_jll.libhsl_path,
            is_quad=false,
            is_cubic=true,
        )
        op = LnL.operators(A=A, G=G, E=E)
        P, Q, cost, ∇cost = LFI.Int_LyapInf(op, X, lyapinf_options; Qi=5.0I(2))
    elseif type == "NI"
        ## Compute the Lyapunov Function using the non-intrusive method
        lyapinf_options = LFI.NonInt_LyapInf_options(
            extra_iter=3,
            optimizer=optimizer,
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
    ρ_est = 1.0
    ρ_est = LFI.skp_stability_rad(P, A, nothing, G; dims=(1,3))
    return P, Q, cost, ∇cost, ρ_min, ρ_max, ρ_est, A, E, G
end


#########################
## Plotting functions
#########################
# Plot the DoA result for a single Lyapunov function
function plot_cstar_convergence(c_all)
    fig1 = Figure(fontsize=20)
    ax1 = Axis(fig1[1,1],
        title="Level Convergence",
        ylabel=L"c_*",
        xlabel="Sample Number",
        xticks=0:(length(c_all)÷10):length(c_all),
    )
    lines!(ax1, 1:length(c_all), c_all)
    return fig1
end


function plot_doa_results(A, F, c_star, x_sample, P, Vdot, xrange, yrange;
         heatmap_lb=-100, meshsize=1e-3, ax2title="Domain of Attraction Estimate", dims=2)
    fig2 = Figure(size=(800,800), fontsize=20)
    ax2 = Axis(fig2[1,1],
        title=ax2title,
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
    rowsize!(fig2.layout, 1, ax2.scene.px_area[].widths[2])

    # Scatter plot of Monte Carlo samples
    if !isnothing(x_sample)
        scatter!(ax2, x_sample[1,:], x_sample[2,:], color=:red, alpha=0.6)
    end

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
    f(x) = dims == 2 ? Point2f( (A*x + F*(x ⊘ x)) ) : Point2f( (A*x + F*⊘(x,x,x)) )
    streamplot!(ax2, f, xi..xf, yi..yf, arrow_size=10, colormap=:gray1)

    return fig2
end

# Plot the DoA comparison for the intrusive and non-intrusive results 
function plot_doa_comparison_results(A, F, c_star1, c_star2, P1, P2, Vdot1, Vdot2, xrange, yrange, ρest; 
        heatmap_lb=-1000, meshsize=1e-3, dims=2)
    fig = Figure(size=(800,800), fontsize=20)
    ax2 = Axis(fig[1,1],
        title="Comparison of DoA",
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
    # θ = acos(dot(V[:,1], [1,0]) / (norm(V[:,1])))
    # xα = (α) -> sqrt(c_star1/Λ[1])*cos(θ)*cos.(α) .+ sqrt(c_star1/Λ[2])*sin(θ)*sin.(α)
    # yα = (α) -> -sqrt(c_star1/Λ[1])*sin(θ)*cos.(α) .+ sqrt(c_star1/Λ[2])*cos(θ)*sin.(α)
    # lines!(ax2, xα(α), yα(α), label="Intrusive", color=:blue, linewidth=2)

    # Plot the Minimal DoA with single radius
    xβ = (β) -> sqrt(c_star1/maximum(Λ))*cos.(β)
    yβ = (β) -> sqrt(c_star1/maximum(Λ))*sin.(β)
    lines!(ax2, xβ(α), yβ(α), label="Intrusive", color=:black, linestyle=:dash, linewidth=3)

    # Plot the Lyapunov function ellipse of non-intrusive
    Λ, V = eigen(P2)
    # θ = acos(dot(V[:,1], [1,0]) / (norm(V[:,1])))
    # xα = (α) -> sqrt(c_star2/Λ[1])*cos(θ)*cos.(α) .+ sqrt(c_star2/Λ[2])*sin(θ)*sin.(α)
    # yα = (α) -> -sqrt(c_star2/Λ[1])*sin(θ)*cos.(α) .+ sqrt(c_star2/Λ[2])*cos(θ)*sin.(α)
    # lines!(ax2, xα(α), yα(α), label="Non-Intrusive", color=:red, linewidth=2)

    # Plot the Minimal DoA with single radius
    xβ = (β) -> sqrt(c_star2/maximum(Λ))*cos.(β)
    yβ = (β) -> sqrt(c_star2/maximum(Λ))*sin.(β)
    lines!(ax2, xβ(α), yβ(α), label="Non-Intrusive", color=:red, linestyle=:dash, linewidth=3)

    # Plot the estimated stability radius
    xγ = (γ) -> ρest*cos.(γ)
    yγ = (γ) -> ρest*sin.(γ)
    lines!(ax2, xγ(α), yγ(α), label="SKP", color=:yellow, linewidth=3)

    # Vector field of state trajectories
    f(x) = dims == 2 ? Point2f( (A*x + F*(x ⊘ x)) ) : Point2f( (A*x + F*⊘(x,x,x)) )
    streamplot!(ax2, f, xi..xf, yi..yf, arrow_size=10, colormap=:bone)

    axislegend(position = :lb)

    return fig
end

# Function to verify the actual domain of attraction using Monte-Carlo sampling
function verify_doa(ρ_int, ρ_nonint, integration_params, domain, max_iter; M=1)
    # Initialize the plot
    fig = Figure(size=(1000,600), fontsize=20)
    ax = Axis(fig[1:3,1:3],
        xlabel=L"x_1",
        ylabel=L"x_2",
        xlabelsize=25,
        ylabelsize=25,
        limits=(domain..., domain...),
        aspect=1
    )
    colsize!(fig.layout, 1, Aspect(1, 1.0))

    ops, ti, tf, dt, type = integration_params
    ρ_est = Inf  # initialize the estimated DoA
    for _ in 1:max_iter
        lb, ub = domain
        x0 = (ub .- lb) .* rand(length(domain)) .+ lb
        data, _, _ = integrate_model(ops, x0, ti, tf, dt, type)
        δ = norm(data[:,1], 2)
        ϵ = maximum(norm.(eachcol(data), 2))
        η = ϵ / δ

        # Plot the sample
        if η <= M
            scatter!(ax, data[1,1], data[2,1], color=:green3, label="", strokewidth=0)
        else
            scatter!(ax, data[1,1], data[2,1], color=:red, label="", strokewidth=0)

            # Update the estimated DoA
            if ρ_est > δ
                ρ_est = δ
            end
        end
    end
    px = ρ_est * cos.(range(0, 2π, length=1000))
    py = ρ_est * sin.(range(0, 2π, length=1000))
    lines!(ax, px, py, color=:black, linestyle=:solid, linewidth=3, label="")
    px = ρ_int * cos.(range(0, 2π, length=1000))
    py = ρ_int * sin.(range(0, 2π, length=1000))
    lines!(ax, px, py, color=:blue3, linestyle=:dash, linewidth=3, label="")
    px = ρ_nonint * cos.(range(0, 2π, length=1000))
    py = ρ_nonint * sin.(range(0, 2π, length=1000))
    lines!(ax, px, py, color=:orange, linestyle=:dash, linewidth=3, label="")

    # Legend 
    elem1 = MarkerElement(color=:green3, markersize=10, marker=:circle, strokewidth=0)
    elem2 = MarkerElement(color=:red, markersize=10, marker=:circle, strokewidth=0)
    elem3 = LineElement(color=:black, linestyle=:solid, linewidth=3)
    elem4 = LineElement(color=:blue3, linestyle=:dash, linewidth=3)
    elem5 = LineElement(color=:orange, linestyle=:dash, linewidth=3)
    if M != 1
        Legend(fig[2,4:5],
            [elem1, elem2, elem3, elem4, elem5], 
            [
                L"$\max\Vert\textbf{x}(t)\Vert_2 ~\leq $ %$(M)$\Vert\textbf{x}_0\Vert_2$", 
                L"$\max\Vert\textbf{x}(t)\Vert_2 ~>$ %$(M)$\Vert\textbf{x}_0\Vert_2$",
                L"True DoA: %$(round(ρ_est,digits=4)) $$",
                L"Intrusive LyapInf DoA: %$(round(ρ_int,digits=4)) $$",
                L"Non-Intrusive LyapInf DoA: %$(round(ρ_nonint,digits=4)) $$"
            ], rowgap = 8
        )
    else
        Legend(fig[2,4:5],
            [elem1, elem2, elem3, elem4, elem5], 
            [
                L"$\max\Vert\textbf{x}(t)\Vert_2 ~\leq~\Vert\textbf{x}_0\Vert_2$", 
                L"$\max\Vert\textbf{x}(t)\Vert_2 ~>~\Vert\textbf{x}_0\Vert_2$",
                L"True DoA: %$(round(ρ_est,digits=4)) $$",
                L"Intrusive LyapInf DoA: %$(round(ρ_int,digits=4)) $$",
                L"Non-Intrusive LyapInf DoA: %$(round(ρ_nonint,digits=4)) $$"
            ], rowgap = 8
        )
    end
    return ρ_est, fig
end


#################################################
## Example 1: Lotka-Volterra Predator-Prey 
#################################################
P1, Q, cost, ∇cost, ρ_min, ρ_max, ρ_est, A, F, H = lvpp_example(method="both", type="I", x0_bnds=(-1.5, 1.5))
##
c_star1, c_all, x_sample = nothing, nothing, nothing
if SAMPLE
    V1(x) = x' * P1 * x
    Vdot1(x) = 2*x' * P1 * A * x + 2*x' * P1 * F * (x ⊘ x)
    c_star1, c_all, x_sample = LFI.doa_sampling(
        V1,
        Vdot1,
        1000, 2, (-5,5);
        method="memory", history=true, n_strata=128, uniform_state_space=true
    )
else
    V1(x) = x' * P1 * x
    Vdot1(x) = 2*x' * P1 * A * x + 2*x' * P1 * F * (x ⊘ x)
    c_star1, _ = LFI.LEDOA(V1, Vdot1, 2; linear_solver="ma86", verbose=true, HSL_lib_path=HSL_jll.libhsl_path,
                                ci=1e2, xi=[10,10], δ=1)
end
##
ρ_star1 = sqrt(c_star1) * ρ_min
println("c_star1 = ", c_star1)
println("ρ_est = ", ρ_est)
println("ρ_min = ", ρ_min)
println("ρ_star = ", ρ_star1)

## Plot Only for Intrusive
fig11 = plot_cstar_convergence(c_all)
fig12 = plot_doa_results(A, F, c_star1, x_sample, P1[1:2,1:2], Vdot1, (-5,5), (-5,5);
                                 heatmap_lb=-5, meshsize=1e-2, ax2title="Intrusive LyapInf: DoA")
display(fig11)
display(fig12)

## Non-Intrusive
P2, Q, cost, ∇cost, ρ_min, ρ_max, ρ_est, A, F, H = lvpp_example(method="both", type="NI", x0_bnds=(-1.5, 1.5))
##
V2 = (x) -> x' * P2 * x
Vdot2 = (x) -> 2*x' * P2 * A * x + 2*x' * P2 * F * (x ⊘ x)
c_star2, c_all, x_sample = LFI.doa_sampling(
    V2,
    Vdot2,
    1000, 2, (-5,5);
    method="memory", history=true, n_strata=8, uniform_state_space=true
)
ρ_star2 = sqrt(c_star2) * ρ_min
println("c_star1 = ", c_star2)
println("ρ_est = ", ρ_est)
println("ρ_min = ", ρ_min)
println("ρ_star = ", ρ_star2)

## Plot Only for Non-Intrusive
fig13 = plot_cstar_convergence(c_all)
fig14 = plot_doa_results(A, F, c_star2, x_sample, P2[1:2,1:2], Vdot2, (-5,5), (-5,5);
                                 heatmap_lb=-5, meshsize=1e-2, ax2title="Non-Intrusive LyapInf: DoA")
display(fig13)
display(fig14)

## Plot the comparison
fig15 = plot_doa_comparison_results(A, F, c_star1, c_star2, P1, P2, Vdot1, Vdot2, (-5,5), (-5,5), ρ_est; 
        heatmap_lb=-5, meshsize=1e-2)
display(fig15)

## Verify the DoA
ρ_mc, fig16 = verify_doa(ρ_star1, ρ_star2, ([A, F], 0.0, 10.0, 0.001, "Q"), (-5,5), 5000)
display(fig16)



#################################################
## Example 2: Van der Pol Oscillator 
#################################################
P1, Q, cost, ∇cost, ρ_min, ρ_max, ρ_est, A, E, G = vpo_example(method="P", type="I", μ=4.0)
##
V1(x) = x' * P1 * x
Vdot1(x) = 2*x' * P1 * A * x + 2*x' * P1 * E * ⊘(x,x,x)
c_star1, c_all, x_sample = LFI.doa_sampling(
    V1,
    Vdot1,
    1000, 2, (-3.0,3.0);
    method="memory", history=true
)
ρ_star1 = sqrt(c_star1) * ρ_min
println("c_star1 = ", c_star1)
println("ρ_est = ", ρ_est)
println("ρ_min = ", ρ_min)
println("ρ_star = ", ρ_star1)

## Plot
fig21 = plot_cstar_convergence(c_all)
fig22 = plot_doa_results(A, E, c_star1, x_sample, P1, Vdot1, (-3.0,3.0), (-3.0,3.0);
                                 heatmap_lb=-5e-2, meshsize=1e-2, ax2title="Intrusive LyapInf: DoA", dims=3)
display(fig21)
display(fig22)

## Non-Intrusive
P2, Q, cost, ∇cost, ρ_min, ρ_max, ρ_est, A, E, G = vpo_example(method="P",type="NI", μ=4.0)
##
V2 = (x) -> x' * P2 * x
Vdot2 = (x) -> 2*x' * P2 * A * x + 2*x' * P2 * E * ⊘(x,x,x)
c_star2, c_all, x_sample = LFI.doa_sampling(
    V2,
    Vdot2,
    1000, 2, (-3.0,3.0);
    method="memory", history=true
)
ρ_star2 = sqrt(c_star2) * ρ_min
println("c_star1 = ", c_star2)
println("ρ_est = ", ρ_est)
println("ρ_min = ", ρ_min)
println("ρ_star = ", ρ_star2)

## Plot
fig23 = plot_cstar_convergence(c_all)
fig24 = plot_doa_results(A, E, c_star2, x_sample, P2, Vdot2, (-3.0,3.0), (-3.0,3.0);
                            heatmap_lb=-5e-2, meshsize=1e-2, ax2title="Non-Intrusive LyapInf: DoA", dims=3)
display(fig23)
display(fig24)

## Plot the comparison
fig25 = plot_doa_comparison_results(A, E, c_star1, c_star2, P1, P2, Vdot1, Vdot2, (-4,4), (-4,4), ρ_est; 
        meshsize=1e-2, dims=3)
display(fig25)

## Verify the DoA
ρ_mc, fig26 = verify_doa(ρ_star1, ρ_star2, ([A, E], 0.0, 10.0, 0.001, "C"), (-3.0,3.0), 5000; M=1.0)
display(fig26)




# ########################################################################################
# Modified quadratic Van Der Pol
# function vpo_example(; method="P", type="I", x0_bnds=(-1.5, 1.5))
#     A = zeros(2,2)
#     A[1,2] = 1.0
#     A[2,1] = -1.0
#     A[2,2] = -0.5
#     H = LnL.makeQuadOp(2, [(1,2,2), (2,2,2)], [-0.5, -0.1])
#     F = LnL.H2F(H)

#     # Generate the data
#     num_ic = 20  # number of initial conditions
#     tf = 10.0
#     dt = 0.001
#     DS = 100  # down-sampling

#     X = []
#     Xdot = []
#     lb, ub = x0_bnds
#     for i in 1:num_ic
#         x0 = (ub - lb) * rand(2) .+ lb
#         data, ddata = integrate_model(A,F,x0,0.0,tf,dt)
#         push!(X, data[:,1:DS:end])
#         push!(Xdot, ddata[:,1:DS:end])
#     end
#     X = reduce(hcat, X)
#     Xdot = reduce(hcat, Xdot)

#     if type == "I"
#         ## Compute the Lyapunov Function using the intrusive method
#         lyapinf_options = LFI.Int_LyapInf_options(
#             extra_iter=3,
#             optimizer="Ipopt",
#             ipopt_linear_solver="ma86",
#             verbose=true,
#             optimize_PandQ=method,
#             HSL_lib_path=HSL_jll.libhsl_path,
#         )
#         op = LnL.operators(A=A, H=H, F=F)
#         P, Q, cost, ∇cost = LFI.Int_LyapInf(op, X, lyapinf_options)
#     elseif type == "NI"
#         ## Compute the Lyapunov Function using the non-intrusive method
#         lyapinf_options = LFI.NonInt_LyapInf_options(
#             extra_iter=3,
#             optimizer="Ipopt",
#             ipopt_linear_solver="ma86",
#             verbose=true,
#             optimize_PandQ=method,
#             HSL_lib_path=HSL_jll.libhsl_path,
#         )
#         P, Q, cost, ∇cost = LFI.NonInt_LyapInf(X, Xdot, lyapinf_options)
#     else
#         error("Invalid type")
#     end
#     ρ_min, ρ_max = LFI.DoA(P)
#     ρ_est = LFI.skp_stability_rad(A, H, Q)
#     return P, Q, cost, ∇cost, ρ_min, ρ_max, ρ_est, A, F, H
# end

