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


function plot_doa_results(ops, c_star, x_sample, P, Vdot, xrange, yrange;
         heatmap_lb=-100, meshsize=1e-3, ax2title="Domain of Attraction Estimate", dims="Q")
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
    f(x) = dims == "Q" ? Point2f( (ops.A*x + ops.F*(x ⊘ x)) ) : (
        (dims == "C") ? Point2f( (ops.A*x + ops.E*⊘(x,x,x)) ) : 
        Point2f( (A*x + ops.F*(x ⊘ x) + ops.E*⊘(x,x,x)) )
    )
    streamplot!(ax2, f, xi..xf, yi..yf, arrow_size=10, colormap=:gray1)

    return fig2
end

# Plot the DoA results for 3D
function plot_doa_results_3D(ops, c_star, x_sample, P, Vdot, xrange, yrange, zrange;
         meshsize=1e-3, ax2title="Domain of Attraction Estimate", dims="Q",
         with_streamplot=true, with_samples=true, animate=false)
    fig2 = Figure(size=(1000,900), fontsize=20)
    ax2 = Axis3(fig2[1,1],
        title=ax2title,
        xlabel=L"x_1", ylabel=L"x_2", zlabel=L"x_3",
        xlabelsize=25, ylabelsize=25, zlabelsize=25,
        limits=(xrange..., yrange..., zrange...),
        perspectiveness=0.2, azimuth=0.8π, elevation=0.05π, 
        aspect=(1,1,1)
    )
    # Heatmap to show area where Vdot <= 0
    xi, xf = xrange[1], xrange[2]
    yi, yf = yrange[1], yrange[2]
    zi, zf = zrange[1], zrange[2]
    xpoints = xi:meshsize:xf
    ypoints = yi:meshsize:yf
    zpoints = zi:meshsize:zf

    data = [Vdot([x,y,z]) for x=xpoints, y=ypoints, z=zpoints]
    data = data .* (data .>= 0)
    contour!(ax2, xpoints, ypoints, zpoints, data, 
        transparent=true,
        alpha = 0.2, levels=35
    )
    # Colorbar(fig2[:, end+1], vol, label=L"\dot{V}(x) \leq 0")
    # rowsize!(fig2.layout, 1, ax2.scene.px_area[].widths[2])

    # # Scatter plot of Monte Carlo samples
    if with_samples
        if !isnothing(x_sample)
            meshscatter!(ax2,
                x_sample[1,:], x_sample[2,:], x_sample[3,:], 
                markersize=0.025, color=:red,
                alpha=0.5,
            )
        end
    end

    # Plot the Lyapunov function ellipse
    Λ, V = eigen(P)
    a = sqrt(c_star/Λ[1])
    b = sqrt(c_star/Λ[2])
    c = sqrt(c_star/Λ[3])
    M(u,v) = [a*cos(u)*sin(v), b*sin(u)*sin(v), c*cos(v)]    # ellipsoid
    RM(u,v) = V * M(u,v)      # rotated ellipsoid
    u, v = range(0, 2π, length=20), range(0, π, length=20)
    xs, ys, zs = [[p[i] for p in RM.(u, v')] for i in 1:3]
    wireframe!(ax2, 
        xs, ys, zs, 
    )

    # Plot the Minimal DoA with single radius
    a = maximum(Λ)
    a = sqrt(c_star/a)
    mesh!(ax2, Sphere(Point3f(0,0,0), a), color=:green, alpha=0.5)

    # Vector field of state trajectories
    if with_streamplot
        f(x) = dims == "Q" ? Point3f( (ops.A*x + ops.E*(x ⊘ x)) ) : (
            (dims == "C") ? Point3f( (ops.A*x + ops.E*⊘(x,x,x)) ) : 
            Point3f( (ops.A*x + ops.F*(x ⊘ x) + ops.E*⊘(x,x,x)) )
        )
        streamplot!(
            ax2, f, xi..xf, yi..yf, zi..zf, arrow_size=0.08, colormap=:gray1,
            gridsize=(9,9),
        )
    end

    if animate
        start_angle = 0.8π
        n_frames = 120
        ax2.viewmode = :fit # Prevent axis from resizing during animation
        record(fig2, "doa.gif", 1:n_frames) do frame
            ax2.azimuth[] = start_angle + 2pi * frame / n_frames
        end
    end

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
function verify_doa(ρ_int, ρ_nonint, integration_params, domain, max_iter; M=1, dim=2)
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
        x0 = (ub .- lb) .* rand(dim) .+ lb
        data, _, _ = integrate_model(ops, x0, ti, tf, dt, type, dim)
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
                L"MC Estimate DoA: %$(round(ρ_est,digits=4)) $$",
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
                L"MC Estimate DoA: %$(round(ρ_est,digits=4)) $$",
                L"Intrusive LyapInf DoA: %$(round(ρ_int,digits=4)) $$",
                L"Non-Intrusive LyapInf DoA: %$(round(ρ_nonint,digits=4)) $$"
            ], rowgap = 8
        )
    end
    return ρ_est, fig
end

function verify_doa_3D(ρ_int, ρ_nonint, integration_params, domain, max_iter; 
                        M=1, dim=3, animate=false)
    # Initialize the plot
    fig = Figure(size=(1100,600), fontsize=20)
    ax = Axis3(fig[1:3,1:3],
        xlabel=L"x_1", ylabel=L"x_2", zlabel=L"x_3",
        xlabelsize=25, ylabelsize=25, zlabelsize=25,
        limits=(domain..., domain..., domain...),
        perspectiveness=0.2, azimuth=0.8π, elevation=0.05π, 
        aspect=(1,1,1)
    )
    colsize!(fig.layout, 1, Aspect(1, 1.0))

    ops, ti, tf, dt, type = integration_params
    ρ_est = Inf  # initialize the estimated DoA
    for _ in 1:max_iter
        lb, ub = domain
        x0 = (ub .- lb) .* rand(dim) .+ lb
        data, _, _ = integrate_model(ops, x0, ti, tf, dt, type, dim)
        δ = norm(data[:,1], 2)
        ϵ = maximum(norm.(eachcol(data), 2))
        η = ϵ / δ

        # Plot the sample
        if η <= M
            meshscatter!(
                ax, data[1,1], data[2,1], data[3,1], 
                color=:green3, label="", strokewidth=0, markersize=0.08, alpha=0.25,
                lightposition=Vec3f(10, 5, 2),
                ambient=Vec3f(0.95, 0.95, 0.95),
                backlight=1.0f0
            )
        else
            meshscatter!(
                ax, data[1,1], data[2,1], data[3,1], 
                color=:red, label="", strokewidth=0, markersize=0.08, alpha=0.25,
                lightposition=Vec3f(10, 5, 2),
                ambient=Vec3f(0.95, 0.95, 0.95),
                backlight=1.0f0
            )

            # Update the estimated DoA
            if ρ_est > δ
                ρ_est = δ
            end
        end
    end
    # Plot the spheres
    S = (a,u,v) -> [a*cos(u)*sin(v), a*sin(u)*sin(v), a*cos(v)]    # Sphere
    u, v = range(0, 2π, length=20), range(0, π, length=20)
    xs, ys, zs = [[p[i] for p in S.(ρ_int, u, v')] for i in 1:3]
    wireframe!(ax, xs, ys, zs, color=:blue3)
    xs, ys, zs = [[p[i] for p in S.(ρ_nonint, u, v')] for i in 1:3]
    wireframe!(ax, xs, ys, zs, color=:orange)
    xs, ys, zs = [[p[i] for p in S.(ρ_est, u, v')] for i in 1:3]
    wireframe!(ax, xs, ys, zs, color=:black)

    # mesh!(Sphere(Point3f(0,0,0), ρ_int), color=:blue3, alpha=0.3)
    # mesh!(Sphere(Point3f(0,0,0), ρ_nonint), color=:orange, alpha=0.3)
    # mesh!(Sphere(Point3f(0,0,0), ρ_est), color="black", alpha=0.3)

    # Legend 
    elem1 = MarkerElement(color=:green3, markersize=10, marker=:circle, strokewidth=0)
    elem2 = MarkerElement(color=:red, markersize=10, marker=:circle, strokewidth=0)
    elem3 = PolyElement(color=:black, strokecolor=:transparent)
    elem4 = PolyElement(color=:blue3, strokecolor=:transparent)
    elem5 = PolyElement(color=:orange, strokecolor=:transparent)
    if M != 1
        Legend(fig[2,4:5],
            [elem1, elem2, elem3, elem4, elem5], 
            [
                L"$\max\Vert\textbf{x}(t)\Vert_2 ~\leq $ %$(M)$\Vert\textbf{x}_0\Vert_2$", 
                L"$\max\Vert\textbf{x}(t)\Vert_2 ~>$ %$(M)$\Vert\textbf{x}_0\Vert_2$",
                L"MC Estimate DoA: %$(round(ρ_est,digits=4)) $$",
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
                L"MC Estimate DoA: %$(round(ρ_est,digits=4)) $$",
                L"Intrusive LyapInf DoA: %$(round(ρ_int,digits=4)) $$",
                L"Non-Intrusive LyapInf DoA: %$(round(ρ_nonint,digits=4)) $$"
            ], rowgap = 8
        )
    end

    if animate
        start_angle = 0.8π
        n_frames = 120
        ax.viewmode = :fit # Prevent axis from resizing during animation
        record(fig, "doa_verify.gif", 1:n_frames) do frame
            ax.azimuth[] = start_angle + 2pi * frame / n_frames
        end
    end

    return ρ_est, fig
end
