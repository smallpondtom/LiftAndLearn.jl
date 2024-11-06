"""
2D heat equation iOpInf example
"""

#================#
## Load Packages
#================#
using CairoMakie
using IncrementalSVD
using LinearAlgebra
using ProgressMeter
using PolynomialModelReductionDataset: Heat2DModel
using LiftAndLearn
using UniqueKronecker: invec
const LnL = LiftAndLearn

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"Streaming-OpInf/") : joinpath(pwd(), "scripts/Streaming-OpInf/")

# #==============================#
# ## Include functions and files
# #==============================#
# include("utilities/plot_theme.jl")
# include("utilities/analysis.jl")
# include("utilities/plotting.jl")


#========================#
## 2D Heat equation setup
#========================#
Ω = ((0.0, 1.0), (0.0, 1.25))
Nx = 32
Ny = 40
heat2d = Heat2DModel(
    spatial_domain=Ω, time_domain=(0,1.0), 
    Δx=(Ω[1][2] + 1/Nx)/Nx, Δy=(Ω[2][2] + 1/Ny)/Ny, Δt=1e-3,
    diffusion_coeffs=0.1, BC=(:dirichlet, :dirichlet)
)
xgrid0 = heat2d.yspan' .* ones(heat2d.spatial_dim[1])
ygrid0 = ones(heat2d.spatial_dim[2])' .* heat2d.xspan
ux0 = sin.(2π * xgrid0) .* cos.(2π * ygrid0)
heat2d.IC = vec(ux0)  # initial condition

# Some options for operator inference
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=1,
        control=1,
        output=1,
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=1e-3,
        deriv_type="BE"
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)

#===============#
## Generate Data
#===============#
# Construct full model
μ = heat2d.diffusion_coeffs
A, B = heat2d.finite_diff_model(heat2d, μ)
C = ones(1, (Int ∘ prod)(heat2d.spatial_dim)) / heat2d.spatial_dim[1] / heat2d.spatial_dim[2]
op_heat = LnL.Operators(A=A, B=B, C=C)

# Generate the input data
U = [1.0, 1.0, -1.0, -1.0]
U = repeat(U, 1, heat2d.time_dim)

# Compute the state snapshot data with backward Euler
X = heat2d.integrate_model(
    heat2d.tspan, heat2d.IC, U; linear_matrix=A, control_matrix=B, 
    system_input=true, integrator_type=:BackwardEuler
)

# Compute the output of the system
Y = C * X

#=====================#
## Plot Data to Check
#=====================#
Xflat = invec.(eachcol(X), heat2d.spatial_dim...)
with_theme(theme_latexfonts()) do
    fig0 = Figure(fontsize=20, size=(1200,1050))
    ax1 = Axis3(fig0[1, 1], xlabel=L"\omega_1", ylabel=L"\omega_2", zlabel=L"x(\omega_1,\omega_2,t)",
                xticks=heat2d.spatial_domain[1][1]:0.2:heat2d.spatial_domain[1][2],
                yticks=heat2d.spatial_domain[2][1]:0.2:heat2d.spatial_domain[2][2],
                xlabelsize=35, ylabelsize=35, zlabelsize=35,
                xticklabelsize=22, yticklabelsize=22, zticklabelsize=22)
    ax2 = Axis(fig0[1, 2], xlabel=L"\omega_1", ylabel=L"\omega_2", aspect=DataAspect(),
               xticks=heat2d.spatial_domain[1][1]:0.2:heat2d.spatial_domain[1][2],
               yticks=heat2d.spatial_domain[2][1]:0.2:heat2d.spatial_domain[2][2],
               xlabelsize=35, ylabelsize=35, xticklabelsize=22, yticklabelsize=22)
    ax3 = Axis3(fig0[2, 1], xlabel=L"\omega_1", ylabel=L"\omega_2", zlabel=L"x(\omega_1,\omega_2,t)",
                xticks=heat2d.spatial_domain[1][1]:0.2:heat2d.spatial_domain[1][2],
                yticks=heat2d.spatial_domain[2][1]:0.2:heat2d.spatial_domain[2][2],
                xlabelsize=35, ylabelsize=35, zlabelsize=35,
                xticklabelsize=22, yticklabelsize=22, zticklabelsize=22)
    ax4 = Axis(fig0[2, 2], xlabel=L"\omega_1", ylabel=L"\omega_2", aspect=DataAspect(),
               xticks=heat2d.spatial_domain[1][1]:0.2:heat2d.spatial_domain[1][2],
               yticks=heat2d.spatial_domain[2][1]:0.2:heat2d.spatial_domain[2][2],
               xlabelsize=35, ylabelsize=35, xticklabelsize=22, yticklabelsize=22)
    Label(fig0[0, :], "Temperature distribution at initial (top) and final time (bottom)", fontsize=35)
    colsize!(fig0.layout, 2, Aspect(1, 0.8))
    sf1 = surface!(ax1, heat2d.xspan, heat2d.yspan, Xflat[1])
    hm1 = heatmap!(ax2, heat2d.xspan, heat2d.yspan, Xflat[1])
    sf2 = surface!(ax3, heat2d.xspan, heat2d.yspan, Xflat[end])
    hm2 = heatmap!(ax4, heat2d.xspan, heat2d.yspan, Xflat[end])
    Colorbar(fig0[1, 3], hm1) 
    Colorbar(fig0[2, 3], hm2)
    display(fig0)
    save(joinpath(FILEPATH, "plots/heat2d/heat2d_initial_final.png"), fig0)
end

#==================================#
## Compute the SVD for the POD basis
#==================================#
r = 12  # order of the reduced form
V, Σ, _ = svd(X)
Vr = V[:, 1:r]
Σr = Σ[1:r]

#========================#
## Compute the iPOD basis
#========================#
isvd = iSVD(x1=X[:,1], algo=:brand1)
full_increment!(isvd, X[:,2:end], tol=1e-9, verbose=true)
iVr = isvd.Q[:,1:r]
iΣ = isvd.Σ
iΣr = sort(iΣ, rev=true)[1:r]

#======================#
## Plot Singular Values
#======================#
with_theme(theme_latexfonts()) do
    fig0 = Figure(fontsize=20, backgroundcolor="#FFFFFF")
    ax = Axis(fig0[1,1], title="Singular Values", xlabel="Index", ylabel="Value", yscale=log10)
    scatterlines!(ax, 1:r, Σr, color=:black, linewidth=3, label="SVD")
    scatterlines!(ax, 1:r, iΣr, color=:red, linewidth=2, linestyle=:dash, label="iSVD")
    axislegend(ax, labelsize=20, position=:rt)
    display(fig0)
    save(joinpath(FILEPATH, "plots/heat2d/singular_values.png"), fig0)
end

#==============#
## POD-Galerkin
#==============#
op = LnL.pod(op_heat, Vr, options.system)

#=======#
## OpInf
#=======#
# Obtain derivative data
op_inf = LnL.opinf(X, Vr, options; U=U, Y=Y)
O_inf = vcat(op_inf.A', op_inf.B')

#============================#
## Tikhonov Regularized OpInf
#============================#
options.with_reg = true
options.λ = LnL.TikhonovParameter(
    A = 1e-6,
    B = 1e-6,
    C = 1e-6
)
op_inf_reg = LnL.opinf(X, Vr, options; U=U, Y=Y)

#==================#
## Streaming-OpInf
#==================#
# Save data 
Xfull = copy(X)
Yfull = copy(Y)
Ufull = copy(U)

# Obtain derivative data and adjust data
Xdot = (X[:, 2:end] - X[:, 1:end-1]) / heat2d.Δt
idx = 2:heat2d.time_dim
X = X[:, idx]  
U = U[:, idx]
Y = Y[:, idx] 

## Streamify the data based on the selected streamsizes
streamsize = 1
X_stream = LnL.streamify(iVr' * X, streamsize)
U_stream = LnL.streamify(U, streamsize)
Y_stream = LnL.streamify(Y, streamsize)
Xdot_stream = LnL.streamify(iVr' * Xdot, streamsize)
num_of_streams = length(X_stream)

## Initialize the stream
# γs = 1e-15
# γo = 1e-15
γs = 1e-9
γo = 1e-9
state_stream, output_stream = LnL.StreamingOpInf(options=options, n=r, m=4, l=1, algorithm=:RLS, γs=γs, γo=γo)

# Placeholders
state_stream_res = (
    true_stream_err = zeros(r, num_of_streams),
    stream_err      = zeros(r, num_of_streams),
    rse             = zeros(r, num_of_streams),
    post_err        = zeros(num_of_streams),
    conv_factor     = zeros(num_of_streams),
)
output_stream_res = (
    true_stream_err = zeros(r, num_of_streams),
    stream_err      = zeros(r, num_of_streams),
    rse             = zeros(r, num_of_streams),
    post_err        = zeros(num_of_streams),
    conv_factor     = zeros(num_of_streams),
)
Es = nothing
Eo = nothing

## Stream one-by-one and collect data
@showprogress for i in 1:num_of_streams
    # Stream, update, and get data matrix for the state system
    D = LnL.stream!(state_stream, X_stream[i], Xdot_stream[i]; U=U_stream[i], final_step=true)

    # Stream and update the output system
    LnL.stream_output!(output_stream, X_stream[i], Y_stream[i])

    # Unpack operators
    tmp = LnL.Operators()
    LnL.unpack_operators!(tmp, state_stream.cache.O', state_stream.termination_settings[:dims], state_stream.termination_settings[:syms])
    tmp.C = output_stream.cache.O'    

    # Error factors
    state_err_fact = 1.0I - state_stream.cache.K * D
    output_err_fact = 1.0I - output_stream.cache.K * X_stream[i]'
    Es_true = O_inf - state_stream.cache.O
    Eo_true = op_inf.C - output_stream.cache.O'

    # Initialize the error factors
    if i == 1
        Es = Es_true
        Eo = Eo_true'
    end
    
    # Update the error factors
    Es = state_err_fact * Es
    Eo = output_err_fact * Eo

    # Loop through each reduced dimension
    for (j, ri) in enumerate(1:r)
        # Relative state and output errors
        Xtmp = heat2d.integrate_model(
            heat2d.tspan, iVr[:,1:ri]' * heat2d.IC, U; linear_matrix=tmp.A[1:ri,1:ri], control_matrix=tmp.B[1:ri,:], 
            system_input=true, integrator_type=:BackwardEuler
        )
        Ytmp = tmp.C[:,1:ri] * Xtmp
        state_stream_res.rse[j, i] = LnL.rel_state_error(Xfull, Xtmp, iVr[:,1:ri])
        output_stream_res.rse[j, i] = LnL.rel_output_error(Yfull, Ytmp)

        # Index for streaming errors
        idx = vcat(collect(1:ri),collect(r+1:r+4))

        # Streaming errors
        O_norm = norm(O_inf[idx,1:ri], 2)
        Es_true_sub = Es_true[idx,1:ri]
        Eo_true_sub = Eo_true[1:ri]
        Es_sub = Es[idx,1:ri]
        Eo_sub = Eo[1:ri]

        state_stream_res.true_stream_err[j, i] = norm(Es_true_sub, 2) / O_norm
        state_stream_res.stream_err[j,i] = norm(Es_sub,2) / O_norm
        output_stream_res.true_stream_err[j,i] = norm(Eo_true_sub, 2) / O_norm
        output_stream_res.stream_err[j,i] = norm(Eo_sub,2) / O_norm
    end

    # A posteriori error and conversion factors
    state_stream_res.post_err[i] = norm(state_stream.cache.ξpost,2)
    state_stream_res.conv_factor[i] = state_stream.cache.C[1]
    output_stream_res.post_err[i] = norm(output_stream.cache.ξpost,2)
    output_stream_res.conv_factor[i] = output_stream.cache.C[1]
end

op_stream = LnL.terminate_stream(state_stream, output_stream)

# ##
# LnL.stream_all!(state_stream, X_stream, Xdot_stream; U=U_stream)
# LnL.stream_output_all!(output_stream, X_stream, Y_stream)

#=================#
## Relative Error 
#=================#
# Collect all operators into a dictionary
op_dict = Dict(
    "POD" => op,
    "OpInf" => op_inf,
    "TR-OpInf" => op_inf_reg,
    "Streaming-OpInf" => op_stream
)

r = size(Vr,2)
rse = Dict{String, Vector{Float64}}()
roe = Dict{String, Vector{Float64}}()
for (key, op) in op_dict
    rse[key] = Vector{Float64}[]
    roe[key] = Vector{Float64}[]
    for i = 1:r
        if key == "Streaming-OpInf"
            Vri = iVr[:, 1:i]
        else
            Vri = Vr[:, 1:i]
        end
        # Integrate the system for reconstruction
        Xtmp = heat2d.integrate_model(
            heat2d.tspan, Vri' * heat2d.IC, Ufull; linear_matrix=op.A[1:i,1:i], control_matrix=op.B[1:i,:], 
            system_input=true, integrator_type=:BackwardEuler
        )

        foo = LnL.rel_state_error(Xfull, Xtmp, Vri)
        Ytmp = op.C[1:end, 1:i] * Xtmp
        bar = LnL.rel_output_error(Yfull, Ytmp)
        push!(rse[key], foo)
        push!(roe[key], bar)
        @info "($key) r = $i, State Error = $(round(foo,sigdigits=4)), Output Error = $(round(bar,sigdigits=4))"
    end
end

## Plot
provided_keys = ["POD", "OpInf", "TR-OpInf", "Streaming-OpInf"]
with_theme(theme_latexfonts()) do
    fig1 = Figure(fontsize=20, size=(1200,600))
    # Relative State Error
    ax1 = Axis(fig1[1, 1], 
        xlabel=L"reduced dimension, $r$",
        ylabel="Relative State Error", 
        # title="Relative State Error", 
        yscale=log10,
        xlabelsize=30,
        ylabelsize=30,
        xticklabelsize=25,
        yticklabelsize=25,
        xticks=1:r
    )
    for key in provided_keys
        scatterlines!(ax1, 1:r, rse[key])
    end
    # Relative Output Error
    lines = []
    labels = []
    ax2 = Axis(fig1[1, 2], 
        xlabel=L"reduced dimensions, $r$", 
        ylabel="Relative Output Error", 
        # title="Relative Output Error", 
        yscale=log10,
        xlabelsize=30,
        ylabelsize=30,
        xticklabelsize=25,
        yticklabelsize=25,
        xticks=1:r
    )
    for key in provided_keys
        l = scatterlines!(ax2, 1:r, roe[key], label=key)
        push!(lines, l)
        push!(labels, key)
    end
    Legend(fig1[2, 1:2], 
        lines, labels,
        orientation=:horizontal, 
        halign=:center, 
        tellwidth=false, 
        tellheight=true,
        labelsize=28
    )
    Label(fig1[0, :], "2D Heat Equation", fontsize=35)
    display(fig1)
    save(joinpath(FILEPATH, "plots/heat2d/relative_error.png"), fig1)
end

#==========================================#
## Plot streaming error and rse per stream
#==========================================#
axis_colors = Makie.categorical_colors(:tab10, 2)
ylimits = [[1e-6, 1e1], [1e-1, 1e1], [1e-6, 1e1], [1e-11, 1e-4]]
with_theme(theme_latexfonts()) do
    fig2 = Figure(size=(1500,900))
    xtick_vals = 0:(num_of_streams ÷ 2):num_of_streams
    lines_ = []
    labels_ = []
    axes = []
    for (j,ri) in enumerate([4,8,12])
        push!(axes, Axis(fig2[1, j], 
            xlabel=L"$k$-th stream", 
            ylabel=j == 1 ? 
                   L"\Vert \mathbf{X}_{\mathrm{true}}-\mathbf{X}_{\mathrm{recon}}\Vert_F / \Vert\mathbf{X}_{\mathrm{true}}\Vert_F" :
                   "", 
            # title=L"Relative State Error & Streaming Error, $r = %$ri$", 
            yscale=log10, xticks=xtick_vals, yticklabelcolor=axis_colors[1],
            xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
            ylabelcolor=axis_colors[1]
        ))
        push!(axes, Axis(fig2[1, j], 
            # ylabel=j==3 ? 
            #           L"\Vert\mathcal{E}_k\Vert_F=\Vert(\mathbf{I}-\mathbf{K}_k\mathbf{D}_k)\mathcal{E}_{k-1}\Vert_F" :
            #           "", 
            ylabel=j==3 ? 
                      L"\Vert\mathbf{O}_* - \mathbf{O}_k\Vert_F / \Vert\mathbf{O}_*\Vert_F" :
                      "", 
            yticklabelcolor=axis_colors[2], yaxisposition=:right, yscale=log10, ygridstyle=:dash,
            xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
            ylabelcolor=axis_colors[2]
        ))
        hidespines!(axes[4*(j-1)+2])
        hidexdecorations!(axes[4*(j-1)+2])
        push!(axes, Axis(fig2[2, j], 
            xlabel=L"$k$-th stream", 
            ylabel=j==1 ? 
                    L"\Vert\mathbf{Y}_{\mathrm{true}}-\mathbf{Y}_{\mathrm{recon}}\Vert_F / \Vert\mathbf{Y}_{\mathrm{true}}\Vert_F" :
                    "", 
            # title=L"Relative Output Error & Streaming Error, $r = %$ri$", 
            yscale=log10, xticks=xtick_vals, yticklabelcolor=axis_colors[1],
            xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
            ylabelcolor=axis_colors[1]
        ))
        push!(axes, Axis(fig2[2, j],
            # ylabel=j==3 ? 
            #         L"\Vert\mathcal{E}_{y_k}\Vert_F=\Vert(\mathbf{I}-\mathbf{K}_{y_k}\hat{\mathbf{X}}_k^\top)\mathcal{E}_{y_{k-1}}\Vert_F" :
            #         "",
            ylabel=j==3 ? 
                      L"\Vert\mathbf{O}_* - \mathbf{O}_k\Vert_F / \Vert\mathbf{O}_*\Vert_F" :
                      "", 
            yticklabelcolor=axis_colors[2], yaxisposition=:right, yscale=log10, ygridstyle=:dash,
            xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
            ylabelcolor=axis_colors[2]
        ))
        hidespines!(axes[4*(j-1)+4])
        hidexdecorations!(axes[4*(j-1)+4])

        ylims!(axes[4*(j-1)+1], ylimits[1]...)
        ylims!(axes[4*(j-1)+2], ylimits[2]...)
        ylims!(axes[4*(j-1)+3], ylimits[3]...)
        ylims!(axes[4*(j-1)+4], ylimits[4]...)

        l = scatterlines!(axes[4*(j-1)+1], 1:num_of_streams, state_stream_res.rse[ri,:], color=axis_colors[1])
        scatterlines!(axes[4*(j-1)+2], 1:num_of_streams, state_stream_res.stream_err[ri,:], color=axis_colors[2])
        scatterlines!(axes[4*(j-1)+3], 1:num_of_streams, output_stream_res.rse[ri,:], color=axis_colors[1])
        scatterlines!(axes[4*(j-1)+4], 1:num_of_streams, output_stream_res.stream_err[ri,:], color=axis_colors[2])
        text!(axes[4*(j-1)+1], 0, ylimits[1][1]*2, text="r = $ri", fontsize=25)
        text!(axes[4*(j-1)+3], 0, ylimits[3][1]*2, text="r = $ri", fontsize=25)
        push!(lines_, l)
        push!(labels_, "r = $ri")
    end
    Label(fig2[0, :], "Relative State/Output Error and Streaming Error per stream for different reduced dimensions", fontsize=32)
    display(fig2)
    save(joinpath(FILEPATH, "plots/heat2d/streaming_error.png"), fig2)
end

#================================================#
## Plot a posteriori error and conversion factor
#================================================#
with_theme(theme_latexfonts()) do 
    fig3 = Figure(size=(900,500))
    axis_colors = Makie.categorical_colors(:tab10, 2)
    xtick_vals = 0:(num_of_streams ÷ 5):num_of_streams
    ax1 = Axis(fig3[1, 1],
        title="A Posteriori Error and Conversion Factor per stream",
        xlabel=L"$k$-th stream", 
        ylabel=L"\Vert\xi_k^+\Vert_2",
        # title=L"Relative State Error & Streaming Error, $r = %$ri$", 
        xticks=xtick_vals, yticklabelcolor=axis_colors[1],
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        ylabelcolor=axis_colors[1], titlesize=30, yscale=log10
    )
    ax2 = Axis(fig3[1, 1],
        ylabel=L"\gamma_k",
        yticklabelcolor=axis_colors[2], yaxisposition=:right, ygridstyle=:dash,
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        ylabelcolor=axis_colors[2]
    )
    hidespines!(ax2)
    hidexdecorations!(ax2)
    scatterlines!(ax1, 1:num_of_streams, state_stream_res.post_err, color=axis_colors[1])
    scatterlines!(ax2, 1:num_of_streams, state_stream_res.conv_factor, color=axis_colors[2])
    display(fig3)
    save(joinpath(FILEPATH, "plots/heat2d/aposteriori_error.png"), fig3)
end

#========================#
## Animate eyeball norm 
#========================#
Xtmp = heat2d.integrate_model(
    heat2d.tspan, iVr' * heat2d.IC, U; linear_matrix=op_stream.A, control_matrix=op_stream.B, 
    system_input=true, integrator_type=:BackwardEuler
)
X2d_stream = invec.(eachcol(iVr * Xtmp), heat2d.spatial_dim...) 
X2d = invec.(eachcol(Xfull), heat2d.spatial_dim...)
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(1300,1000))
    ax1 = Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="u(x,y,t)")
    ax2 = Axis3(fig[2, 1], xlabel="x", ylabel="y", zlabel="u(x,y,t)",
                limits=(nothing, nothing, nothing, nothing, -2e-5, 2e-5))
    ax3 = Axis(fig[1, 2], xlabel="x", ylabel="y", aspect=DataAspect())
    ax4 = Axis(fig[2, 2], xlabel="x", ylabel="y", aspect=DataAspect())
    colsize!(fig.layout, 2, Aspect(1, 0.8))
    sf1 = surface!(ax1, heat2d.xspan, heat2d.yspan, X2d_stream[1])
    sf2 = surface!(ax2, heat2d.xspan, heat2d.yspan, X2d[1] - X2d_stream[1], colorrange=(-2e-5,2e-5))
    hm1 = heatmap!(ax3, heat2d.xspan, heat2d.yspan, X2d_stream[1])
    hm2 = heatmap!(ax4, heat2d.xspan, heat2d.yspan, X2d[1] - X2d_stream[1], colorrange=(-2e-5,2e-5))
    Colorbar(fig[1, 3], hm1)
    Colorbar(fig[2, 3], hm2)
    record(fig, joinpath(FILEPATH, "plots/heat2d/eyeball_norm.mp4"), 1:heat2d.time_dim) do i
        sf1[3] = X2d_stream[i]
        sf2[3] = X2d[i] - X2d_stream[i]
        hm1[3] = X2d_stream[i]
        hm2[3] = X2d[i] - X2d_stream[i]
        autolimits!(ax1) # update limits
        # autolimits!(ax2) # update limits
        autolimits!(ax3) # update limits
        autolimits!(ax4) # update limits
    end
end