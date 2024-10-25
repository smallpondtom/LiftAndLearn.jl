"""
viscous Burgers' equation iOpInf example
"""

#===============#
## Load Packages
#===============#
using CairoMakie
using LinearAlgebra
using IncrementalSVD
using PolynomialModelReductionDataset: BurgersModel
using ProgressMeter
using LiftAndLearn
const LnL = LiftAndLearn

#===================#
## Import functions
#===================#
include("utilities/extract_operators.jl")

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"Streaming-OpInf/") : joinpath(pwd(), "scripts/Streaming-OpInf/")

#========================#
## Burgers equation setup
#========================#
Ω = (0.0, 1.0)
Nx = 2^7; dt = 1e-4
burgers = BurgersModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    diffusion_coeffs=0.5, BC=:dirichlet,
)
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=[1,2],
        control=1,
        output=1
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=dt,
        deriv_type="SI"
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)
num_of_inputs = 3
rmax = 15

#===============#
## Generate Data
#===============#
μ = burgers.diffusion_coeffs
A, F, B = burgers.finite_diff_model(burgers, μ)
C = ones(1, burgers.spatial_dim) / burgers.spatial_dim
op_burgers = LnL.Operators(A=A, B=B, C=C, A2u=F)

# Reference solution
Uref = ones(burgers.time_dim, 1)  # Reference input/boundary condition
Xref = burgers.integrate_model(burgers.tspan, burgers.IC, Uref; linear_matrix=A,
                               control_matrix=B, quadratic_matrix=F, system_input=true)
Yref = C * Xref

Urand = rand(burgers.time_dim, num_of_inputs)  # uniformly random input
Xall = Vector{Matrix{Float64}}(undef, num_of_inputs)
Xdotall = Vector{Matrix{Float64}}(undef, num_of_inputs)
Xstore = nothing  # store one data trajectory for plotting
for j in 1:num_of_inputs
    @info "Generating data for input $j"
    states = burgers.integrate_model(
        burgers.tspan, burgers.IC, Urand[:, j]; linear_matrix=A,
        control_matrix=B, quadratic_matrix=F, system_input=true
    )
    Xall[j] = states[:, 2:end]
    Xdotall[j] = (states[:, 2:end] - states[:, 1:end-1]) / burgers.Δt
    if j == 1
        Xstore = states
    end
end
X = reduce(hcat, Xall)
Xdot = reduce(hcat, Xdotall)
U = reshape(Urand[2:end,:], (burgers.time_dim-1) * num_of_inputs, 1)'
Y = C * X

#==================================#
## Compute the SVD for the POD basis
#==================================#
r = rmax  # order of the reduced form
V, Σ, _ = svd(X)
Vr = V[:, 1:r]
Σr = Σ[1:r]

#=======================#
## Compute the iPOD basis
#=======================#
isvd = iSVD(x1=X[:,1], algo=:baker, kselect=r)
full_increment!(isvd, X[:,2:end], tol=1e-12, verbose=true)
iVr = isvd.Q[:,1:r]
iΣ = isvd.Σ
iΣr = sort(iΣ, rev=true)[1:r]

#====================#
## Plot Data to Check
#====================#
with_theme(theme_latexfonts()) do
    fig0 = Figure(fontsize=20, size=(1300,500), backgroundcolor="#FFFFFF")
    ax1 = Axis3(fig0[1, 1], xlabel="x", ylabel="t", zlabel="u(x,t)")
    surface!(ax1, burgers.xspan, burgers.tspan, Xstore)
    ax2 = Axis(fig0[1, 2], xlabel="x", ylabel="t")
    hm = heatmap!(ax2, burgers.xspan, burgers.tspan, Xstore)
    Colorbar(fig0[1, 3], hm)
    display(fig0)
end

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
    save(joinpath(FILEPATH, "plots/burgers/singular_values.png"), fig0)
end

#=====================#
## Intrusive-POD model
#=====================#
op = LnL.pod(op_burgers, Vr, options.system)

#=============#
## OpInf model
#=============#
op_inf = LnL.opinf(X, Vr, options; U=U, Y=Y, Xdot=Xdot)

#============================#
## Tikhonov Regularized OpInf
#============================#
options.with_reg = true
options.λ = LnL.TikhonovParameter(
    A = 1e-15,
    A2 = 1e-15,
    B = 1e-15,
    C = 1e-15
)
op_inf_reg = LnL.opinf(X, Vr, options; U=U, Y=Y, Xdot=Xdot)
O_inf = vcat(op_inf_reg.A', op_inf_reg.B', op_inf_reg.A2u')

#==================#
## Streaming-OpInf
#==================#
# Streamify the data based on the selected streamsizes
streamsize = 1
X_stream = LnL.streamify(iVr' * X, streamsize)
U_stream = LnL.streamify(U, streamsize)
Y_stream = LnL.streamify(Y, streamsize)
Xdot_stream = LnL.streamify(iVr' * Xdot, streamsize)
num_of_streams = length(X_stream)

# Initialize the stream
γs = 1e-9
γo = 1e-12
state_stream, output_stream = LnL.StreamingOpInf(options=options, n=rmax, m=1, l=1, γs=γs, γo=γo, algorithm=:iQRRLS)

# Placeholders
step = 100
used_num_of_streams = num_of_streams ÷ step + 1
state_stream_res = (
    true_stream_err = zeros(r, used_num_of_streams),
    stream_err      = zeros(r, used_num_of_streams),
    rse             = zeros(r, used_num_of_streams),
    post_err        = zeros(used_num_of_streams),
    conv_factor     = zeros(used_num_of_streams),
)
output_stream_res = (
    true_stream_err = zeros(r, used_num_of_streams),
    stream_err      = zeros(r, used_num_of_streams),
    rse             = zeros(r, used_num_of_streams),
    post_err        = zeros(used_num_of_streams),
    conv_factor     = zeros(used_num_of_streams),
)
Es_full = nothing
Eo_full = nothing

## Stream one-by-one and collect data
ct = 1
used_streams = []
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
    state_err_fact = I - state_stream.cache.K * D
    output_err_fact = I - output_stream.cache.K * X_stream[i]'
    Es = O_inf - state_stream.cache.O
    Eo = op_inf.C - output_stream.cache.O'

    # Loop through each reduced dimension
    if i % step == 0 || i == 1 || i == num_of_streams
        for (j, ri) in enumerate(1:r)
            # Relative state and output errors
            quad_idx = quad_indices(r, ri)
            Xtmp = burgers.integrate_model(
                burgers.tspan, iVr[:,1:ri]' * burgers.IC, Uref; linear_matrix=tmp.A[1:ri,1:ri],
                control_matrix=tmp.B[1:ri,:], quadratic_matrix=tmp.A2u[1:ri,quad_idx], system_input=true
            )

            Ytmp = tmp.C[:,1:ri] * Xtmp
            state_stream_res.rse[j, ct] = LnL.rel_state_error(Xref, Xtmp, iVr[:,1:ri])
            output_stream_res.rse[j, ct] = LnL.rel_output_error(Yref, Ytmp)

            # Index for streaming errors
            idx = extract_indices(state_stream, r, ri, options.system)

            # Streaming errors
            O_norm = norm(O_inf[idx,1:ri], 2)
            state_stream_res.true_stream_err[j, ct] = norm(Es[idx,1:ri], 2) / O_norm
            Es_full = ct == 1 ? Es[idx,1:ri] : (state_err_fact * Es)[idx]
            state_stream_res.stream_err[j,ct] = norm(Es_full,2) / O_norm
            output_stream_res.true_stream_err[j,ct] = norm(Eo[1:ri], 2) / O_norm
            Eo_full = ct == 1 ? Eo[1:ri] : (output_err_fact * Eo')[1:ri]
            output_stream_res.stream_err[j,ct] = norm(Eo_full,2) / O_norm
        end
        # A posteriori error and conversion factors
        state_stream_res.post_err[ct] = norm(state_stream.cache.ξpost,2)
        state_stream_res.conv_factor[ct] = state_stream.cache.C[1]
        output_stream_res.post_err[ct] = norm(output_stream.cache.ξpost,2)
        output_stream_res.conv_factor[ct] = output_stream.cache.C[1]

        ct += 1
        push!(used_streams, i)
    end
end

op_stream = LnL.terminate_stream(state_stream, output_stream)

##
# LnL.stream_all!(state_stream, X_stream, Xdot_stream; U=U_stream, verbose=true)
# LnL.stream_output_all!(output_stream, X_stream, Y_stream, true)
# op_stream = LnL.terminate_stream(state_stream, output_stream)

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
        quad_idx = quad_indices(r, i)
        Xtmp = burgers.integrate_model(
            burgers.tspan, Vri' * burgers.IC, Uref; linear_matrix=op.A[1:i,1:i],
            control_matrix=op.B[1:i,:], quadratic_matrix=op.A2u[1:i,quad_idx], system_input=true
        )

        foo = LnL.rel_state_error(Xref, Xtmp, Vri)
        Ytmp = op.C[1:end, 1:i] * Xtmp
        bar = LnL.rel_output_error(Yref, Ytmp)
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
    Label(fig1[0, :], "Burgers' Equation", fontsize=35)
    display(fig1)
    save(joinpath(FILEPATH, "plots/burgers/relative_error.png"), fig1)
end

#==========================================#
## Plot streaming error and rse per stream
#==========================================#
axis_colors = Makie.categorical_colors(:tab10, 2)
ylimits = [[1e-5, 1e1], [1e-5, 1e1], [1e-2, 1e1], [1e-21, 1e-17]]
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

        l = scatterlines!(axes[4*(j-1)+1], used_streams, state_stream_res.rse[ri,:], color=axis_colors[1])
        scatterlines!(axes[4*(j-1)+2], used_streams, state_stream_res.stream_err[ri,:], color=axis_colors[2])
        scatterlines!(axes[4*(j-1)+3], used_streams, output_stream_res.rse[ri,:], color=axis_colors[1])
        scatterlines!(axes[4*(j-1)+4], used_streams, output_stream_res.stream_err[ri,:], color=axis_colors[2])
        text!(axes[4*(j-1)+1], 0, ylimits[1][1]*2, text="r = $ri", fontsize=25)
        text!(axes[4*(j-1)+3], 0, ylimits[3][1]*2, text="r = $ri", fontsize=25)
        push!(lines_, l)
        push!(labels_, "r = $ri")
    end
    Label(fig2[0, :], "Relative State/Output Error and Streaming Error per stream for different reduced dimensions", fontsize=32)
    display(fig2)
    save(joinpath(FILEPATH, "plots/burgers/streaming_error.png"), fig2)
end

#================================================#
## Plot a posteriori error and conversion factor
#================================================#
with_theme(theme_latexfonts()) do 
    fig3 = Figure(size=(900,800))
    axis_colors = Makie.categorical_colors(:tab10, 2)
    xtick_vals = 0:(num_of_streams ÷ 5):num_of_streams
    ax1 = Axis(fig3[1, 1],
        title="A Posteriori Error and Conversion Factor per stream",
        xlabel=L"$k$-th stream", 
        ylabel=L"\Vert(\xi_{\mathrm{post}})_k\Vert_2",
        # title=L"Relative State Error & Streaming Error, $r = %$ri$", 
        xticks=xtick_vals, yticklabelcolor=axis_colors[1],
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        ylabelcolor=axis_colors[1], titlesize=30
    )
    ax2 = Axis(fig3[1, 1],
        ylabel=L"\gamma_k",
        yticklabelcolor=axis_colors[2], yaxisposition=:right, ygridstyle=:dash,
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        ylabelcolor=axis_colors[2]
    )
    hidespines!(ax2)
    hidexdecorations!(ax2)
    scatterlines!(ax1, used_streams, state_stream_res.post_err, color=axis_colors[1])
    scatterlines!(ax2, used_streams, state_stream_res.conv_factor, color=axis_colors[2])
    display(fig3)
    save(joinpath(FILEPATH, "plots/burgers/aposteriori_error.png"), fig3)
end