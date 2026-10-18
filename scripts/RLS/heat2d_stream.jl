"""
    Streaming-OpInf example of the 2D heat equation.
"""

#===========#
## Packages
#===========#
using BenchmarkTools
using CairoMakie
using LinearAlgebra
using ProgressMeter
using PolynomialModelReductionDataset: Heat2DModel
using UniqueKronecker: invec

#=============#
## My modules
#=============#
using LiftAndLearn
const LnL = LiftAndLearn

#=================#
## Global Settings
#=================#
SAVEFIG = true

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"RLS/") : joinpath(pwd(), "scripts/RLS/")

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

# Copy the data for later analysis
Xfull = copy(X)
Yfull = copy(Y)
Ufull = copy(U)

#=====================#
## Plot Data to Check
#=====================#
Xflat = invec.(eachcol(X), heat2d.spatial_dim...)
with_theme(theme_latexfonts()) do
    fig0 = Figure(fontsize=20, size=(1200,1050))
    ax1 = Axis3(fig0[1, 1], xlabel=L"x", ylabel=L"y", zlabel=L"u(x,y,t)",
                xticks=heat2d.spatial_domain[1][1]:0.2:heat2d.spatial_domain[1][2],
                yticks=heat2d.spatial_domain[2][1]:0.2:heat2d.spatial_domain[2][2],
                xlabelsize=35, ylabelsize=35, zlabelsize=35,
                xticklabelsize=22, yticklabelsize=22, zticklabelsize=22)
    ax2 = Axis(fig0[1, 2], xlabel=L"x", ylabel=L"y", aspect=DataAspect(),
               xticks=heat2d.spatial_domain[1][1]:0.2:heat2d.spatial_domain[1][2],
               yticks=heat2d.spatial_domain[2][1]:0.2:heat2d.spatial_domain[2][2],
               xlabelsize=35, ylabelsize=35, xticklabelsize=22, yticklabelsize=22)
    ax3 = Axis3(fig0[2, 1], xlabel=L"x", ylabel=L"y", zlabel=L"u(x,y,t)",
                xticks=heat2d.spatial_domain[1][1]:0.2:heat2d.spatial_domain[1][2],
                yticks=heat2d.spatial_domain[2][1]:0.2:heat2d.spatial_domain[2][2],
                xlabelsize=35, ylabelsize=35, zlabelsize=35,
                xticklabelsize=22, yticklabelsize=22, zticklabelsize=22)
    ax4 = Axis(fig0[2, 2], xlabel=L"x", ylabel=L"y", aspect=DataAspect(),
               xticks=heat2d.spatial_domain[1][1]:0.2:heat2d.spatial_domain[1][2],
               yticks=heat2d.spatial_domain[2][1]:0.2:heat2d.spatial_domain[2][2],
               xlabelsize=35, ylabelsize=35, xticklabelsize=22, yticklabelsize=22)
    Label(fig0[0, :], "2D Heat Equation at initial (top) and final time (bottom)", fontsize=35)
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

#========================#
## Animate Data to Check
#========================#
X2d = LnL.invec.(eachcol(X), heat2d.spatial_dim...)
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(1300,500))
    ax1 = Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="u(x,y,t)")
    ax2 = Axis(fig[1, 2], xlabel="x", ylabel="y", aspect=DataAspect())
    colgap!(fig.layout, 0)
    sf = surface!(ax1, heat2d.xspan, heat2d.yspan, X2d[1])
    hm = heatmap!(ax2, heat2d.xspan, heat2d.yspan, X2d[1])
    Colorbar(fig[1, 3], hm)
    record(fig, "scripts/streaming/plots/heat2d/temperature.mp4", 1:heat2d.time_dim) do i
        sf[3] = X2d[i]
        hm[3] = X2d[i]
        autolimits!(ax1) # update limits
        autolimits!(ax2) # update limits
    end
end
##
X2d = nothing
GC.gc()

#==================================#
## Compute the SVD for the POD basis
#==================================#
r = 12  # order of the reduced form
V, Σ, _ = svd(X)
Vr = V[:, 1:r]
Σr = Σ[1:r]

#===========#
## Intrusive
#===========#
op_int = LnL.pod(op_heat, Vr, options.system)

#====================#
## Operator Inference
#====================#
# Obtain derivative data
Xdot = (X[:, 2:end] - X[:, 1:end-1]) / heat2d.Δt
idx = 2:heat2d.time_dim
X = X[:, idx]  
U = U[:, idx]
Y = Y[:, idx] 
op_inf = LnL.opinf(X, Vr, options; U=U, Y=Y, Xdot=Xdot)

#============================#
## Tikhonov Regularized OpInf
#============================#
options.with_reg = true
options.λ = LnL.TikhonovParameter(
    A = 1e-6,
    B = 1e-6,
    C = 1e-3
)
op_inf_reg = LnL.opinf(X, Vr, options; U=U, Y=Y, Xdot=Xdot)

#=================#
## Streaming-OpInf
#=================#
# Streamify the data based on the selected streamsizes
streamsize = 1
X_stream = LnL.streamify(Vr' * X, streamsize)
U_stream = LnL.streamify(U, streamsize)
Y_stream = LnL.streamify(Y, streamsize)
R_stream = LnL.streamify(Vr' * Xdot, streamsize)
num_of_streams = length(X_stream)

## RLS
γs = 5e-10
γo = 1e-10
algo = :RLS
state_stream, output_stream = LnL.StreamingOpInf(options=options, n=r, m=size(U,1), l=size(Y,1); γs=γs, γo=γo, algorithm=algo)
LnL.stream_all!(state_stream, X_stream, R_stream; U=U_stream)
LnL.stream_output_all!(output_stream, X_stream, Y_stream)
op_stream_rls = LnL.terminate_stream(state_stream, output_stream)

## QRRLS
γs = 1e-15
γo = 1e-15
algo = :QRRLS
state_stream, output_stream = LnL.StreamingOpInf(options=options, n=r, m=size(U,1), l=size(Y,1); γs=γs, γo=γo, algorithm=algo)
LnL.stream_all!(state_stream, X_stream, R_stream; U=U_stream)
LnL.stream_output_all!(output_stream, X_stream, Y_stream)
op_stream_qrrls = LnL.terminate_stream(state_stream, output_stream)

## iQRRLS
γs = 1e-15
γo = 1e-15
algo = :iQRRLS
state_stream, output_stream = LnL.StreamingOpInf(options=options, n=r, m=size(U,1), l=size(Y,1); γs=γs, γo=γo, algorithm=algo)
LnL.stream_all!(state_stream, X_stream, R_stream; U=U_stream)
LnL.stream_output_all!(output_stream, X_stream, Y_stream)
op_stream_iqrrls = LnL.terminate_stream(state_stream, output_stream)

#=============================#
## (Analysis 1) Relative Error 
#=============================#
# Collect all operators into a dictionary
op_dict = Dict(
    "POD" => op_int,
    "OpInf" => op_inf,
    "TR-OpInf" => op_inf_reg,
    "RLS-Streaming-OpInf" => op_stream_rls,
    "QRRLS-Streaming-OpInf" => op_stream_qrrls,
    "iQRRLS-Streaming-OpInf" => op_stream_iqrrls,
)

r = size(Vr,2)
rse = Dict{String, Vector{Float64}}()
roe = Dict{String, Vector{Float64}}()
for (key, op) in op_dict
    rse[key] = Vector{Float64}[]
    roe[key] = Vector{Float64}[]
    for i = 1:r
        Vri = Vr[:, 1:i]

        # Integrate the system for reconstruction
        Xtmp = heat2d.integrate_model(
            heat2d.tspan, Vri' * heat2d.IC, Ufull; linear_matrix=op.A[1:i,1:i], control_matrix=op.B[1:i,:], 
            system_input=true, integrator_type=:BackwardEuler
        )

        foo = LnL.rel_state_error(Xfull, Xtmp, Vri)
        Y = op.C[1:end, 1:i] * Xtmp
        bar = LnL.rel_output_error(Yfull, Y)
        push!(rse[key], foo)
        push!(roe[key], bar)
        @info "($key) r = $i, State Error = $(round(foo,sigdigits=4)), Output Error = $(round(bar,sigdigits=4))"
    end
end

## Plot
provided_keys = ["POD", "OpInf", "TR-OpInf", "RLS-Streaming-OpInf",
                 "QRRLS-Streaming-OpInf", "iQRRLS-Streaming-OpInf"]
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
        yticklabelsize=25
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
        yticklabelsize=25
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
        labelsize=28,
        nbanks=2
    )
    display(fig1)
    save(joinpath(FILEPATH, "plots/heat2d/heat2d_error.png"), fig1)
end

#==============================#
## (Analysis 2) execution time 
#==============================#
runtimes = Dict{String, Vector{Float64}}(
    "POD" => Float64[],
    "OpInf" => Float64[],
    "TR-OpInf" => Float64[],
    "RLS-Streaming-OpInf" => Float64[],
    "QRRLS-Streaming-OpInf" => Float64[],
    "iQRRLS-Streaming-OpInf" => Float64[]
)
options.system.output = 0
for ri in 1:r
    # pod
    t_pod = @benchmark LnL.pod(op_heat, Vr[:,1:$ri], options.system)
    tmp = mean(t_pod).time / 1e9
    push!(runtimes["POD"], tmp)

    # opinf
    options.with_reg = false
    t_opinf = @benchmark LnL.opinf(X, Vr[:,1:$ri], options; U=U, Xdot=Xdot)
    tmp = mean(t_opinf).time / 1e9
    push!(runtimes["OpInf"], tmp)

    # tr-opinf
    options.with_reg = true
    t_opinf_reg = @benchmark LnL.opinf(X, Vr[:,1:$ri], options; U=U, Xdot=Xdot) 
    tmp = mean(t_opinf_reg).time / 1e9
    push!(runtimes["TR-OpInf"], tmp)

    # Streamify the data based on the reduced dimension
    X_stream = LnL.streamify(Vr[:,1:ri]' * X, streamsize)
    R_stream = LnL.streamify(Vr[:,1:ri]' * Xdot, streamsize)

    # rls-streaming-opinf
    γs = 5e-10
    γo = 1e-10
    algo = :RLS
    state_stream, output_stream = LnL.StreamingOpInf(options=options, n=ri, m=size(U,1), l=size(Y,1); γs=γs, γo=γo, algorithm=algo)
    t_stream_rls = @benchmark LnL.stream_all!(state_stream, X_stream, R_stream; U=U_stream)
    tmp = mean(t_stream_rls).time / 1e9
    push!(runtimes["RLS-Streaming-OpInf"], tmp)

    # qrrls-streaming-opinf
    γs = 1e-15
    γo = 1e-15
    algo = :QRRLS
    state_stream, output_stream = LnL.StreamingOpInf(options=options, n=ri, m=size(U,1), l=size(Y,1); γs=γs, γo=γo, algorithm=algo)
    t_stream_qrrls = @benchmark LnL.stream_all!(state_stream, X_stream, R_stream; U=U_stream)
    tmp = mean(t_stream_qrrls).time / 1e9
    push!(runtimes["QRRLS-Streaming-OpInf"], tmp)

    # iqrrls-streaming-opinf
    γs = 1e-15
    γo = 1e-15
    algo = :iQRRLS
    state_stream, output_stream = LnL.StreamingOpInf(options=options, n=ri, m=size(U,1), l=size(Y,1); γs=γs, γo=γo, algorithm=algo)
    t_stream_iqrrls = @benchmark LnL.stream_all!(state_stream, X_stream, R_stream; U=U_stream)
    tmp = mean(t_stream_iqrrls).time / 1e9
    push!(runtimes["iQRRLS-Streaming-OpInf"], tmp)
end

## Plot
provided_keys = ["POD", "OpInf", "TR-OpInf", "RLS-Streaming-OpInf",
                 "QRRLS-Streaming-OpInf", "iQRRLS-Streaming-OpInf"]
with_theme(theme_latexfonts()) do
    fig2 = Figure(fontsize=20, size=(1250,600))
    # Relative State Error
    ax1 = Axis(fig2[1, 1], 
        xlabel=L"reduced dimension, $r$",
        ylabel="average runtimes", 
        # title="", 
        yscale=log10,
        xlabelsize=30,
        ylabelsize=30,
        xticklabelsize=25,
        yticklabelsize=25,
        xticks=1:r
    )
    lines = []
    labels = []
    for key in provided_keys
        l = scatterlines!(ax1, 1:r, runtimes[key])
        push!(lines, l)
        push!(labels, key)
    end
    Legend(fig2[2, 1], 
        lines, labels,
        orientation=:horizontal, 
        halign=:center, 
        tellwidth=false, 
        tellheight=true,
        labelsize=28,
        nbanks=2
    )
    display(fig2)
    save(joinpath(FILEPATH, "plots/heat2d/heat2d_runtime.png"), fig2)
end