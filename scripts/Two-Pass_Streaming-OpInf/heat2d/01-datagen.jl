"""
2D heat equation: generate data
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using IncrementalSVD
using LinearAlgebra
using ProgressMeter
using PolynomialModelReductionDataset: Heat2DModel
using Printf
using Random
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"Two-Pass_Streaming-OpInf/heat2d") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/heat2d")

#========================#
## 2D Heat equation setup
#========================#
Ω = ((0.0, 1.0), (0.0, 1.0))  # or ω1 ∈ [0,1], ω2 ∈ [0,1.25]
Nx = 32
Ny = 32 # or 40
M = 10
μs = range(0.1, 1.0, length=M)
heat2d = Heat2DModel(
    spatial_domain=Ω, time_domain=(0,1.0), 
    Δx=(Ω[1][2] + 1/Nx)/Nx, Δy=(Ω[2][2] + 1/Ny)/Ny, Δt=1e-3,
    diffusion_coeffs=μs, BC=(:dirichlet, :dirichlet)
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
        # output=1,
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
    use_backslash=true,
)

#=========================#
## Generate training data
#=========================#
# Generate the (reference) input data (same for all parameters)
Uref = [-1.0, -1.0, 1.0, 1.0]
Uref = repeat(Uref, 1, heat2d.time_dim)

# @showprogress Threads.@threads for (i, μ) in collect(enumerate(heat2d.diffusion_coeffs))
@showprogress for (i, μ) in collect(enumerate(heat2d.diffusion_coeffs))
    A, B = heat2d.finite_diff_model(heat2d, μ)
    op_heat = LnL.Operators(A=A, B=B)

    # Compute the (reference) state snapshot data with backward Euler
    Xref = heat2d.integrate_model(
        heat2d.tspan, heat2d.IC, Uref; linear_matrix=A, control_matrix=B, 
        system_input=true, integrator_type=:BackwardEuler
    )
    Xdot = (Xref[:, 2:end] - Xref[:, 1:end-1]) / heat2d.Δt
    X = Xref[:, 2:end]
    U = Uref[:, 2:end]

    data = Dict(
        "X" => X, "U" => U, "Xdot" => Xdot, 
        "Xref" => Xref, "Uref" => Uref,
        "A" => A, "B" => B,
        "mu" => μ,
    )
    t2 = time()
    mu_str = @sprintf("%1.4f", μ)
    save(joinpath(FILEPATH, "data/training/0$(i)_mu$(mu_str).jld2"), data)
end

#=====================#
## Plot Data to Check
#=====================#
X = load(joinpath(FILEPATH, "data/training/01_mu0.1000.jld2"), "X")
using UniqueKronecker: invec
using CairoMakie
Xflat = invec.(eachcol(X), heat2d.spatial_dim...)
with_theme(theme_latexfonts()) do
    fig0 = Figure(fontsize=20, size=(1200,1050))
    ax1 = Axis3(fig0[1, 1], xlabel=L"x", ylabel=L"y", zlabel=L"s(x,y,t)",
                xticks=heat2d.spatial_domain[1][1]:0.2:heat2d.spatial_domain[1][2],
                yticks=heat2d.spatial_domain[2][1]:0.2:heat2d.spatial_domain[2][2],
                xlabelsize=35, ylabelsize=35, zlabelsize=35,
                xticklabelsize=22, yticklabelsize=22, zticklabelsize=22)
    ax2 = Axis(fig0[1, 2], xlabel=L"x", ylabel=L"y", aspect=DataAspect(),
               xticks=heat2d.spatial_domain[1][1]:0.2:heat2d.spatial_domain[1][2],
               yticks=heat2d.spatial_domain[2][1]:0.2:heat2d.spatial_domain[2][2],
               xlabelsize=35, ylabelsize=35, xticklabelsize=22, yticklabelsize=22)
    ax3 = Axis3(fig0[2, 1], xlabel=L"x", ylabel=L"y", zlabel=L"s(x,y,t)",
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
    save(joinpath(FILEPATH, "plots/heat2d_initial_final.png"), fig0)
end

#========================#
## Generate testing data
#========================#
Mtest = 5
seed = 1234
randn_gen = Random.MersenneTwister(seed)
μs_test = rand(randn_gen, Mtest) * (heat2d.param_domain[2] - heat2d.param_domain[1]) .+ heat2d.param_domain[1]
@showprogress for (i,μ) in collect(enumerate(μs_test))
    A, B = heat2d.finite_diff_model(heat2d, μ)
    op_heat = LnL.Operators(A=A, B=B)

    # Compute the state snapshot data with backward Euler
    X = heat2d.integrate_model(
        heat2d.tspan, heat2d.IC, Uref; linear_matrix=A, control_matrix=B, 
        system_input=true, integrator_type=:BackwardEuler
    )

    # NOTE: Remove the last state to make size nice
    data = Dict(
        "X" => X, "U" => Uref, 
        "A" => A, "B" => B,
        "mu" => μ
    )
    mu_str = @sprintf("%1.4f", μ)
    save(joinpath(FILEPATH, "data/testing/0$(i)_mu$(mu_str).jld2"), data)
end

#===============================#
## Save the options for system
#===============================#
save(joinpath(FILEPATH, "data/setup.jld2"), "options", options, "heat2d", heat2d)