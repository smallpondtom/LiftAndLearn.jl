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
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"Streaming-OpInf/heat2d") : joinpath(pwd(), "scripts/Streaming-OpInf/heat2d")

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

#=========================#
## Generate training data
#=========================#
# Generate the input data (same for all parameters)
U = [1.0, 1.0, -1.0, -1.0]
U = repeat(U, 1, heat2d.time_dim)

# Construct the output matrix (same for all parameters)
# C = ones(1, (Int ∘ prod)(heat2d.spatial_dim)) / heat2d.spatial_dim[1] / heat2d.spatial_dim[2]

@showprogress Threads.@threads for (i, μ) in collect(enumerate(heat2d.diffusion_coeffs))
    A, B = heat2d.finite_diff_model(heat2d, μ)
    # op_heat = LnL.Operators(A=A, B=B, C=C)
    op_heat = LnL.Operators(A=A, B=B)

    # Compute the state snapshot data with backward Euler
    X = heat2d.integrate_model(
        heat2d.tspan, heat2d.IC, U; linear_matrix=A, control_matrix=B, 
        system_input=true, integrator_type=:BackwardEuler
    )

    # Compute the output of the system
    # Y = C * X

    data = Dict(
        "X" => X, "U" => U, # "Y" => Y,
        "A" => A, "B" => B, # "C" => C,
        "mu" => μ,
    )
    mu_str = @sprintf("%1.4f", μ)
    save(joinpath(FILEPATH, "data/training/0$(i)_mu$(mu_str).jld2"), data)
end

#========================#
## Generate testing data
#========================#
Mtest = 5
seed = 1234
randn_gen = Random.MersenneTwister(seed)
μs_test = rand(randn_gen, Mtest) * (heat2d.param_domain[2] - heat2d.param_domain[1]) .+ heat2d.param_domain[1]
@showprogress Threads.@threads for (i,μ) in collect(enumerate(μs_test))
    A, B = heat2d.finite_diff_model(heat2d, μ)
    # op_heat = LnL.Operators(A=A, B=B, C=C)
    op_heat = LnL.Operators(A=A, B=B)

    # Compute the state snapshot data with backward Euler
    X = heat2d.integrate_model(
        heat2d.tspan, heat2d.IC, U; linear_matrix=A, control_matrix=B, 
        system_input=true, integrator_type=:BackwardEuler
    )

    # Compute the output of the system
    # Y = C * X

    data = Dict(
        "X" => X, "U" => U, # "Y" => Y,
        "A" => A, "B" => B, # "C" => C,
        "mu" => μ
    )
    mu_str = @sprintf("%1.4f", μ)
    save(joinpath(FILEPATH, "data/testing/0$(i)_mu$(mu_str).jld2"), data)
end

#===============================#
## Save the options for system
#===============================#
save(joinpath(FILEPATH, "data/setup.jld2"), "options", options, "heat2d", heat2d)