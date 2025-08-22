"""
1D Viscous Burgers' equation: generate data
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using IncrementalSVD
using LinearAlgebra
using ProgressMeter
using PolynomialModelReductionDataset: BurgersModel
using Printf
using Random
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"Two-Pass_Streaming-OpInf/burgers") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/burgers")

#=========================#
## Burgers equation setup
#=========================#
Ω = (0.0, 1.0)
Nx = 2^7; dt = 1e-4
burgers = BurgersModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    diffusion_coeffs=range(0.1, 1.0, length=10), BC=:dirichlet,
)
burgers.IC=0.1*sin.(2π*burgers.xspan)
num_inputs = 10  # number of random inputs for training data
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=[1,2],
        control=1,
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=dt,
        deriv_type="SI",
        DS=20,  # downsampling factor
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)

#=========================#
## Generate training data
#=========================#
seed = 1234
rgen = Random.MersenneTwister(seed)
Urand_all = randn(rgen, burgers.time_dim, num_inputs, burgers.param_dim) # Random input/boundary condition for training data
@showprogress Threads.@threads for (i, μ) in collect(enumerate(burgers.diffusion_coeffs))
    A, F, B = burgers.finite_diff_model(burgers, μ)
    op_burgers = LnL.Operators(A=A, B=B, A2u=F)

    # Compute the reference data with the reference input
    Uref = ones(burgers.time_dim, 1);  # Reference input/boundary condition for OpInf testing 
    Xref = burgers.integrate_model(
        burgers.tspan, burgers.IC, Uref; linear_matrix=A,
        control_matrix=B, quadratic_matrix=F, system_input=true
    )

    # Compute the training with random input 
    Urand = Urand_all[:, :, i]
    Xall = Vector{Matrix{Float64}}(undef, num_inputs)
    Xdotall = Vector{Matrix{Float64}}(undef, num_inputs)
    for j in 1:num_inputs
        states = burgers.integrate_model(
            burgers.tspan, burgers.IC, Urand[:, j], linear_matrix=A,
            control_matrix=B, quadratic_matrix=F, system_input=true
        ) 
        Xall[j] = states[:, 2:end]
        Xdotall[j] = (states[:, 2:end] - states[:, 1:end-1]) / burgers.Δt
    end
    X = reduce(hcat, Xall)
    Xdot = reduce(hcat, Xdotall)
    U = reshape(Urand[2:end,:], (burgers.time_dim - 1) * num_inputs, 1)

    # Down sample the training data
    X = X[:, 1:options.data.DS:end]
    Xdot = Xdot[:, 1:options.data.DS:end]
    U = U[1:options.data.DS:end]

    data = Dict(
        "X" => X, "U" => U, "Xdot" => Xdot, "Xref" => Xref, "Uref" => Uref,
        "A" => A, "B" => B, "F" => F, "mu" => μ, "num_inputs" => num_inputs
    )
    mu_str = @sprintf("%1.4f", μ)
    save(joinpath(FILEPATH, "data/training/0$(i)_mu$(mu_str).jld2"), data)
end

#========================#
## Generate testing data
#========================#
Mtest = 5
μs_test = rand(rgen, Mtest) * (burgers.param_domain[2] - burgers.param_domain[1]) .+ burgers.param_domain[1]
@showprogress Threads.@threads for (i,μ) in collect(enumerate(μs_test))
    A, F, B = burgers.finite_diff_model(burgers, μ)
    op_burgers = LnL.Operators(A=A, B=B, A2u=F)

    # Compute the testing data with the reference input
    Uref = ones(burgers.time_dim, 1);  # Reference input/boundary condition for OpInf testing 
    Xref = burgers.integrate_model(
        burgers.tspan, burgers.IC, Uref; linear_matrix=A,
        control_matrix=B, quadratic_matrix=F, system_input=true
    )

    data = Dict(
        "X" => Xref, "U" => Uref,
        "A" => A, "B" => B, "F" => F, "mu" => μ
    )
    mu_str = @sprintf("%1.4f", μ)
    save(joinpath(FILEPATH, "data/testing/0$(i)_mu$(mu_str).jld2"), data)
end

#===============================#
## Save the options for system
#===============================#
save(joinpath(FILEPATH, "data/setup.jld2"), "options", options, "burgers", burgers) 
