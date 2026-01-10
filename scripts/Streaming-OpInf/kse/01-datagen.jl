"""
Kuramoto-Sivashinsky equation: generate data
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using LinearAlgebra
using ProgressMeter
using PolynomialModelReductionDataset: KuramotoSivashinskyModel
using Printf
using Random
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
FILEPATH = @__DIR__

#============#
## KSE setup
#============#
Ω = (0.0, 22.0); L = Ω[2] - Ω[1]
Nx = 2^9; dt = 1e-3; 
kse = KuramotoSivashinskyModel(
    spatial_domain=Ω, time_domain=(0.0, 300.0), Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt,
    diffusion_coeffs=1.0, BC=:periodic, conservation_type=:NC
)

# Downsampling rate
DS = 100

#=====================#
## Initial conditions
#=====================#
# Parameters of the initial condition
ic_a = [0.2, 0.7, 1.2]
ic_b = [0.1, 0.5, 0.9]
num_ic_params = Int(length(ic_a) * length(ic_b))

# Parameterized function for the initial condition
u0 = (a,b) -> a * cos.((2*π*kse.xspan)/L) .+ b * cos.((4*π*kse.xspan)/L)  # initial condition

# Some options for operator inference
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=[1,2],
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=dt,
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
    use_backslash=true,
)

#=========================#
## Generate training data
#=========================#
@info "Generating training data..."
A, F = kse.finite_diff_model(kse, kse.diffusion_coeffs[1])  # system matrices

# Generate the data for all combinations of the initial condition parameters
ic_combos = collect(Iterators.product(ic_a, ic_b))

# Total number of streams for training
num_of_streams = zeros(length(ic_combos))
    
@showprogress for (j, ic) in collect(enumerate(ic_combos))
    a, b = ic
    IC = u0(a, b)
    states = kse.integrate_model(
        kse.tspan, IC; linear_matrix=A, quadratic_matrix=F, 
        system_input=false, const_stepsize=true
    )

    Xref = states
    tmp = states[:, 2:end]
    X = tmp[:, 1:DS:end]  # downsample data
    tmp = (states[:, 2:end] - states[:, 1:end-1]) / kse.Δt
    Xdot = tmp[:, 1:DS:end]  # downsample data

    num_of_streams[j] = size(X,2)  # increment number of streams

    data = Dict(
        "X" => X, "Xdot" => Xdot, 
        "IC" => IC, "a" => a, "b" => b,
        "Xref" => Xref,
    )
    a_str = @sprintf("%1.4f", a)
    b_str = @sprintf("%1.4f", b)

    save(joinpath(FILEPATH, "data/training/0$(j)_a$(a_str)_b$(b_str).jld2"), data)
end
num_of_streams = sum(num_of_streams)
@info "Done."

#============================#
## Generate the testing data
#============================#
@info "Generating testing data..."
# Number of random test inputs
num_test_ic = 50 # <-------- CHANGE THIS TO DESIRED NUMBER OF TEST INPUTS

# Random number generator for reproducibility
seed = 1234
randn_gen = Random.MersenneTwister(seed)

# Generate random initial condition parameters
ic_a_rand_in = (maximum(ic_a) - minimum(ic_a)) .* rand(num_test_ic) .+ minimum(ic_a)
ic_b_rand_in = (maximum(ic_b) - minimum(ic_b)) .* rand(num_test_ic) .+ minimum(ic_b)

@showprogress for j in 1:num_test_ic  
    # generate test 1 data
    a = ic_a_rand_in[j]
    b = ic_b_rand_in[j]
    IC = u0(a,b)
    X = kse.integrate_model(
        kse.tspan, IC; linear_matrix=A, quadratic_matrix=F, 
        system_input=false, const_stepsize=true
    )

    data = Dict(
        "X" => X, "IC" => IC, "a" => a, "b" => b,
    )
    a_str = @sprintf("%1.4f", a)
    b_str = @sprintf("%1.4f", b)

    save(joinpath(FILEPATH, "data/testing/0$(j)_a$(a_str)_b$(b_str).jld2"), data)
end
@info "Done."

#===============================#
## Save the options for system
#===============================#
save(
    joinpath(FILEPATH, "data/setup.jld2"), "options", options, 
    "kse", kse, "num_of_streams", num_of_streams,
    "FOM", Dict("A" => A, "F" => F), "DS", DS,
    "rrange" => 9:3:24
)
