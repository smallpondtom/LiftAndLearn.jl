#=================#
## Load packages ##
#=================#
using LinearAlgebra
using BlockDiagonals
using FileIO
using JLD2
import LiftAndLearn as LnL

#=============#
## Load data ##
#=============#
FILEPATH = occursin("scripts", @__DIR__) ? 
           joinpath(@__DIR__, "Two-Pass_Streaming-OpInf/supernova") : 
           joinpath(@__DIR__, "scripts/Two-Pass_Streaming-OpInf/supernova")
X = load(joinpath(FILEPATH, "data/preprocessed_data.jld2"))["X"]
tspan = load(joinpath(FILEPATH, "data/preprocessed_data.jld2"))["dimensions"]["t"]
dims = load(joinpath(FILEPATH, "data/preprocessed_data.jld2"))["dimensions"]
shapes = load(joinpath(FILEPATH, "data/preprocessed_data.jld2"))["shape"]
V = load(joinpath(FILEPATH, "data/streaming/basis.jld2"))["V"]

#===============#
## Set Options ##
#===============#
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=[1,2],
        control=0,
        constant=1,
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
    data=LnL.DataStructure(
        Δt=sum(diff(tspan)) / (length(tspan)-1),
        deriv_type="FBCT4"
    ),
)

#===================#
## Train Operators ##
#===================#
op = LnL.opinf(X, V, options)

#===================#
## Check Operators ##
#===================#
include("integrate.jl")
x_rom = rk4_integrate(V' * X[:,1], tspan, op.A, op.A2u, op.K)

# Compute the relative state error 
rse = norm(X - V * x_rom) / norm(X)
println("Relative state error: $rse")