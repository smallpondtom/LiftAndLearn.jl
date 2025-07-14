#=================#
## Load packages ##
#=================#
using LinearAlgebra
using BlockDiagonals
using FileIO
using JLD2

#=============#
## Load data ##
#=============#
FILEPATH = occursin("scripts", @__DIR__) ? 
           joinpath(@__DIR__, "Two-Pass_Streaming-OpInf/supernova") : 
           joinpath(@__DIR__, "scripts/Two-Pass_Streaming-OpInf/supernova")
X = load(joinpath(FILEPATH, "data/preprocessed_data.jld2"))["X"]
dims = load(joinpath(FILEPATH, "data/preprocessed_data.jld2"))["dimensions"]
shapes = load(joinpath(FILEPATH, "data/preprocessed_data.jld2"))["shape"]
V = load(joinpath(FILEPATH, "data/streaming/basis.jld2"))["V"]

#===================#
## Train Operators ##
#===================#
# Compute reduced data 
Xhat = V' * X

# Compute the derivative data
Xhatdot = 