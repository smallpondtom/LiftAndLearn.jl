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

#=========================#
## Compute the POD bases ##
#=========================#
# Load the target ranks 
target_r = load(joinpath(FILEPATH, "data/target_ranks.jld2"))["target_ranks"]
V_p = svd(X["p"]).U[:, 1:target_r["pressure"]]
V_z = svd(X["z"]).U[:, 1:target_r["specific volume"]]
V_u = svd(X["u"]).U[:, 1:target_r["u-velocity"]]
V_v = svd(X["v"]).U[:, 1:target_r["v-velocity"]]
V_w = svd(X["w"]).U[:, 1:target_r["w-velocity"]]
V = BlockDiagonal([V_p, V_z, V_u, V_v, V_w])
println("POD basis of size $(size(V))")

# Save basis 
basis_file = joinpath(FILEPATH, "data/streaming/basis.jld2")
save(basis_file, "V", V)