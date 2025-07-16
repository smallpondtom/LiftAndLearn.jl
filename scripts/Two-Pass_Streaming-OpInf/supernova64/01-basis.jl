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
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(), "Two-Pass_Streaming-OpInf/supernova64") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/supernova64")
DATAPATH = "../../../../DATA/THE_WELL/supernova_explosion_64/train"
train_files = readdir(DATAPATH, join=true)
fn = train_files[1]
X = load(joinpath(FILEPATH, "data/preprocessed_data.jld2"))["X"]

# Include the data sourcing module for data access
include(joinpath(FILEPATH, "datasource.jl"))

# Load data source 
ds = DataSource(fn)
nx, ny, nz, n_fields, n_time, n_traj = ds.dims
nxyz = nx * ny * nz

#=========================#
## Compute the POD bases ##
#=========================#
# Load the target ranks 
target_r = load(joinpath(FILEPATH, "data/target_ranks.jld2"))["target_ranks"]
V_all = Dict(
    fld => svd(X[fld]).U[:, 1:target_r[fld]]
    for fld in ds.fields
)
V = BlockDiagonal([V_all["p"], V_all["z"], V_all["u"], V_all["v"], V_all["w"]])
println("POD basis of size $(size(V))")

## Save basis 
basis_file = joinpath(FILEPATH, "data/streaming/basis.jld2")
save(basis_file, "V", V)

#=============================#
## Compute projection errors ##
#=============================#
# Processed data
for fld in ds.fields
    tmp = norm(X[fld] - V_all[fld] * (V_all[fld]' * X[fld])) / norm(X[fld])
    println("Projection error for each field: $(tmp)")
end
X_all = reduce(vcat, [X[fld] for fld in ds.fields])
tmp = norm(X_all - V * (V' * X_all)) / norm(X_all)
println("Overall projection error: $(tmp)")

## Original data (unscaled and uncentered)
X_orig = load(joinpath(FILEPATH, "data/original_data.jld2"))["X"]
shift  = load(joinpath(FILEPATH, "data/minmax.jld2"))["shift"]
scale  = load(joinpath(FILEPATH, "data/minmax.jld2"))["scale"]
mean   = load(joinpath(FILEPATH, "data/mean.jld2"))["mean"]

unscale = (X, scale, shift) -> (scale .* X) .+ shift
uncenter = (X, Xbar) -> X .+ Xbar

X_proj_all = Array[]
for (i, field) in enumerate(ds.fields)
    V_field = V_all[field]

    # Project the processed data
    X_proj = V_field * (V_field' * X[field])

    # Unscale and uncenter the projected data
    X_proj = unscale(X_proj, scale[field], shift[field])
    X_proj = uncenter(X_proj, mean[field])

    # Compute the relative projection error
    X_orig_field = Float64.(X_orig[field])
    rpe = norm(X_orig_field - X_proj) / norm(X_orig_field)
    println("Relative projection error for $field: $rpe")
    push!(X_proj_all, X_proj)
end
X_proj_all = reduce(vcat, X_proj_all)
X_orig_all = Float64.(X_orig["all"])
rpe = norm(X_orig_all - X_proj_all) / norm(X_orig_all)
println("Overall relative projection error: $rpe")
