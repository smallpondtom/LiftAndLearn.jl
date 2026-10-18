"""
Supernova 64^3 example: Computing the POD basis
"""

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
n = n_time * n_traj

#=========================#
## Compute the POD bases ##
#=========================#
FIELD_WISE = false
# Load the target ranks 
target_r = load(joinpath(FILEPATH, "data/target_ranks.jld2"))["target_ranks"]
extra_ranks = 0

if FIELD_WISE
    @info "Computing POD basis field-wise"
    V = BlockDiagonal([
        svd(X[fld]).U[:, 1:target_r[fld]+extra_ranks]
        for fld in ds.fields
    ])
else
    @info "Computing POD basis for all fields combined"
    V = svd(X["all"]).U[:, 1:sum(values(target_r))+extra_ranks*length(ds.fields)]
end
println("POD basis of size $(size(V))")

## Save basis 
basis_file = joinpath(FILEPATH, "data/bases/basis.jld2")
save(basis_file, "V", V)

#=============================#
## Compute projection errors ##
#=============================#
# Processed data
X_perp = X["all"] - V * (V' * X["all"])
rpe_processed = Dict(
    fld => 0.0 for fld in [ds.fields, "all"]
)
for i in eachindex(ds.fields)
    fld = ds.fields[i]
    idx_start = (i-1) * nxyz + 1
    idx_end = i * nxyz
    num = norm(X_perp[idx_start:idx_end, :], 2)
    den = norm(X[fld], 2)
    rpe_fld = num / den 
    rpe_processed[ds.fields[i]] = rpe_fld
    println("Projection error for field $(fld): $rpe_fld")
end
tmp = norm(X_perp, 2) / norm(X["all"], 2)
rpe_processed["all"] = tmp
println("Overall projection error: $(tmp)")

## Original data (unscaled and uncentered)
X_orig = load(joinpath(FILEPATH, "data/original_data.jld2"))["X"]
shift  = load(joinpath(FILEPATH, "data/minmax.jld2"))["shift"]
scale  = load(joinpath(FILEPATH, "data/minmax.jld2"))["scale"]
mean   = load(joinpath(FILEPATH, "data/mean.jld2"))["mean"]

unscale = (X, scale, shift) -> (scale .* X) .+ shift
uncenter = (X, Xbar) -> X .+ Xbar

scale_all = reduce(vcat, [scale[fld] for fld in ds.fields])
shift_all = reduce(vcat, [shift[fld] for fld in ds.fields])
mean_all  = reduce(vcat, [mean[fld] for fld in ds.fields])

##

X_proj = V * (V' * X["all"])
X_proj = unscale(X_proj, scale_all, shift_all)
X_proj = uncenter(X_proj, mean_all)
X_perp = X_orig["all"] - X_proj

rpe_orig = Dict(
    fld => 0.0 for fld in [ds.fields, "all"]
)

for i in eachindex(ds.fields)
    fld = ds.fields[i]
    idx_start = (i-1) * nxyz + 1
    idx_end = i * nxyz
    X_perp_field = X_perp[idx_start:idx_end, :] 
    num = norm(X_perp_field, 2)
    den = norm(X_orig[fld], 2)
    rpe_fld = num / den
    rpe_orig[fld] = rpe_fld
    println("Relative projection error for $fld: $rpe_fld")
end
rpe_orig["all"] = norm(X_perp) / norm(X_orig["all"])
println("Overall relative projection error: $(rpe_orig["all"])")

## Save projection errors
rpe_file = joinpath(FILEPATH, "data/results/rpe.jld2")
if !isfile(rpe_file)
    @info "Saving relative projection errors to file"
    save(rpe_file, "rpe", Dict("processed" => rpe_processed, "original" => rpe_orig))
end