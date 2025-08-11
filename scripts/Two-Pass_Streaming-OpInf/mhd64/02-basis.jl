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
           joinpath(pwd(), "Two-Pass_Streaming-OpInf/mhd64") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/mhd64")
DATAPATH = "../../../../DATA/THE_WELL/mhd64"
train_files = readdir(DATAPATH, join=true)
fn = train_files[1]
X = load(joinpath(FILEPATH, "data/preprocessed_data.jld2"))["X"]["all"]

# Include the data sourcing module for data access
include(joinpath(FILEPATH, "datasource.jl"))

# Load data source 
ds = DataSource(fn)
nx, ny, nz, n_fields, n_time, n_traj = ds.dims
nxyz = nx * ny * nz
n = n_time * n_traj

#========================================#
## Compute the POD bases with batch SVD ##
#========================================#
@info "Computing POD basis for all fields combined"
target_r = load(joinpath(FILEPATH, "data/target_ranks.jld2"))["target_ranks"]
extra_ranks = 0
V = svd(X).U[:, 1:min(sum(values(target_r))+extra_ranks*n_fields, n)]
println("POD basis of size $(size(V))")

## Save basis 
basis_file = joinpath(FILEPATH, "data/bases/basis.jld2")
save(basis_file, "V", V)

## Load the basis instead
basis_file = joinpath(FILEPATH, "data/bases/basis.jld2")
V = load(basis_file)["V"]


#====================================================#
## Compute the POD bases using streaming algorithms ##
#====================================================#
using IncrementalSVD
rmax = 50

## (Dry) Run it once due to JUlia's JIT compilation
# Baker
baker = iSVD(x1=X[:,1], algo=:baker, max_rank=rmax) 
full_increment!(baker, X[:,2:3], verbose=true, runtime=true)
## Brand
brand = iSVD(x1=X[:,1], algo=:brand1, reorth_method=:qr, max_rank=rmax)
full_increment!(brand, X[:,2:3], verbose=true, tol=1e-10, runtime=true)
## SketchySVD
sketchy = iSVD(algo=:sketchy; m=nxyz*n_fields, n=n, r=rmax, ReduxMap=:Sparse)
full_increment!(sketchy, X[:,2:3], verbose=true, runtime=true, dump_all=true)

## Main run 
# Baker
baker = iSVD(x1=X[:,1], algo=:baker, max_rank=rmax)
full_increment!(baker, X[:,2:end], verbose=true, runtime=false)

## Brand
brand = iSVD(x1=X[:,1], algo=:brand1, reorth_method=:qr, max_rank=rmax)
full_increment!(brand, X[:,2:end], verbose=true, tol=1e-10)

## SketchySVD
sketchy = iSVD(algo=:sketchy; m=nxyz*n_fields, n=n, r=rmax, ReduxMap=:Sparse)
full_increment!(sketchy, X, verbose=true, runtime=true, dump_all=true)

## Save the bases
save(joinpath(FILEPATH, "data/bases/baker_basis.jld2"), "baker", baker)
save(joinpath(FILEPATH, "data/bases/brand_basis.jld2"), "brand", brand)
save(joinpath(FILEPATH, "data/bases/sketchy_basis.jld2"), "sketchy", sketchy)

#=============================#
## Compute projection errors ##
#=============================#
# Data projected onto orthogonal complement of the basis
X_perp_batch = X - V[:,1:rmax] * (V[:,1:rmax]' * X)
X_perp_baker = X - baker.Q * (baker.Q' * X)
X_perp_brand = X - brand.Q * (brand.Q' * X)
X_perp_sketchy = X - sketchy.Q * (sketchy.Q' * X)

# Preallocate relative projection errors
rpe = Dict(
    "batch" => Dict(
        fld => 0.0 for fld in [ds.fields, "all"]
    ),
    "baker" => Dict(
        fld => 0.0 for fld in [ds.fields, "all"]
    ),
    "brand" => Dict(
        fld => 0.0 for fld in [ds.fields, "all"]
    ),
    "sketchy" => Dict(
        fld => 0.0 for fld in [ds.fields, "all"]
    )
)

for i in eachindex(ds.fields)
    fld = ds.fields[i]
    idx_start = (i-1) * nxyz + 1
    idx_end = i * nxyz

    num_batch = @views norm(X_perp_batch[idx_start:idx_end, :], 2)
    num_baker = @views norm(X_perp_baker[idx_start:idx_end, :], 2)
    num_brand = @views norm(X_perp_brand[idx_start:idx_end, :], 2)
    num_sketchy = @views norm(X_perp_sketchy[idx_start:idx_end, :], 2)
    den = @views norm(X[idx_start:idx_end, :], 2)

    rpe["batch"][fld] = num_batch / den
    rpe["baker"][fld] = num_baker / den
    rpe["brand"][fld] = num_brand / den
    rpe["sketchy"][fld] = num_sketchy / den

    println("Relative projection error for field $(fld):")
    println("  Batch:   $num_batch / $den = $(rpe["batch"][fld])")
    println("  Baker:   $num_baker / $den = $(rpe["baker"][fld])")
    println("  Brand:   $num_brand / $den = $(rpe["brand"][fld])")
    println("  Sketchy: $num_sketchy / $den = $(rpe["sketchy"][fld])")
end
Xnorm = norm(X, 2)
rpe["batch"]["all"] = norm(X_perp_batch) / Xnorm
rpe["baker"]["all"] = norm(X_perp_baker) / Xnorm
rpe["brand"]["all"] = norm(X_perp_brand) / Xnorm
rpe["sketchy"]["all"] = norm(X_perp_sketchy) / Xnorm
println("Overall relative projection error:")
println("  Batch:   $(rpe["batch"]["all"])")
println("  Baker:   $(rpe["baker"]["all"])")
println("  Brand:   $(rpe["brand"]["all"])")
println("  Sketchy: $(rpe["sketchy"]["all"])")

## Save results
save(joinpath(FILEPATH, "data/projection_errors.jld2"), rpe)