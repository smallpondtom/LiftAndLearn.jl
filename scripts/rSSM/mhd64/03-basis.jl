"""
MHD64 example: Computing the POD basis
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
           joinpath(pwd(), "rSSM/mhd64") : 
           joinpath(pwd(), "scripts/rSSM/mhd64")
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

#====================================================#
## Compute the POD bases using streaming algorithms ##
#====================================================#
import SingularSubspaceMerging as SSM
rmax = 50

# Setup
prow = (nxyz*n_fields) ÷ 256
pcol = 2
Nrow, Ncol = (nxyz*n_fields) ÷ prow, n ÷ pcol 
energy = 1e-3
fix_rank = false

# Function to determine rank based on energy threshold
Σσ = (s_all) -> begin 
    s_total = sum(s_all.^2)
    s_cumsum = cumsum(s_all.^2)
    tmp = s_cumsum / s_total
    rtmp = searchsortedfirst(tmp, 1 - energy)
    return rtmp
end

## Memory-efficient streaming merge
# Only store current column of merged results
ζ_total = 0.0  # Accumulate tail singular values
τ_total = 0.0  # Accumulate merge errors
block_ranks = Int[]
merge_ranks = Int[]

# Process column by column to avoid storing all blocks
column_results = Vector{LinearAlgebra.SVD}(undef, pcol)

for j in 1:pcol
    @info "Processing column $j"
    
    # Process first block in column
    Xi = @view X[1:Nrow, (j-1)*Ncol+1:j*Ncol]
    Vi, Si, Wi = svd(Xi; full=false)
    ri = fix_rank ? rmax : Σσ(Si)
    ri = min(ri, length(Si))
    push!(block_ranks, ri)
    
    ζ_total += (ri < length(Si) ? Si[ri+1]^2 : 0.0)
    
    # Start with truncated SVD
    current_svd = LinearAlgebra.SVD(Vi[:, 1:ri], Si[1:ri], Wi[:, 1:ri]')
    
    # Sequentially merge rows within this column
    for i in 2:prow
        Xi = @view X[(i-1)*Nrow+1:i*Nrow, (j-1)*Ncol+1:j*Ncol]
        Vi, Si, Wi = svd(Xi; full=false)
        ri = fix_rank ? rmax : Σσ(Si)
        ri = min(ri, length(Si))
        
        ζ_total += (ri < length(Si) ? Si[ri+1]^2 : 0.0)
        
        # Merge with current result
        merge_rank = fix_rank ? rmax : energy
        Vprev, Sprev, Wprev = current_svd
        
        current_svd, τ_, rsel = SSM.right_ssm(
            merge_rank, Vprev, view(Vi, :, 1:ri), 
            Sprev, view(Si, 1:ri), 
            Wprev, view(Wi, :, 1:ri), 
            left_singular_vectors=true, error=true
        )
        push!(merge_ranks, rsel)
        
        τ_total += τ_^2
        @info "  Merged row block $i with rank $rsel"
    end
    
    column_results[j] = current_svd
end

## Merge columns
@info "Merging columns"
F = column_results[1]
for j in 2:pcol
    merge_rank = fix_rank ? rmax : energy
    Vjm1, Sjm1, Wjm1 = F
    Vj, Sj, Wj = column_results[j]
    
    F, τ_, rsel = SSM.left_ssm(
        merge_rank, Vjm1, Vj, Sjm1, Sj, Wjm1, Wj,
        right_singular_vectors=true, error=true
    )
    if rsel < rmax
        F, τ_, rsel = SSM.left_ssm(
            rmax, Vjm1, Vj, Sjm1, Sj, Wjm1, Wj,
            right_singular_vectors=true, error=true
        )
    end
    
    τ_total += τ_^2
    @info "  Merged column $j with rank $rsel"
end

## Final merged SVD
Vrhat = F.U[:, 1:rmax]
Srhat = F.S[1:rmax]
Wtrhat = F.Vt[1:rmax, :]
Wrhat = Matrix(F.V[:, 1:rmax])

# Compute final truncation error 
ϕ = length(F.S) > rmax ? F.S[rmax+1] : 0.0
# Compute error bound
ssm_bnd = sqrt(ϕ^2 + τ_total + ζ_total)

## Errors 
errors = Dict(
    "true" => Float64[],
    "bounds" => Float64[],
)
push!(errors["true"], opnorm(X - (Vrhat .* Srhat') * Wtrhat))
push!(errors["bounds"], ssm_bnd)

## Save SSM basis
save(
    joinpath(FILEPATH, "data/bases/basis_ssm.jld2"), 
    "Vrhat", Vrhat, "Srhat", Srhat, "Wrhat", Wrhat
)
save(
    joinpath(FILEPATH, "data/bases/ssm_ranks.jld2"), 
    "block_ranks", block_ranks, "merge_ranks", merge_ranks
)


#========================================#
## Compute the POD bases with batch SVD ##
#========================================#
@info "Computing POD basis for all fields combined"
# target_r = load(joinpath(FILEPATH, "data/target_ranks.jld2"))["target_ranks"]
target_r = 80
extra_ranks = 0
F = svd(X)
V = F.U[:, 1:min(sum(values(target_r))+extra_ranks*n_fields, n)]
Σ = F.S[1:min(sum(values(target_r))+extra_ranks*n_fields, n)]
W = F.V[:, 1:min(sum(values(target_r))+extra_ranks*n_fields, n)]
F = nothing  # Free memory
GC.gc()
println("POD basis of size $(size(V))")

## Save basis 
basis_file = joinpath(FILEPATH, "data/bases/basis.jld2")
save(basis_file, "Vr", V, "Σr", Σ, "Wr", W)

## Errors of standard SVD  
@views push!(errors["true"], 
             opnorm(X - (V[:,1:rmax] .* Σ[1:rmax]') * W[:,1:rmax]'))
push!(errors["bounds"], Σ[rmax+1])

save(
    joinpath(FILEPATH, "data/bases/basis_errors.jld2"), 
    "errors", errors
)

# Print out errors and bounds 
println("SSM true error: $(errors["true"][1])")
println("SSM error bound: $(errors["bounds"][1])")
println("Standard SVD true error: $(errors["true"][2])")
println("Standard SVD error bound: $(errors["bounds"][2])")

#=============================#
## Compute projection errors ##
#=============================#
rspan = 5:5:rmax
# Preallocate relative projection errors
rpe = Dict(
    "batch" => Dict(
        fld => zeros(length(rspan)) for fld in vcat(ds.fields, "all")
    ),
    "ssm" => Dict(
        fld => zeros(length(rspan)) for fld in vcat(ds.fields, "all")
    ),
)

for (ri, r) in enumerate(rspan)
    @info "Computing projection errors for r = $r"
    # Data projected onto orthogonal complement of the basis
    X_perp_batch = X - V[:,1:r] * (V[:,1:r]' * X)
    X_perp_ssm = X - Vrhat[:,1:r] * (Vrhat[:,1:r]' * X)

    for i in eachindex(ds.fields)
        fld = ds.fields[i]
        idx_start = (i-1) * nxyz + 1
        idx_end = i * nxyz

        num_batch = @views norm(X_perp_batch[idx_start:idx_end, :], 2)
        num_ssm = @views norm(X_perp_ssm[idx_start:idx_end, :], 2)
        den = @views norm(X[idx_start:idx_end, :], 2)

        rpe["batch"][fld][ri] = num_batch / den
        rpe["ssm"][fld][ri] = num_ssm / den

        println("Relative projection error for field $(fld):")
        println("  Batch:   $num_batch / $den = $(rpe["batch"][fld][ri])")
        println("    SSM:   $num_ssm / $den = $(rpe["ssm"][fld][ri])")
    end
    Xnorm = norm(X, 2)
    rpe["batch"]["all"][ri] = norm(X_perp_batch) / Xnorm
    rpe["ssm"]["all"][ri] = norm(X_perp_ssm) / Xnorm
    println("Overall relative projection error:")
    println("  Batch:   $(rpe["batch"]["all"][ri])")
    println("    SSM:   $(rpe["ssm"]["all"][ri])")
end

## Save results
save(joinpath(FILEPATH, "data/results/projection_errors.jld2"), 
     "rpe", rpe, "rspan", rspan)
