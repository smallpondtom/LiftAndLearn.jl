"""
2D heat equation: generate data
"""

#================#
## Load Packages
#================#
using CairoMakie
using FileIO
using JLD2
using IncrementalSVD
using LinearAlgebra
using ProgressMeter
using LiftAndLearn
const LnL = LiftAndLearn

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"Streaming-OpInf/heat2d") : joinpath(pwd(), "scripts/Streaming-OpInf/heat2d")

#======================================#
## Obtain all the saved training files
#======================================#
training_data_files = readdir(joinpath(FILEPATH, "data/training"), join=true)

# #============================================================#
# ## Generate the POD basis using iSVD using Baker's algorithm
# #============================================================#
# rmax = 12
# Xall = Array[]

# # Initialize the iSVD object with the first dataset
# data = load(training_data_files[1])
# isvd = iSVD(x1=data["X"][:,1], algo=:baker, max_rank=rmax)
# # isvd = iSVD(x1=data["X"][:,1], algo=:brand1, reorth_method=:qr, max_rank=rmax)
# # isvd = iSVD(x1=data["X"][:,1], algo=:sketchy; m=32*40, n=10010, r=rmax, ReduxMap=:Sparse)
# full_increment!(isvd, data["X"][:,2:end], verbose=true, tol=1e-12)
# push!(Xall, data["X"])

# # Increment for the rest of the data
# for (i,data_file) in enumerate(training_data_files[2:end])
#     jldopen(data_file, "r") do data
#         # Load the data
#         X = data["X"]
#         # Compute the POD basis using the incremental SVD
#         full_increment!(isvd, X, verbose=true, tol=1e-12)
#         # Save the data for batch SVD
#         push!(Xall, X)
#     end
# end

# #============================================#
# ## Compute the POD basis using the batch SVD 
# #============================================#
# F = svd(reduce(hcat, Xall))

# #=====================================================================#
# ## Save the POD basis and singular values from the iSVD and batch SVD
# #=====================================================================#
# save(
#     joinpath(FILEPATH, "data/basis.jld2"),
#     "iVr", isvd.Q[:,1:rmax], "iΣr", isvd.Σ[1:rmax], 
#     "Vr", F.U[:,1:rmax], "Σr", F.S[1:rmax], "r", rmax
# )

#=========================================================#
## Generate the POD basis using iSVD using all algorithms
#=========================================================#
rmax = 12
Xall = Array[]

# Execution times 
time_baker = []
time_brand = []
time_sketchy = []

# Initialize the iSVD object with the first dataset
data = load(training_data_files[1])
# baker
baker = iSVD(x1=data["X"][:,1], algo=:baker, max_rank=rmax)
tmp = full_increment!(baker, data["X"][:,2:end], verbose=true, tol=1e-12, runtime=true)
push!(time_baker, tmp)
# brand
brand = iSVD(x1=data["X"][:,1], algo=:brand1, reorth_method=:qr, max_rank=rmax)
tmp = full_increment!(brand, data["X"][:,2:end], verbose=true, tol=1e-12, runtime=true)
push!(time_brand, tmp)
# sketchy
sketchy = iSVD(x1=data["X"][:,1], algo=:sketchy; m=32*40, n=10010, r=rmax, ReduxMap=:Sparse)
_, tmp = full_increment!(sketchy, data["X"][:,2:end], verbose=true, runtime=true)
push!(time_sketchy, tmp)

push!(Xall, data["X"])

# Increment for the rest of the data
for (i,data_file) in enumerate(training_data_files[2:end])
    jldopen(data_file, "r") do data
        # Load the data
        X = data["X"]
        # Compute the POD basis using Baker's algorithm
        tmp = full_increment!(baker, X, verbose=true, tol=1e-12, runtime=true)
        push!(time_baker, tmp)
        # Comput the POD basis using Brand's algorithm
        tmp = full_increment!(brand, X, verbose=true, tol=1e-12, runtime=true)
        push!(time_brand, tmp)
        # Compute the POD basis using SketchySVD
        _, tmp = full_increment!(sketchy, X, verbose=true, runtime=true)
        push!(time_sketchy, tmp)
        # Save the data for batch SVD
        push!(Xall, X)
    end
end

#============================================#
## Compute the POD basis using the batch SVD 
#============================================#
time_batch = @elapsed F = svd(reduce(hcat, Xall))

#=====================================================================#
## Save the POD basis and singular values from the iSVD and batch SVD
#=====================================================================#
bases = Dict(
    "baker" => (iVr=baker.Q[:,1:rmax], iΣr=baker.Σ[1:rmax]),
    "brand" => (iVr=brand.Q[:,1:rmax], iΣr=brand.Σ[1:rmax]),
    "sketchy" => (iVr=sketchy.Q[:,1:rmax], iΣr=sketchy.Σ[1:rmax]),
    "batch" => (Vr=F.U[:,1:rmax], Σr=F.S[1:rmax]),
)
save(joinpath(FILEPATH, "data/streaming/basis.jld2"), bases)

#============================================================#
## Save the runtime of the iSVD algorithms over all streams
#============================================================#
time_baker = reduce(vcat, time_baker)
time_brand = reduce(vcat, time_brand)
time_sketchy = reduce(vcat, time_sketchy)
save(
    joinpath(FILEPATH, "data/streaming/basis_runtime.jld2"),
    "baker", time_baker, "brand", time_brand, "sketchy", time_sketchy, "batch", time_batch,
)

#============================================================#
## Plot the error between the batch and iSVD singular values
#============================================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 600))
    ax = Axis(
        fig[1, 1], xlabel=L"singular value index, $i$", ylabel="relative error of singular values",
        yscale=log10, xticks=1:rmax, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        # title="Relative error between batch and \n incremental singular values",
    )
    lines = []
    labels = []
    marker_styles = [:circle, :diamond, :cross, :rect]
    line_styles = [:solid, :dash, :dot, :dashdot]
    i = 1
    for (algo, basis) in bases
        if algo == "batch"
            continue
        end
        Σr = basis.iΣr
        Σr_batch = bases["batch"].Σr
        error = abs.(Σr - Σr_batch) ./ Σr_batch
        l = scatterlines!(
            ax, 1:rmax, error, 
            marker=marker_styles[i], markersize=(35-(i-1)*2),
            linestyle=line_styles[i], linewidth=7,
        )
        i += 1
        push!(lines, l)
        push!(labels, algo)
    end
    axislegend(ax, 
        lines, labels,
        position=:rb,
        # orientation=:horizontal, 
        # halign=:center, 
        # tellwidth=false, 
        # tellheight=true,
        labelsize=30
    )
    # Label(fig[0, :], "Relative error between batch and incremental singular values", fontsize=35)
    display(fig)
    save(joinpath(FILEPATH, "plots/relative_sval_error.pdf"), fig)
end

#=======================================================#
## Plot the runtime of the iSVD algorithms over streams
#=======================================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(550, 600))
    ax = Axis(
        fig[1, 1], xlabel="Algorithm", ylabel="runtime per stream (s)",
        xticks = (1:3, ["Baker", "Brand", "Sketchy"]), yscale=log10,
        titlesize=30, xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        # title="Runtime of iSVD algorithms over streams",
    )
    # Baker
    foo = fill(1, length(time_baker))
    boxplot!(ax, foo, time_baker; whiskerwidth=1.0, width=0.6, mediancolor=:black)
    # Brand
    foo = fill(2, length(time_brand))
    boxplot!(ax, foo, time_brand; whiskerwidth=1.0, width=0.6, mediancolor=:black)
    # Sketchy
    foo = fill(3, length(time_sketchy))
    boxplot!(ax, foo, time_sketchy; whiskerwidth=1.0, width=0.6, mediancolor=:black)
    display(fig)
    save(joinpath(FILEPATH, "plots/basis_runtime.pdf"), fig)
end