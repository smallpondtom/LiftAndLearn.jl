"""
3D Channel flow: Compute the energy spectrum
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using IncrementalSVD
using LinearAlgebra
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
DATAPATH = "../../../../../DATA/NREL/3D_CHANNEL"
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"One-Pass_Streaming-OpInf/3d_channel") : 
           joinpath(pwd(), "scripts/One-Pass_Streaming-OpInf/3d_channel")
fn = "channel_5200_data_0_10000.h5"
datafile = joinpath(DATAPATH, fn)

#==========================================#
## Load struct to read data in HDF5 format 
#==========================================#
include(joinpath(FILEPATH, "datasource.jl"))

#=============================#
## Load the training dataset
#=============================#
ds = ChannelDataSource(datafile, ["z", "y", "x", "fields", "times"])
nz, ny, nx, n_fields, n = ds.dims
dim_per_field = nz * ny * nx
field_names = ["u", "v", "w", "p"]

#===============================================#
## Plot the singular value computed using iSVD ##
#===============================================#
# Load the singular values
rmax = 500
basis_file = joinpath(FILEPATH, "data/results/onepass_stream.jld2")
iSVD = load(basis_file)["stream"]
iΣrmax = iSVD.Σ[1:rmax]
iSVD = nothing
GC.gc()

## Plot
using CairoMakie
with_theme(theme_latexfonts()) do
    fig = Figure(size=(1000, 500))
    c1, c2 = Makie.wong_colors()[1:2]
    ax = Axis(
        fig[1, 1],
        xlabel=L"singular value index, $i$",
        ylabel=L"singular value, $\sigma_i$",
        yscale=log10,
        xlabelsize=38,
        ylabelsize=38,
        xticklabelsize=30,
        yticklabelsize=30,
        xticks=0:100:rmax,
        limits=(nothing, nothing, 9e2, 2e6)
    )
    lines!(ax, 1:rmax, iΣrmax, linewidth=12, color=c1)
    vlines!(ax, [300], color=c2, linewidth=5, linestyle=:dash, alpha=0.7)
    hlines!(ax, [iΣrmax[300]], color=c2, linewidth=5, linestyle=:dash, alpha=0.7)
    scatter!(ax, [300], [iΣrmax[300]], color=c2, markersize=20)
    text!(
        ax,
        L"$r=300$",
        position=(310, 10^5),
        align = (:left, :bottom),
        fontsize=40
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/svdvals.pdf"), fig, px_per_inch=200)
end