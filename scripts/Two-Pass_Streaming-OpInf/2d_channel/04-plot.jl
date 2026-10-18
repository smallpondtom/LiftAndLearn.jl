"""
2D Channel (wall-normal) flow data: plotting results
"""

#================#
## Load Packages
#================#
using CairoMakie
using FileIO
using JLD2
using LinearAlgebra
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"Two-Pass_Streaming-OpInf/2d_channel") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/2d_channel")

#===================#
## Load the options
#===================#
setup_file = joinpath(FILEPATH, "data/setup.jld2")
setup = load(setup_file)
# options = setup["options"]
# channel = setup["2dchannel"]
basis_file = joinpath(FILEPATH, "data/streaming/basis.jld2")
bases = load(basis_file)
rmax = size(bases["batch"].Vr, 2)

#============================================================#
## Plot the error between the batch and iSVD singular values
#============================================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 600))
    indices = vcat(1, 10:10:rmax)
    label_indices = vcat(1, 50:50:rmax)
    ax = Axis(
        fig[1, 1], xlabel=L"singular value index, $i$", ylabel="absolute error of singular values",
        yscale=log10, xticks=label_indices, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        # title="Relative error between batch and \n incremental singular values",
    )
    lines = []
    labels = []
    marker_styles = [:diamond, :cross, :circle, :rect]
    line_styles = [:solid, :solid, :solid, :solid]
    Algorithms = ["Baker", "Brand", "Sketchy"]
    i = 1
    for Algo in Algorithms
        algo = lowercase(Algo)
        basis = bases[algo]
        Σr = basis.iΣr
        Σr_batch = bases["batch"].Σr
        error = abs.(Σr - Σr_batch)
        l = scatterlines!(
            ax, indices, error[indices], 
            marker=marker_styles[i], markersize=(35-(i-1)*2),
            linestyle=line_styles[i], linewidth=7,
        )
        i += 1
        push!(lines, l)
        push!(labels, Algo)
    end
    axislegend(ax, 
        lines, labels,
        position=:lt,
        # orientation=:horizontal, 
        # halign=:center, 
        # tellwidth=false, 
        # tellheight=true,
        labelsize=30
    )
    # Label(fig[0, :], "Relative error between batch and incremental singular values", fontsize=35)
    display(fig)
    save(joinpath(FILEPATH, "plots/absolute_sval_error.pdf"), fig)
end

##

with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 600))
    indices = vcat(1, 10:10:rmax)
    label_indices = vcat(1, 50:50:rmax)
    ax = Axis(
        fig[1, 1], xlabel=L"singular value index, $i$", ylabel="relative error of singular values",
        yscale=log10, xticks=label_indices, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        # title="Relative error between batch and \n incremental singular values",
    )
    lines = []
    labels = []
    marker_styles = [:diamond, :cross, :circle, :rect]
    line_styles = [:solid, :solid, :solid, :solid]
    Algorithms = ["Baker", "Brand", "Sketchy"]
    i = 1
    for Algo in Algorithms
        algo = lowercase(Algo)
        basis = bases[algo]
        Σr = basis.iΣr
        Σr_batch = bases["batch"].Σr
        error = abs.(Σr - Σr_batch) ./ Σr_batch
        l = scatterlines!(
            ax, indices, error[indices], 
            marker=marker_styles[i], markersize=(35-(i-1)*2),
            linestyle=line_styles[i], linewidth=7,
        )
        i += 1
        push!(lines, l)
        push!(labels, Algo)
    end
    axislegend(ax, 
        lines, labels,
        position=:lt,
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

#====================================================#
## Plot the subspace angle errors between the bases ##
#====================================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 600))      
    indices = vcat(1, 10:10:rmax)
    label_indices = vcat(1, 50:50:rmax)
    ax = Axis(
        fig[1, 1], xlabel=L"singular value index, $i$", ylabel=L"subspace angle error, $|\cos(\theta_i)-1|$",
        yscale=log10, xticks=label_indices, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        # title="2D Heat", limits=(nothing, nothing, 1e-18, 1e-1)
    )
    lines = []
    labels = []
    marker_styles = [:diamond, :cross, :circle, :rect]
    line_styles = [:solid, :solid, :solid, :dash]
    Algorithms = ["Baker", "Brand", "Sketchy" ]
    i = 1
    for Algo in Algorithms
        algo = lowercase(Algo)
        if algo == "mergingsketchy"
            for blk in blksizes
                basis = bases[algo][blk]
                angle_errs = abs.(svdvals(basis.Q[:,1:r]' * bases["batch"].U[:,1:r]) .- 1)
                l = scatterlines!(
                    ax, indices, angle_errs[indices], 
                    marker=marker_styles[i], markersize=(35-(i-1)*2),
                    linestyle=line_styles[i], linewidth=7,
                )
                push!(lines, l)
                B = n ÷ blk
                push!(labels, L"MergingSketchy ($B=%$B$)")
            end
        else
            basis = bases[algo]
            angle_errs = abs.(svdvals(basis.iVr[:,1:rmax]' * bases["batch"].Vr[:,1:rmax]) .- 1)
            l = scatterlines!(
                ax, indices, angle_errs[indices], 
                marker=marker_styles[i], markersize=(35-(i-1)*2),
                linestyle=line_styles[i], linewidth=7,
            )
            i += 1
            push!(lines, l)
            push!(labels, Algo)
        end
    end
    # Legend(
    #     fig[2,1], lines, labels,
    #     position=:rb, orientation=:horizontal, labelsize=30,
    #     patchsize=(60,20), nbanks=1, framevisible=false
    # )
    axislegend(ax, 
        lines, labels,
        position=:lt,
        # orientation=:horizontal, 
        # halign=:center, 
        # tellwidth=false, 
        # tellheight=true,
        labelsize=30
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/subspace_angle_error.pdf"), fig)
end

#=======================================================#
## Plot the runtime of the iSVD algorithms over streams
#=======================================================#
basis_runtime = load(joinpath(FILEPATH, "data/streaming/basis_runtime.jld2"))
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 1000))
    algorithms = ["Baker", "Brand", "Sketchy", "MergingSketchy"]
    yticks = -5.0:1.0:0.0
	yticklabels = [L"10^{%$(Int(y))}" for y in yticks]
    ax = Axis(
        fig[1, 1], xlabel="Algorithm", ylabel="runtime per stream (s)",
        xticks=(1:length(algorithms), algorithms),
        yticks=(yticks, yticklabels),
        titlesize=30, xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        # title="Runtime of iSVD algorithms over streams",
    )
    for (i, algo) in enumerate(algorithms)
        algo = lowercase(algo)
        foo = fill(i, length(basis_runtime[algo]))
        boxplot!(ax, foo, log10.(basis_runtime[algo]); whiskerwidth=1.0, width=0.8, mediancolor=:black)
    end
    display(fig)
    save(joinpath(FILEPATH, "plots/basis_runtime.pdf"), fig)
end

#=================================================#
## Plot the total runtime of the iSVD algorithms
#=================================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1050, 800))
    algorithms = ["Batch", "Baker", "Brand", "Sketchy", "MergingSketchy"]
    ax = Axis(
        fig[1, 1], xlabel="Algorithm", ylabel="total runtime (s)",
        xticks = (1:5, algorithms), yscale=log10,
        titlesize=30, xlabelsize=35, ylabelsize=30, xticklabelsize=30, yticklabelsize=25,
        xgridvisible=false, # ygridvisible=false,
        # title="Runtime of iSVD algorithms over streams",
    )
    tbl = (
        cat = collect(1:5),
        height = [
            sum(basis_runtime[lowercase(algo)]) for algo in algorithms
        ],
        grp = collect(1:5),
    )
    barplot!(ax, tbl.cat, tbl.height, bar_labels=:y, label_size=30, label_offset=2, 
             color=vcat(:black, Makie.wong_colors()[tbl.grp][1:end-1]))

    # # inset for excluding sketchy
    # inset_ax = Axis(fig[1, 1],
    #     width=Relative(0.5),
    #     height=Relative(0.5),
    #     halign=0.3,
    #     valign=0.8,
    #     xticks = (1:3, ["Batch", "Baker", "Brand"]),
    #     xgridvisible=false, ygridvisible=false,
    #     xlabelsize=22, ylabelsize=22, xticklabelsize=18, yticklabelsize=18)
    # tbl = (
    #     cat = collect(1:3),
    #     height = [
    #         sum(basis_runtime["batch"]), sum(basis_runtime["baker"]),
    #         sum(basis_runtime["brand"]),
    #     ],
    #     grp = collect(1:3),
    # )
    # barplot!(inset_ax, tbl.cat, tbl.height, color=Makie.wong_colors()[tbl.grp])
    # bracket!(ax, 0.8, 300, 3.2, 300, offset=5, text="Zoom-in", fontsize=20)
    display(fig)
    save(joinpath(FILEPATH, "plots/basis_total_runtime.pdf"), fig)
end

#=============================#
## Plot the projection errors
#=============================#
proj_error = load(joinpath(FILEPATH, "data/projection_errors.jld2"))
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 600))
    indices = vcat(1, 10:10:rmax)
    label_indices = vcat(1, 50:50:rmax)
    ax = Axis(
        fig[1, 1], xlabel=L"reduced dimension, $r$", ylabel="mean projection error",
        xticks=label_indices, yscale=log10,
        titlesize=30, xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        # title="Projection error of the POD basis",
    )
    lines = []
    algos = ["batch", "baker", "brand", "sketchy"]
    marker_styles = [:rect, :diamond, :cross, :circle, :rect]
    line_styles = [:solid, :dot, :dash, :dashdot, :dashdotdot]
    colors = vcat(:black, Makie.wong_colors()[1:4])
    i = 1
    for algo in algos
        l = scatterlines!(
            ax, indices, proj_error[algo][indices],
            marker=marker_styles[i], markersize=(35-(i-1)*2),
            linestyle=line_styles[i], linewidth=7, color=colors[i],
        )
        push!(lines, l)
        i += 1
    end
    axislegend(
        ax, lines, algos,
        position=:rt,
        # orientation=:horizontal, 
        # halign=:center, 
        # tellwidth=false, 
        # tellheight=true,
        labelsize=30
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/projection_errors.pdf"), fig)
end