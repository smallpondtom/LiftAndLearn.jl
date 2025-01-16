"""
2D heat equation: plotting results
"""

#================#
## Load Packages
#================#
using CairoMakie
using FileIO
using JLD2
using LinearAlgebra
using PolynomialModelReductionDataset: Heat2DModel
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"Two-Pass_Streaming-OpInf/heat2d") : joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/heat2d")

#===================#
## Load the options
#===================#
setup_file = joinpath(FILEPATH, "data/setup.jld2")
setup = load(setup_file)
options = setup["options"]
heat2d = setup["heat2d"]
basis_file = joinpath(FILEPATH, "data/streaming/basis.jld2")
basis_data = load(basis_file)
Vrmax = basis_data["batch"].Vr
rmax = size(Vrmax, 2)

#============================================================#
## Plot the error between the batch and iSVD singular values
#============================================================#
basis_file = joinpath(FILEPATH, "data/streaming/basis.jld2")
bases = load(basis_file)
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 600))
    ax = Axis(
        fig[1, 1], xlabel=L"singular value index, $i$", ylabel="absolute error of singular values",
        yscale=log10, xticks=1:rmax, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        # title="Relative error between batch and \n incremental singular values",
    )
    lines = []
    labels = []
    marker_styles = [:diamond, :cross, :circle, :rect]
    line_styles = [:solid, :solid, :solid, :solid]
    i = 1
    for Algo in ["Baker", "Brand", "Sketchy", "MergingSketchy"]
        algo = lowercase(Algo)
        basis = bases[algo]
        Σr = basis.iΣr
        Σr_batch = bases["batch"].Σr
        error = abs.(Σr - Σr_batch)
        l = scatterlines!(
            ax, 1:rmax, error, 
            marker=marker_styles[i], markersize=(35-(i-1)*2),
            linestyle=line_styles[i], linewidth=7,
        )
        i += 1
        push!(lines, l)
        push!(labels, Algo)
    end
    axislegend(ax, 
        lines, labels,
        position=:lb,
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
    ax = Axis(
        fig[1, 1], xlabel=L"reduced dimension, $r$", ylabel="mean projection error",
        xticks=1:rmax, yscale=log10,
        titlesize=30, xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        # title="Projection error of the POD basis",
    )
    lines = []
    algos = ["batch", "baker", "brand", "sketchy", "mergingsketchy"]
    marker_styles = [:rect, :diamond, :cross, :circle, :rect]
    line_styles = [:solid, :dot, :dash, :dashdot, :dashdotdot]
    colors = vcat(:black, Makie.wong_colors()[1:4])
    i = 1
    for algo in algos
        l = scatterlines!(
            ax, 1:rmax, proj_error[algo],
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

#==========================================#
## Plot the training relative state errors
#==========================================#
training_errors = load(joinpath(FILEPATH, "data/training_errors.jld2"))
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 600))
    ax = Axis(
        fig[1, 1], xlabel=L"reduced dimension, $r$", ylabel="mean relative state error",
        xticks=1:rmax, yscale=log10, limits=(nothing, nothing, 1e-8, 1e1),
        titlesize=30, xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        # limits=(nothing, nothing, 1e-5, 2e-1),
        # title="Relative state error of the training data",
    )
    lines = []
    labels = ["pod", "opinf", "tropinf", "stream_rls", "stream_iqrrls", "stream_qrrls"]
    marker_styles = [:diamond, :cross, :circle, :rect, :star5, :hexagon]
    line_styles = [:dot, :dash, :solid, :dashdot, :dashdotdot, :dash]
    i = 1
    for method in labels
        l = scatterlines!(
            ax, 1:rmax, vec(training_errors[method]),
            marker=marker_styles[i], markersize=(35-(i-1)*2),
            linestyle=line_styles[i], linewidth=7,
        )
        push!(lines, l)
        i += 1
    end
    axislegend(
        ax, lines, labels,
        position=:lb,
        # orientation=:horizontal, 
        # halign=:center, 
        # tellwidth=false, 
        # tellheight=true,
        labelsize=30
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/training_rse_errors.pdf"), fig)
end

#============================================#
## Plot the relative state errors per stream
#============================================#
stream_res = load(joinpath(FILEPATH, "data/streaming/stream_results.jld2"))["stream_res"]
with_theme(theme_latexfonts()) do
    num_of_streams = size(stream_res[:rls].rse, 2)
    line_colors = Makie.resample_cmap(:viridis, rmax)
    fig = Figure(size=(1800,700))
    xtick_vals = 0:(num_of_streams ÷ 4):num_of_streams
    # Standard RLS
    ax1 = Axis(fig[1, 1], 
        xlabel=L"$k$-th stream", 
        ylabel="mean relative state error", 
        title="RLS", 
        yscale=log10, xticks=xtick_vals, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        limits=(nothing, nothing, 1e-5, 1e1),
    )
    for (j,ri) in enumerate(1:rmax)  # over all reduced dimensions
        scatterlines!(ax1, 1:num_of_streams, stream_res[:rls].rse[ri,:], color=line_colors[j])
    end
    # iQRRLS
    ax2 = Axis(fig[1, 2], 
        xlabel=L"$k$-th stream", 
        title="iQRRLS", 
        yscale=log10, xticks=xtick_vals, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        limits=(nothing, nothing, 1e-5, 1e1),
    )
    for (j,ri) in enumerate(1:rmax)  # over all reduced dimensions
        scatterlines!(ax2, 1:num_of_streams, stream_res[:iqrrls].rse[ri,:], color=line_colors[j])
    end
    # QRRLS
    ax3 = Axis(fig[1, 3], 
        xlabel=L"$k$-th stream", 
        title="QRRLS", 
        yscale=log10, xticks=xtick_vals, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        limits=(nothing, nothing, 1e-5, 1e1),
    )
    lines = []
    labels = []
    for (j,ri) in enumerate(1:rmax)  # over all reduced dimensions
        l = scatterlines!(ax3, 1:num_of_streams, stream_res[:qrrls].rse[ri,:], color=line_colors[j])
        push!(lines, l)
        push!(labels, "r = $ri")
    end
    Legend(fig[1,4], lines, labels, labelsize=30)
    display(fig)
    save(joinpath(FILEPATH, "plots/rel_state_err_per_stream.pdf"), fig)
end

#================================================#
## Plot the relative streaming errors per stream
#================================================#
with_theme(theme_latexfonts()) do
    num_of_streams = size(stream_res[:rls].stream_err, 2)
    line_colors = Makie.resample_cmap(:viridis, rmax)
    fig = Figure(size=(1800,700))
    xtick_vals = 0:(num_of_streams ÷ 4):num_of_streams
    # Standard RLS
    ax1 = Axis(fig[1, 1], 
        xlabel=L"$k$-th stream", 
        ylabel="meann relative streaming errors", 
        title="RLS", 
        yscale=log10, 
        xticks=xtick_vals, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    for (j,ri) in enumerate(1:rmax)  # over all reduced dimensions
        scatterlines!(ax1, 1:num_of_streams, stream_res[:rls].stream_err[ri,:], color=line_colors[j])
    end
    # iQRRLS
    ax2 = Axis(fig[1, 2], 
        xlabel=L"$k$-th stream", 
        title="iQRRLS", 
        yscale=log10, 
        xticks=xtick_vals, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    for (j,ri) in enumerate(1:rmax)  # over all reduced dimensions
        scatterlines!(ax2, 1:num_of_streams, stream_res[:iqrrls].stream_err[ri,:], color=line_colors[j])
    end
    # QRRLS
    ax3 = Axis(fig[1, 3], 
        xlabel=L"$k$-th stream", 
        title="QRRLS", 
        yscale=log10, 
        xticks=xtick_vals, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    lines = []
    labels = []
    for (j,ri) in enumerate(1:rmax)  # over all reduced dimensions
        l = scatterlines!(ax3, 1:num_of_streams, stream_res[:qrrls].stream_err[ri,:], color=line_colors[j])
        push!(lines, l)
        push!(labels, "r = $ri")
    end
    Legend(fig[1,4], lines, labels, labelsize=30)
    display(fig)
    save(joinpath(FILEPATH, "plots/rel_stream_err_per_stream.pdf"), fig)
end

#=====================================================#
## Plot the conversion factor and a posteriori errors
#=====================================================#
with_theme(theme_latexfonts()) do 
    num_of_streams = length(stream_res[:rls].post_err)
    fig = Figure(size=(1000,700))
    xtick_vals = 0:(num_of_streams ÷ 4):num_of_streams
    algos = ["RLS", "iQRRLS", "QRRLS"]
    algos_lower = (Symbol ∘ lowercase).(algos)
    # A posteriori error norm
    ax1 = Axis(fig[1, 1],
        xlabel=L"$k$-th stream", 
        ylabel=L"mean a posteriori error norm, $\Vert\mathbf{\xi}_k^+\Vert_2$",
        xticks=xtick_vals, 
        xlabelsize=30, ylabelsize=35, xticklabelsize=25, yticklabelsize=25,
        titlesize=30, yscale=log10
    )
    marker_styles = Dict(:rls => :rect, :iqrrls => :star5, :qrrls => :hexagon)
    line_styles = Dict(:rls => :dot, :iqrrls => :dash, :qrrls => :dashdot)
    for (i, algo) in enumerate(algos_lower)
        scatterlines!(
            ax1, 1:num_of_streams, stream_res[algo].post_err,
            linestyle=line_styles[algo], linewidth=7-2*(i-1), marker=marker_styles[algo], markersize=25-8*(i-1),
        )
    end
    # Conversion factor
    ax2 = Axis(fig[1, 2],
        xlabel=L"$k$-th stream", 
        ylabel=L"mean conversion factor, $c_k$",
        xlabelsize=30, ylabelsize=35, xticklabelsize=25, yticklabelsize=25,
    )
    lines = []
    for (i, algo) in enumerate(algos_lower)
        l = scatterlines!(
            ax2, 1:num_of_streams, stream_res[algo].conv_factor,
            linestyle=line_styles[algo], linewidth=7-2*(i-1), marker=marker_styles[algo], markersize=25-8*(i-1),
        )
        push!(lines, l)
    end
    axislegend(
        ax2, lines, algos, 
        position=:rb,
        labelsize=30
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/posterror_convfact.pdf"), fig)
end

#=============================================#
## Plot the relative state errors for testing
#=============================================#
testing_errors = load(joinpath(FILEPATH, "data/testing_errors.jld2"))
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 600))
    ax = Axis(
        fig[1, 1], xlabel=L"reduced dimension, $r$", ylabel="mean relative state error",
        xticks=1:rmax, yscale=log10,
        titlesize=30, xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        # limits=(nothing, nothing, 1e-6, 1e0),
        # title="Relative state error of the training data",
    )
    lines = []
    labels = ["pod", "opinf", "tropinf", "stream_rls", "stream_iqrrls", "stream_qrrls"]
    marker_styles = [:diamond, :cross, :circle, :rect, :star5, :hexagon]
    line_styles = [:dot, :dash, :solid, :dashdot, :dashdotdot, :dash]
    i = 1
    for method in labels
        l = scatterlines!(
            ax, 1:rmax, vec(testing_errors[method]),
            marker=marker_styles[i], markersize=(35-(i-1)*2),
            linestyle=line_styles[i], linewidth=7,
        )
        push!(lines, l)
        i += 1
    end
    axislegend(
        ax, lines, labels,
        position=:lb,
        # orientation=:horizontal, 
        # halign=:center, 
        # tellwidth=false, 
        # tellheight=true,
        labelsize=30
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/testing_rse_errors.pdf"), fig)
end