"""
Kuramoto-Sivashinsky equation: plotting results
"""

#================#
## Load Packages
#================#
using CairoMakie
using FileIO
using JLD2
using LinearAlgebra
using PolynomialModelReductionDataset: KuramotoSivashinskyModel
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"Two-Pass_Streaming-OpInf/kse") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/kse")

#===================#
## Load the options
#===================#
setup_file = joinpath(FILEPATH, "data/setup.jld2")
setup = load(setup_file)
options = setup["options"]
kse = setup["kse"]
rrange = setup["rrange"]
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
        fig[1, 1], xlabel=L"singular value index, $i$", 
        ylabel="relative error of singular values",
        yscale=log10, xticks=1:2:rmax, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    lines = []
    labels = []
    marker_styles = [:diamond, :cross, :circle, :rect]
    line_styles = [:solid, :solid, :solid, :solid]
    Algorithms = ["Baker", "Brand", "Sketchy" ]
    i = 1
    for Algo in Algorithms
        algo = lowercase(Algo)
        basis = bases[algo]
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
        push!(labels, Algo)
    end
    axislegend(ax, 
        lines, labels,
        position=:lt,
        labelsize=30,
        patchsize=(80,20)
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/relative_sval_error.pdf"), fig)
end

#============================================================#
## Plot the error between the batch and iSVD singular values
#============================================================#
basis_file = joinpath(FILEPATH, "data/streaming/basis.jld2")
bases = load(basis_file)
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 600))
    ax = Axis(
        fig[1, 1], xlabel=L"singular value index, $i$", 
        ylabel="absolute error of singular values",
        yscale=log10, xticks=1:2:rmax, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    lines = []
    labels = []
    marker_styles = [:diamond, :cross, :circle, :rect]
    line_styles = [:solid, :solid, :solid, :solid]
    Algorithms = ["Baker", "Brand", "Sketchy" ]
    i = 1
    for Algo in Algorithms
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
        position=:lt,
        labelsize=30,
        patchsize=(80,20)
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/absolute_sval_error.pdf"), fig)
end


# #====================================================#
# ## Plot the subspace angle errors between the bases ##
# #====================================================#
# with_theme(theme_latexfonts()) do 
#     fig = Figure(size=(800, 600))      
#     ax = Axis(
#         fig[1, 1], xlabel=L"singular value index, $i$", 
#         ylabel=L"subspace angle error, $|\cos(\theta_i)-1|$",
#         yscale=log10, xticks=1:2:rmax, titlesize=30, 
#         xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
#         # title="2D Heat", 
#         limits=(nothing, nothing, 1e-17, 1e+1)
#     )
#     lines = []
#     labels = []
#     marker_styles = [:diamond, :cross, :circle, :rect]
#     line_styles = [:solid, :solid, :solid, :dash]
#     Algorithms = ["Baker", "Brand", "Sketchy" ]
#     i = 1
#     for Algo in Algorithms
#         algo = lowercase(Algo)
#         if algo == "mergingsketchy"
#             for blk in blksizes
#                 basis = bases[algo][blk]
#                 angle_errs = abs.(svdvals(basis.Q[:,1:r]' * bases["batch"].U[:,1:r]) .- 1)
#                 l = scatterlines!(
#                     ax, 1:rmax, angle_errs, 
#                     marker=marker_styles[i], markersize=(35-(i-1)*2),
#                     linestyle=line_styles[i], linewidth=7,
#                 )
#                 push!(lines, l)
#                 B = n ÷ blk
#                 push!(labels, L"MergingSketchy ($B=%$B$)")
#             end
#         else
#             basis = bases[algo]
#             angle_errs = abs.(svdvals(basis.iVr[:,1:rmax]' * bases["batch"].Vr[:,1:rmax]) .- 1)
#             l = scatterlines!(
#                 ax, 1:rmax, angle_errs, 
#                 marker=marker_styles[i], markersize=(35-(i-1)*2),
#                 linestyle=line_styles[i], linewidth=7,
#             )
#             i += 1
#             push!(lines, l)
#             push!(labels, Algo)
#         end
#     end
#     axislegend(ax, 
#         lines, labels,
#         position=:lt,
#         labelsize=30
#     )
#     display(fig)
#     save(joinpath(FILEPATH, "plots/subspace_angle_error.pdf"), fig)
# end

# #=======================================================#
# ## Plot the runtime of the iSVD algorithms over streams
# #=======================================================#
# basis_runtime = load(joinpath(FILEPATH, "data/streaming/basis_runtime.jld2"))
# with_theme(theme_latexfonts()) do 
#     fig = Figure(size=(800, 1000))
#     algorithms = ["Baker", "Brand", "Sketchy", "MergingSketchy"]
#     yticks = -5.0:1.0:0.0
# 	yticklabels = [L"10^{%$(Int(y))}" for y in yticks]
#     ax = Axis(
#         fig[1, 1], xlabel="Algorithm", ylabel="runtime per stream (s)",
#         xticks=(1:length(algorithms), algorithms),
#         yticks=(yticks, yticklabels),
#         titlesize=30, xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
#         # title="Runtime of iSVD algorithms over streams",
#     )
#     for (i, algo) in enumerate(algorithms)
#         algo = lowercase(algo)
#         foo = fill(i, length(basis_runtime[algo]))
#         boxplot!(ax, foo, log10.(basis_runtime[algo]); whiskerwidth=1.0, width=0.8, mediancolor=:black)
#     end
#     display(fig)
#     save(joinpath(FILEPATH, "plots/basis_runtime.pdf"), fig)
# end

# #=================================================#
# ## Plot the total runtime of the iSVD algorithms
# #=================================================#
# with_theme(theme_latexfonts()) do 
#     fig = Figure(size=(1050, 800))
#     algorithms = ["Batch", "Baker", "Brand", "Sketchy", "MergingSketchy"]
#     ax = Axis(
#         fig[1, 1], xlabel="Algorithm", ylabel="total runtime (s)",
#         xticks = (1:5, algorithms), yscale=log10,
#         titlesize=30, xlabelsize=35, ylabelsize=30, xticklabelsize=30, yticklabelsize=25,
#         xgridvisible=false, # ygridvisible=false,
#         # title="Runtime of iSVD algorithms over streams",
#     )
#     tbl = (
#         cat = collect(1:5),
#         height = [
#             sum(basis_runtime[lowercase(algo)]) for algo in algorithms
#         ],
#         grp = collect(1:5),
#     )
#     barplot!(ax, tbl.cat, tbl.height, bar_labels=:y, label_size=30, label_offset=2, 
#              color=vcat(:black, Makie.wong_colors()[tbl.grp][1:end-1]))

#     # # inset for excluding sketchy
#     # inset_ax = Axis(fig[1, 1],
#     #     width=Relative(0.5),
#     #     height=Relative(0.5),
#     #     halign=0.3,
#     #     valign=0.8,
#     #     xticks = (1:3, ["Batch", "Baker", "Brand"]),
#     #     xgridvisible=false, ygridvisible=false,
#     #     xlabelsize=22, ylabelsize=22, xticklabelsize=18, yticklabelsize=18)
#     # tbl = (
#     #     cat = collect(1:3),
#     #     height = [
#     #         sum(basis_runtime["batch"]), sum(basis_runtime["baker"]),
#     #         sum(basis_runtime["brand"]),
#     #     ],
#     #     grp = collect(1:3),
#     # )
#     # barplot!(inset_ax, tbl.cat, tbl.height, color=Makie.wong_colors()[tbl.grp])
#     # bracket!(ax, 0.8, 300, 3.2, 300, offset=5, text="Zoom-in", fontsize=20)
#     display(fig)
#     save(joinpath(FILEPATH, "plots/basis_total_runtime.pdf"), fig)
# end



#====================================================#
## Plot the subspace angle errors between the bases ##
#====================================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 400))      
    ax = Axis(
        fig[1, 1], xlabel=L"reduced dimension, $r$", 
        ylabel=L"subspace angle error$$",
        yscale=log10, xticks=2:2:rmax, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    lines = []
    labels = []
    # marker_styles = [:diamond, :cross, :circle, :rect]
    # line_styles = [:solid, :solid, :solid, :dash]
    # Algorithms = ["Baker", "Brand", "Sketchy" ]
    marker_styles = [:cross, :rect, :rect]
    line_styles = [:solid, :solid, :dashdot]
    Algorithms = ["Baker", "Sketchy" ]
    colors = Makie.wong_colors()[1:4]

    i = 1
    for Algo in Algorithms
        algo = lowercase(Algo)
        basis = bases[algo]
        angle_errs = zeros(rmax)
        for r in 1:rmax
            angle_errs[r] = norm(
                bases["batch"].Vr[:, 1:r] * bases["batch"].Vr[:, 1:r]' - 
                basis.iVr[:, 1:r] * basis.iVr[:, 1:r]', 2
            ) / sqrt(2)
        end
        l = scatterlines!(
            ax, 1:rmax, angle_errs, 
            marker=marker_styles[i], markersize=(35-(i-1)*2),
            linestyle=line_styles[i], linewidth=7,
            markercolor=:transparent, strokewidth=2.5,
            strokecolor=colors[i],
        )
        i += 1
        push!(lines, l)
        push!(labels, Algo)
    end
    axislegend(ax, 
        lines, labels,
        position=:lt,
        labelsize=30,
        patchsize=(100,20)
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/subspace_angle_error.pdf"), fig)
end

#=============================#
## Plot the projection errors
#=============================#
proj_error = load(joinpath(FILEPATH, "data/projection_errors.jld2"))
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 400))
    ax = Axis(
        fig[1, 1], xlabel=L"reduced dimension, $r$", 
        ylabel=L"relative projection error$$",
        xticks=2:2:rmax, yscale=log10,
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25,
    )
    lines = []
    # algos = ["batch", "baker", "brand", "sketchy"]
    # marker_styles = [:rect, :diamond, :cross, :circle, :rect]
    # line_styles = [:solid, :dot, :dash, :dashdot, :dashdotdot]
    Algorithms = ["Batch", "Baker", "Sketchy"]
    marker_styles = [:circle, :cross, :rect, :rect]
    line_styles = [:solid, :dash, :dashdot, :dashdotdot]
    colors = vcat(:black, Makie.wong_colors()[1:4])
    i = 1
    for Algo in Algorithms
        algo = lowercase(Algo)
        l = scatterlines!(
            ax, 1:rmax, proj_error[algo],
            marker=marker_styles[i], markersize=(35-(i-1)*2),
            linestyle=line_styles[i], linewidth=7, color=colors[i],
            markercolor=:transparent, strokewidth=2.5,
            strokecolor=colors[i],
        )
        push!(lines, l)
        i += 1
    end
    axislegend(
        ax, lines, Algorithms,
        position=:rt,
        labelsize=30,
        patchsize=(100,20),
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
        fig[1, 1], xlabel="Algorithm", ylabel="relative state error",
        xticks=1:2:rmax, yscale=log10,
        titlesize=30, xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        # limits=(nothing, nothing, 1e-5, 4e0),
        # title="Relative state error of the training data",
    )
    lines = []
    labels = ["pod", "opinf", "tropinf", "stream_rls", "stream_iqrrls", "stream_qrrls"]
    marker_styles = [:diamond, :cross, :circle, :rect, :star5, :hexagon]
    line_styles = [:dot, :dash, :solid, :dashdot, :dashdotdot, :dash]
    i = 1
    for method in labels
        l = scatterlines!(
            ax, rrange, vec(training_errors[method]),
            marker=marker_styles[i], markersize=(35-(i-1)*2),
            linestyle=line_styles[i], linewidth=7,
        )
        push!(lines, l)
        i += 1
    end
    axislegend(
        ax, lines, labels,
        position=:lb,
        labelsize=30
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/training_rse_errors.pdf"), fig)
end

#================================================#
## Plot the relative streaming errors per stream
#================================================#
stream_res = load(joinpath(FILEPATH, "data/streaming/stream_results.jld2"))["stream_res"]
with_theme(theme_latexfonts()) do
    num_of_streams = size(stream_res[:rls].stream_err, 2)
    line_colors = Makie.resample_cmap(:viridis, length(rrange))
    fig = Figure(size=(1800,500))
    xtick_vals = 0:(num_of_streams ÷ 3):num_of_streams
    ytick_vals = 10.0 .^ (-16:4:2)
    # Standard RLS
    ax1 = Axis(fig[1, 1], 
        xlabel=L"$k$-th stream", 
        ylabel=L"MR-SOE($k,r$)", 
        title="RLS", 
        yscale=log10, 
        xticks=xtick_vals, titlesize=35, 
        xlabelsize=33, ylabelsize=33, xticklabelsize=30, yticklabelsize=30,
        limits=(nothing, nothing, 1e-17, 1),
        yticks=(ytick_vals, [L"10^{%$(Int(log10(y)))}" for y in ytick_vals]),
    )
    for j in eachindex(rrange)  # over all reduced dimensions
        rj = rrange[j]
        dr = (rj + rj*(rj+1)/2) * rj
        lines!(ax1, 
            1:num_of_streams, 
            stream_res[:rls].true_stream_err[j,:] / dr, 
            color=line_colors[j], linewidth=8)
    end
    # iQRRLS
    ax2 = Axis(fig[1, 2], 
        xlabel=L"$k$-th stream", 
        title="iQRRLS", 
        yscale=log10, 
        xticks=xtick_vals, titlesize=35, 
        xlabelsize=33, ylabelsize=33, xticklabelsize=30, yticklabelsize=30,
        limits=(nothing, nothing, 1e-17, 1),
        yticks=(ytick_vals, [L"10^{%$(Int(log10(y)))}" for y in ytick_vals]),
    )
    lines = []
    labels = []
    for (j,rj) in enumerate(rrange)  # over all reduced dimensions
        dr = (rj + rj*(rj+1)/2) * rj
        l = lines!(ax2, 
            1:num_of_streams, 
            stream_res[:iqrrls].true_stream_err[j,:] / dr, 
            color=line_colors[j], linewidth=8)
        push!(lines, l)
        push!(labels, "r = $rj")
    end
    # # QRRLS
    # ax3 = Axis(fig[1, 3], 
    #     xlabel=L"$k$-th stream", 
    #     title="QRRLS", 
    #     yscale=log10, 
    #     xticks=xtick_vals, titlesize=30, 
    #     xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    #     limits=(nothing, nothing, 2e-15, 100),
    #     yticks=(ytick_vals, [L"10^{%$(Int(log10(y)))}" for y in ytick_vals]),
    # )
    # lines = []
    # labels = []
    # for (j,rj) in enumerate(rrange)  # over all reduced dimensions
    #     l = scatterlines!(ax3, 
    #         1:num_of_streams, 
    #         stream_res[:qrrls].true_stream_err[j,:], color=line_colors[j])
    #     push!(lines, l)
    #     push!(labels, "r = $rj")
    # end
    Legend(fig[1,3], lines, labels, labelsize=30, patchsize=(30,10))
    display(fig)
    save(joinpath(FILEPATH, "plots/rel_stream_err_per_stream.pdf"), fig)
end

#=====================================================#
## Plot the conversion factor and a posteriori errors
#=====================================================#
with_theme(theme_latexfonts()) do 
    num_of_streams = length(stream_res[:rls].post_err)
    fig = Figure(size=(1000,600))
    xtick_vals = 0:(num_of_streams ÷ 2):num_of_streams
    algos = ["RLS", "iQRRLS", "QRRLS"]
    algos_lower = (Symbol ∘ lowercase).(algos)
    # A posteriori error norm
    ax1 = Axis(fig[1, 1],
        xlabel=L"$k$-th stream", 
        ylabel=L"a posteriori error norm, $\Vert\mathbf{\xi}_k^+\Vert_2$",
        # xticks=xtick_vals, 
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
        ylabel=L"conversion factor, $c_k$", yticks=0:5,
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
    hlines!(ax2, [1.0], color=:red, linestyle=:dashdot, linewidth=3)
    axislegend(
        ax2, lines, algos, 
        position=:lt,
        labelsize=30
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/posterror_convfact.pdf"), fig)
end

#===========================================================================#
## Normalized autocorrelation function for training and test data for r_max
#===========================================================================#
training_stats = load(joinpath(FILEPATH, "data/training_statistics.jld2"))
test_stats = load(joinpath(FILEPATH, "data/testing_statistics.jld2"))
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1400, 600), figure_padding=(1,30,1,1))
    ax1 = Axis(
        fig[1, 1], xlabel="lag", ylabel="mean normalized \n autocorrelation", title=L"Training ($r = %$(rmax)$)",
        titlesize=30, xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25, 
    )
    ax2 = Axis(
        fig[1, 2], xlabel="lag", title=L"Test ($r = %$(rmax)$)", 
        titlesize=30, xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )

    lines = []
    labels = ["fom", "pod", "opinf", "tropinf", "stream_rls", "stream_iqrrls", "stream_qrrls"]
    marker_styles = [:xcross, :diamond, :cross, :circle, :rect, :star5, :hexagon]
    line_styles = [:solid, :dot, :dash, :solid, :dashdot, :dashdotdot, :dash]
    colors = vcat(:black, Makie.wong_colors()[1:length(labels)-1])

    for (i, algo) in enumerate(labels)
        algo = Symbol(algo)
        if algo == :fom
            l = scatterlines!(
                ax1, training_stats["AC_lags"], training_stats["AC"][algo][:],
                marker=marker_styles[i], markersize=20-2*i, linestyle=line_styles[i], linewidth=20-2*i, color=colors[i]
            )
            scatterlines!(
                ax2, test_stats["AC_lags"], test_stats["AC"][algo][:],
                marker=marker_styles[i], markersize=20-2*i, linestyle=line_styles[i], linewidth=20-2*i, color=colors[i]
            )
        else
            l = scatterlines!(
                ax1, training_stats["AC_lags"], training_stats["AC"][algo][:,end],
                marker=marker_styles[i], markersize=20-2*i, linestyle=line_styles[i], linewidth=20-2*i, color=colors[i]
            )
            scatterlines!(
                ax2, test_stats["AC_lags"], test_stats["AC"][algo][:,end],
                marker=marker_styles[i], markersize=20-2*i, linestyle=line_styles[i], linewidth=20-2*i, color=colors[i]
            )
        end
        push!(lines, l)
    end
    Legend(
        fig[2,1:2], lines, 
        [
            "Full", "POD", "OpInf", "TrOpInf", 
            "Stream-RLS", "Stream-iQRRLS", "Stream-QRRLS"
        ],
        orientation=:horizontal, 
        halign=:center, 
        # tellwidth=false, 
        # tellheight=true,
        colgap=30,
        labelsize=30,
        nbanks=2,
        framevisible=false,
        patchsize=(80,20)
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/autocorr.pdf"), fig)
end

#===========================================================#
## Normalized autocorrelation error over reduced dimensions
#===========================================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1400, 600))
    ax1 = Axis(
        fig[1, 1], xlabel=L"reduced dimension, $r$", 
        ylabel="mean normalized \n autocorrelation error",
        title="Training", titlesize=30, xlabelsize=30, ylabelsize=30,
        xticklabelsize=25, yticklabelsize=25, xticks=0:2:24,
    )
    ax2 = Axis(
        fig[1, 2], xlabel=L"reduced dimension, $r$", title="Test", xticks=0:2:24,
        titlesize=30, xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    lines = []
    labels = ["pod", "opinf", "tropinf", "stream_rls", "stream_iqrrls", "stream_qrrls"]
    marker_styles = [:diamond, :cross, :circle, :rect, :star5, :hexagon]
    line_styles = [:dot, :dash, :solid, :dashdot, :dashdotdot, :dash]
    colors = Makie.wong_colors()[1:length(labels)]
    for (i, algo) in enumerate(labels)
        algo = Symbol(algo)
        l = scatterlines!(
            ax1, rrange, training_stats["AC_ERR"][algo][:],
            marker=marker_styles[i], markersize=30, linestyle=:solid, linewidth=10, color=colors[i]
        )
        scatterlines!(
            ax2, rrange, test_stats["AC_ERR"][algo][:],
            marker=marker_styles[i], markersize=30, linestyle=:solid, linewidth=10, color=colors[i]
        )
        push!(lines, l)
    end
    Legend(
        fig[end+1,1:end], lines,
        [
            "POD", "OpInf", "TrOpInf", 
            "Stream-RLS", "Stream-iQRRLS", "Stream-QRRLS"
        ],
        colgap = 30,
        orientation=:horizontal, 
        halign=:center, 
        # tellwidth=false, 
        # tellheight=true,
        labelsize=30,
        nbanks=2,
        framevisible=false,
        patchsize=(80,20)
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/autocorr_error.pdf"), fig)
end

#============================================#
## Lyapunov Exponents over reduced dimensions
#============================================#
training_stats = load(joinpath(FILEPATH, "data/training_statistics.jld2"))
test_stats = load(joinpath(FILEPATH, "data/testing_statistics.jld2"))
using ChaosGizmo: kaplan_yorke_dim
# Reference values
edson = [
    0.043, 0.003, 0.002, -0.004, -0.008, 
    -0.185, -0.253, -0.296, -0.309, -1.965
]
cvitanovic = [
    0.048, 0, 0, -0.003, -0.189,
    -0.256, -0.290, -0.310, -1.963, -1.967
]
edson_ky = kaplan_yorke_dim(edson)
cvitanovic_ky = kaplan_yorke_dim(cvitanovic)

with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1800, 920))  # Increased width for colorbar

    lines = []
    # labels = [
    #     "pod", "opinf", "tropinf", "stream_rls", 
    #     "stream_iqrrls", "stream_qrrls"
    # ]
    # marker_styles = [:diamond, :cross, :circle, :rect, :star5, :hexagon]
    # line_styles = [:dot, :dash, :solid, :dashdot, :dashdotdot, :dash]
    # labels = [
    #     "pod", "opinf", "tropinf", "stream_rls", 
    #     "stream_iqrrls", 
    # ]
    # marker_styles = [:diamond, :cross, :circle, :rect, :star5]
    # line_styles = [:dot, :dash, :solid, :dashdot, :dashdotdot]

    labels = [
        "pod", "tropinf", "stream_rls", 
        "stream_iqrrls", 
    ]
    marker_styles = [:circle, :cross, :rect, :star5]
    line_styles = [:solid, :dash, :dot, :dashdot]

    colors = Makie.wong_colors()[1:length(labels)]

    # --- GRID: 2 rows (Train/Test) x (r-dimensions) + 1 column (D_KY) ---
    rdims = [21, 24]
    ax_le = Matrix{Axis}(undef, 2, length(rdims))
    for (row, splitname, stats) in zip(1:2, ["Training","Test"], [training_stats, test_stats])
        for (col, r) in enumerate(rdims)
            ax = Axis(fig[row, col], 
                    title = (
                        row == 1 ?
                        L"Lyapunov exponent \n %$(splitname) ($r$ = %$r)" :
                        L"%$(splitname) ($r$ = %$r)"
                    ),
                    xlabel = (row==2 ? L"Lyapunov index $i$" : ""),
                    ylabel = (col==1 ? L"Lyapunov exponent $\lambda_i$" : ""),
                    titlesize=30, xlabelsize=30, ylabelsize=30,
                    xticklabelsize=25, yticklabelsize=25,)
            ax_le[row, col] = ax
            if row == 1 && col == 1
                axislegend(ax, 
                    [MarkerElement(color=:black, marker=:star8)],
                    ["Edson et al."], position=:lb, framevisible=false,
                    labelsize=35, markersize=40,
                    patchlabelgap=20
                )
            end
            for (i, algo) in enumerate(labels)
                algo = Symbol(algo)
                l = scatterlines!(
                    ax, 1:length(edson), 
                    stats["LE"][algo][:,findfirst(isequal(r), rrange)],
                    marker=marker_styles[i], markersize=35, 
                    linestyle=line_styles[i], 
                    linewidth=10, color=colors[i],
                    markercolor=:transparent, strokewidth=2.5,
                    strokecolor=colors[i]
                )
                # Reference values
                scatter!(ax, 
                    1:length(edson), edson, color=:black, 
                    markersize=30, marker=:star8)
                # scatter!(ax, 
                #     1:length(edson), cvitanovic, color=:red, 
                #     markersize=25)
                if row == 1 && col == 1
                    push!(lines, l)
                end
            end
        end
    end

    axKYtrain = Axis(
        fig[1, length(rdims)+1], 
        ylabel=L"mean $D_{KY}$",
        title=L"Kaplan-Yorke dimension \n Training$$", xticks=rrange,
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25,
    )
    axKYtest = Axis(
        fig[2, length(rdims)+1], 
        xlabel=L"reduced dimension $r$", ylabel=L"mean $D_{KY}$",
        title=L"Test$$", xticks=rrange,
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25,
    )

    axislegend(axKYtrain, 
        [LineElement(color=:black, linestyle=:solid)],
        ["Edson et al."], position=:rb, framevisible=false,
        labelsize=35, linewidth=10, patchsize=(80,20),
        patchlabelgap=10
    )

    for (i, a) in enumerate(Symbol.(labels))
        c = colors[i]
        lines!(axKYtrain, rrange, training_stats["KY"][a]; color=c, linewidth=3)
        lines!(axKYtest, rrange, test_stats["KY"][a]; color=c, linewidth=3)
    end

    # Reference values
    hlines!(axKYtrain, [edson_ky], color=:black, linestyle=:solid, linewidth=5)
    # hlines!(axKYtrain, [cvitanovic_ky], color=:red, linestyle=:dashdot, linewidth=3)
    hlines!(axKYtest, [edson_ky], color=:black, linestyle=:solid, linewidth=5)
    # hlines!(axKYtest, [cvitanovic_ky], color=:red, linestyle=:dashdot, linewidth=3)

    Legend(
        fig[end+1,1:end], lines,
        [
            "POD", "OpInf",
            "iSVD-Projection-RLS", "iSVD-Projection-iQRRLS", 
            # "Stream-QRRLS",
            # "Edson", "Cvitanovic"
        ],
        colgap = 30,
        rowgap = 20,
        orientation=:horizontal, 
        halign=:center, 
        # tellwidth=false, 
        # tellheight=true,
        labelsize=30,
        nbanks=1,
        framevisible=false,
        patchsize=(130,30),
        patchlabelgap=10,
    )
    save(joinpath(FILEPATH, "plots/lyapunov_exponent_and_ky.pdf"), fig)
    display(fig)
end

#==================================================================#
## Plot the flow field predictions for the training and test data ##
#==================================================================#
training_data_files = readdir(joinpath(FILEPATH, "data/training"), join=true)
test_data_files = readdir(joinpath(FILEPATH, "data/testing"), join=true)
Xtrain = load(training_data_files[4])["Xref"]
Xtest = load(test_data_files[1])["X"]
iVrmax = basis_data["baker"].iVr[:, 1:rmax]
models = load(joinpath(FILEPATH, "data/models/op_mu1.0.jld2"))

## Predict flow field
algos = [
    "pod", "opinf", "tropinf",
    "stream_rls", "stream_iqrrls"
]
## Training
Xtrain_algos = Dict(
    algo => zeros(rmax, size(Xtrain, 2)) for algo in algos
)
for algo in algos
    x0 = view(Xtrain, :, 1)
    Xtrain_algos[algo] = kse.integrate_model(
        kse.tspan, iVrmax' * x0,
        linear_matrix=models[algo].A,
        quadratic_matrix=models[algo].A2u,
        system_input=false, const_stepsize=true
    )
    @info "Completed training reconstructions for r=$rmax using $algo"
end
save(joinpath(FILEPATH, "data/plot_recon/train_r$(rmax).jld2"), Xtrain_algos)

## Testing 
Xtest_algos = Dict(
    algo => zeros(rmax, size(Xtest, 2)) for algo in algos
)
for algo in algos
    x0 = view(Xtest, :, 1)
    Xtest_algos[algo] = kse.integrate_model(
        kse.tspan, iVrmax' * x0,
        linear_matrix=models[algo].A,
        quadratic_matrix=models[algo].A2u,
        system_input=false, const_stepsize=true
    )
    @info "Completed test reconstructions for r=$rmax using $algo"
end
# save(joinpath(FILEPATH, "data/plot_recon/test_r$(rmax).jld2"), Xtest_algos)

## Load predicted flow fields
Xtrain_algos = load(joinpath(FILEPATH, "data/plot_recon/train_r$(rmax).jld2"))
Xtest_algos = load(joinpath(FILEPATH, "data/plot_recon/test_r$(rmax).jld2"))

## Remove the OpInf (temporary since it's not necessary for plotting)
delete!(Xtrain_algos, "opinf")
delete!(Xtest_algos, "opinf")

## Create plots
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(2000, 1000))
    rows, cols = 2, 11
    hp = (cols - 1) ÷ 2

    # algos = [
    #     "pod", "opinf", "tropinf",
    #     "stream_rls", "stream_iqrrls"
    # ]
    # labels = [
    #     "POD", "OpInf", "TR-OpInf",
    #     "Stream-RLS", "Stream-iQRRLS"
    # ]

    algos = [
        "pod", "tropinf",
        "stream_rls", "stream_iqrrls"
    ]
    labels = [
        "POD", "OpInf",
        "Stream-RLS", "Stream-iQRRLS"
    ]

    ds = 100 # Downsampling factor for visualization

    # Pre-compute all matrices to find global colorranges
    all_errors = []
    all_flow_fields = []
    
    # Create axes for all positions
    axes_flow = Matrix{Axis}(undef, 1, cols)
    axes_error = Matrix{Axis}(undef, 1, cols-1)  # No error axes for cols 1 and 8
    
    for col in 1:cols
        # Flow field axes (top row)
        axes_flow[1, col] = Axis(
            fig[1, col], 
            xgridvisible=false, ygridvisible=false,
            xticklabelsvisible=false, yticklabelsvisible=false,
            xticksvisible=false, yticksvisible=false,
            xlabelsize=20, ylabelsize=20, titlesize=25,
        )
        
        # Error axes (bottom row) - skip for cols 1 and 8 (ground truth)
        if col != 1 && col != hp+1
            error_col_idx = col > hp+1 ? col - 2 : col - 1  # Adjust index for skipped columns
            axes_error[1, error_col_idx] = Axis(
                fig[2, col], 
                xgridvisible=false, ygridvisible=false,
                xticklabelsvisible=false, yticklabelsvisible=false,
                xticksvisible=false, yticksvisible=false,
                xlabelsize=20, ylabelsize=20, titlesize=25
            )
        end
    end

    # First pass: compute all data to find global ranges
    for col in 1:cols
        X = col <= hp ? Xtrain : Xtest
        
        # Collect flow field data
        if col == 1 || col == hp+1
            # Ground truth
            push!(all_flow_fields, X[:, 1:ds:end])
        elseif col != cols
            # ROM predictions - just use first algorithm since they should be similar for visualization
            algo = algos[(col - 1) % length(algos) + 1]
            Xrom = col <= hp ? Xtrain_algos[algo] : Xtest_algos[algo]
            Xrecon = iVrmax * Xrom
            push!(all_flow_fields, Xrecon[:, 1:ds:end])
            error_matrix = abs.(X - Xrecon)[:, 1:ds:end]
            push!(all_errors, error_matrix)
        end
    end
    
    # Find global ranges
    global_flow_min = minimum(minimum.(all_flow_fields))
    global_flow_max = maximum(maximum.(all_flow_fields))
    flow_colorrange = (global_flow_min, global_flow_max)
    
    global_error_min = minimum(minimum.(all_errors))
    global_error_max = maximum(maximum.(all_errors))
    error_colorrange = (global_error_min, global_error_max)
    
    # Second pass: create all plots with consistent coloring
    error_idx = 1
    for col in 1:cols
        X = col <= hp ? Xtrain : Xtest
        
        if col == 1 || col == hp+1
            # Ground truth plots
            heatmap!(axes_flow[1, col], X[:,1:ds:end]; 
                    colormap=:viridis, colorrange=flow_colorrange)
            # if col == 1
            #     axes_flow[1,col].ylabel = "Full (Train)"
            # else
            #     axes_flow[1,col].ylabel = "Full (Test)"
            # end
        elseif col != cols
            # ROM predictions and errors
            idx = (col - 1) % length(algos) + 1 
            algo = algos[idx]
            label = labels[idx] 
            Xrom = col <= hp ? Xtrain_algos[algo] : Xtest_algos[algo]
            Xrecon = iVrmax * Xrom
                
            heatmap!(axes_flow[1, col], Xrecon[:,1:ds:end]; 
                    colormap=:viridis, colorrange=flow_colorrange)
                
            error_col_idx = col > hp+1 ? col - 2 : col - 1
            hm_error = heatmap!(
                axes_error[1, error_col_idx], 
                all_errors[error_idx];
                colormap=:matter,
                colorrange=error_colorrange
            )

            # if col < hp+1
            #     axes_flow[1,col].ylabel = "$label (Train)"
            # else
            #     axes_flow[1,col].ylabel = "$label (Test)"
            # end

            error_idx += 1
        end
    end
    
    # Add colorbar for flow fields in column 15, row 1
    Colorbar(fig[1, cols], 
        colormap=:viridis, 
        colorrange=flow_colorrange,
        label="Flow Field Value",
        labelsize=30,
        ticklabelsize=25,
        width=20
    )
    
    # Add colorbar for errors in column 15, row 2
    Colorbar(fig[2, cols], 
        colormap=:matter, 
        colorrange=error_colorrange,
        label="Absolute Error",
        labelsize=30,
        ticklabelsize=25,
        width=20
    )
    
    display(fig)
    save(joinpath(FILEPATH, "plots/flow_field_comparison.png"), fig)
end