"""
1D Viscous Burgers equation: plotting results
"""

#================#
## Load Packages
#================#
using CairoMakie
using FileIO
using JLD2
using LinearAlgebra
using PolynomialModelReductionDataset: BurgersModel
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"Two-Pass_Streaming-OpInf/burgers") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/burgers")

#===================#
## Load the options
#===================#
setup_file = joinpath(FILEPATH, "data/setup.jld2")
setup = load(setup_file)
options = setup["options"]
burgers = setup["burgers"]
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
        yscale=log10, xticks=1:rmax, titlesize=30, 
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
        patchsize=(80,20),
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
        yscale=log10, xticks=1:rmax, titlesize=30, 
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
        patchsize=(80,20),
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/absolute_sval_error.pdf"), fig)
end


#====================================================#
## Plot the subspace angle errors between the bases ##
#====================================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 400))      
    ax = Axis(
        fig[1, 1], xlabel=L"reduced dimension, $r$", 
        ylabel=L"subspace angle error$$",
        yscale=log10, xticks=1:rmax, titlesize=30, 
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
        xticks=1:rmax, yscale=log10,
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
training_errors = load(joinpath(FILEPATH, "data/training_errors.jld2"), "train_errors")
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 550))
    ax = Axis(
        fig[1, 1], xlabel=L"reduced dimension, $r$", 
        ylabel="relative state error",
        xticks=1:rmax, yscale=log10,
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25,
        limits=(nothing, nothing, 2e-5, 2e0),
        title="Training"
    )
    lines = []
    # labels = [
    #     "pod", "opinf", "tropinf", 
    #     "stream_rls", "stream_iqrrls", "stream_qrrls",
    #     "stream"
    # ]
    # marker_styles = [:diamond, :cross, :circle, :rect, :star5, :hexagon, :utriangle]
    # line_styles = [:dot, :dash, :solid, :dashdot, :dashdotdot, :dash, :dot]

    # labels = [
    #     "pod", "opinf", "tropinf", 
    #     "stream_rls", "stream_iqrrls",
    #     "stream"
    # ]
    # legend_labels = [
    #     "POD", "OpInf", "TR-OpInf", 
    #     "Stream-RLS", "Stream-iQRRLS",
    #     "Stream"
    # ]
    # marker_styles = [:diamond, :cross, :circle, :rect, :star5, :hexagon]
    # line_styles = [:dot, :dash, :solid, :dashdot, :dashdotdot, :dash]

    labels = [
        "pod", "opinf", "tropinf", 
        "stream_rls", "stream_iqrrls",
        "stream"
    ]
    legend_labels = [
        "POD", "OpInf", "TR-OpInf", 
        "Stream-RLS", "Stream-iQRRLS",
        "Stream"
    ]
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
        ax, lines, legend_labels,
        position=:lb,
        labelsize=30,
        patchsize=(80,20),
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/training_rse_errors.pdf"), fig)
end

#=======================================================#
## Plot the training relative state errors (version 2) ##
#=======================================================#
training_errors = load(joinpath(FILEPATH, "data/training_errors.jld2"), "train_errors")
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 1000))
    
    # Common settings
    common_xlabelsize = 29
    common_ylabelsize = 29
    common_titlesize = 30
    common_ticklabelsize = 24
    
    # Define the three comparisons
    comparisons = [
        (methods=["pod", "opinf", "stream_rls"], 
         legends=["POD", "OpInf", "iSVD-RLS"], 
         title="iSVD-RLS"),
        (methods=["pod", "opinf", "stream_iqrrls"], 
         legends=["POD", "OpInf", "iSVD-iQRRLS"], 
         title="iSVD-iQRRLS"),
        (methods=["pod", "opinf", "stream"], 
         legends=["POD", "OpInf", "iSVD-Compact LS"], 
         title="iSVD-Compact LS")
    ]

    colors = Makie.wong_colors()[1:5]
    
    # Marker and line styles for each method type
    pod_style = (
        marker=:circle, linestyle=:solid, color=:transparent,
        strokecolor=colors[1], strokewidth=2.5
    )
    opinf_style = (
        marker=:cross, linestyle=:dash, color=:transparent,
        strokecolor=colors[2], strokewidth=2.5
    )
    stream_styles = [
        (
            marker=:rect, linestyle=:dashdot, color=:transparent,
            strokecolor=colors[3], strokewidth=2.5
        ),   # Stream-RLS
        (
            marker=:star5, linestyle=:dashdot, color=:transparent,
            strokecolor=colors[4], strokewidth=2.5
        ),  # Stream-iQRRLS
        (
            marker=:hexagon, linestyle=:dashdot, color=:transparent,
            strokecolor=colors[5], strokewidth=2.5
        ) # Stream
    ]
    
    for (i, comp) in enumerate(comparisons)
        ax = Axis(
            fig[i, 1], 
            xlabel=i == 3 ? L"reduced dimension, $r$" : "",
            ylabel=L"MR-SSE($K, r$)",
            xticks=1:rmax, yscale=log10,
            titlesize=common_titlesize, xlabelsize=common_xlabelsize, 
            ylabelsize=common_ylabelsize, 
            xticklabelsize=common_ticklabelsize, 
            yticklabelsize=common_ticklabelsize,
            xticksvisible=i == 3 ? true : false,
            xticklabelsvisible=i == 3 ? true : false,
            limits=(nothing, nothing, 2e-5, 2e0),
            title=comp.title
        )
        
        lines = []
        
        for (j, method) in enumerate(comp.methods)
            if method == "pod"
                style = pod_style
            elseif method == "opinf"
                style = opinf_style
            else # streaming method
                style = stream_styles[i]
            end
            
            l = scatterlines!(
                ax, 1:rmax, vec(training_errors[method]),
                linestyle=style.linestyle, linewidth=8,
                color=style.strokecolor, markercolor=style.color,
                marker=style.marker, markersize=28,
                strokecolor=style.strokecolor, strokewidth=style.strokewidth
            )
            push!(lines, l)
        end
        
        # axislegend(
        #     ax, lines, comp.legends,
        #     position=:rt,
        #     labelsize=26,
        #     patchsize=(120,15),
        # )
    end
    
    display(fig)
    save(joinpath(FILEPATH, "plots/training_rse_errors_comparison.pdf"), fig)
end

#============================================#
## Plot the relative state errors per stream
#============================================#
stream_res = load(joinpath(FILEPATH, "data/streaming/stream_results.jld2"))["stream_res"]
with_theme(theme_latexfonts()) do
    num_of_streams = size(stream_res[:rls].rse, 2)
    line_colors = Makie.resample_cmap(:viridis, rmax)
    fig = Figure(size=(1800,520))
    xtick_vals = 0:(num_of_streams ÷ 4):num_of_streams
    # Standard RLS
    ax1 = Axis(fig[1, 1], 
        xlabel=L"$k$-th stream", 
        ylabel="mean relative state error", 
        title="RLS", 
        yscale=log10, xticks=xtick_vals, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        limits=(nothing, nothing, 1e-3, 3),
    )
    for (j,ri) in enumerate(1:rmax)  # over all reduced dimensions
        scatterlines!(
            ax1, 1:num_of_streams, stream_res[:rls].rse[ri,:], 
            color=line_colors[j])
    end
    # iQRRLS
    ax2 = Axis(fig[1, 2], 
        xlabel=L"$k$-th stream", 
        # ylabel="Relative state error", 
        title="iQRRLS", 
        yscale=log10, xticks=xtick_vals, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        limits=(nothing, nothing, 1e-3, 3),
    )
    lines = []
    labels = []
    for (j,ri) in enumerate(1:rmax)  # over all reduced dimensions
        l = scatterlines!(
            ax2, 1:num_of_streams, stream_res[:iqrrls].rse[ri,:], 
            color=line_colors[j])
        push!(lines, l)
        push!(labels, "r = $ri")
    end
    # # QRRLS
    # ax3 = Axis(fig[1, 3], 
    #     xlabel=L"$k$-th stream", 
    #     # ylabel="Relative state error", 
    #     title="QRRLS", 
    #     yscale=log10, xticks=xtick_vals, titlesize=30, 
    #     xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    #     limits=(nothing, nothing, 1e-3, 3),
    # )
    # lines = []
    # labels = []
    # for (j,ri) in enumerate(1:rmax)  # over all reduced dimensions
    #     l = scatterlines!(
    #         ax3, 1:num_of_streams, stream_res[:qrrls].rse[ri,:], 
    #         color=line_colors[j])
    #     push!(lines, l)
    #     push!(labels, "r = $ri")
    # end
    Legend(fig[1,3], lines, labels, labelsize=30, patchsize=(30,10))
    display(fig)
    save(joinpath(FILEPATH, "plots/rel_state_err_per_stream.pdf"), fig)
end

#================================================#
## Plot the relative streaming errors per stream
#================================================#
with_theme(theme_latexfonts()) do
    num_of_streams = size(stream_res[:rls].stream_err, 2)
    line_colors = Makie.resample_cmap(:viridis, rmax)
    fig = Figure(size=(1800,520))
    xtick_vals = 0:(num_of_streams ÷ 4):num_of_streams
    ytick_vals = 10.0 .^ (-14:2:0)
    # Standard RLS
    ax1 = Axis(fig[1, 1], 
        xlabel=L"$k$-th stream", 
        ylabel="mean relative streaming errors", 
        title="RLS", 
        yscale=log10,
        limits=(nothing, nothing, 1e-15, 50),
        yticks=(ytick_vals, [L"10^{%$(Int(log10(y)))}" for y in ytick_vals]),
        xticks=xtick_vals, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    for (j,ri) in enumerate(1:rmax)  # over all reduced dimensions
        dr = (ri + ri*(ri+1)/2 + 1) * ri
        scatterlines!(
            ax1, 1:num_of_streams, 
            stream_res[:rls].true_stream_err[ri,:] / dr, 
            color=line_colors[j])
    end
    # iQRRLS
    ax2 = Axis(fig[1, 2], 
        xlabel=L"$k$-th stream", 
        title="iQRRLS", 
        yscale=log10, 
        limits=(nothing, nothing, 1e-15, 50),
        yticks=(ytick_vals, [L"10^{%$(Int(log10(y)))}" for y in ytick_vals]),
        xticks=xtick_vals, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    lines = []
    labels = []
    for (j,ri) in enumerate(1:rmax)  # over all reduced dimensions
        dr = (ri + ri*(ri+1)/2 + 1) * ri
        l = scatterlines!(
            ax2, 1:num_of_streams, 
            stream_res[:iqrrls].true_stream_err[ri,:] / dr, 
            color=line_colors[j])
        push!(lines, l)
        push!(labels, "r = $ri")
    end
    # # QRRLS
    # ax3 = Axis(fig[1, 3], 
    #     xlabel=L"$k$-th stream", 
    #     title="QRRLS", 
    #     yscale=log10, 
    #     limits=(nothing, nothing, 1e-15, 50),
    #     yticks=(ytick_vals, [L"10^{%$(Int(log10(y)))}" for y in ytick_vals]),
    #     xticks=xtick_vals, titlesize=30, 
    #     xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    # )
    # lines = []
    # labels = []
    # for (j,ri) in enumerate(1:rmax)  # over all reduced dimensions
    #     l = scatterlines!(
    #         ax3, 1:num_of_streams, stream_res[:qrrls].true_stream_err[ri,:], 
    #         color=line_colors[j])
    #     push!(lines, l)
    #     push!(labels, "r = $ri")
    # end
    Legend(fig[1,3], lines, labels, labelsize=30, patchsize=(30,10))
    display(fig)
    # save(joinpath(FILEPATH, "plots/rel_stream_err_per_stream.pdf"), fig)
end


#============================================#
## Plot the relative state errors per stream
#============================================#
stream_res = load(joinpath(FILEPATH, "data/streaming/stream_results.jld2"))["stream_res"]
with_theme(theme_latexfonts()) do
    # RSEs
    num_of_streams = size(stream_res[:rls].rse, 2)
    line_colors = Makie.resample_cmap(:viridis, rmax)
    fig = Figure(size=(1880,900))
    xtick_vals = 0:(num_of_streams ÷ 4):num_of_streams
    # Standard RLS
    ax1 = Axis(fig[1, 1], 
        xlabel=L"$k$-th stream", 
        ylabel=L"MR-SSE($k,r$)", 
        title="RLS", 
        yscale=log10, xticks=xtick_vals, titlesize=32, 
        xlabelsize=32, ylabelsize=32, xticklabelsize=28, yticklabelsize=28,
        limits=(nothing, nothing, 1e-3, 3),
    )
    for (j,ri) in enumerate(1:rmax)  # over all reduced dimensions
        lines!(
            ax1, 1:num_of_streams, stream_res[:rls].rse[ri,:], 
            color=line_colors[j], linewidth=8)
    end
    # iQRRLS
    ax2 = Axis(fig[1, 2], 
        xlabel=L"$k$-th stream", 
        # ylabel="Relative state error", 
        title="iQRRLS", 
        yscale=log10, xticks=xtick_vals, titlesize=32, 
        xlabelsize=32, ylabelsize=32, xticklabelsize=28, yticklabelsize=28,
        yticksvisible=false, yticklabelsvisible=false,
        limits=(nothing, nothing, 1e-3, 3),
    )
    lines = []
    labels = []
    for (j,ri) in enumerate(1:rmax)  # over all reduced dimensions
        l = lines!(
            ax2, 1:num_of_streams, stream_res[:iqrrls].rse[ri,:], 
            color=line_colors[j], linewidth=8)
        push!(lines, l)
        push!(labels, "r = $ri")
    end

    # Streaming errors
    num_of_streams = size(stream_res[:rls].stream_err, 2)
    line_colors = Makie.resample_cmap(:viridis, rmax)
    xtick_vals = 0:(num_of_streams ÷ 4):num_of_streams
    ytick_vals = 10.0 .^ (-14:2:0)
    # Standard RLS
    ax3 = Axis(fig[2, 1], 
        xlabel=L"$k$-th stream", 
        ylabel=L"MR-SOE($k,r$)", 
        title="RLS", 
        yscale=log10,
        limits=(nothing, nothing, 1e-15, 50),
        yticks=(ytick_vals, [L"10^{%$(Int(log10(y)))}" for y in ytick_vals]),
        xticks=xtick_vals, titlesize=32, 
        xlabelsize=32, ylabelsize=32, xticklabelsize=28, yticklabelsize=28,
    )
    for (j,ri) in enumerate(1:rmax)  # over all reduced dimensions
        dr = (ri + ri*(ri+1)/2 + 1) * ri
        lines!(
            ax3, 1:num_of_streams, 
            stream_res[:rls].true_stream_err[ri,:] / dr, 
            color=line_colors[j], linewidth=8)
    end
    # iQRRLS
    ax4 = Axis(fig[2, 2], 
        xlabel=L"$k$-th stream", 
        title="iQRRLS", 
        yscale=log10, 
        limits=(nothing, nothing, 1e-15, 50),
        yticks=(ytick_vals, [L"10^{%$(Int(log10(y)))}" for y in ytick_vals]),
        xticks=xtick_vals, titlesize=32, 
        xlabelsize=32, ylabelsize=32, xticklabelsize=28, yticklabelsize=28,
        yticksvisible=false, yticklabelsvisible=false,
    )
    lines = []
    labels = []
    for (j,ri) in enumerate(1:rmax)  # over all reduced dimensions
        dr = (ri + ri*(ri+1)/2 + 1) * ri
        l = lines!(
            ax4, 1:num_of_streams, 
            stream_res[:iqrrls].true_stream_err[ri,:] / dr, 
            color=line_colors[j], linewidth=8)
        push!(lines, l)
        push!(labels, "r = $ri")
    end
    Legend(fig[:,3], lines, labels, labelsize=30, patchsize=(30,10))
    display(fig)
    save(joinpath(FILEPATH, "plots/rel_stream_and_rse_err_per_stream.png"), fig,
         px_per_unit=4)
end

#=====================================================#
## Plot the conversion factor and a posteriori errors
#=====================================================#
with_theme(theme_latexfonts()) do 
    num_of_streams = length(stream_res[:rls].post_err)
    fig = Figure(size=(1000,600))
    xtick_vals = 0:(num_of_streams ÷ 4):num_of_streams
    algos = ["RLS", "iQRRLS", "QRRLS"]
    algos_lower = (Symbol ∘ lowercase).(algos)
    # A posteriori error norm
    ax1 = Axis(fig[1, 1],
        xlabel=L"$k$-th stream", 
        ylabel=L"a posteriori error norm, $\Vert\mathbf{\xi}_k^+\Vert_2$",
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
        ylabel=L"conversion factor, $c_k$",
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
    fig = Figure(size=(800, 550))
    ax = Axis(
        fig[1, 1], xlabel=L"reduced dimension, $r$", 
        ylabel="relative state error",
        xticks=1:rmax, yscale=log10,
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25,
        limits=(nothing, nothing, 2e-5, 2e0),
        title="Testing"
    )
    lines = []
    # labels = [
    #     "pod", "opinf", "tropinf", 
    #     "stream_rls", "stream_iqrrls", "stream_qrrls",
    #     "stream"
    # ]
    # marker_styles = [:diamond, :cross, :circle, :rect, :star5, :hexagon, :utriangle]
    # line_styles = [:dot, :dash, :solid, :dashdot, :dashdotdot, :dash, :dot]

    labels = [
        "pod", "opinf", "tropinf", 
        "stream_rls", "stream_iqrrls",
        "stream"
    ]
    legend_labels = [
        "POD", "OpInf", "Tr-OpInf", 
        "Stream-RLS", "Stream-iQRRLS",
        "Stream"
    ]
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
    # axislegend(
    #     ax, lines, legend_labels,
    #     position=:lb,
    #     labelsize=30,
    #     patchsize=(80,20),
    # )
    display(fig)
    save(joinpath(FILEPATH, "plots/testing_rse_errors.pdf"), fig)
end

#=============================================#
## Plot the relative state errors for testing
#=============================================#
testing_errors = load(joinpath(FILEPATH, "data/testing_errors.jld2"))
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 1000))
    
    # Common settings
    common_xlabelsize = 29
    common_ylabelsize = 29
    common_titlesize = 30
    common_ticklabelsize = 24
    
    # Define the three comparisons
    comparisons = [
        (methods=["pod", "opinf", "stream_rls"], 
         legends=["POD", "OpInf", "iSVD-RLS"], 
         title="iSVD-RLS"),
        (methods=["pod", "opinf", "stream_iqrrls"], 
         legends=["POD", "OpInf", "iSVD-iQRRLS"], 
         title="iSVD-iQRRLS"),
        (methods=["pod", "opinf", "stream"], 
         legends=["POD", "OpInf", "iSVD-Compact LS"], 
         title="iSVD-Compact LS")
    ]

    colors = Makie.wong_colors()[1:5]
    
    # Marker and line styles for each method type
    pod_style = (
        marker=:circle, linestyle=:solid, color=:transparent,
        strokecolor=colors[1], strokewidth=2.5
    )
    opinf_style = (
        marker=:cross, linestyle=:dash, color=:transparent,
        strokecolor=colors[2], strokewidth=2.5
    )
    stream_styles = [
        (
            marker=:rect, linestyle=:dashdot, color=:transparent,
            strokecolor=colors[3], strokewidth=2.5
        ),   # Stream-RLS
        (
            marker=:star5, linestyle=:dashdot, color=:transparent,
            strokecolor=colors[4], strokewidth=2.5
        ),  # Stream-iQRRLS
        (
            marker=:hexagon, linestyle=:dashdot, color=:transparent,
            strokecolor=colors[5], strokewidth=2.5
        ) # Stream
    ]
    
    for (i, comp) in enumerate(comparisons)
        ax = Axis(
            fig[i, 1], 
            xlabel=i == 3 ? L"reduced dimension, $r$" : "",
            # ylabel=L"MR-SSE($K, r$)",
            xticks=1:rmax, yscale=log10,
            titlesize=common_titlesize, xlabelsize=common_xlabelsize, 
            ylabelsize=common_ylabelsize, 
            xticklabelsize=common_ticklabelsize, 
            yticklabelsize=common_ticklabelsize,
            xticksvisible=i == 3 ? true : false,
            xticklabelsvisible=i == 3 ? true : false,
            limits=(nothing, nothing, 2e-5, 2e0),
            title=comp.title
        )
        
        lines = []
        
        for (j, method) in enumerate(comp.methods)
            if method == "pod"
                style = pod_style
            elseif method == "opinf"
                style = opinf_style
            else # streaming method
                style = stream_styles[i]
            end
            
            l = scatterlines!(
                ax, 1:rmax, vec(testing_errors[method]),
                linestyle=style.linestyle, linewidth=8,
                color=style.strokecolor, markercolor=style.color,
                marker=style.marker, markersize=28,
                strokecolor=style.strokecolor, strokewidth=style.strokewidth
            )
            push!(lines, l)
        end
        
        axislegend(
            ax, lines, comp.legends,
            position=:rt,
            labelsize=26,
            patchsize=(120,15),
        )
    end
    
    display(fig)
    save(joinpath(FILEPATH, "plots/testing_rse_errors_comparison.pdf"), fig)
end