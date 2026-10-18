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
FILEPATH = @__DIR__

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

@info "Generating all plots..."

@info "Loading basis data..."
basis_file = joinpath(FILEPATH, "data/streaming/basis.jld2")
bases = load(basis_file)

#====================================================#
## Plot the subspace angle errors between the bases ##
#====================================================#
@info "Plotting subspace angle errors..."
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
@info "Plotting projection errors..."
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

#================================================#
## Plot the relative streaming errors per stream
#================================================#
@info "Plotting relative streaming errors per stream..."
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
    Legend(fig[1,3], lines, labels, labelsize=30, patchsize=(30,10))
    display(fig)
    save(joinpath(FILEPATH, "plots/rel_stream_err_per_stream.pdf"), fig)
end

#============================================#
## Lyapunov Exponents over reduced dimensions
#============================================#
@info "Plotting Lyapunov exponents and Kaplan-Yorke dimension..."
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
    fig = Figure(size=(1800, 980))
    
    # Both columns use r=24, but different methods
    labels_col1 = ["pod", "tropinf", "stream_rls"]
    labels_col2 = ["pod", "tropinf", "stream_iqrrls"]
    labels = [labels_col1, labels_col2]
    marker_styles1 = [:circle, :cross, :rect]
    marker_styles2 = [:circle, :cross, :star5]
    marker_styles = [marker_styles1, marker_styles2]
    marker_sizes = [48, 50, 60]
    line_styles1 = [:solid, :dash, :dot]
    line_styles2 = [:solid, :dash, :dashdot]
    line_styles = [line_styles1, line_styles2]
    line_widths = [10, 13, 16]
    colors = [Makie.wong_colors()[[1,2,3]], Makie.wong_colors()[[1,2,4]]]

    # --- GRID: 2 rows (Train/Test) x 2 columns (RLS, iQRRLS) + 1 column (D_KY) ---
    r = 24  # Both columns use r=24
    r_idx = findfirst(isequal(r), rrange)
    ax_le = Matrix{Axis}(undef, 2, 2)
    
    for (row, splitname, stats) in zip(1:2, ["Train","Test"], [training_stats, test_stats])
        for (col, method_name) in zip(1:2, ["iSVD-Project-RLS", "iSVD-Project-iQRRLS"])
            ax = Axis(fig[row, col], 
                    title = (
                        row == 1 ?
                        L"Lyapunov exponent \n %$(splitname) ($r$ = %$r, %$method_name)" :
                        L"%$(splitname) ($r$ = %$r, %$method_name)"
                    ),
                    xlabel = (row==2 ? L"Lyapunov index $i$" : ""),
                    ylabel = (col==1 ? L"Lyapunov exponent $\lambda_i$" : ""),
                    titlesize=34, xlabelsize=35, ylabelsize=35,
                    xticklabelsize=30, yticklabelsize=30,)
            ax_le[row, col] = ax
            
            if row == 1 && col == 1
                axislegend(ax, 
                    [MarkerElement(
                        color=:transparent, marker=:star8,
                        strokecolor=:black, strokewidth=3.5
                    )],
                    ["Edson et al."], position=:lb, framevisible=false,
                    labelsize=35, markersize=40, 
                    patchlabelgap=20
                )
            end

            for (i, algo) in enumerate(labels[col])
                algo = Symbol(algo)
                l = scatterlines!(
                    ax, 1:length(edson), 
                    stats["LE"][algo][:, r_idx],
                    marker=marker_styles[col][i], 
                    markersize=marker_sizes[i],
                    linestyle=line_styles[col][i], 
                    linewidth=line_widths[i], #color=colors[col][i],
                    color= :transparent,
                    markercolor=:transparent, strokewidth=3.5,
                    strokecolor=colors[col][i]
                )
                
                # Reference values
                scatter!(ax, 
                    1:length(edson), edson, color=:transparent,
                    strokewidth=2.5,
                    markersize=35, marker=:star8, strokecolor=:black) 
            end
        end
    end

    axKYtrain = Axis(
        fig[1, 3], 
        ylabel=L"mean $D_{KY}$",
        title=L"Kaplan-Yorke dimension \n Training$$", xticks=rrange,
        titlesize=35, xlabelsize=35, ylabelsize=35, 
        xticklabelsize=30, yticklabelsize=30,
    )
    axKYtest = Axis(
        fig[2, 3], 
        xlabel=L"reduced dimension $r$", ylabel=L"mean $D_{KY}$",
        title=L"Test$$", xticks=rrange,
        titlesize=35, xlabelsize=35, ylabelsize=35, 
        xticklabelsize=30, yticklabelsize=30,
    )

    axislegend(axKYtrain, 
        [LineElement(color=:black, linestyle=:dash)],
        ["Edson et al."], position=:rb, framevisible=false,
        labelsize=35, linewidth=10, patchsize=(100,20),
        patchlabelgap=10
    )

    # Plot all methods on KY dimension plot
    all_labels = ["pod", "tropinf", "stream_rls", "stream_iqrrls"]
    colors = Makie.wong_colors()[1:length(all_labels)]
    for (i, a) in enumerate(Symbol.(all_labels))
        c = colors[i]
        lines!(axKYtrain, rrange, training_stats["KY"][a]; color=c, linewidth=3)
        lines!(axKYtest, rrange, test_stats["KY"][a]; color=c, linewidth=3)
    end

    # Reference values
    hlines!(axKYtrain, [edson_ky], color=:black, linestyle=:dash, linewidth=5)
    hlines!(axKYtest, [edson_ky], color=:black, linestyle=:dash, linewidth=5)

    # Elements for the Legend
    elem1 = [
        MarkerElement(color=:transparent, marker=:circle, markersize=45,
            strokecolor=colors[1], strokewidth=5.0),
        LineElement(color=colors[1], linestyle=:solid, linewidth=8,
            strokecolor=colors[1], strokewidth=5.0),
    ]
    elem2 = [
        MarkerElement(color=:transparent, marker=:cross, markersize=45,
            strokecolor=colors[2], strokewidth=5),
        LineElement(color=colors[2], linestyle=:solid, linewidth=8,
            strokecolor=colors[2], strokewidth=5),
    ]
    elem3 = [
        MarkerElement(color=:transparent, marker=:rect, markersize=45,
            strokecolor=colors[3], strokewidth=5),
        LineElement(color=colors[3], linestyle=:solid, linewidth=8,
            strokecolor=colors[3], strokewidth=5),
    ]
    elem4 = [
        MarkerElement(color=:transparent, marker=:star5, markersize=45,
            strokecolor=colors[4], strokewidth=5),
        LineElement(color=colors[4], linestyle=:solid, linewidth=8,
            strokecolor=colors[4], strokewidth=5),
    ]

    elements = [elem1, elem2, elem3, elem4]

    Legend(
        fig[end+1,1:end], elements,
        [
            "POD", "OpInf", "iSVD-Project-RLS", "iSVD-Project-iQRRLS" 
        ],
        colgap = 30,
        rowgap = 20,
        orientation=:horizontal, 
        halign=:center,
        labelsize=35,
        nbanks=1,
        framevisible=false,
        patchsize=(169,30),
        patchlabelgap=10,
    )
    save(joinpath(FILEPATH, "plots/lyapunov_exponent_and_ky.pdf"), fig)
    display(fig)
end 


#==================================================================#
## Plot the flow field predictions for the training and test data ##
#==================================================================#
@info "Plotting flow field predictions..."
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

train_file = joinpath(FILEPATH, "data/plot_recon/train_r$(rmax).jld2")
test_file = joinpath(FILEPATH, "data/plot_recon/test_r$(rmax).jld2")

if isfile(train_file) && isfile(test_file)
    ## Load predicted flow fields
    @info "Loading existing predicted flow fields..."
    Xtrain_algos = load(train_file)
    Xtest_algos = load(test_file)
else
    @info "Computing predicted flow fields..."
    
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
    save(train_file, Xtrain_algos)

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
    save(test_file, Xtest_algos)
end

## Remove the OpInf (temporary since it's not necessary for plotting)
delete!(Xtrain_algos, "opinf")
delete!(Xtest_algos, "opinf")

## Create plots
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(2000, 1000))
    rows, cols = 2, 11
    hp = (cols - 1) ÷ 2

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

@info "All plots generated and saved."