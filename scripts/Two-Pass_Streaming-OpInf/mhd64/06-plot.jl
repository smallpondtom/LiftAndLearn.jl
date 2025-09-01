"""
MHD64: Plotting results
"""

#================#
## Load Packages
#================#
using CairoMakie
using FileIO
using JLD2
using LinearAlgebra
using IncrementalSVD
import LiftAndLearn as LnL

#============#
## Settings ##
#============#
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(), "Two-Pass_Streaming-OpInf/mhd64") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/mhd64")
DATAPATH = "../../../../DATA/THE_WELL/mhd64"
rmax = 50

#============================================================#
## Plot the error between the batch and iSVD singular values
#============================================================#
# Load the bases
basis = load(joinpath(FILEPATH, "data/bases/basis.jld2"))
brand = load(joinpath(FILEPATH, "data/bases/brand_basis.jld2"))["brand"]
baker = load(joinpath(FILEPATH, "data/bases/baker_basis.jld2"))["baker"]
sketchy = load(joinpath(FILEPATH, "data/bases/sketchy_basis.jld2"))["sketchy"]
bases = Dict(
    "batch" => basis, 
    "baker" => baker, 
    "brand" => brand, 
    "sketchy" => sketchy
)

##

with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 600))
    ax = Axis(
        fig[1, 1], xlabel=L"singular value index, $i$", 
        ylabel="relative error of singular values",
        yscale=log10, xticks=vcat(1, 5:5:rmax), titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
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
        Σr = basis.Σ
        Σr_batch = bases["batch"]["Σr"][1:rmax]
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
        position=:rb,
        labelsize=30,
        patchsize=(80,20)
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/relative_sval_error.pdf"), fig)
end

#====================================================#
## Plot the subspace angle errors between the bases ##
#====================================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 600))      
    ax = Axis(
        fig[1, 1], xlabel=L"reduced dimension, $r$", 
        ylabel="subspace angle error",
        yscale=log10, xticks=vcat(1, 5:5:rmax), titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    lines = []
    labels = []
    marker_styles = [:diamond, :cross, :circle, :rect]
    line_styles = [:solid, :solid, :solid, :dash]
    Algorithms = ["Baker", "Brand", "Sketchy" ]
    i = 1
    for Algo in Algorithms
        algo = lowercase(Algo)
        basis = bases[algo]
        angle_errs = zeros(rmax)
        for r in 1:rmax
            # angle_errs[r] = norm(
            #     bases["batch"]["Vr"][:, 1:r] * bases["batch"]["Vr"][:, 1:r]' - 
            #     basis.Q[:, 1:r] * basis.Q[:, 1:r]', 2
            # ) / sqrt(2)

            # More memory-efficient calculation using SVD of the cross-correlation
            U_batch = bases["batch"]["Vr"][:, 1:r]
            U_approx = basis.Q[:, 1:r]
            
            # Compute cross-correlation matrix (much smaller: r×r instead of n×n)
            C = U_batch' * U_approx
            σ = svdvals(C)
            
            # Principal angles from singular values
            angle_errs[r] = sqrt(r - sum(σ.^2))
        end
        l = scatterlines!(
            ax, 1:rmax, angle_errs, 
            marker=marker_styles[i], markersize=(35-(i-1)*2),
            linestyle=line_styles[i], linewidth=7,
        )
        i += 1
        push!(lines, l)
        push!(labels, Algo)
    end
    axislegend(ax, 
        lines, labels,
        position=:rb,
        labelsize=30,
        patchsize=(80,20)
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/subspace_angle_error.pdf"), fig)
end


#====================================================#
## Plot the subspace angle errors between the bases ##
#====================================================#
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 600))      
    ax = Axis(
        fig[1, 1], xlabel=L"singular value index, $i$", 
        ylabel=L"subspace angle error, $|\cos(\theta_i)-1|$",
        yscale=log10, titlesize=30, xticks=vcat(1, 5:5:rmax), 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
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
                angle_errs = abs.(
                    svdvals(basis.Q[:,1:r]' * bases["batch"]["Vr"][:,1:r]) .- 1
                )
                l = scatterlines!(
                    ax, 1:rmax, angle_errs, 
                    marker=marker_styles[i], markersize=(35-(i-1)*2),
                    linestyle=line_styles[i], linewidth=7,
                )
                push!(lines, l)
                B = n ÷ blk
                push!(labels, L"MergingSketchy ($B=%$B$)")
            end
        else
            basis = bases[algo]
            angle_errs = abs.(
                svdvals(basis.Q[:,1:rmax]' * bases["batch"]["Vr"][:,1:rmax]) .- 1
            )
            l = scatterlines!(
                ax, 1:rmax, angle_errs, 
                marker=marker_styles[i], markersize=(35-(i-1)*2),
                linestyle=line_styles[i], linewidth=7,
            )
            i += 1
            push!(lines, l)
            push!(labels, Algo)
        end
    end
    axislegend(ax, 
        lines, labels,
        position=:rb,
        labelsize=30
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/subspace_angle_error.pdf"), fig)
end

#=============================#
## Plot the projection errors
#=============================#
proj_error = load(joinpath(FILEPATH, "data/results/projection_errors.jld2"))["rpe"]
rspan = load(joinpath(FILEPATH, "data/results/projection_errors.jld2"))["rspan"]
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 600))
    ax = Axis(
        fig[1, 1], xlabel=L"reduced dimension, $r$", 
        ylabel="mean projection error",
        xticks=vcat(1, rspan),
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25,
    )
    lines = []
    algos = ["batch", "baker", "brand", "sketchy"]
    marker_styles = [:rect, :diamond, :cross, :circle, :rect]
    line_styles = [:solid, :dot, :dash, :dashdot, :dashdotdot]
    colors = vcat(:black, Makie.wong_colors()[1:4])
    i = 1
    for algo in algos
        l = scatterlines!(
            ax, rspan, proj_error[algo]["all"],
            marker=marker_styles[i], markersize=(35-(i-1)*2),
            linestyle=line_styles[i], linewidth=7, color=colors[i],
        )
        push!(lines, l)
        i += 1
    end
    axislegend(
        ax, lines, algos,
        position=:rt,
        labelsize=30,
        patchsize=(80,20)
    )
    display(fig)
    save(joinpath(FILEPATH, "plots/projection_errors.pdf"), fig)
end

#=============================#
## Plot the streaming errors ##
#=============================#
streaming_errors = load(
    joinpath(FILEPATH, "data/results/streaming_errors.jld2"),
    "stream_errors"
)

with_theme(theme_latexfonts()) do 
    fig = Figure(size=(800, 600))
    ytick_vals = 10.0 .^ (-14:2:0)
    ax = Axis(
        fig[1, 1], 
        xlabel=L"$k$-th stream", 
        ylabel="relative streaming error",
        yscale=log10,
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25,
        yticks=(ytick_vals, [L"10^{%$(Int(log10(y)))}" for y in ytick_vals]),
    )
    
    lines = []
    labels = ["RLS", "iQRRLS", "QRRLS"]
    methods = ["rls", "iqrrls", "qrrls"]
    marker_styles = [:rect, :star5, :hexagon]
    line_styles = [:dot, :dash, :dashdot]
    colors = Makie.wong_colors()[1:3]
    
    for (i, method) in enumerate(methods)
        num_streams = length(streaming_errors[Symbol(method)])
        l = scatterlines!(
            ax, 1:num_streams, streaming_errors[Symbol(method)],
            marker=marker_styles[i], markersize=(35-(i-1)*5),
            linestyle=line_styles[i], linewidth=7,
            color=colors[i]
        )
        push!(lines, l)
    end
    
    axislegend(
        ax, lines, labels,
        position=:lb,
        labelsize=30,
        patchsize=(80,20)
    )
    
    display(fig)
    save(joinpath(FILEPATH, "plots/streaming_errors.pdf"), fig)
end


#==========================#
## Plot the power spectra ##
#==========================#
# Load the power spectrum results
power_data = load(joinpath(FILEPATH, "data/results/power_spectrum.jld2"))

# Extract all the data
k_orig_train = power_data["k_orig_train"]
Pv_orig_train = power_data["Pv_orig_train"]
Pb_orig_train = power_data["Pb_orig_train"]

k_rom_train = power_data["k_rom_train"]
Pv_rom_train = power_data["Pv_rom_train"]
Pb_rom_train = power_data["Pb_rom_train"]

k_orig_test = power_data["k_orig_test"]
Pv_orig_test = power_data["Pv_orig_test"]
Pb_orig_test = power_data["Pb_orig_test"]

k_rom_test = power_data["k_rom_test"]
Pv_rom_test = power_data["Pv_rom_test"]
Pb_rom_test = power_data["Pb_rom_test"]

# Plot velocity and magnetic power spectra
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1200, 1000))
    
    # Velocity Power Spectrum - Training
    ax1 = Axis(
        fig[1, 1], 
        xlabel=L"Wavenumber $k$", 
        ylabel=L"Velocity Power,  $\frac{1}{2}\langle |u|^2 \rangle$",
        title="Velocity Power Spectrum - Training",
        xscale=log10, yscale=log10,
        titlesize=25, xlabelsize=22, ylabelsize=22, 
        xticklabelsize=18, yticklabelsize=18
    )

    c2 = Makie.wong_colors()[2]

    lines1 = []
    l1 = lines!(ax1, k_orig_train, Pv_orig_train, 
               color=:black, linewidth=4, label="Original")
    l2 = lines!(ax1, k_rom_train, Pv_rom_train, 
               color=c2, linewidth=3, linestyle=:dash, label="ROM")
    
    # Add Kolmogorov reference slope
    k_theory = k_orig_train[2:end-1]
    kolmogorov_slope = k_theory .^ (-5/3) .* Pv_orig_train[2] * k_orig_train[2]^(5/3)
    l3 = lines!(ax1, k_theory, kolmogorov_slope, 
               color=:gray, linewidth=3, linestyle=:dot, label=L"k^{-5/3}")
    
    push!(lines1, l1, l2, l3)
    # axislegend(ax1, lines1, ["Original", "ROM", L"k^{-5/3}"], 
    #            position=:lb, labelsize=28, patchsize=(60,20))
    
    # Velocity Power Spectrum - Testing
    ax2 = Axis(
        fig[1, 2], 
        xlabel=L"Wavenumber $k$", 
        title="Velocity Power Spectrum - Testing",
        xscale=log10, yscale=log10,
        titlesize=25, xlabelsize=22, ylabelsize=22, 
        xticklabelsize=18, yticklabelsize=18
    )
    
    lines2 = []
    l4 = lines!(ax2, k_orig_test, Pv_orig_test, 
               color=:black, linewidth=4, label="Original")
    l5 = lines!(ax2, k_rom_test, Pv_rom_test, 
               color=c2, linewidth=3, linestyle=:dash, label="ROM")
    
    # Add Kolmogorov reference slope
    k_theory_test = k_orig_test[2:end-1]
    kolmogorov_slope_test = k_theory_test .^ (-5/3) .* Pv_orig_test[2] * k_orig_test[2]^(5/3)
    l6 = lines!(ax2, k_theory_test, kolmogorov_slope_test, 
               color=:gray, linewidth=3, linestyle=:dot, label=L"k^{-5/3}")
    
    push!(lines2, l4, l5, l6)
    # axislegend(ax2, lines2, ["Original", "ROM", L"k^{-5/3}"], 
    #            position=:lb, labelsize=28, patchsize=(60,20))
    
    # Magnetic Power Spectrum - Training
    ax3 = Axis(
        fig[2, 1], 
        xlabel=L"Wavenumber $k$", 
        ylabel=L"Magnetic Power,  $\frac{1}{2}\langle |B|^2 \rangle$",
        title="Magnetic Power Spectrum - Training",
        xscale=log10, yscale=log10,
        titlesize=25, xlabelsize=22, ylabelsize=22, 
        xticklabelsize=18, yticklabelsize=18
    )
    
    lines3 = []
    l7 = lines!(ax3, k_orig_train, Pb_orig_train, 
               color=:black, linewidth=4, label="Original")
    l8 = lines!(ax3, k_rom_train, Pb_rom_train, 
               color=c2, linewidth=3, linestyle=:dash, label="ROM")
    
    # Add reference slope for magnetic field (often k^(-5/3) as well)
    mag_slope = k_theory .^ (-5/3) .* Pb_orig_train[2] * k_orig_train[2]^(5/3)
    l9 = lines!(ax3, k_theory, mag_slope, 
               color=:gray, linewidth=3, linestyle=:dot, label=L"k^{-5/3}")
    
    push!(lines3, l7, l8, l9)
    # axislegend(ax3, lines3, ["Original", "ROM", L"k^{-5/3}"], 
    #            position=:lb, labelsize=28, patchsize=(60,20))
    
    # Magnetic Power Spectrum - Testing
    ax4 = Axis(
        fig[2, 2], 
        xlabel=L"Wavenumber $k$", 
        title="Magnetic Power Spectrum - Testing",
        xscale=log10, yscale=log10,
        titlesize=25, xlabelsize=22, ylabelsize=22, 
        xticklabelsize=18, yticklabelsize=18
    )
    
    lines4 = []
    l10 = lines!(ax4, k_orig_test, Pb_orig_test, 
                color=:black, linewidth=4, label="Original")
    l11 = lines!(ax4, k_rom_test, Pb_rom_test, 
                color=c2, linewidth=3, linestyle=:dash, label="ROM")
    
    # Add reference slope
    mag_slope_test = k_theory_test .^ (-5/3) .* Pb_orig_test[2] * k_orig_test[2]^(5/3)
    l12 = lines!(ax4, k_theory_test, mag_slope_test, 
                color=:gray, linewidth=3, linestyle=:dot, label=L"k^{-5/3}")
    
    push!(lines4, l10, l11, l12)
    axislegend(ax4, lines4, ["Original", "Streaming-OpInf", L"k^{-5/3}"], 
               position=:lb, labelsize=32, patchsize=(60,20))
    
    display(fig)
    save(joinpath(FILEPATH, "plots/power_spectra_comparison.pdf"), fig)
end

## Create error analysis plot
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1000, 400))

    c1, c2 = Makie.wong_colors()[1:2]
    
    # Velocity error
    ax1 = Axis(
        fig[1, 1], 
        xlabel=L"Wavenumber $k$", 
        ylabel="Relative Error",
        title="Velocity Power Spectrum Error",
        xscale=log10, yscale=log10,
        titlesize=24, xlabelsize=22, ylabelsize=22, 
        xticklabelsize=18, yticklabelsize=18
    )
    
    # Calculate relative errors
    vel_error_train = abs.(Pv_rom_train .- Pv_orig_train) ./ (Pv_orig_train .+ 1e-15)
    vel_error_test = abs.(Pv_rom_test .- Pv_orig_test) ./ (Pv_orig_test .+ 1e-15)
    
    lines!(ax1, k_orig_train, vel_error_train, 
           color=c1, linewidth=3, label="Training")
    lines!(ax1, k_orig_test, vel_error_test, 
           color=c1, linewidth=3, label="Testing", linestyle=:dash)
    
    axislegend(ax1, position=:rb, labelsize=22, patchsize=(60,20))
    
    # Magnetic error
    ax2 = Axis(
        fig[1, 2], 
        xlabel=L"Wavenumber $k$", 
        ylabel="Relative Error",
        title="Magnetic Power Spectrum Error",
        xscale=log10, yscale=log10,
        titlesize=24, xlabelsize=22, ylabelsize=22, 
        xticklabelsize=18, yticklabelsize=18
    )
    
    # Calculate relative errors
    mag_error_train = abs.(Pb_rom_train .- Pb_orig_train) ./ (Pb_orig_train .+ 1e-15)
    mag_error_test = abs.(Pb_rom_test .- Pb_orig_test) ./ (Pb_orig_test .+ 1e-15)
    
    lines!(ax2, k_orig_train, mag_error_train, 
           color=c2, linewidth=3, label="Training")
    lines!(ax2, k_orig_test, mag_error_test, 
           color=c2, linewidth=3, label="Testing", linestyle=:dash)
    
    axislegend(ax2, position=:rb, labelsize=22, patchsize=(60,20))
    
    display(fig)
    save(joinpath(FILEPATH, "plots/power_spectra_errors.pdf"), fig)
end



#=================#
## Plot the 3PCF ##
#=================#
include(joinpath(FILEPATH, "analysis.jl"))
# Load all the 3PCF data
npcf_files = readdir(joinpath(FILEPATH, "data/results"), join=true)
npcf_files = filter(f -> occursin("3pcf", f), npcf_files)
npcf3 = Dict()
for (i, npcf_file) in enumerate(npcf_files)
    npcf = load(npcf_file)
    if i == 1
        for (key, value) in npcf
            npcf3[key] = value
        end
    else
        for (key, value) in npcf
            npcf3[key] .+= value
        end
    end 
end
for key in keys(npcf3)
    npcf3[key] ./= length(npcf_files)  # Average over all files
end

## Combined plot: 3PCF coefficients and relative errors
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1800, 1700))
    
    # Data keys and row labels for main heatmaps
    data_keys = ["npcf3_orig", "npcf3_rom_train", "npcf3_orig_test", 
                 "npcf3_rom_test"]
    row_labels = ["Original (Train)", "Streaming-OpInf (Train)", 
                  "Original (Test)", "Streaming-OpInf (Test)"]
    
    # Process all matrices to find global min/max for colorbar
    all_processed = []
    for data_key in data_keys
        for ell in 1:6  # ℓ = 0, 1, 2, 3, 4, 5
            processed = process_3pcf_matrix(npcf3[data_key], ell)
            push!(all_processed, processed)
        end
    end
    
    # Find global min/max for consistent colorbar
    global_min = minimum([minimum(m) for m in all_processed])
    global_max = maximum([maximum(m) for m in all_processed])
    color_limit = max(abs(global_min), abs(global_max))
    
    # Create main 3PCF heatmaps (rows 1-4)
    heatmaps = []
    for (row_idx, data_key) in enumerate(data_keys)
        for ell in 1:6  # ℓ = 0, 1, 2, 3, 4, 5
            ax = Axis(
                fig[row_idx, ell],
                xlabel = "",
                ylabel = ell == 1 ? L"%$(row_labels[row_idx]) \n $r_2$ bin" : "",
                title = row_idx == 1 ? L"$\ell = %$(ell-1)$" : "",
                titlesize = 24,
                xlabelsize = 18,
                ylabelsize = 18,
                xticklabelsize = 14,
                yticklabelsize = 14,
                xticksvisible = false,
                yticksvisible = ell == 1 ? true : false,
                xticklabelsvisible = false,
                yticklabelsvisible = ell == 1 ? true : false,
                yreversed = true,
                # ylabelpadding = ell == 1 ? 20.0 : 3.0,
                aspect = 1
            )
            
            # Process matrix for this multipole
            processed_matrix = process_3pcf_matrix(npcf3[data_key], ell)
            
            # Create heatmap
            hm = heatmap!(
                ax, processed_matrix,
                colormap = CairoMakie.Reverse(:RdBu),
                colorrange = (-color_limit, color_limit)
            )
            
            # Store first heatmap for colorbar reference
            if row_idx == 1 && ell == 1
                push!(heatmaps, hm)
            end
            
            # Set ticks to show bin indices
            n_bins = size(processed_matrix, 1)
            tick_positions = 1:n_bins
            tick_labels = string.(n_bins:-1:1)
            ax.xticks = (tick_positions, tick_labels)
            ax.yticks = (tick_positions, tick_labels)
        end
    end
    
    # Calculate relative errors for error plots (rows 6-7)
    error_train = []
    error_test = []
    
    for ell in 1:6
        # Training error
        orig_train = process_3pcf_matrix(npcf3["npcf3_orig"], ell)
        rom_train = process_3pcf_matrix(npcf3["npcf3_rom_train"], ell)
        err_train = abs.(rom_train .- orig_train) # ./ (abs.(orig_train) .+ 1e-10)
        push!(error_train, err_train)
        
        # Testing error
        orig_test = process_3pcf_matrix(npcf3["npcf3_orig_test"], ell)
        rom_test = process_3pcf_matrix(npcf3["npcf3_rom_test"], ell)
        err_test = abs.(rom_test .- orig_test) # ./ (abs.(orig_test) .+ 1e-10)
        push!(error_test, err_test)
    end
    
    # Find global max for error colorbar
    error_max = maximum([maximum(e) for e in vcat(error_train, error_test)])
    
    # Plot training errors (row 6)
    for ell in 1:6
        ax = Axis(
            fig[6, ell],
            xlabel = "",
            ylabel = ell == 1 ? L"Training Error \n $r_2$ bin" : "",
            titlesize = 22,
            xlabelsize = 18,
            ylabelsize = 18,
            xticklabelsize = 14,
            yticklabelsize = 14,
            xticksvisible = false,
            yticksvisible = ell == 1 ? true : false,
            xticklabelsvisible = false,
            yticklabelsvisible = ell == 1 ? true : false,
            yreversed = true,
            # ylabelpadding = ell == 1 ? 23.0 : 3.0,
            aspect = 1
        )
        
        hm = heatmap!(
            ax, error_train[ell],
            colormap = :plasma,
            colorrange = (0, error_max)
        )
        
        n_bins = size(error_train[ell], 1)
        tick_positions = 1:n_bins
        tick_labels = string.(1:(n_bins))
        ax.xticks = (tick_positions, tick_labels)
        ax.yticks = (tick_positions, tick_labels)
    end
    
    # Plot testing errors (row 7)
    hm_err = nothing
    for ell in 1:6
        ax = Axis(
            fig[7, ell],
            xlabel = L"$r_1$ bin",
            ylabel = ell == 1 ? L"Testing Error \n $r_2$ bin" : "",
            titlesize = 22,
            xlabelsize = 18,
            ylabelsize = 18,
            xticklabelsize = 14,
            yticklabelsize = 14,
            xticksvisible = true,
            yticksvisible = ell == 1 ? true : false,
            xticklabelsvisible = true,
            yticklabelsvisible = ell == 1 ? true : false,
            yreversed = true,
            # xlabelpadding = ell != 1 ? 25.0 : 3.0,
            # ylabelpadding = ell == 1 ? 23.0 : 3.0,
            aspect = 1
        )
        
        hm_err = heatmap!(
            ax, error_test[ell],
            colormap = :plasma,
            colorrange = (0, error_max)
        )
        
        n_bins = size(error_test[ell], 1)
        tick_positions = 1:n_bins
        tick_labels = string.(1:(n_bins))
        ax.xticks = (tick_positions, tick_labels)
        ax.yticks = (tick_positions, reverse(tick_labels))
    end
    
    # Add colorbars
    cb1 = Colorbar(
        fig[1:4, 7], 
        heatmaps[1],
        label = "Normalized 3PCF",
        labelsize = 20,
        ticklabelsize = 16,
        width = 25
    )
    
    cb2 = Colorbar(
        fig[6:7, 7], 
        hm_err,
        label = "Absolute Error",
        labelsize = 18,
        ticklabelsize = 14,
        width = 25
    )
    
    # Add section titles
    Label(
        fig[0, 1:6], 
        "3-Point Correlation Function Normalized by Standard Deviation for Multipole",
        fontsize = 28,
        font = "TeX Gyre Termes Bold"
    )
    
    # Add a separator label between sections (using row 5)
    Label(
        fig[5, 1:6], 
        "Absolute Errors",
        fontsize = 24,
        font = "TeX Gyre Termes Bold"
    )
    
    # Adjust layout
    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 15)
    
    display(fig)
    save(joinpath(FILEPATH, "plots/3pcf_combined.pdf"), fig)
end

#=====================#
## 3PCF coefficients ##
#=====================#
npcf_files = readdir(joinpath(FILEPATH, "data/results"), join=true)
npcf_files = filter(f -> occursin("3pcf", f), npcf_files)

# Filter files with time values smaller than t50
# filtered_files = filter(npcf_files) do file
#     # Extract the time number from the filename
#     match_result = match(r"3pcf_t(\d+)\.jld2", basename(file))
#     if match_result !== nothing
#         time_value = parse(Int, match_result.captures[1])
#         return time_value > 50
#     end
#     return false
# end
# npcf_files = filtered_files

zeta_l_orig_train = nothing
zeta_l_rom_train  = nothing
zeta_l_orig_test  = nothing
zeta_l_rom_test   = nothing
for (i, npcf_file) in enumerate(npcf_files)
    npcf = load(npcf_file)
    if i == 1
        zeta_l_orig_train = project_to_legendre(npcf["npcf3_orig"])
        zeta_l_rom_train  = project_to_legendre(npcf["npcf3_rom_train"])
        zeta_l_orig_test  = project_to_legendre(npcf["npcf3_orig_test"])
        zeta_l_rom_test   = project_to_legendre(npcf["npcf3_rom_test"])
    else
        foo = project_to_legendre(npcf["npcf3_orig"])
        bar = project_to_legendre(npcf["npcf3_rom_train"])
        baz = project_to_legendre(npcf["npcf3_orig_test"])
        qux = project_to_legendre(npcf["npcf3_rom_test"])
        for ell in keys(zeta_l_orig_train)
            zeta_l_orig_train[ell] .+= foo[ell]
            zeta_l_rom_train[ell]  .+= bar[ell]
            zeta_l_orig_test[ell]  .+= baz[ell]
            zeta_l_rom_test[ell]   .+= qux[ell]
        end
    end
end
for ell in keys(zeta_l_orig_train)
    zeta_l_orig_train[ell] ./= length(npcf_files)
    zeta_l_rom_train[ell]  ./= length(npcf_files)
    zeta_l_orig_test[ell]  ./= length(npcf_files)
    zeta_l_rom_test[ell]   ./= length(npcf_files)
end

## Combined plot: Legendre coefficients and relative errors
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1800, 1700))
    
    # Data dictionaries and row labels for main heatmaps
    data_dicts = [zeta_l_orig_train, zeta_l_rom_train, zeta_l_orig_test, 
                  zeta_l_rom_test]
    row_labels = ["Original (Train)", "Streaming-OpInf (Train)", 
                  "Original (Test)", "Streaming-OpInf (Test)"]
    
    # Process all matrices to find global min/max for colorbar
    all_processed = []
    for data_dict in data_dicts
        for ell in 0:5  # ℓ = 0, 1, 2, 3, 4, 5
            processed = process_legendre_matrix(data_dict, ell)
            push!(all_processed, processed)
        end
    end
    
    # Find global min/max for consistent colorbar
    global_min = minimum([minimum(m) for m in all_processed])
    global_max = maximum([maximum(m) for m in all_processed])
    color_limit = max(abs(global_min), abs(global_max))
    
    # Create main Legendre coefficient heatmaps (rows 1-4)
    heatmaps = []
    for (row_idx, data_dict) in enumerate(data_dicts)
        for ell in 0:5  # ℓ = 0, 1, 2, 3, 4, 5
            col = ell + 1
            ax = Axis(
                fig[row_idx, col],
                xlabel = "",
                ylabel = ell == 0 ? L"%$(row_labels[row_idx]) \n $r_2$ bin" : "",
                title = row_idx == 1 ? L"$\ell = %$ell$" : "",
                titlesize = 24,
                xlabelsize = 18,
                ylabelsize = 18,
                xticklabelsize = 14,
                yticklabelsize = 14,
                xticksvisible = false,
                yticksvisible = ell == 0 ? true : false,
                xticklabelsvisible = false,
                yticklabelsvisible = ell == 0 ? true : false,
                yreversed = true,
                aspect = 1
            )
            
            # Process matrix for this multipole
            processed_matrix = process_legendre_matrix(data_dict, ell)
            
            # Create heatmap
            hm = heatmap!(
                ax, processed_matrix,
                colormap = CairoMakie.Reverse(:RdBu),
                colorrange = (-color_limit, color_limit)
            )
            
            # Store first heatmap for colorbar reference
            if row_idx == 1 && ell == 0
                push!(heatmaps, hm)
            end
            
            # Set ticks to show bin indices
            n_bins = size(processed_matrix, 1)
            tick_positions = 1:n_bins
            tick_labels = string.(n_bins:-1:1)
            ax.xticks = (tick_positions, tick_labels)
            ax.yticks = (tick_positions, tick_labels)
        end
    end
    
    # Calculate relative errors for error plots (rows 6-7)
    error_train = []
    error_test = []
    
    for ell in 0:5
        # Training error
        orig_train = process_legendre_matrix(zeta_l_orig_train, ell)
        rom_train = process_legendre_matrix(zeta_l_rom_train, ell)
        err_train = abs.(rom_train .- orig_train) # ./ (abs.(orig_train))
        push!(error_train, err_train)
        
        # Testing error
        orig_test = process_legendre_matrix(zeta_l_orig_test, ell)
        rom_test = process_legendre_matrix(zeta_l_rom_test, ell)
        err_test = abs.(rom_test .- orig_test) # ./ (abs.(orig_test))
        push!(error_test, err_test)
    end
    
    # Find global max for error colorbar
    error_max = maximum([maximum(e) for e in vcat(error_train, error_test)])
    
    # Plot training errors (row 6)
    for ell in 0:5
        col = ell + 1
        ax = Axis(
            fig[6, col],
            xlabel = "",
            ylabel = ell == 0 ? L"Training Error \n $r_2$ bin" : "",
            titlesize = 22,
            xlabelsize = 18,
            ylabelsize = 18,
            xticklabelsize = 14,
            yticklabelsize = 14,
            xticksvisible = false,
            yticksvisible = ell == 0 ? true : false,
            xticklabelsvisible = false,
            yticklabelsvisible = ell == 0 ? true : false,
            yreversed = true,
            aspect = 1,
        )
        
        hm = heatmap!(
            ax, error_train[ell + 1],
            colormap = :plasma,
            colorrange = (0, error_max),
        )
        
        n_bins = size(error_train[ell + 1], 1)
        tick_positions = 1:n_bins
        tick_labels = string.(1:n_bins)
        ax.xticks = (tick_positions, tick_labels)
        ax.yticks = (tick_positions, tick_labels)
    end
    
    # Plot testing errors (row 7)
    hm_err = nothing
    for ell in 0:5
        col = ell + 1
        ax = Axis(
            fig[7, col],
            xlabel = L"$r_1$ bin",
            ylabel = ell == 0 ? L"Testing Error \n $r_2$ bin" : "",
            titlesize = 22,
            xlabelsize = 18,
            ylabelsize = 18,
            xticklabelsize = 14,
            yticklabelsize = 14,
            xticksvisible = true,
            yticksvisible = ell == 0 ? true : false,
            xticklabelsvisible = true,
            yticklabelsvisible = ell == 0 ? true : false,
            yreversed = true,
            aspect = 1,
        )
        
        hm_err = heatmap!(
            ax, error_test[ell + 1],
            colormap = :plasma,
            colorrange = (0, error_max),
        )
        
        n_bins = size(error_test[ell + 1], 1)
        tick_positions = 1:n_bins
        tick_labels = string.(1:n_bins)
        ax.xticks = (tick_positions, tick_labels)
        ax.yticks = (tick_positions, reverse(tick_labels))
    end
    
    # Add colorbars
    cb1 = Colorbar(
        fig[1:4, 7], 
        heatmaps[1],
        label = "Normalized Legendre Coefficients",
        labelsize = 20,
        ticklabelsize = 16,
        width = 25
    )
    
    cb2 = Colorbar(
        fig[6:7, 7], 
        hm_err,
        label = "Absolute Error",
        labelsize = 18,
        ticklabelsize = 14,
        width = 25,
    )
    
    # Add section titles
    Label(
        fig[0, 1:6], 
        "3PCF Legendre Coefficients Normalized by Standard Deviation for Multipole",
        fontsize = 28,
        font = "TeX Gyre Termes Bold"
    )
    
    # Add a separator label between sections (using row 5)
    Label(
        fig[5, 1:6], 
        "Absolute Errors",
        fontsize = 24,
        font = "TeX Gyre Termes Bold",
    )
    
    # Adjust layout
    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 15)
    
    display(fig)
    save(joinpath(FILEPATH, "plots/3pcf_legendre_combined.pdf"), fig)
end