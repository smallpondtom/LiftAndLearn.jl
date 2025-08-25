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
# train_files = readdir(DATAPATH, join=true)
# fn = train_files[1]

# # Include the data sourcing module for data access
# include(joinpath(FILEPATH, "datasource.jl"))

# # Load data source 
# ds = DataSource(fn)
# nx, ny, nz, n_fields, n_time, n_traj = ds.dims
# nxyz = nx * ny * nz
# n = n_time * n_traj


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