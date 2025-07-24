"""
Supernova 64^3 example: Computing the energy spectrum
"""

#=================#
## Load packages ##
#=================#
using LinearAlgebra
using FileIO
using JLD2
using Statistics

#=============#
## Load data ##
#=============#
# Get paths and data file name
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(), "Two-Pass_Streaming-OpInf/mhd64") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/mhd64")
DATAPATH = "../../../../DATA/THE_WELL/mhd64"
train_files = readdir(DATAPATH, join=true)
fn = train_files[1]

# Include the data sourcing module for data access
include(joinpath(FILEPATH, "datasource.jl"))

# Load data source 
ds = DataSource(fn)
nx, ny, nz, n_fields, n_time, n_traj = ds.dims
nxyz = nx * ny * nz
n = n_time * n_traj

#=============#
## Load data ##
#=============#
preprocessed_file = joinpath(FILEPATH, "data/preprocessed_data.jld2")
X = load(preprocessed_file, "X")

#===========================#
## Compute singular values ##
#===========================#
singular_values = Dict(
    fld => svdvals(X[fld]) for fld in ds.fields
)

#===============================#
## Check the energy retainment ##
#===============================#
function check_energy_retainment(svals, target=0.999)
    energy = sum(svals.^2)
    energy_ret = cumsum(svals.^2) / energy
    r = findfirst(x -> x > target, energy_ret) + 1
    return energy_ret, r
end

spectrum = Dict(fn => zeros(length(singular_values[fn])) for fn in ds.fields)
target_r = Dict(fn => 0 for fn in ds.fields)

target_energy = 0.85
for fld in ds.fields
    svals = singular_values[fld]
    spectrum[fld], target_r[fld] = check_energy_retainment(svals, target_energy)
end

# Save target ranks
target_r_file = joinpath(FILEPATH, "data/target_ranks.jld2")
save(target_r_file, "target_ranks", target_r)


#============================#
## Plot the energy spectrum ##
#============================#
using CairoMakie

with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1400, 1000))
    ax1 = Axis(
        fig[2, 1], # xlabel=L"singular value index, $i$", 
        ylabel="energy retainment",
        limits=(-50, length(spectrum["rho"])+10, nothing, nothing),
        # xgridvisible=false, ygridvisible=false,
        topspinevisible=false, rightspinevisible=false,
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25,
        yticks=(0:0.1:1, ["0%", "10%", "20%", "30%", "40%", 
                          "50%", "60%", "70%", "80%", "90%", "100%"])
    )

    colors = Dict(
        key => color for (key, color) in zip(
            ds.fields, Makie.categorical_colors(
                :seaborn_colorblind, length(ds.fields)
            )
        )
    )

    # Plot energy retainment for each field
    for fld in ds.fields
        svals = singular_values[fld]
        lines!(ax1, 1:length(svals), spectrum[fld], 
               linewidth=4, color=colors[fld])
    end

    # Add horizontal line at 99.9% energy retention
    hlines!(ax1, [target_energy], color=:black, linestyle=:dash, linewidth=4)

    ax2 = Axis(
        fig[3, 1], xlabel="singular value index", 
        # xgridvisible=false, ygridvisible=false,
        ylabel="singular value", yscale=log10,
        topspinevisible=false, rightspinevisible=false, ylabelpadding=30,
        limits=(-50, length(spectrum["rho"])+10, nothing, nothing),
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25,
    )

    for fld in ds.fields
        svals = singular_values[fld]
        lines!(ax2, 1:length(svals), svals, label=fld, linewidth=4, 
               color=colors[fld])
    end

    # Add vertical lines for target ranks
    for (key, r) in target_r
        if r > 0
            vlines!(ax1, [r], linestyle=:dash, color=colors[key],
                    linewidth=4, label="target rank for $key")
            vlines!(ax2, [r], linestyle=:dash, color=colors[key],
                    linewidth=4, label="target rank for $key")
        end
    end

    # Add legend
    # Legend(fig[:, 2], ax1, framevisible=false, labelsize=25)
    line_elements = [
        [LineElement(color=colors[fn], linewidth=5)]
        for fn in ds.fields
    ]
    labels = [
        fld == "rho" ? L"$\rho$" : 
        fld == "z" ? L"$\zeta$" :
        occursin("m", fld) ? L"$m_{%$(fld[2])}$" :
        occursin("B", fld) ? L"$B_{%$(fld[2])}$" :
        L"$%$(fld)$" for fld in ds.fields
    ]
    Legend(fig[1, :],
        line_elements, labels,
        framevisible=false, patchsize=(70, 20),
        labelsize=40, rowgap=10, colgap=50, orientation=:horizontal,
    )
    
    # Display figure
    display(fig)
    save(joinpath(FILEPATH, "plots/energy_spectrum.png"), fig, px_per_inch=200)
end