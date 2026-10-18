"""
ABL flow: Compute the energy spectrum
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using LinearAlgebra

#================================#
## Configure filepath for saving
#================================#
DATAPATH = "../../../../../DATA/NREL/ABL"
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"Two-Pass_Streaming-OpInf/ABL") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/ABL")
fn = "ABL_0_10000.h5"
datafile = joinpath(DATAPATH, fn)

#==========================================#
## Load struct to read data in HDF5 format 
#==========================================#
include(joinpath(FILEPATH, "datasource.jl"))

#========================#
## Additional functions
#========================#
include(joinpath(FILEPATH, "preprocess.jl"))

#=============================#
## Load the training dataset
#=============================#
ds = ChannelDataSource(
    datafile, ["z", "y", "x", "fields", "times"],
    x_subsample=3, 
    y_subsample=3, 
    z_subsample=2,
    time_downsample=10,
)
nz, ny, nx, n_fields, nt = ds.dims
nxyz = nz * ny * nx

#=============================#
## Load the mean and scalings
#=============================#
means  = load(joinpath(FILEPATH, "data/mean.jld2"))["xbar"]
shifts = load(joinpath(FILEPATH, "data/minmax.jld2"))["minmax"]["shifts"]
scales = load(joinpath(FILEPATH, "data/minmax.jld2"))["minmax"]["scales"]

#===========================#
## Compute singular values ##
#===========================#
singular_values = Dict()
for (i, fld) in enumerate(ds.fields)
    t1 = time()
    idx_start = nxyz * (i - 1) + 1
    idx_end = nxyz * i

    # Extract field data
    field_data = ds[fld][1:nxyz, 1:nt]
    @info "Loading field: $fld, size: $(size(field_data))"
    
    # Center the data
    centered_data = center!(field_data, means[idx_start:idx_end])
    # centered_data = center!(field_data, means)
    @info "Centered data for field: $fld"
    
    # Normalize the data
    scaled_data = scale!(centered_data, shifts[idx_start:idx_end], 
                         scales[idx_start:idx_end])
    # scaled_data = scale!(centered_data, shifts, scales)
    @info "Scaled data for field: $fld"
    
    # Compute singular values
    singular_values[fld] = svdvals(scaled_data)
    @info "Computed singular values for field: $fld"

    t2 = time()
    @info "Time taken for field $fld: $(t2 - t1) seconds"
    GC.gc()  # Force garbage collection to free memory
end

# save the singular values
save(joinpath(FILEPATH, "data/svdvals.jld2"), 
     "singular_values", singular_values)

#===============================#
## Check the energy retainment ##
#===============================#
function check_energy_retainment(svals, target=0.999)
    energy = sum(svals.^2)
    energy_ret = cumsum(svals.^2) / energy
    r = findfirst(x -> x > target, energy_ret) + 1
    return energy_ret, r
end

spectrum = Dict(fld => zeros(length(singular_values["u"])) for fld in ds.fields)
target_r = Dict(fld => 0 for fld in ds.fields)

target_energy = 0.99
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
        limits=(-50, length(spectrum["p"])+10, nothing, nothing),
        # xgridvisible=false, ygridvisible=false,
        topspinevisible=false, rightspinevisible=false,
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25,
        yticks=(0:0.1:1, ["0%", "10%", "20%", "30%", "40%", 
                          "50%", "60%", "70%", "80%", "90%", "100%"])
    )

    colors = Dict(
        key => color for (key, color) in zip(
            ds.fields, Makie.wong_colors()[1:5]
        )
    )

    # Plot energy retainment for each field
    for fld in ds.fields
        svals = spectrum[fld]
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
        limits=(-50, length(spectrum["p"])+10, nothing, nothing),
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