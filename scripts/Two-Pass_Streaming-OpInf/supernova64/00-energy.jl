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
           joinpath(pwd(), "Two-Pass_Streaming-OpInf/supernova64") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/supernova64")
DATAPATH = "../../../../DATA/THE_WELL/supernova_explosion_64/train"
train_files = readdir(DATAPATH, join=true)
fn = train_files[1]

# Include the data sourcing module for data access
include(joinpath(FILEPATH, "datasource.jl"))

# Load data source 
ds = DataSource(fn)
nx, ny, nz, n_fields, n_time, n_traj = ds.dims
nxyz = nx * ny * nz

# Flag for float64
USE_FLOAT64 = false

#====================#
## Preprocess data  ##
#====================#
n = n_time * n_traj
if USE_FLOAT64
    Xp = Float64.(reshape(ds["p"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xz = Float64.(reshape(ds["z"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xu = Float64.(reshape(ds["u"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xv = Float64.(reshape(ds["v"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xw = Float64.(reshape(ds["w"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
else
    Xp = Float32.(reshape(ds["p"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xz = Float32.(reshape(ds["z"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xu = Float32.(reshape(ds["u"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xv = Float32.(reshape(ds["v"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
    Xw = Float32.(reshape(ds["w"][1:nxyz, 1:n_time, 1:n_traj], nxyz, n))
end

## Save unscaled/unshifted data
original_file = joinpath(FILEPATH, "data/original_data.jld2")
# if !isfile(original_file)
    save(original_file, 
        "X", Dict(
            "p" => Xp, "z" => Xz, "u" => Xu, "v" => Xv, "w" => Xw,
            "all" => vcat(Xp, Xz, Xu, Xv, Xw)
        ),
    )
# end

## Center the data
Xpbar = mean(Xp, dims=2)
Xzbar = mean(Xz, dims=2)
Xubar = mean(Xu, dims=2)
Xvbar = mean(Xv, dims=2)
Xwbar = mean(Xw, dims=2)

Xp .-= Xpbar
Xz .-= Xzbar
Xu .-= Xubar
Xv .-= Xvbar
Xw .-= Xwbar

save(joinpath(FILEPATH, "data/mean.jld2"),
    "mean", Dict(
        "p" => Xpbar, "z" => Xzbar, 
        "u" => Xubar, "v" => Xvbar, "w" => Xwbar
    )
)

## Normalize to [0,1] with (row-wise) minmax scaling
function minmax_shift_scale!(X)
    X_min = minimum(X, dims=2)
    X_max = maximum(X, dims=2)
    X .-= X_min 
    X ./= (X_max - X_min)
    return X_min, X_max
end
Xp_min, Xp_max = minmax_shift_scale!(Xp)
Xz_min, Xz_max = minmax_shift_scale!(Xz)
Xu_min, Xu_max = minmax_shift_scale!(Xu)
Xv_min, Xv_max = minmax_shift_scale!(Xv)
Xw_min, Xw_max = minmax_shift_scale!(Xw)

## Save preprocessed data
preprocessed_file = joinpath(FILEPATH, "data/preprocessed_data.jld2")
save(preprocessed_file, 
    "X", Dict(
        "p" => Xp, "z" => Xz, "u" => Xu, "v" => Xv, "w" => Xw
    )
)

## Save shift and scaling 
shift_scale_file = joinpath(FILEPATH, "data/minmax.jld2")
save(shift_scale_file, 
    "shift", Dict(
       "p" => Xp_min, "z" => Xz_min,
       "u" => Xu_min, "v" => Xv_min, "w" => Xw_min
    ),
    "scale", Dict(
       "p" => Xp_max - Xp_min, "z" => Xz_max - Xz_min,
       "u" => Xu_max - Xu_min, "v" => Xv_max - Xv_min, "w" => Xw_max - Xw_min
    ),
)

#===========================#
## Compute singular values ##
#===========================#
singular_values = Dict(fn => zeros(n) for fn in ds.fields)
sp = svdvals(Xp)
sz = svdvals(Xz)
su = svdvals(Xu)
sv = svdvals(Xv)
sw = svdvals(Xw)

#===============================#
## Check the energy retainment ##
#===============================#
function check_energy_retainment(svals, target=0.999)
    energy = sum(svals.^2)
    energy_ret = cumsum(svals.^2) / energy
    r = findfirst(x -> x > target, energy_ret) + 1
    return energy_ret, r
end

spectrum = Dict(fn => zeros(length(sp)) for fn in ds.fields)
target_r = Dict(fn => 0 for fn in ds.fields)

target_energy = 0.99
for (field, svals) in zip(ds.fields, [sp, sz, su, sv, sw])
    spectrum[field], target_r[field] = check_energy_retainment(svals, target_energy)
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
    for (fld, svals) in zip(ds.fields, [sp, sz, su, sv, sw])
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

    for (fld, svals) in zip(ds.fields, [sp, sz, su, sv, sw])
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
    Legend(fig[1, :],
        line_elements,
        [L"$p$", L"$\zeta$", L"$u_x$", L"$u_y$", L"$u_z$"],
        framevisible=false, patchsize=(70, 20),
        labelsize=40, rowgap=10, colgap=50, orientation=:horizontal,
    )
    
    # Display figure
    display(fig)
    save(joinpath(FILEPATH, "plots/energy_spectrum.png"), fig, px_per_inch=200)
end