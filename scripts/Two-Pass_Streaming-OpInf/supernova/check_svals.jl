#=================#
## Load packages ##
#=================#
using LinearAlgebra
using HDF5

#=============#
## Load data ##
#=============#
FILEPATH = occursin("scripts", @__DIR__) ? 
           joinpath(@__DIR__, "Two-Pass_Streaming-OpInf/supernova") : 
           joinpath(@__DIR__, "scripts/Two-Pass_Streaming-OpInf/supernova")
DATAPATH = "../../../../DATA/THE_WELL/supernova_explosion_64/train"
train_files = readdir(DATAPATH, join=true)
fn = train_files[1]
Xp = h5read(fn, "t0_fields")["pressure"] # pressure field data
Xd = h5read(fn, "t0_fields")["density"]  # density field data
Xvel = h5read(fn, "t1_fields")["velocity"] # velocity field data
Xu = Xvel[1, :, :, :, :, :]
Xv = Xvel[2, :, :, :, :, :]
Xw = Xvel[3, :, :, :, :, :]
xspan = h5read(fn, "dimensions")["x"]
yspan = h5read(fn, "dimensions")["y"]
zspan = h5read(fn, "dimensions")["z"]
tspan = h5read(fn, "dimensions")["time"]
Nx, Ny, Nz, n_time, num_of_traj = size(Xp)
Xvel = nothing

#====================#
## Preprocess data  ##
#====================#
n = n_time * num_of_traj
Xp = reshape(Xp, Nx, Ny, Nz, n)
Xp = reshape(Xp, :, n)
Xd = reshape(Xd, Nx, Ny, Nz, n)
Xd = reshape(Xd, :, n)
Xz = 1 ./ Xd # specific volume
Xu = reshape(Xu, Nx, Ny, Nz, n)
Xu = reshape(Xu, :, n)
Xv = reshape(Xv, Nx, Ny, Nz, n)
Xv = reshape(Xv, :, n)
Xw = reshape(Xw, Nx, Ny, Nz, n)
Xw = reshape(Xw, :, n)

# Normalize to [0,1] with minmax scaling
function minmax_shift_scale(X)
    X_min = minimum(X, dims=2)
    X_max = maximum(X, dims=2)
    return (X .- X_min) ./ (X_max .- X_min)
end
Xp = minmax_shift_scale(Xp)
Xz = minmax_shift_scale(Xz)
Xu = minmax_shift_scale(Xu)
Xv = minmax_shift_scale(Xv)
Xw = minmax_shift_scale(Xw)

#=========================#
## Check singular values ##
#=========================#
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

field_names = [
    "pressure", "specific volume", "u-velocity", 
    "v-velocity", "w-velocity"
]

spectrum = Dict(
    "pressure" => zeros(length(sp)),
    "specific volume" => zeros(length(sz)),
    "u-velocity" => zeros(length(su)),
    "v-velocity" => zeros(length(sv)),
    "w-velocity" => zeros(length(sw))
)

target_r = Dict(
    "pressure" => 0,
    "specific volume" => 0,
    "u-velocity" => 0,
    "v-velocity" => 0,
    "w-velocity" => 0
)

target_energy = 0.95
for (key, svals) in zip(keys(spectrum), [sp, sz, su, sv, sw])
    spectrum[key], target_r[key] = check_energy_retainment(svals, target_energy)
end

#============================#
## Plot the energy spectrum ##
#============================#
using CairoMakie

with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1400, 1000))
    ax1 = Axis(
        fig[2, 1], # xlabel=L"singular value index, $i$", 
        ylabel="energy retainment",
        limits=(-50, length(spectrum["pressure"])+10, nothing, nothing),
        # xgridvisible=false, ygridvisible=false,
        topspinevisible=false, rightspinevisible=false,
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25,
        yticks=(0:0.1:1, ["0%", "10%", "20%", "30%", "40%", 
                          "50%", "60%", "70%", "80%", "90%", "100%"])
    )

    colors = Dict(
        key => color for (key, color) in zip(
            field_names, Makie.wong_colors()[1:5]
        )
    )

    # Plot energy retainment for each field
    for (key, svals) in zip(keys(spectrum), [sp, sz, su, sv, sw])
        lines!(ax1, 1:length(svals), spectrum[key], 
               linewidth=4, color=colors[key])
    end

    # Add horizontal line at 99.9% energy retention
    hlines!(ax1, [target_energy], color=:black, linestyle=:dash, linewidth=4)

    ax2 = Axis(
        fig[3, 1], xlabel="singular value index", 
        # xgridvisible=false, ygridvisible=false,
        ylabel="singular value", yscale=log10,
        topspinevisible=false, rightspinevisible=false, ylabelpadding=30,
        limits=(-50, length(spectrum["pressure"])+10, nothing, nothing),
        titlesize=30, xlabelsize=30, ylabelsize=30, 
        xticklabelsize=25, yticklabelsize=25,
    )

    for (key, svals) in zip(keys(spectrum), [sp, sz, su, sv, sw])
        lines!(ax2, 1:length(svals), svals, label=key, linewidth=4, 
               color=colors[key])
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
        for fn in field_names
    ]
    Legend(fig[1, :],
        line_elements,
        [L"$p$", L"$\zeta$", L"$u_x$", L"$u_y$", L"$u_z$"],
        framevisible=false, patchsize=(70, 20),
        labelsize=40, rowgap=10, colgap=50, orientation=:horizontal,
    )
    
    # Display figure
    display(fig)
    save(joinpath(FILEPATH, "plots/energy_spectrum.pdf"), fig, px_per_inch=200)
end