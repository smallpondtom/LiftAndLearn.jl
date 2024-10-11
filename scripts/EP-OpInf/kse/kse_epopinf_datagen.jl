"""
Kuramoto–Sivashinsky equation training and testing data generation for EP-OpInf
"""

#================#
## Load packages
#================#
using FileIO
using JLD2
using LaTeXStrings
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using Printf
using ProgressMeter
using Random
using SparseArrays
using Statistics

#========================#
## Load the model module
#========================#
using PolynomialModelReductionDataset: KuramotoSivashinskyModel, AbstractModel

#====================#
## Load LiftAndLearn
#====================#
using LiftAndLearn
const LnL = LiftAndLearn

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"EP-OpInf/") : joinpath(pwd(), "scripts/EP-OpInf/")

#======================#
## Setup the KSE model
#======================#
Ω = (0.0, 22.0)
Nx = 2^9; dt = 1e-3
KSE = KuramotoSivashinskyModel(
    spatial_domain=Ω, time_domain=(0.0, 300.0), Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt,
    diffusion_coeffs=1.0, BC=:periodic, conservation_type=:EP
)

#===========#
## Settings
#===========#
# Downsampling rate
DS = 100

# Down-sampled dimension of the time data
Tdim_ds = size(1:DS:KSE.time_dim, 1)  # downsampled time dimension

# Prune data to get only the chaotic region
prune_data = false
prune_idx = KSE.time_dim ÷ 2
t_prune = KSE.tspan[prune_idx-1:end]

# Number of random test inputs
num_test_ic = 50 # <-------- CHANGE THIS TO DESIRED NUMBER OF TEST INPUTS (!!! 50 FOR PAPER !!!)

#=====================#
## Initial conditions
#=====================#
# Parameters of the initial condition
ic_a = [0.8, 1.0, 1.2]
ic_b = [0.2, 0.4, 0.6]

num_ic_params = Int(length(ic_a) * length(ic_b))
L = KSE.spatial_domain[2] - KSE.spatial_domain[1]  # length of the domain

# Parameterized function for the initial condition
u0 = (a,b) -> a * cos.((2*π*KSE.xspan)/L) .+ b * cos.((4*π*KSE.xspan)/L)  # initial condition

#================#
## Generate data
#================#
# Data placeholders
Xtr = Vector{Matrix{Float64}}(undef, KSE.param_dim)  # training state data 
Rtr = Vector{Matrix{Float64}}(undef, KSE.param_dim)  # training derivative data
Vr = Vector{Matrix{Float64}}(undef, KSE.param_dim)  # POD basis
Σr = Vector{Vector{Float64}}(undef, KSE.param_dim)  # singular values 
op_fom_tr = Vector{LnL.Operators}(undef, KSE.param_dim)  # FOM operators 

@info "Generate the FOM system matrices and training data."
for i in eachindex(KSE.diffusion_coeffs)
    μ = KSE.diffusion_coeffs[i]

    # Generate the FOM system matrices (ONLY DEPENDS ON μ)
    A, F = KSE.finite_diff_model(KSE, μ)
    op_fom_tr[i] = LnL.Operators(A=A, A2u=F)

    # Store the training data 
    Xall = Vector{Matrix{Float64}}(undef, num_ic_params)
    Xdotall = Vector{Matrix{Float64}}(undef, num_ic_params)
    
    # Generate the data for all combinations of the initial condition parameters
    ic_combos = collect(Iterators.product(ic_a, ic_b))
    prog = Progress(length(ic_combos))
    Threads.@threads for (j, ic) in collect(enumerate(ic_combos))
        a, b = ic
        IC = u0(a, b)
        states = KSE.integrate_model(KSE.tspan, IC; linear_matrix=A, quadratic_matrix=F, 
                                     system_input=false, const_stepsize=true)
        if prune_data
            Xsave = states[:, prune_idx-1:end]

            tmp = states[:, prune_idx:end]
            Xall[j] = tmp[:, 1:DS:end]  # downsample data
            tmp = (states[:, prune_idx:end] - states[:, prune_idx-1:end-1]) / KSE.Δt
            Xdotall[j] = tmp[:, 1:DS:end]  # downsample data
        else
            Xsave = states

            tmp = states[:, 2:end]
            Xall[j] = tmp[:, 1:DS:end]  # downsample data
            tmp = (states[:, 2:end] - states[:, 1:end-1]) / KSE.Δt
            Xdotall[j] = tmp[:, 1:DS:end]  # downsample data
        end

        save_data = Dict(
            "X" => Xsave,
            "X_ds" => Xall[j],
            "Xdot_ds" => Xdotall[j],
            "IC" => IC,
            "DS" => DS,
        )
        save(joinpath(FILEPATH, "data/kse_train/00$(j)_a$(a)_b$(b).jld2"), save_data)
        next!(prog)
    end

    # Combine all initial condition data to form on big training data matrix
    Xtr[i] = reduce(hcat, Xall) 
    Rtr[i] = reduce(hcat, Xdotall)
    
    # Compute the POD basis from the training data
    tmp = svd(Xtr[i])
    Vr[i] = tmp.U
    Σr[i] = tmp.S

    @info "Finished generating training data for parameter $i"
end

#======================#
## Check energy levels
#======================#
nice_orders_all = Vector{Vector{Int}}(undef, KSE.param_dim)
energy_level_all = Vector{Vector{Float64}}(undef, KSE.param_dim)

for i in eachindex(KSE.diffusion_coeffs)
    nice_orders_all[i], energy_level_all[i] = LnL.choose_ro(Σr[i]; en_low=-12)
end
nice_orders = Int.(round.(mean(nice_orders_all)))
energy_level = mean(energy_level_all)

#=====================#
## Plot energy levels
#=====================#
p = plot(energy_level[1:nice_orders[end]], yaxis=:log10, label="energy loss", fontfamily="Computer Modern", lw=2,
    ylabel="relative energy loss from truncation", xlabel="number of retained modes", legend=:topright, majorgrid=true, grid=true)
plot!(nice_orders, energy_level[nice_orders], seriestype=:scatter, label=nothing, markersize=6)
for (i,order) in enumerate(nice_orders)
    if i == 8
        annotate!((order+4, energy_level[order], text(L"r_{\mathrm{max}}="*string(order), :bottom, 18, "Computer Modern")))
    end
end
yticks!(10.0 .^ (0:-2:-10))
plot!(fg_legend=:false)
plot!(guidefontsize=14, tickfontsize=12, titlefontsize=15,  legendfontsize=15, right_margin=2mm)
savefig(p, joinpath(FILEPATH, "plots/kse/kse_energy_levels.pdf"))

#========================#
## Select reduced orders
#========================#
ro = nice_orders[2:8]

#============#
## Save data
#============#
@info "Saving training setting information..."
tmp = joinpath(FILEPATH, "data/kse_epopinf_reduction_info.jld2")
save(tmp, "REDUCTION_INFO", 
    Dict(
    "Xtr" => Xtr, "Rtr" => Rtr, "Vr" => Vr, "op_fom_tr" => op_fom_tr, "Σr" => Σr, "ro" => ro,
    "num_ic_params" => num_ic_params, "ic_a" => ic_a, "ic_b" => ic_b
    )
)
@info "Done. Saved at $(tmp)"
@info "Saving model setting information..."
tmp = joinpath(FILEPATH, "data/kse_epopinf_model_setting.jld2")
save(tmp, "KSE", KSE)
@info "Done. Saved at $(tmp)"

#============================#
## Generate the test 1 data
#============================#
# Generate random initial condition parameters
ic_a_rand_in = (maximum(ic_a) - minimum(ic_a)) .* rand(num_test_ic) .+ minimum(ic_a)
ic_b_rand_in = (maximum(ic_b) - minimum(ic_b)) .* rand(num_test_ic) .+ minimum(ic_b)

for i in eachindex(KSE.diffusion_coeffs)
    μ = KSE.diffusion_coeffs[i]
    A = op_fom_tr[i].A
    F = op_fom_tr[i].A2u

    prog = Progress(num_test_ic)
    Threads.@threads for j in 1:num_test_ic  
        # generate test 1 data
        a = ic_a_rand_in[j]
        b = ic_b_rand_in[j]
        IC = u0(a,b)
        Xtest_in = KSE.integrate_model(KSE.tspan, IC; linear_matrix=A, quadratic_matrix=F, 
                                    system_input=false, const_stepsize=true)

        save_data = Dict(
            "X" => Xtest_in,
            "IC" => IC,
            "a" => a,
            "b" => b,
        )
        a_str = @sprintf("%1.5f", a)
        b_str = @sprintf("%1.5f", b)
        save(joinpath(FILEPATH, "data/kse_test1/00$(j)_a$(a_str)_b$(b_str).jld2"), save_data)
        next!(prog)
    end

    @info "Finished generating test 1 data for parameter $i"
end

#============================#
## Generate the test 2 data
#============================#
# Generate random initial condition parameters
ic_a_out = [0.0, 2.0]
ic_b_out = [0.0, 0.8]
ϵ=1e-2
half_num_test_ic = Int(num_test_ic/2)
ic_a_rand_out = ((minimum(ic_a) - ϵ) - ic_a_out[1]) .* rand(half_num_test_ic) .+ ic_a_out[1]
append!(ic_a_rand_out, (ic_a_out[2] - (maximum(ic_a) + ϵ)) .* rand(half_num_test_ic) .+ (maximum(ic_a) + ϵ))
ic_b_rand_out = ((minimum(ic_b) - ϵ) - ic_b_out[1]) .* rand(half_num_test_ic) .+ ic_b_out[1]
append!(ic_b_rand_out, (ic_b_out[2] - (maximum(ic_b) + ϵ)) .* rand(half_num_test_ic) .+ (maximum(ic_b) + ϵ))

for i in eachindex(KSE.diffusion_coeffs)
    μ = KSE.diffusion_coeffs[i]
    A = op_fom_tr[i].A
    F = op_fom_tr[i].A2u

    prog = Progress(num_test_ic)
    Threads.@threads for j in 1:num_test_ic  
        # generate test 1 data
        a = ic_a_rand_out[j]
        b = ic_b_rand_out[j]
        IC = u0(a,b)
        Xtest_out = KSE.integrate_model(KSE.tspan, IC; linear_matrix=A, quadratic_matrix=F, 
                                        system_input=false, const_stepsize=true)
        
        save_data = Dict(
            "X" => Xtest_out,
            "IC" => IC,
            "a" => a,
            "b" => b,
        )
        a_str = @sprintf("%1.5f", a)
        b_str = @sprintf("%1.5f", b)
        save(joinpath(FILEPATH, "data/kse_test2/00$(j)_a$(a_str)_b$(b_str).jld2"), save_data)
        next!(prog)
    end

    @info "Finished generating test 2 data for parameter $i"
end