"""
Kuramoto–Sivashinsky equation example for EP-OpInf
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
using ProgressMeter
using Random
using SparseArrays
using Statistics
using StatsBase
using UniqueKronecker
import HSL_jll

#========================#
## Load the model module
#========================#
using PolynomialModelReductionDataset: KuramotoSivashinskyModel

#====================#
## Load LiftAndLearn
#====================#
using LiftAndLearn
const LnL = LiftAndLearn

#===================#
## Helper functions
#===================#
include("utils/ep_analyze.jl")
include("utils/kse_analyze.jl")

#======================#
## Setup the KSE model
#======================#
Ω = (0.0, 22.0)
Nx = 2^9; dt = 1e-3
KSE = KuramotoSivashinskyModel(
    spatial_domain=Ω, time_domain=(0.0, 300.0), Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt,
    diffusion_coeffs=1.0, BC=:periodic,
)

#===========#
## Settings
#===========#
KSE_system = LnL.SystemStructure(
    state=[1,2],
)
KSE_vars = LnL.VariableStructure(
    N=1,
)
KSE_data = LnL.DataStructure(
    Δt=KSE.Δt,
    DS=100,  # keep every 100th data point
)
KSE_optim = LnL.OptimizationSetting(
    verbose=false,
    initial_guess=false,
    max_iter=1000,
    reproject=false,
    SIGE=false,
    HSL_lib_path=HSL_jll.libhsl_path,
    linear_solver="ma97",
)

options = LnL.LSOpInfOption(
    system=burger_system,
    vars=burger_vars,
    data=burger_data,
    optim=burger_optim,
)

# Downsampling rate
DS = KSE_data.DS

# Down-sampled dimension of the time data
Tdim_ds = size(1:DS:KSE.time_dim, 1)  # downsampled time dimension

# Number of random test inputs
num_test_ic = 2 # <-------- CHANGE THIS TO DESIRED NUMBER OF TEST INPUTS (!!! 50 FOR PAPER !!!)

# Prune data to get only the chaotic region
prune_data = false
prune_idx = KSE.time_dim ÷ 2
t_prune = KSE.tspan[prune_idx-1:end]

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
Xtr_all = Matrix{Matrix{Float64}}(undef, KSE.param_dim, num_ic_params)  # all training data
IC_train = Vector{Vector{Float64}}(undef, num_ic_params)  # all initial conditions 
Vr = Vector{Matrix{Float64}}(undef, KSE.param_dim)  # POD basis
Σr = Vector{Vector{Float64}}(undef, KSE.param_dim)  # singular values 
op_fom_tr = Vector{LnL.Operators}(undef, KSE.param_dim)  # FOM operators 

@info "Generate the FOM system matrices and training data."
@showprogress for i in eachindex(KSE.diffusion_coeffs)
    μ = KSE.μs[i]

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

        states = KSE.integrate_model(KSE.tspan, u0(a,b); linear_matrix=A, quadratic_matrix=F, 
                                     system_input=false, const_stepsize=true)
        if prune_data
            Xtr_all[i,j] = states[:, prune_idx-1:end]

            tmp = states[:, prune_idx:end]
            Xall[j] = tmp[:, 1:DS:end]  # downsample data
            tmp = (states[:, prune_idx:end] - states[:, prune_idx-1:end-1]) / KSE.Δt
            Xdotall[j] = tmp[:, 1:DS:end]  # downsample data

            if i == 1
                IC_train[j] = states[:, prune_idx-1]
            end
        else
            Xtr_all[i,j] = states

            tmp = states[:, 2:end]
            Xall[j] = tmp[:, 1:DS:end]  # downsample data
            tmp = (states[:, 2:end] - states[:, 1:end-1]) / KSE.Δt
            Xdotall[j] = tmp[:, 1:DS:end]  # downsample data

            if i == 1
                IC_train[j] = u0(a, b)
            end
        end

        next!(prog)
    end
    # Combine all initial condition data to form on big training data matrix
    Xtr[i] = reduce(hcat, Xall) 
    Rtr[i] = reduce(hcat, Xdotall)
    
    # Compute the POD basis from the training data
    tmp = svd(Xtr[i])
    Vr[i] = tmp.U
    Σr[i] = tmp.S
end

#======================#
## Check energy levels
#======================#
nice_orders_all = Vector{Vector{Int}}(undef, KSE.param_dim)
energy_level_all = Vector{Vector{Float64}}(undef, KSE.param_dim)

for i in eachindex(KSE.μs)
    nice_orders_all[i], energy_level_all[i] = LnL.choose_ro(Σr[i]; en_low=-12)
end
nice_orders = Int.(round.(mean(nice_orders_all)))
energy_level = mean(energy_level_all)

#=====================#
## Plot energy levels
#=====================#
plot(energy_level[1:nice_orders[end]], yaxis=:log10, label="energy loss", fontfamily="Computer Modern", lw=2,
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

#========================#
## Select reduced orders
#========================#
ro = nice_orders[2:8]

#============#
## Save data
#============#
DATA = Dict("Xtr" => Xtr, "Rtr" => Rtr, "Vr" => Vr, 
            "op_fom_tr" => op_fom_tr, "Σr" => Σr,
            "Xtr_all" => Xtr_all, "IC_train" => IC_train,
            "ro" => ro
        )
save("./scripts/EP-OpInf/data/kse_epopinf_data.jld2", DATA)

#==================================#
## Obtain standard OpInf operators
#==================================#
@info "Compute the least-squares solution"
options = LnL.LSOpInfOption(
    system=KSE_system,
    vars=KSE_vars,
    data=KSE_data,
    optim=KSE_optim,
)

# Store values
op_LS = Array{LnL.Operators}(undef, KSE.param_dim)

@showprogress for i in eachindex(KSE.μs)
    op_LS[i] = LnL.opinf(Xtr[i], Vr[i][:, 1:ro[end]], options; Xdot=Rtr[i])
end

#==============================#
## Compute intrusive operators
#==============================#
@info "Compute the intrusive model"

# Store values
op_int = Array{LnL.Operators}(undef, KSE.param_dim)

@showprogress for i in eachindex(KSE.μs)
    op_int[i] = LnL.pod(op_fom_tr[i], Vr[i][:, 1:ro[end]], options.system)
end

#===============================#
## Compute EPHEC-OpInf operators
#===============================#
@info "Compute the EPHEC model"

options = LnL.EPHECOpInfOption(
    system=KSE_system,
    vars=KSE_vars,
    data=KSE_data,
    optim=KSE_optim,
    A_bnds=(-1000.0, 1000.0),
    ForH_bnds=(-100.0, 100.0),
)

# Store values
op_ephec =  Array{LnL.Operators}(undef, KSE.param_dim)

@showprogress for i in eachindex(KSE.μs)
    op_ephec[i] = LnL.epopinf(Xtr[i], Vr[i][:, 1:ro[end]], options; Xdot=Rtr[i])
end

#===============================#
## Compute EPSIC-OpInf operators
#===============================#
@info "Compute the EPSIC model"

options = LnL.EPSICOpInfOption(
    system=KSE_system,
    vars=KSE_vars,
    data=KSE_data,
    optim=KSE_optim,
    ϵ = 1e-3,
    A_bnds=(-1000.0, 1000.0),
    ForH_bnds=(-100.0, 100.0),
)

# Store values
op_epsic = Array{LnL.Operators}(undef, KSE.param_dim)

@showprogress for i in eachindex(KSE.μs)
    op_epsic[i] = LnL.epopinf(Xtr[i], Vr[i][:, 1:ro[end]], options; Xdot=Rtr[i])
end

#==============================#
## Compute EPP-OpInf operators
#==============================#
@info "Compute the EPP OpInf."

options = LnL.EPPOpInfOption(
    system=KSE_system,
    vars=KSE_vars,
    data=KSE_data,
    optim=KSE_optim,
    α=1e6,
    A_bnds=(-1000.0, 1000.0),
    ForH_bnds=(-100.0, 100.0),
)

# Store values
op_epp =  Array{LnL.Operators}(undef, KSE.param_dim)

@showprogress for i in eachindex(KSE.μs)
    op_epp[i] = LnL.epopinf(Xtr[i], Vr[i][:, 1:ro[end]], options; Xdot=Rtr[i])
end

#=================#
## Save operators
#=================#
ops = Dict("op_LS" => op_LS, "op_int" => op_int, "op_ephec" => op_ephec,
                "op_epsic" => op_epsic, "op_epp" => op_epp)
save("./scripts/EP-OpInf/data/kse_epopinf_ops.jld2", ops)

#==========================#
## Prepare to save results
#==========================#
RES = Dict{String, Any}()
RES["train_proj_err"] = Array{Float64}(undef, length(ro), KSE.param_dim);

#=================#
## Project errors
#=================#
RES["train_proj_err"] = kse_analyze_proj_err(KSE, DATA["Xtr_all"], DATA["Vr"], DATA["IC_train"], DATA["ro"]);

#=========================#
## Plot projection errors
#=========================#
mean_train_proj_err = mean(RES["train_proj_err"], dims=2)
plot(DATA["ro"], mean_train_proj_err, marker=(:rect), fontfamily="Computer Modern")
plot!(yscale=:log10, majorgrid=true, legend=false)
yticks!([1e-1, 1e-2, 1e-3, 1e-4])
xticks!(DATA["ro"][1]:2:DATA["ro"][end])
xlabel!("reduced model dimension " * L"r")
ylabel!("projection error")
plot!(guidefontsize=16, tickfontsize=13,  legendfontsize=13)

#========================#
## Relative state errors
#========================#
RES["train_state_err"] = Dict(
    :int => Array{Float64}(undef, length(ro), KSE.param_dim),
    :LS => Array{Float64}(undef, length(ro), KSE.param_dim),
    :ephec => Array{Float64}(undef, length(ro), KSE.param_dim),
    :epsic => Array{Float64}(undef, length(ro), KSE.param_dim),
    :epp => Array{Float64}(undef, length(ro), KSE.param_dim),
)

# Standard OpInf
RES["train_state_err"][:LS] = kse_analyze_rse(OPS["op_LS"], KSE, DATA["Xtr_all"], DATA["Vr"], DATA["IC_train"], DATA["ro"], DS, KSE.integrate_FD)
# Intrusive
RES["train_state_err"][:int] = kse_analyze_rse(OPS["op_int"], KSE, DATA["Xtr_all"], DATA["Vr"], DATA["IC_train"], DATA["ro"], DS, KSE.integrate_FD)
# EPHEC
RES["train_state_err"][:ephec] = kse_analyze_rse(OPS["op_ephec"], KSE, DATA["Xtr_all"], DATA["Vr"], DATA["IC_train"], DATA["ro"], DS, KSE.integrate_FD)
# EPSIC
RES["train_state_err"][:epsic] = kse_analyze_rse(OPS["op_epsic"], KSE, DATA["Xtr_all"], DATA["Vr"], DATA["IC_train"], DATA["ro"], DS, KSE.integrate_FD)
# EPP
RES["train_state_err"][:epp] = kse_analyze_rse(OPS["op_epp"], KSE, DATA["Xtr_all"], DATA["Vr"], DATA["IC_train"], DATA["ro"], DS, KSE.integrate_FD)

#=============================#
## Plot relative state errors
#=============================#
mean_LS_state_err = mean(RES["train_state_err"][:LS], dims=2)
mean_int_state_err = mean(RES["train_state_err"][:int], dims=2)
mean_ephec_state_err = mean(RES["train_state_err"][:ephec], dims=2)
mean_epsic_state_err = mean(RES["train_state_err"][:epsic], dims=2)
mean_epp_state_err = mean(RES["train_state_err"][:epp], dims=2)

plot!(DATA["ro"], mean_int_state_err, c=:orange, marker=(:cross, 10, :orange), markerstrokewidth=2.5, label="Intrusive")
plot(DATA["ro"], mean_LS_state_err, c=:crimson, marker=(:circle, 5, :crimson), markerstrokecolor=:red3, label="OpInf")
# plot!(DATA["ro"], mean_ephec_state_err, c=:blue, markerstrokecolor=:blue, marker=(:rect, 3), ls=:dash, lw=2, label="EP-OpInf")
plot!(DATA["ro"], mean_ephec_state_err, c=:blue, markerstrokecolor=:blue, marker=(:rect, 3), ls=:dash, lw=2, label="EPHEC-OpInf")
plot!(DATA["ro"], mean_epsic_state_err, c=:purple, markerstrokecolor=:purple, marker=(:dtriangle, 5), ls=:dot, label="EPSIC-OpInf")
plot!(DATA["ro"], mean_epp_state_err, c=:brown, markerstrokecolor=:brown, marker=(:star, 5), lw=2, ls=:dashdot, label="EPP-OpInf")
plot!(majorgrid=true, legend=:topright)
# yticks!([1e-0, 1e-1])
xticks!(DATA["ro"][1]:2:DATA["ro"][end])
xlabel!("reduced model dimension " * L" r")
ylabel!("average relative state error")
title!("Training")
plot!(guidefontsize=16, tickfontsize=13, legendfontsize=13, fontfamily="Computer Modern")

#=======================#
## Constraint residuals
#=======================#
RES["train_CR"] = Dict(
    :int => Array{Float64}(undef, length(ro), KSE.Pdim),
    :LS => Array{Float64}(undef, length(ro), KSE.Pdim),
    :ephec => Array{Float64}(undef, length(ro), KSE.Pdim),
    :epsic => Array{Float64}(undef, length(ro), KSE.Pdim),
    :epp => Array{Float64}(undef, length(ro), KSE.Pdim),
    :fom => Array{Float64}(undef, KSE.Pdim)
)

# Compute CR of full order model
RES["train_CR"][:fom] = kse_fom_CR(DATA["op_fom_tr"], KSE)
# Standard OpInf
RES["train_CR"][:LS] = analyze_cr(OPS["op_LS"], KSE, DATA["IC_train"], DATA["ro"])
# Intrusive
RES["train_CR"][:int] = analyze_cr(OPS["op_int"], KSE, DATA["IC_train"], DATA["ro"])
# EPHEC
RES["train_CR"][:ephec] = analyze_cr(OPS["op_ephec"], KSE, DATA["IC_train"], DATA["ro"])
# EPSIC
RES["train_CR"][:epsic] = analyze_cr(OPS["op_epsic"], KSE, DATA["IC_train"], DATA["ro"])
# EPP
RES["train_CR"][:epp] = analyze_cr(OPS["op_epp"], KSE, DATA["IC_train"], DATA["ro"])

#============================#
## Plot constraint residuals
#============================#
mean_LS_CR_tr = mean(RES["train_CR"][:LS], dims=2)
mean_int_CR_tr = mean(RES["train_CR"][:int], dims=2)
mean_ephec_CR_tr = mean(RES["train_CR"][:ephec], dims=2)
mean_epsic_CR_tr = mean(RES["train_CR"][:epsic], dims=2)
mean_epp_CR_tr = mean(RES["train_CR"][:epp], dims=2)

plot(DATA["ro"], mean_int_CR_tr, c=:orange, marker=(:cross, 10), markerstrokewidth=2.5, label="Intrusive")
plot!(DATA["ro"], mean_LS_CR_tr, marker=(:circle, 5), c=:crimson, markerstrokecolor=:crimson, lw=2, label="OpInf")
# plot!(DATA["ro"], mean_ephec_CR_tr, c=:blue, markerstrokecolor=:blue, marker=(:rect, 3), lw=2, ls=:dash, label="EP-OpInf")
plot!(DATA["ro"], mean_ephec_CR_tr, c=:blue, markerstrokecolor=:blue, marker=(:rect, 3), lw=2, ls=:dash, label="EPHEC-OpInf")
plot!(DATA["ro"], mean_epsic_CR_tr, c=:purple, markerstrokecolor=:purple, marker=(:dtriangle, 5), ls=:dot, label="EPSIC-OpInf")
plot!(DATA["ro"], mean_epp_CR_tr, c=:brown, markerstrokecolor=:brown, marker=(:star, 5), ls=:dashdot, lw=1, label="EPP-OpInf")
plot!(yscale=:log10, majorgrid=true, legend=:right, minorgridalpha=0.03)
yticks!(10.0 .^ [-15, -12, -9, -6, -3, 0, 3])
xticks!(DATA["ro"][1]:2:DATA["ro"][end])
xlabel!("reduced model dimension " * L" r")
ylabel!("energy-preserving constraint violation")
plot!(xlabelfontsize=16, ylabelfontsize=13, tickfontsize=13, legendfontsize=14, fontfamily="Computer Modern")

#=======================================#
## Normalized Autocorrelation functions
#=======================================#
# Time lag
lags = 0:KSE_data.DS:(KSE.time_dim ÷ 2)

RES["train_AC"] = Dict(
    :int   => Array{Float64}(undef, length(lags), length(ro)),
    :LS    => Array{Float64}(undef, length(lags), length(ro)),
    :ephec => Array{Float64}(undef, length(lags), length(ro)),
    :epsic => Array{Float64}(undef, length(lags), length(ro)),
    :epp   => Array{Float64}(undef, length(lags), length(ro)),
    :fom   => Array{Float64}(undef, length(lags))
)
RES["train_AC_ERR"] = Dict(
    :int   => Array{Float64}(undef, length(ro), KSE.param_dim),
    :LS    => Array{Float64}(undef, length(ro), KSE.param_dim),
    :ephec => Array{Float64}(undef, length(ro), KSE.param_dim),
    :epsic => Array{Float64}(undef, length(ro), KSE.param_dim),
    :epp   => Array{Float64}(undef, length(ro), KSE.param_dim),
)

# Compute autocorrelation functions
ac_fom = zeros(length(lags))
ac_LS = zeros(length(lags), length(DATA["ro"]))
ac_LS_err = zeros(length(DATA["ro"]))
ac_int = zeros(length(lags), length(DATA["ro"]))
ac_int_err = zeros(length(DATA["ro"]))
ac_ephec = zeros(length(lags), length(DATA["ro"]))
ac_ephec_err = zeros(length(DATA["ro"]))
ac_epsic = zeros(length(lags), length(DATA["ro"]))
ac_epsic_err = zeros(length(DATA["ro"]))
ac_epp = zeros(length(lags), length(DATA["ro"]))
ac_epp_err = zeros(length(DATA["ro"]))

bar = length(DATA["IC_train"])
prog = Progress(bar)
Threads.@threads for (i,initcond) in collect(enumerate(DATA["IC_train"]))
    ac_fom_tmp = kse_analyze_autocorr(KSE, DATA["Xtr_all"], i, lags)[1]
    ac_fom .+= ac_fom_tmp

    ac_LS_tmp = kse_analyze_autocorr(OPS["op_LS"], KSE, DATA["Vr"], initcond, DATA["ro"], KSE.integrate_model, lags)
    ac_int_tmp = kse_analyze_autocorr(OPS["op_int"], KSE, DATA["Vr"], initcond, DATA["ro"], KSE.integrate_model, lags)
    ac_ephec_tmp = kse_analyze_autocorr(OPS["op_ephec"], KSE, DATA["Vr"], initcond, DATA["ro"], KSE.integrate_model, lags)
    ac_epsic_tmp = kse_analyze_autocorr(OPS["op_epsic"], KSE, DATA["Vr"], initcond, DATA["ro"], KSE.integrate_model, lags)
    ac_epp_tmp = kse_analyze_autocorr(OPS["op_epp"], KSE, DATA["Vr"], initcond, DATA["ro"], KSE.integrate_model, lags)
    for r in eachindex(DATA["ro"])
        ac_LS[:,r] .+= ac_LS_tmp[r,1]
        ac_LS_err[r] += norm(ac_LS_tmp[r,1] - ac_fom_tmp, 2) / norm(ac_fom_tmp, 2)

        ac_int[:,r] .+= ac_int_tmp[r,1]
        ac_int_err[r] += norm(ac_int_tmp[r,1] - ac_fom_tmp, 2) / norm(ac_fom_tmp, 2)

        ac_ephec[:,r] .+= ac_ephec_tmp[r,1]
        ac_ephec_err[r] += norm(ac_ephec_tmp[r,1] - ac_fom_tmp, 2) / norm(ac_fom_tmp, 2)

        ac_epsic[:,r] .+= ac_epsic_tmp[r,1]
        ac_epsic_err[r] += norm(ac_epsic_tmp[r,1] - ac_fom_tmp, 2) / norm(ac_fom_tmp, 2)

        ac_epp[:,r] .+= ac_epp_tmp[r,1]
        ac_epp_err[r] += norm(ac_epp_tmp[r,1] - ac_fom_tmp, 2) / norm(ac_fom_tmp, 2)
    end
    next!(prog)
end
# save the mean normalized autocorrelation
RES["train_AC"][:fom] = ac_fom ./ bar
for r in eachindex(DATA["ro"])
    RES["train_AC"][:LS][:,r] = ac_LS[:,r] ./ bar
    RES["train_AC"][:int][:,r] = ac_int[:,r] ./ bar
    RES["train_AC"][:ephec][:,r] = ac_ephec[:,r] ./ bar
    RES["train_AC"][:epsic][:,r] = ac_epsic[:,r] ./ bar
    RES["train_AC"][:epp][:,r] = ac_epp[:,r] ./ bar
end
# average over all initial conditions
ac_LS_err ./= bar  
ac_int_err ./= bar
ac_ephec_err ./= bar
ac_epsic_err ./= bar
ac_epp_err ./= bar
RES["train_AC_ERR"][:LS] = reshape(ac_LS_err, length(ac_LS_err), 1)
RES["train_AC_ERR"][:int] = reshape(ac_int_err, length(ac_int_err), 1)
RES["train_AC_ERR"][:ephec] = reshape(ac_ephec_err, length(ac_ephec_err), 1)
RES["train_AC_ERR"][:epsic] = reshape(ac_epsic_err, length(ac_epsic_err), 1)
RES["train_AC_ERR"][:epp] = reshape(ac_epp_err, length(ac_epp_err), 1)