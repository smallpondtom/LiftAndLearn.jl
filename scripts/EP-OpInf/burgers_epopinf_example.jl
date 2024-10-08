"""
Viscous Burgers' equation exmaple for EP-OpInf
"""

## Load packages
using LaTeXStrings
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using ProgressMeter
using Random
using SparseArrays
using Statistics
using UniqueKronecker
import HSL_jll

#========================#
## Load the model module
#========================#
using PolynomialModelReductionDataset: BurgersModel

#====================#
## Load LiftAndLearn
#====================#
using LiftAndLearn
const LnL = LiftAndLearn

#===================#
## Helper functions
#===================#
include("utils/ep_analyze.jl")

#===========================#
## Setup the Burgers' model
#===========================#
Ω = (0.0, 1.0)
Nx = 2^7; dt = 1e-4
burger = BurgersModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt,
    diffusion_coeffs=0.1, BC=:periodic,
)

#===========#
## Settings
#===========#
rmin = 1
rmax = 10

burger_system = LnL.SystemStructure(
    state=[1,2],
)
burger_vars = LnL.VariableStructure(
    N=1,
)
burger_data = LnL.DataStructure(
    Δt=1e-4,
    DS=100,
)
burger_optim = LnL.OptimizationSetting(
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
DS = burger_data.DS

# Down-sampled dimension of the time data
Tdim_ds = size(1:DS:burger.time_dim, 1)  # downsampled time dimension

# Number of random test inputs
num_test_ic = 2 # <-------- CHANGE THIS TO DESIRED NUMBER OF TEST INPUTS (!!! 50 FOR PAPER !!!)

#=====================#
## Initial conditions
#=====================#
# training parameter values
ic_a = range(0.8, 1.2, length=5)
ic_b = [1, 2, 3]
ic_c = range(-0.25, 0.25, length=5)

# test parameter domains
ic_a_out = [0.5, 1.5]
ic_b_out = [3, 6]
ic_c_out = [-0.5, 0.5]
num_ic_params = Int(length(ic_a) * length(ic_b) * length(ic_c))

# Parameterized function for the initial condition
init_wave = (A,a,b) -> A .* sin.(2 * pi * ceil(a) * burger.xspan .+ b) 

#=========================#
## Generate training data
#=========================#
# Data placeholders
Xtr = Vector{Matrix{Float64}}(undef, burger.param_dim)  # training state data 10x1
Rtr = Vector{Matrix{Float64}}(undef, burger.param_dim)  # training derivative data 10x1
Σr = Vector{Vector{Float64}}(undef, burger.param_dim)  # singular values 
Xtr_all = Matrix{Matrix{Float64}}(undef, burger.param_dim, num_ic_params)  # all training data 10x27
IC_train = Vector{Vector{Float64}}(undef, num_ic_params)  # all initial conditions 
Vrmax = Vector{Matrix{Float64}}(undef, burger.param_dim)  # POD basis 10x1
op_fom_tr = Vector{LnL.Operators}(undef, burger.param_dim)  # FOM operators 10x1

@info "Generate the FOM system matrices and training data."
@showprogress for i in 1:length(burger.diffusion_coeffs)
    μ = burger.diffusion_coeffs[i]

    # Generate the FOM system matrices (ONLY DEPENDS ON μ)
    A, F = burger.finite_diff_model(burger, μ)
    op_fom_tr[i] = LnL.Operators(A=A, A2u=F)

    # Store the training data 
    Xall = Vector{Matrix{Float64}}(undef, num_ic_params)
    Xdotall = Vector{Matrix{Float64}}(undef, num_ic_params)
    
    # Generate the data for all combinations of the initial condition parameters
    ct = 1  # set/reset counter
    for a in ic_a, b in ic_b, c in ic_c
        if i == 1
            IC_train[ct] = init_wave(a, b, c)
        end

        states = burger.integrate_model(burger.tspan, IC_train[ct]; linear_matrix=A, quadratic_matrix=F, system_input=false)
        Xtr_all[i,ct] = states
        
        tmp = states[:, 2:end]
        Xall[ct] = tmp[:, 1:DS:end]  # downsample data
        tmp = (states[:, 2:end] - states[:, 1:end-1]) / burger.Δt
        Xdotall[ct] = tmp[:, 1:DS:end]  # downsample data

        ct += 1  # increment counter
    end
    # Combine all initial condition data to form on big training data matrix
    Xtr[i] = reduce(hcat, Xall) 
    Rtr[i] = reduce(hcat, Xdotall)
    
    # Compute the POD basis from the training data
    tmp = svd(Xtr[i])
    Vrmax[i] = tmp.U[:, 1:rmax]
    Σr[i] = tmp.S
end

Data = Dict(
        "Xtr" => Xtr, "Rtr" => Rtr, "Vrmax" => Vrmax, 
        "op_fom_tr" => op_fom_tr, "Σr" => Σr,
        "Xtr_all" => Xtr_all, "IC_train" => IC_train
        )

#=====================#
## Plot data to check
#=====================#
p1 = plot()
timestamps = 1:1000:burger.time_dim
plot!(
    p1, burger.xspan, Xtr_all[1,1][:, timestamps], color=:roma, line_z=(burger.Δt*timestamps)', 
    lw=1.5, label="", colorbar_ticks=0:0.1:1.0, clims=(0,1.0),
) 
plot!(p1, xlabel=L"\omega"*", spatial coordinate", ylabel=L"x(\omega,t)", grid=true, minorgrid=true)
plot!(p1, guidefontsize=16, tickfontsize=13,  legendfontsize=13, fontfamily="Computer Modern")
plot!(p1, colorbar_title=L"t", colorbar_titlefontsize=16)
display(p1)

p2 = plot()
plot!(contourf!(p2, burger.tspan, burger.xspan, Xtr_all[1,1], lw=0), xlabel=L"t", ylim=[0, 1.0],
    ylabel=L"\omega, \mathrm{spatial~coordinate}", zlabel=L"u(x,t)")
yticks!(p2, 0:0.2:1.0)
plot!(p2, guidefontsize=13, fontfamily="Computer Modern")
display(p2)

#=====================#
## Check enery levels
#=====================#
nice_orders_all = Vector{Vector{Int}}(undef, burger.param_dim)
energy_level_all = Vector{Vector{Float64}}(undef, burger.param_dim)

for i in eachindex(burger.diffusion_coeffs)
    nice_orders_all[i], energy_level_all[i] = LnL.choose_ro(Σr[i]; en_low=-12)
end
nice_orders = Int.(round.(mean(nice_orders_all)))
energy_level = mean(energy_level_all)

#=====================#
## Plot energy levels
#=====================#
p3 = plot(energy_level[1:nice_orders[end]], yaxis=:log10, label="energy loss", fontfamily="Computer Modern", lw=2,
    ylabel="relative energy loss from truncation", xlabel="number of retained modes", legend=:topright, majorgrid=true, grid=true)
plot!(nice_orders, energy_level[nice_orders], seriestype=:scatter, label=nothing, markersize=6)
for (i,order) in enumerate(nice_orders)
    if i == 7
        annotate!((order+2.5, energy_level[order], text(L"r_{\mathrm{max}}="*string(order), :bottom, 18, "Computer Modern")))
    end
end
yticks!(10.0 .^ (0:-2:-10))
xticks!(0:2:20)
plot!(fg_legend=:false)
plot!(guidefontsize=14, tickfontsize=12, titlefontsize=15,  legendfontsize=15, right_margin=2mm)
display(p3)

#=======================#
## Generate Test 1 data
#=======================#
# Store values
Xtest_in = Matrix{Matrix{Float64}}(undef, burger.param_dim, num_test_ic)  # all training data 
IC_test_in = Vector{Vector{Float64}}(undef, num_test_ic)  # all initial conditions

@info "Generate test data for parameters within the training region."

# Generate 10 random initial condition parameters
ic_a_rand_in = (maximum(ic_a) - minimum(ic_a)) .* rand(num_test_ic) .+ minimum(ic_a)
ic_b_rand_in = (maximum(ic_b) - minimum(ic_b)) .* rand(num_test_ic) .+ minimum(ic_b)
ic_c_rand_in = (maximum(ic_c) - minimum(ic_c)) .* rand(num_test_ic) .+ minimum(ic_c)

@showprogress for i in 1:length(burger.diffusion_coeffs)
    μ = burger.diffusion_coeffs[i]

    # Generate the FOM system matrices (ONLY DEPENDS ON μ)
    A = op_fom_tr[i].A
    F = op_fom_tr[i].A2u

    # Generate the data for all combinations of the initial condition parameters
    prog = Progress(num_test_ic)
    Threads.@threads for j in 1:num_test_ic  
        a = ic_a_rand_in[j]
        b = ic_b_rand_in[j]
        c = ic_c_rand_in[j]

        if i == 1
            IC_test_in[j] = init_wave(a, b, c)
        end
        Xtest_in[i,j] = burger.integrate_model(burger.tspan, IC_test_in[j]; linear_matrix=A, quadratic_matrix=F, system_input=false)
        next!(prog)
    end
end
Data["Xtest_in"] = Xtest_in
Data["IC_test_in"] = IC_test_in

#=======================#
## Generate Test 2 data
#=======================#
# Store values
Xtest_out = Matrix{Matrix{Float64}}(undef, burger.param_dim, num_test_ic)  # all training data 
IC_test_out = Vector{Vector{Float64}}(undef, num_test_ic)  # all initial conditions

@info "Generate test data for parameters outside the training region."

# Generate 10 random initial condition parameters outside the training region
ϵ=1e-2
half_num_test_ic = Int(num_test_ic/2)
ic_a_rand_out = ((minimum(ic_a) - ϵ) - ic_a_out[1]) .* rand(half_num_test_ic) .+ ic_a_out[1]
append!(ic_a_rand_out, (ic_a_out[2] - (maximum(ic_a) + ϵ)) .* rand(half_num_test_ic) .+ (maximum(ic_a) + ϵ))
ic_b_rand_out = (ic_b_out[2] - ic_b_out[1] + ϵ) .* rand(num_test_ic) .+ ic_b_out[1]
ic_c_rand_out = ((minimum(ic_c) - ϵ) - ic_c_out[1]) .* rand(half_num_test_ic) .+ ic_c_out[1]
append!(ic_c_rand_out, (ic_c_out[2] - (maximum(ic_c) + ϵ)) .* rand(half_num_test_ic) .+ (maximum(ic_c) + ϵ))

@showprogress for i in 1:length(burger.diffusion_coeffs)
    μ = burger.diffusion_coeffs[i]

    # Generate the FOM system matrices (ONLY DEPENDS ON μ)
    A = op_fom_tr[i].A
    F = op_fom_tr[i].A2u

    # Generate the data for all combinations of the initial condition parameters
    prog = Progress(num_test_ic)
    Threads.@threads for j in 1:num_test_ic
        a = ic_a_rand_out[j]
        b = ic_b_rand_out[j]
        c = ic_c_rand_out[j]

        if i == 1
            IC_test_out[j] = init_wave(a, b, c)
        end
        Xtest_out[i,j] = burger.integrate_model(burger.tspan, IC_test_out[j]; linear_matrix=A, quadratic_matrix=F, system_input=false)
        next!(prog)
    end
end
Data["Xtest_out"] = Xtest_out
Data["IC_test_out"] = IC_test_out

#==================================#
## Obtain standard OpInf operators
#==================================#
@info "Compute the Least Squares solution."
options = LnL.LSOpInfOption(
    system=burger_system,
    vars=burger_vars,
    data=burger_data,
    optim=burger_optim,
)

# Data storage
op_LS = Array{LnL.Operators}(undef, burger.param_dim)

@showprogress for i in 1:length(burger.diffusion_coeffs)
    op_LS[i] = LnL.opinf(Xtr[i], Vrmax[i], options; Xdot=Rtr[i])
end
Data["op_LS"] = op_LS

#=============================#
## Obtain intrusive operators
#=============================#
@info "Compute the intrusive model"

# Data storage
op_int = Array{LnL.Operators}(undef, burger.param_dim)

@showprogress for i in 1:length(burger.diffusion_coeffs)
    # Compute the values for the intrusive model from the basis of the training data
    op_int[i] = LnL.pod(op_fom_tr[i], Vrmax[i], options.system)
end
Data["op_int"] = op_int

#===============================#
## Obtain EPHEC-OpInf operators
#===============================#
@info "Compute the EPHEC model"

options = LnL.EPHECOpInfOption(
    system=burger_system,
    vars=burger_vars,
    data=burger_data,
    optim=burger_optim,
)

op_ephec = Array{LnL.Operators}(undef, burger.param_dim)

@showprogress for i in 1:length(burger.diffusion_coeffs)
    # Use Ipopt default initial guess
    options.optim.initial_guess = false 
    op_ephec[i] = LnL.epopinf(Xtr[i], Vrmax[i], options; Xdot=Rtr[i])
end
Data["op_ephec"] = op_ephec

#===============================#
## Obtain EPSIC-OpInf operators
#===============================#
@info "Compute the EPSIC OpInf."

options = LnL.EPSICOpInfOption(
    system=burger_system,
    vars=burger_vars,
    data=burger_data,
    optim=burger_optim,
    ϵ=1e-3,
)

op_epsic = Array{LnL.Operators}(undef, burger.param_dim)

@showprogress for i in 1:length(burger.diffusion_coeffs)
    # Use Ipopt default initial guess
    options.optim.initial_guess = false 
    op_epsic[i] = LnL.epopinf(Xtr[i], Vrmax[i], options; Xdot=Rtr[i])

    @info "Loop $(i) out of $(burger.param_dim) completed..."
end
Data["op_epsic"] = op_epsic

#=============================#
## Obtain EPP-OpInf operators
#=============================#
@info "Compute the EPUC OpInf."

options = LnL.EPPOpInfOption(
    system=burger_system,
    vars=burger_vars,
    data=burger_data,
    optim=burger_optim,
    α=1e6,
)

op_epp = Array{LnL.Operators}(undef, burger.param_dim)

for i in 1:length(burger.diffusion_coeffs)
    # Use Ipopt default initial guess
    options.optim.initial_guess = false 
    op_epp[i] = LnL.epopinf(Xtr[i], Vrmax[i], options; Xdot=Rtr[i])

    @info "Loop $(i) out of $(burger.param_dim) completed..."
end
Data["op_epp"] = op_epp

#===========================#
## Analyze training results
#===========================#
Data = ep_analyze(Data,burger,options,Data["Xtr_all"],Data["IC_train"],rmin,rmax)

#=========================#
## Analyze test 1 results
#=========================#
Data["test1_state_err"] = ep_analyze(Data,burger,options,Data["Xtest_in"],Data["IC_test_in"],rmin,rmax; is_train=false)

#=========================#
## Analyze test 2 results
#=========================#
Data["test2_state_err"] = ep_analyze(Data,burger,options,Data["Xtest_out"],Data["IC_test_out"],rmin,rmax; is_train=false)

#===================#
## Projection error
#===================#
# Training data
mean_train_proj_err = mean(Data["train_proj_err"], dims=2)
plot(rmin:rmax, mean_train_proj_err, marker=(:rect))
plot!(yscale=:log10, majorgrid=true, legend=false)
tmp = log10.(mean_train_proj_err)
yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
xticks!(rmin:rmax)
xlabel!(L"\mathrm{reduced~model~dimension~} r")
ylabel!(L"\mathrm{average~~projection~~error}")
plot!(guidefontsize=16, tickfontsize=13,  legendfontsize=13)

#============================#
## Mean relative state error
#============================#
# Training data
mean_LS_state_err = mean(Data["train_state_err"][:LS], dims=2)
mean_int_state_err = mean(Data["train_state_err"][:int], dims=2)
mean_ephec_state_err = mean(Data["train_state_err"][:ephec], dims=2)
mean_epsic_state_err = mean(Data["train_state_err"][:epsic], dims=2)  
mean_epp_state_err = mean(Data["train_state_err"][:epp], dims=2)

plot(rmin:rmax, mean_int_state_err, c=:orange, marker=(:cross, 10, :orange), markerstrokewidth=2.5, label="Intrusive")
plot!(rmin:rmax, mean_LS_state_err, c=:crimson, marker=(:circle, 5, :crimson), label="OpInf", markerstrokecolor=:red3, lw=2)
# plot!(rmin:rmax, mean_ephec_state_err, c=:blue, markerstrokecolor=:blue, marker=(:rect, 3), ls=:dash, lw=2, label="EP-OpInf")
plot!(rmin:rmax, mean_ephec_state_err, c=:blue, markerstrokecolor=:blue, marker=(:rect, 3), ls=:dash, lw=2, label="EPHEC-OpInf")
plot!(rmin:rmax, mean_epsic_state_err, c=:purple, markerstrokecolor=:purple, marker=(:dtriangle, 5), ls=:dot, label="EPSIC-OpInf")
plot!(rmin:rmax, mean_epp_state_err, c=:brown, markerstrokecolor=:brown, marker=(:star, 5), lw=2, ls=:dashdot, label="EPP-OpInf")
plot!(yscale=:log10, majorgrid=true, legend=:bottomleft)
tmp = log10.(mean_int_state_err)
yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
xticks!(rmin:rmax)
xlabel!("reduced model dimension " *L"r")
ylabel!("average relative state error")
title!("Training")
# ylims!(1e-4,2)
plot!(guidefontsize=16, tickfontsize=13, legendfontsize=13, titlefontsize=18, fontfamily="Computer Modern")

#============================#
## Test relative state errors
#============================#
# Test 1
mean_LS_state_err = mean(Data["test1_state_err"][:LS], dims=2)
mean_int_state_err = mean(Data["test1_state_err"][:int], dims=2)
mean_ephec_state_err = mean(Data["test1_state_err"][:ephec], dims=2)
mean_epsic_state_err = mean(Data["test1_state_err"][:epsic], dims=2)
mean_epp_state_err = mean(Data["test1_state_err"][:epp], dims=2)

p = plot(rmin:rmax, mean_int_state_err, c=:orange, marker=(:cross, 10), markerstrokewidth=2.5, label="Intrusive")
plot!(rmin:rmax, mean_LS_state_err, c=:crimson, marker=(:circle, 5), label="OpInf", markerstrokecolor=:crimson, lw=2)
# plot!(rmin:rmax, mean_ephec_state_err, c=:blue, markerstrokecolor=:blue, marker=(:rect, 3), ls=:dash, lw=2, label="EP-OpInf")
plot!(rmin:rmax, mean_ephec_state_err, c=:blue, markerstrokecolor=:blue, marker=(:rect, 3), ls=:dash, lw=2, label="EPHEC-OpInf")
plot!(rmin:rmax, mean_epsic_state_err, c=:purple, markerstrokecolor=:purple, marker=(:dtriangle, 5), ls=:dot, label="EPSIC-OpInf")
plot!(rmin:rmax, mean_epp_state_err, c=:brown, markerstrokecolor=:brown, marker=(:star, 5), lw=2, ls=:dash, label="EPP-OpInf")

# Test 2
mean_LS_state_err = mean(Data["test2_state_err"][:LS], dims=2)
mean_int_state_err = mean(Data["test2_state_err"][:int], dims=2)
mean_ephec_state_err = mean(Data["test2_state_err"][:ephec], dims=2)
mean_epsic_state_err = mean(Data["test2_state_err"][:epsic], dims=2)
mean_epp_state_err = mean(Data["test2_state_err"][:epp], dims=2)

plot!(rmin:rmax, mean_LS_state_err, c=:crimson, marker=(:circle, 5), label="", lw=2, markerstrokecolor=:crimson)
plot!(rmin:rmax, mean_int_state_err, c=:orange, marker=(:cross, 10), markerstrokewidth=2.5, label="")
plot!(rmin:rmax, mean_ephec_state_err, c=:blue, markerstrokecolor=:blue, marker=(:rect, 3), ls=:dash, lw=2, label="")
plot!(rmin:rmax, mean_epsic_state_err, c=:purple, markerstrokecolor=:purple, marker=(:dtriangle, 5), ls=:dot, label="")
plot!(rmin:rmax, mean_epp_state_err, c=:brown, markerstrokecolor=:brown, marker=(:star, 5), lw=1, ls=:dash, label="")

plot!(yscale=:log10, majorgrid=true, legend=:bottomleft)
# tmp = log10.(mean_int_state_err)
# yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
xticks!(rmin:rmax)
xlabel!("reduced model dimension " * L" r")
ylabel!("average relative state error")
title!("Test", fontsize=18)
ylims!(1e-4,2)
# yticks!([10.0^i for i in -4:1:0])
plot!(guidefontsize=16, tickfontsize=13, legendfontsize=13, titlefontsize=18, fontfamily="Computer Modern")
annotate!(rmax-3.5, 5e-4, text("(1) Interpolation", 13, :left, :bottom, :white, "Computer Modern"))
annotate!(rmax-3.0, 1.5e-1, text("(2) Extrapolation", 13, :left, :top, :white, "Computer Modern"))

#===============================#
## Energy-preserving constraint
#===============================#
mean_LS_CR_tr = mean(Data["train_CR"][:LS], dims=2)
mean_int_CR_tr = mean(Data["train_CR"][:int], dims=2)
mean_ephec_CR_tr = mean(Data["train_CR"][:ephec], dims=2)
mean_epsic_CR_tr = mean(Data["train_CR"][:epsic], dims=2)
mean_epp_CR_tr = mean(Data["train_CR"][:epp], dims=2)

plot(rmin:rmax, mean_int_CR_tr, c=:orange, marker=(:cross, 10), markerstrokewidth=2.5, label="Intrusive")
plot!(rmin:rmax, mean_LS_CR_tr, marker=(:circle, 5), c=:crimson, label="OpInf", markerstrokecolor=:crimson, lw=2)
# plot!(rmin:rmax, mean_ephec_CR_tr, c=:blue, markerstrokecolor=:blue, marker=(:rect, 3), lw=2, ls=:dash, label="EP-OpInf")
plot!(rmin:rmax, mean_ephec_CR_tr, c=:blue, markerstrokecolor=:blue, marker=(:rect, 3), lw=2, ls=:dash, label="EPHEC-OpInf")
plot!(rmin:rmax, mean_epsic_CR_tr, c=:purple, markerstrokecolor=:purple, marker=(:dtriangle, 5), ls=:dot, label="EPSIC-OpInf")
plot!(rmin:rmax, mean_epp_CR_tr, c=:brown, markerstrokecolor=:brown, marker=(:star, 5), ls=:dash, lw=2, label="EPP-OpInf")
plot!(yscale=:log10, majorgrid=true, legend=:bottomright, minorgridalpha=0.03)
yticks!(10.0 .^ [-30, -25, -20, -15, -10, -5, 0])
xticks!(1:15)
xlabel!("reduced model dimension " * L"r")
ylabel!("energy-preserving constraint violation")
plot!(xlabelfontsize=15, ylabelfontsize=13, tickfontsize=13,  legendfontsize=13, fontfamily="Computer Modern")