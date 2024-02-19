## Setup
using FileIO
using JLD2
using LinearAlgebra

using LiftAndLearn
const LnL = LiftAndLearn
const CG = LiftAndLearn.ChaosGizmo

# Settings for the KS equation
KSE = LnL.ks(
    [0.0, 22.0], [0.0, 300.0], [1.0, 1.0],
    512, 0.001, 1, "ep"
)

# Create file name to save data
datafile = "examples/data/kse_data_L22.jld2"
opfile = "examples/data/kse_operators_L22.jld2"
resultfile = "examples/data/kse_results_L22.jld2"
testresultfile = "examples/data/kse_test_results_L22.jld2"
testICfile = "examples/data/kse_test_ics_L22.jld2"

# Settings for Operator Inference
KSE_system = LnL.sys_struct(
    is_lin=true,
    is_quad=true,
)
KSE_vars = LnL.vars(
    N=1,
)
KSE_data = LnL.data(
    Δt=KSE.Δt,
    DS=100,
)
KSE_optim = LnL.opt_settings(
    verbose=true,
    initial_guess=false,
    max_iter=1000,
    reproject=false,
    SIGE=false,
    with_bnds=true,
    linear_solver="ma86",
)

options = LnL.LS_options(
    system=KSE_system,
    vars=KSE_vars,
    data=KSE_data,
    optim=KSE_optim,
)


# num_ic_params = Int(length(ic_a) * length(ic_b))
L = KSE.Omega[2] - KSE.Omega[1]  # length of the domain

# # Parameterized function for the initial condition
u0 = (a,b) -> a * cos.((2*π*KSE.x)/L) .+ b * cos.((4*π*KSE.x)/L)  # initial condition

## Load data
DATA = load(datafile)
OPS = load(opfile)
RES = load(resultfile)
rmax = size(OPS["op_LS"][1].A, 1)
ro = DATA["ro"]

## Training: Compute the Lyapunov exponent and Kaplan-Yorke dimension
num_IC = length(DATA["IC_train"])
RES["train_LE"] = Dict(
    :int   => Array{Array{Float64}}(undef, length(ro), KSE.Pdim, num_IC),
    :LS    => Array{Array{Float64}}(undef, length(ro), KSE.Pdim, num_IC),
    :ephec => Array{Array{Float64}}(undef, length(ro), KSE.Pdim, num_IC),
    :fom   => Array{Array{Float64}}(undef, KSE.Pdim)
)
RES["train_dky"] = Dict(
    :int   => Array{Float64}(undef, length(ro), KSE.Pdim, num_IC),
    :LS    => Array{Float64}(undef, length(ro), KSE.Pdim, num_IC),
    :ephec => Array{Float64}(undef, length(ro), KSE.Pdim, num_IC),
    :fom   => 0.0
)


## Function to compute the Lyapunov exponent and Kaplan-Yorke dimension for one initial condition
function compute_LE_oneIC!(RES, type, keys, model, op, IC, Vr, ro, integrator, jacobian, options, idx)
    for i in eachindex(model.μs)
        for (j, r) in collect(enumerate(ro))
            Ar = op[i].A[1:r,1:r]
            Hr = LnL.extractH(op[i].H, r)
            Fr = LnL.extractF(op[i].F, r)
            op_tmp = LnL.operators(A=Ar, H=Hr, F=Fr)
            if options.history
                _, foo = CG.lyapunovExponentJacobian(op_tmp, integrator, jacobian, Vr[i][:,1:r]' * IC, options)
                RES[keys[1]][type][j,i,idx] = foo
                RES[keys[2]][type][j,i,idx] = CG.kaplanYorkeDim(foo[:,end]; sorted=false)
            else
                foo = lyapunovExponentJacobian(op_tmp, integrator, jacobian, Vr[i][:,1:r]' * IC, options)
                RES[keys[1]][type][j,i,idx] = foo[:,end]
                RES[keys[2]][type][j,i,idx] = CG.kaplanYorkeDim(foo; sorted=false)
            end
            @info "Reduced order of $(r) completed..."
        end
        @debug "Loop $(i) out of $(model.Pdim) completed..."
    end
end

function compute_LE_allIC!(RES, type, keys, model, op, ICs, Vr, ro, integrator, jacobian, options)
    for (idx, IC) in collect(enumerate(ICs))
        compute_LE_oneIC!(RES, type, keys, model, op, IC, Vr, ro, integrator, jacobian, options, idx)
        @info "Initial condition $(idx) out of $(length(ICs)) completed..."
    end
end

# FOM dispatch
function compute_LE_oneIC!(RES, type, keys, model, op, IC, integrator, options, idx)
    for i in eachindex(model.μs)    
        if options.history
            _, foo = CG.lyapunovExponent(op[i], integrator, IC, options)
            RES[keys[1]][type][i,idx] = foo
            RES[keys[2]][type] = CG.kaplanYorkeDim(foo[:,end]; sorted=false)
        else
            foo = lyapunovExponent(op[i], integrator, IC, options)
            RES[keys[1]][type][i,idx] = foo
            RES[keys[2]][type] = CG.kaplanYorkeDim(foo; sorted=false)
        end
        @debug "Loop $(i) out of $(model.Pdim) completed..."

    end
end

function compute_LE_allIC!(RES, type, keys, model, op, ICs, integrator, options)
    for (idx, IC) in collect(enumerate(ICs))
        compute_LE_oneIC!(RES, type, keys, model, op, IC, integrator, options, idx)
        @info "Initial condition $(idx) out of $(length(ICs)) completed..."
    end
end


## Options
options_fom = CG.LE_options(N=1e4, τ=1e3, Δt=0.01, Δτ=KSE.Δt, m=11, T=0.05, verbose=true, history=true)
options_rom = CG.LE_options(N=1e4, τ=1e3, Δt=5*KSE.Δt, m=9, T=5*KSE.Δt, verbose=true, history=true)

## Compute the LE and Dky for all models 
@info "Computing the LE and Dky for training..."
compute_LE_oneIC!(RES, :fom, ["train_LE", "train_dky"], KSE, 
        DATA["op_fom_tr"], DATA["IC_train"][1], KSE.integrate_FD, options_fom, 1)
compute_LE_allIC!(RES, :int, ["train_LE", "train_dky"], KSE, 
        OPS["op_int"], DATA["IC_train"], DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)
compute_LE_allIC!(RES, :LS, ["train_LE", "train_dky"], KSE, 
        OPS["op_LS"], DATA["IC_train"], DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)
compute_LE_allIC!(RES, :ephec, ["train_LE", "train_dky"], KSE, 
        OPS["op_ephec"], DATA["IC_train"], DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)

## Save data
save(resultfile, "RES", RES)

## Testing
GC.gc()

TEST_RES = load(testresultfile)
TEST_IC = load(testICfile)

## Organize the initial conditions into a matrix 
TEST1_ICs = [u0(a,b) for (a,b) in zip(TEST_IC["ic_a_rand_in"], TEST_IC["ic_b_rand_in"])]
TEST2_ICs = [u0(a,b) for (a,b) in zip(TEST_IC["ic_a_rand_out"], TEST_IC["ic_b_rand_out"])]

num_IC = length(TEST_IC["ic_a_rand_in"])

TEST_RES["test1_LE"] = Dict(
    :int   => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :LS    => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :ephec => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :fom   => Array{Array{Float64}}(undef, KSE.Pdim)
)
TEST_RES["test2_LE"] = Dict(
    :int   => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :LS    => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :ephec => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :fom   => Array{Array{Float64}}(undef, KSE.Pdim)
)
TEST_RES["test1_dky"] = Dict(
    :int   => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :LS    => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :ephec => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :fom   => 0.0
)
TEST_RES["test2_dky"] = Dict(
    :int   => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :LS    => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :ephec => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :fom   => 0.0
)


## Test 1
@info "Computing the LE and Dky for test 1..."
compute_LE_oneIC!(TEST_RES, :fom, ["test1_LE", "test1_dky"], KSE, 
    DATA["op_fom_tr"], TEST1_ICs[1], KSE.integrate_FD, options_fom, 1)
compute_LE_allIC!(TEST_RES, :int, ["test1_LE", "test1_dky"], KSE, 
    OPS["op_int"], TEST1_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)
compute_LE_allIC!(TEST_RES, :LS, ["test1_LE", "test1_dky"], KSE, 
    OPS["op_LS"], TEST1_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)
compute_LE_allIC!(TEST_RES, :ephec, ["test1_LE", "test1_dky"], KSE, 
    OPS["op_ephec"], TEST1_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)

## Test 2
@debug "Computing the LE and Dky for test 2..."
compute_LE_oneIC!(TEST_RES, :fom, ["test2_LE", "test2_dky"], KSE, 
    DATA["op_fom_tr"], TEST2_ICs[1], KSE.integrate_FD, options_fom, 1)
compute_LE_allIC!(TEST_RES, :int, ["test2_LE", "test2_dky"], KSE, 
    OPS["op_int"], TEST2_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)
compute_LE_allIC!(TEST_RES, :LS, ["test2_LE", "test2_dky"], KSE, 
    OPS["op_LS"], TEST2_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)
compute_LE_allIC!(TEST_RES, :ephec, ["test2_LE", "test2_dky"], KSE, 
    OPS["op_ephec"], TEST2_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)

## Save data
save(testresultfile, "TEST_RES", TEST_RES)

## Plotting Results 
# Normal Distribution
using Distributions: Normal, pdf
using LaTeXStrings
using Plots
using Statistics
using StatsPlots

# Function to plot histogram with bell curve
function plot_dky_distribution(dky_data, fom_dky, ridx, title; bins=30, annote_loc=(3.3, 1.2))
    rom_dky = vec(dky_data[ridx,:,:])
    # Gather some info
    median_rom_dky = median(rom_dky)
    mean_rom_dky = mean(rom_dky)
    std_rom_dky = std(rom_dky)

    p1 = histogram(rom_dky, bins=bins, normed=true, alpha=0.6, label="")
    plot!(Normal(mean_rom_dky, std_rom_dky), label="", lw=2)
    vline!(p1, [median_rom_dky], color=:red, label="Median")
    # vline!(p1, [fom_dky], color=:black, label="Full")
    vline!(p1, [5.198], label="Edson et al.", linestyle=:dash)
    vline!(p1, [4.2381], label="Cvitanovic et al.", linestyle=:dash)
    vspan!(p1, [mean_rom_dky - std_rom_dky, mean_rom_dky + std_rom_dky], color=:green, alpha=0.1, label=L"\pm 1\sigma")
    plot!(p1, fontfamily="Computer Modern", legendfont=9, tickfont=12, guidefontsize=15,
        legend=:topleft, xlabel=L"D_{ky}", ylabel="Normal Distribution", title=title)
    annotate!(p1, annote_loc..., text("r = $(DATA["ro"][ridx])", 14, "Computer Modern"))
    display(p1)
end

function plot_LEmax_distribution(rom_LE, fom_LE, ridx, title; bins=30, annote_loc=(0.01, 30))
    # Gather some info
    LEmax = []
    _, _, n = size(rom_LE)
    for i in 1:n
        push!(LEmax, maximum(rom_LE[ridx,1,i][:,end]))
    end
    LEmax_fom = maximum(fom_LE[1][:,end])

    median_rom_LEmax = median(LEmax)
    mean_rom_LEmax = mean(LEmax)
    std_rom_LEmax = std(LEmax)

    p1 = histogram(LEmax, bins=bins, normed=true, alpha=0.6, label="")
    plot!(Normal(mean_rom_LEmax, std_rom_LEmax), label="", lw=2)
    vline!(p1, [median_rom_LEmax], color=:red, label="Median")
    # vline!(p1, [LEmax_fom], color=:black, label="Full")
    vline!(p1, [0.043], label="Edson et al.", linestyle=:dash)
    vline!(p1, [0.048], label="Cvitanovic et al.", linestyle=:dash)
    vspan!(p1, [mean_rom_LEmax - std_rom_LEmax, mean_rom_LEmax + std_rom_LEmax], color=:green, alpha=0.1, label=L"\pm 1\sigma")
    plot!(p1, fontfamily="Computer Modern", legendfont=9, tickfont=12, guidefontsize=15,
        legend=:topleft, xlabel=L"\lambda_{\text{max}}", ylabel="Normal Distribution", title=title)
    annotate!(p1, annote_loc..., text("r = $(DATA["ro"][ridx])", 14, "Computer Modern"))
    display(p1)
end

function plot_LE_convergence(LE_data, ridx, ICidx, C, title; ylimits=(1e-7, 2e+2), ytickvalues=10.0 .^ (-7:2:2))
    p = plot()
    data = LE_data[ridx,1,ICidx]
    m, n = size(data)
    for i in 1:m
        plot!(p, 
            (1:n-1)[1:100:end], 
            abs.(data[i,1:100:end-1] .- data[i,end]), 
            lw=1.5, label=false
        )
    end
    plot!(p, 1:n, C ./ (1:n), c=:black, ls=:dash, lw=1.5, label=L"C/{i}")
    plot!(p, 1:n, C ./ sqrt.(1:n), c=:red, ls=:dash, lw=1.5, label=L"C/\sqrt{i}")
    plot!(xscale=:log10, yscale=:log10)
    ylims!(ylimits...)
    xticks!(10 .^ (0:floor(Int, log10(n))))
    yticks!(ytickvalues)
    xlabel!(L"i" * "-th reorthonormalization step " * L"\mathrm{log}_{10} " * " scale")
    ylabel!(L"\mathrm{log}_{10}(|\lambda_i - \lambda_N|)")
    plot!(fontfamily="Computer Modern", guidefontsize=13, tickfontsize=13, legendfontsize=13, legend=:bottomleft)
    title!(title)
    annotate!(p, 1e+2, 1e-4, text("r = $(DATA["ro"][ridx])", 14, "Computer Modern"))
    display(p)
end


## Plot histograms with bell curve
ridx = 7
mdl = :ephec
plot_dky_distribution(RES["train_dky"][mdl], RES["train_dky"][:fom], ridx, "Train: EP-OpInf"; annote_loc=(3.3, 0.7))
plot_dky_distribution(TEST_RES["test1_dky"][mdl], TEST_RES["test1_dky"][:fom], ridx, "Test 1: EP-OpInf"; annote_loc=(3.3, 0.7))
plot_dky_distribution(TEST_RES["test2_dky"][mdl], TEST_RES["test2_dky"][:fom], ridx, "Test 2: EP-OpInf"; annote_loc=(3.3, 0.7))

## Plot LEmax distribution
ridx = 2
mdl = :ephec
plot_LEmax_distribution(RES["train_LE"][mdl], RES["train_LE"][:fom], ridx, "Train: EP-OpInf"; annote_loc=(-0.1, 10))
plot_LEmax_distribution(TEST_RES["test1_LE"][mdl], TEST_RES["test1_LE"][:fom], ridx, "Test 1: EP-OpInf"; annote_loc=(-0.1, 10))
plot_LEmax_distribution(TEST_RES["test2_LE"][mdl], TEST_RES["test2_LE"][:fom], ridx, "Test 2: EP-OpInf"; annote_loc=(-0.1, 10))

## Plot LE convergence
ridx = 2
mdl = :ephec
plot_LE_convergence(RES["train_LE"][mdl], ridx, 1, 200, "Train: EP-OpInf"; ylimits=(1e-5, 1e3))
plot_LE_convergence(TEST_RES["test1_LE"][mdl], ridx, 1, 200, "Test 1: EP-OpInf"; ylimits=(1e-5, 1e3))
plot_LE_convergence(TEST_RES["test2_LE"][mdl], ridx, 1, 200, "Test 2: EP-OpInf"; ylimits=(1e-5, 1e3))

## Plot LEmax errors
function plot_LEmax_error(data, ref, ro, title)
    p = plot()
    model_type = [:int, :LS, :ephec]
    r, _, n = size(data[model_type[1]])
    labels = ["Intru", "OpInf", "EP-OpInf"]
    errs = zeros(r,length(model_type))
    for (mi,model) in enumerate(model_type)
        for ri in 1:r
            err = 0.0
            for ni in 1:n
                err += abs(maximum(data[model][ri,1,ni][:,end]) - ref) / abs(ref)
            end
            errs[ri,mi] = err / n
        end
        plot!(p, ro, errs[:,mi], label=labels[mi], marker=true)
    end
    xticks!(ro)
    plot!(p, fontfamily="Computer Modern", legendfont=9, tickfont=12, guidefontsize=15,
        legend=:topright, xlabel=L"r", ylabel="Relative Error", title=title)
    display(p)
end

plot_LEmax_error(RES["train_LE"], 0.043, DATA["ro"], "Train: Max LE Error")
plot_LEmax_error(TEST_RES["test1_LE"], 0.043, DATA["ro"], "Test 1: Max LE Error")
plot_LEmax_error(TEST_RES["test2_LE"], 0.043, DATA["ro"], "Test 2: Max LE Error")

## Plot Dky errors
function plot_dky_error(data, ref, ro, title)
    p = plot()
    model_type = [:int, :LS, :ephec]
    r, _, n = size(data[model_type[1]])
    labels = ["Intru", "OpInf", "EP-OpInf"]
    errs = zeros(r,length(model_type))
    for (mi,model) in enumerate(model_type)
        for ri in 1:r
            err = 0.0
            for ni in 1:n
                err += abs(data[model][ri,1,ni] - ref) / abs(ref)
            end
            errs[ri,mi] = err / n
        end
        plot!(p, ro, errs[:,mi], label=labels[mi], marker=true)
    end
    xticks!(ro)
    ylims!(9e-3, 1e+0)
    yticks!(10.0 .^ (-3:1))
    plot!(p, yscale=:log10, fontfamily="Computer Modern", legendfont=9, tickfont=12, guidefontsize=15,
        legend=:topright, xlabel=L"r", ylabel="Relative Error", title=title)
    display(p)
end

plot_dky_error(RES["train_dky"], 4.2381, DATA["ro"], "Train: "*L"D_{ky}"*" Error")
plot_dky_error(TEST_RES["test1_dky"], 4.2381, DATA["ro"], "Test 1: "*L"D_{ky}"*" Error")
plot_dky_error(TEST_RES["test2_dky"], 4.2381, DATA["ro"], "Test 2: "*L"D_{ky}"*" Error")