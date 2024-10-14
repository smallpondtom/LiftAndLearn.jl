"""
Kuramoto–Sivashinsky equation testing for EP-OpInf
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

#================#
## Load the data
#================#
DATA = load("./scripts/EP-OpInf/data/kse_epopinf_data.jld2")
RES = load("./scripts/EP-OpInf/data/kse_epopinf_results.jld2")
KSE = DATA["KSE"]
OPS = load("./scripts/EP-OpInf/data/kse_epopinf_ops.jld2")

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

# Parameters of the training initial condition
ic_a = [0.8, 1.0, 1.2]
ic_b = [0.2, 0.4, 0.6]

#============================#
## Generate the test 1 data
#============================#
Xtest1 = Vector{Matrix{Float64}}(undef, KSE.param_dim, num_test_ic)  # training state data 
IC_test1 = Vector{Vector{Float64}}(undef, num_test_ic)  # all initial conditions 

# Generate random initial condition parameters
ic_a_rand_in = (maximum(ic_a) - minimum(ic_a)) .* rand(num_test_ic) .+ minimum(ic_a)
ic_b_rand_in = (maximum(ic_b) - minimum(ic_b)) .* rand(num_test_ic) .+ minimum(ic_b)

i = 1
μ = KSE.diffusion_coeffs[i]

# Generate the FOM system matrices (ONLY DEPENDS ON μ)
A = DATA["op_fom_tr"][i].A
F = DATA["op_fom_tr"][i].A2u

prog = Progress(num_test_ic)
Threads.@threads for j in 1:num_test_ic  
    # generate test 1 data
    a = ic_a_rand_in[j]
    b = ic_b_rand_in[j]
    IC = u0(a,b)
    Xtest_in = KSE.integrate_model(KSE.tspan, IC; linear_matrix=A, quadratic_matrix=F, 
                                   system_input=false, const_stepsize=true)
    Xtest1[i,j] = Xtest_in
    IC_test1[j] = IC
    next!(prog)
end

#====================#
## Save test 1 data
#====================#
Xtest = Dict("Xtest1" => Xtest1, "IC_test1" => IC_test1)
save("./scripts/EP-OpInf/data/kse_epopinf_test1_data.jld2", Xtest)

#==============================================#
## Test 1: Normalized autocorrelation function
#==============================================#
TEST1_RES = Dict{String, Any}()
TEST1_RES["test1_AC"] = Dict(
    :int   => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.param_dim),
    :LS    => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.param_dim),
    :ephec => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.param_dim),
    :epsic => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.param_dim),
    :epp => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.param_dim),
    :fom   => Array{Array{Float64}}(undef, KSE.param_dim)
)
TEST1_RES["test1_AC_ERR"] = Dict(
    :int   => Array{Float64}(undef, length(DATA["ro"]), KSE.param_dim),
    :LS    => Array{Float64}(undef, length(DATA["ro"]), KSE.param_dim),
    :ephec => Array{Float64}(undef, length(DATA["ro"]), KSE.param_dim),
    :epsic => Array{Float64}(undef, length(DATA["ro"]), KSE.param_dim),
    :epp => Array{Float64}(undef, length(DATA["ro"]), KSE.param_dim),
)

# lag for autocorrelation
lags = 0:DS:(KSE.time_dim ÷ 2)
idx = length(lags)

# Store some arrays
fom_ac1 = zeros(length(lags))
int_ac1 = zeros(length(lags), length(DATA["ro"]))
LS_ac1 = zeros(length(lags), length(DATA["ro"]))
ephec_ac1 = zeros(length(lags), length(DATA["ro"]))
epsic_ac1 = zeros(length(lags), length(DATA["ro"]))
epp_ac1 = zeros(length(lags), length(DATA["ro"]))

int_ac_err1 = zeros(length(DATA["ro"]))
LS_ac_err1 = zeros(length(DATA["ro"]))
ephec_ac_err1 = zeros(length(DATA["ro"]))
ephsic_ac_err1 = zeros(length(DATA["ro"]))
epp_ac_err1 = zeros(length(DATA["ro"]))

# Generate the data for all combinations of the initial condition parameters
prog = Progress(num_test_ic)
Threads.@threads for j in 1:num_test_ic  
    # compute autocorrelation of FOM
    fom_ac_j = tmean_autocorr(Xtest1[1,j], lags)
    fom_ac1 .+= fom_ac_j

    IC = IC_test1[j]

    # compute autocorrelation of ROMs
    int_ac_tmp = kse_analyze_autocorr(OPS["op_int"], KSE, DATA["Vr"], IC, DATA["ro"], KSE.integrate_model, lags)
    LS_ac_tmp = kse_analyze_autocorr(OPS["op_LS"], KSE, DATA["Vr"], IC, DATA["ro"], KSE.integrate_model, lags)
    ephec_ac_tmp = kse_analyze_autocorr(OPS["op_ephec"], KSE, DATA["Vr"], IC, DATA["ro"], KSE.integrate_model, lags)
    epsic_ac_tmp = kse_analyze_autocorr(OPS["op_epsic"], KSE, DATA["Vr"], IC, DATA["ro"], KSE.integrate_model, lags)
    epp_ac_tmp = kse_analyze_autocorr(OPS["op_epp"], KSE, DATA["Vr"], IC, DATA["ro"], KSE.integrate_model, lags)
    for r in eachindex(DATA["ro"])
        int_ac_err1[r] += norm(int_ac_tmp[r,1] - fom_ac_j, 2) / norm(fom_ac_j, 2)
        int_ac1[:,r] .+= int_ac_tmp[r,1]

        LS_ac_err1[r] += norm(LS_ac_tmp[r,1] - fom_ac_j, 2) / norm(fom_ac_j, 2)
        LS_ac1[:,r] .+= LS_ac_tmp[r,1]

        ephec_ac_err1[r] += norm(ephec_ac_tmp[r,1] - fom_ac_j, 2) / norm(fom_ac_j, 2)
        ephec_ac1[:,r] .+= ephec_ac_tmp[r,1]

        epsic_ac_err1[r] += norm(epsic_ac_tmp[r,1] - fom_ac_j, 2) / norm(fom_ac_j, 2)
        epsic_ac1[:,r] .+= epsic_ac_tmp[r,1]

        epp_ac_err1[r] += norm(epp_ac_tmp[r,1] - fom_ac_j, 2) / norm(fom_ac_j, 2)
        epp_ac1[:,r] .+= epp_ac_tmp[r,1]
    end
    next!(prog)
end

# Compute the average of all initial conditions as the final result
TEST11_RES["test1_AC"][:fom][1] = fom_ac1 ./ num_test_ic
for r in eachindex(DATA["ro"])
    TEST1_RES["test1_AC"][:int][r,1] = int_ac1[:,r] / num_test_ic
    TEST1_RES["test1_AC"][:LS][r,1] = LS_ac1[:,r] / num_test_ic
    TEST1_RES["test1_AC"][:ephec][r,1] = ephec_ac1[:,r] / num_test_ic
    TEST1_RES["test1_AC"][:epsic][r,1] = epsic_ac1[:,r] / num_test_ic
    TEST1_RES["test1_AC"][:epp][r,1] = epp_ac1[:,r] / num_test_ic

    TEST1_RES["test1_AC_ERR"][:int][r,1] = int_ac_err1[r] / num_test_ic
    TEST1_RES["test1_AC_ERR"][:LS][r,1] = LS_ac_err1[r] / num_test_ic
    TEST1_RES["test1_AC_ERR"][:ephec][r,1] = ephec_ac_err1[r] / num_test_ic
    TEST1_RES["test1_AC_ERR"][:epsic][r,1] = epsic_ac_err1[r] / num_test_ic
    TEST1_RES["test1_AC_ERR"][:epp][r,1] = epp_ac_err1[r] / num_test_ic
end

#======================#
## Save test 1 results
#======================#
save("./scripts/EP-OpInf/data/kse_epopinf_test1_results.jld2", TEST1_RES)

#============================#
## Generate the test 2 data
#============================#
Xtest2 = Vector{Matrix{Float64}}(undef, KSE.param_dim, num_test_ic)  # training state data 
IC_test2 = Vector{Vector{Float64}}(undef, num_test_ic)  # all initial conditions 

# Generate random initial condition parameters
ic_a_out = [0.0, 2.0]
ic_b_out = [0.0, 0.8]
ϵ=1e-2
half_num_test_ic = Int(num_test_ic/2)
ic_a_rand_out = ((minimum(ic_a) - ϵ) - ic_a_out[1]) .* rand(half_num_test_ic) .+ ic_a_out[1]
append!(ic_a_rand_out, (ic_a_out[2] - (maximum(ic_a) + ϵ)) .* rand(half_num_test_ic) .+ (maximum(ic_a) + ϵ))
ic_b_rand_out = ((minimum(ic_b) - ϵ) - ic_b_out[1]) .* rand(half_num_test_ic) .+ ic_b_out[1]
append!(ic_b_rand_out, (ic_b_out[2] - (maximum(ic_b) + ϵ)) .* rand(half_num_test_ic) .+ (maximum(ic_b) + ϵ))

i = 1
μ = KSE.μs[i]

# Generate the FOM system matrices (ONLY DEPENDS ON μ)
A = DATA["op_fom_tr"][i].A
F = DATA["op_fom_tr"][i].A2u

prog = Progress(num_test_ic)
Threads.@threads for j in 1:num_test_ic  
    # generate test 1 data
    a = ic_a_rand_out[j]
    b = ic_b_rand_out[j]
    IC = u0(a,b)
    Xtest_out = KSE.integrate_model(KSE.tspan, IC; linear_matrix=A, quadratic_matrix=F, 
                                    system_input=false, const_stepsize=true)
    Xtest2[i,j] = Xtest_out
    IC_test2[j] = IC
    next!(prog)
end

#====================#
## Save test 2 data
#====================#
Xtest = Dict("Xtest1" => Xtest2, "IC_test1" => IC_test2)
save("./scripts/EP-OpInf/data/kse_epopinf_test2_data.jld2", Xtest)

#==============================================#
## Test 2: Normalized autocorrelation function
#==============================================#
# lag for autocorrelation
lags = 0:DS:(KSE.time_dim ÷ 2)
idx = length(lags)

# Store some arrays
fom_ac2 = zeros(length(lags))
int_ac2 = zeros(length(lags), length(DATA["ro"]))
LS_ac2 = zeros(length(lags), length(DATA["ro"]))
ephec_ac2 = zeros(length(lags), length(DATA["ro"]))
epsic_ac2 = zeros(length(lags), length(DATA["ro"]))
epp_ac2 = zeros(length(lags), length(DATA["ro"]))

int_ac_err2 = zeros(length(DATA["ro"]))
LS_ac_err2 = zeros(length(DATA["ro"]))
ephec_ac_err2 = zeros(length(DATA["ro"]))
epsic_ac_err2 = zeros(length(DATA["ro"]))
epp_ac_err2 = zeros(length(DATA["ro"]))

# Generate the data for all combinations of the initial condition parameters
prog = Progress(num_test_ic)
Threads.@threads for j in 1:num_test_ic  
    # compute autocorrelation of FOM
    fom_ac_j = tmean_autocorr(Xtest2[1,j], lags)
    fom_ac2 .+= fom_ac_j

    IC = IC_test2[j]

    # compute autocorrelation of ROMs
    int_ac_tmp = analyze_autocorr(OPS["op_int"], KSE, DATA["Vr"], IC, DATA["ro"], KSE.integrate_model, lags)
    LS_ac_tmp = analyze_autocorr(OPS["op_LS"], KSE, DATA["Vr"], IC, DATA["ro"], KSE.integrate_model, lags)
    ephec_ac_tmp = analyze_autocorr(OPS["op_ephec"], KSE, DATA["Vr"], IC, DATA["ro"], KSE.integrate_model, lags)
    epsic_ac_tmp = analyze_autocorr(OPS["op_epsic"], KSE, DATA["Vr"], IC, DATA["ro"], KSE.integrate_model, lags)
    epp_ac_tmp = analyze_autocorr(OPS["op_epp"], KSE, DATA["Vr"], IC, DATA["ro"], KSE.integrate_model, lags)
    for r in eachindex(DATA["ro"])
        int_ac_err2[r] += norm(int_ac_tmp[r,1] - fom_ac_j, 2) / norm(fom_ac_j, 2)
        int_ac2[:,r] .+= int_ac_tmp[r,1]

        LS_ac_err2[r] += norm(LS_ac_tmp[r,1] - fom_ac_j, 2) / norm(fom_ac_j, 2)
        LS_ac2[:,r] .+= LS_ac_tmp[r,1]

        ephec_ac_err2[r] += norm(ephec_ac_tmp[r,1] - fom_ac_j, 2) / norm(fom_ac_j, 2)
        ephec_ac2[:,r] .+= ephec_ac_tmp[r,1]
        
        epsic_ac_err2[r] += norm(epsic_ac_tmp[r,1] - fom_ac_j, 2) / norm(fom_ac_j, 2)
        epsic_ac2[:,r] .+= epsic_ac_tmp[r,1]

        epp_ac_err2[r] += norm(epp_ac_tmp[r,1] - fom_ac_j, 2) / norm(fom_ac_j, 2)
        epp_ac2[:,r] .+= epp_ac_tmp[r,1]
    end
    next!(prog)
end

TEST2_RES["test2_AC"][:fom][1] = fom_ac2 ./ num_test_ic
for r in eachindex(DATA["ro"])
    TEST2_RES["test2_AC"][:int][r,1] = int_ac2[:,r] / num_test_ic
    TEST2_RES["test2_AC"][:LS][r,1] = LS_ac2[:,r] / num_test_ic
    TEST2_RES["test2_AC"][:ephec][r,1] = ephec_ac2[:,r] / num_test_ic
    TEST2_RES["test2_AC"][:epsic][r,1] = epsic_ac2[:,r] / num_test_ic
    TEST2_RES["test2_AC"][:epp][r,1] = epp_ac2[:,r] / num_test_ic

    TEST2_RES["test2_AC_ERR"][:int][r,1] = int_ac_err2[r] / num_test_ic
    TEST2_RES["test2_AC_ERR"][:LS][r,1] = LS_ac_err2[r] / num_test_ic
    TEST2_RES["test2_AC_ERR"][:ephec][r,1] = ephec_ac_err2[r] / num_test_ic
    TEST2_RES["test2_AC_ERR"][:epsic][r,1] = epsic_ac_err2[r] / num_test_ic
    TEST2_RES["test2_AC_ERR"][:epp][r,1] = epp_ac_err2[r] / num_test_ic
end
