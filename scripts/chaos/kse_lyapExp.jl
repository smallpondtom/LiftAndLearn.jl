"""
Energy-Preserving OpInf of Kuramoto-Kuramoto–Sivashinsky Equation and its analysis of chaos.
"""

#################
## Load packages
#################
using DelimitedFiles: writedlm, readdlm
using LinearAlgebra: svd
using ProgressMeter: @showprogress, Progress, next!


#####################
## Load main modules
#####################
using LiftAndLearn
const LnL = LiftAndLearn
const CG = LiftAndLearn.ChaosGizmo


#############
## Constants
#############
LOAD_DATA = false
SAVE_DATA = LOAD_DATA ? false : true
PRUNE_DATA = false  # data pruning flag to get only the chaotic region


#########
## Setup
#########
include("settings.jl")


########################################
## Generate training data and POD-basis
########################################
include("generate_kse_data.jl")
Xtr, Rtr, pod_basis, sing_vals, Xtr_all, training_IC = generate_kse_data(KSE, u0, ic_a, ic_b, num_ic_params, DS, PRUNE_DATA, prune_idx)

# Save the data
if SAVE_DATA
    writedlm("data/Xtr.csv", Xtr, ",")
    writedlm("data/Rtr.csv", Rtr, ",")
    writedlm("data/Vr.csv", pod_basis, ",")
    writedlm("data/Σr.csv", sing_vals, ",")
end

# *Only completed up to data generation stage. For full implementation, refer to the Jupyter notebook #10


# ## Training: Compute the Lyapunov exponent and Kaplan-Yorke dimension
# num_IC = length(DATA["IC_train"])
# RES["train_LE"] = Dict(
#     :int   => Array{Array{Float64}}(undef, length(ro), KSE.Pdim, num_IC),
#     :LS    => Array{Array{Float64}}(undef, length(ro), KSE.Pdim, num_IC),
#     :ephec => Array{Array{Float64}}(undef, length(ro), KSE.Pdim, num_IC),
#     :fom   => Array{Array{Float64}}(undef, KSE.Pdim)
# )
# RES["train_dky"] = Dict(
#     :int   => Array{Float64}(undef, length(ro), KSE.Pdim, num_IC),
#     :LS    => Array{Float64}(undef, length(ro), KSE.Pdim, num_IC),
#     :ephec => Array{Float64}(undef, length(ro), KSE.Pdim, num_IC),
#     :fom   => 0.0
# )

# ## Options
# options_fom = CG.LE_options(N=1e4, τ=1e3, Δt=0.01, Δτ=KSE.Δt, m=11, T=0.05, verbose=true, history=true)
# options_rom = CG.LE_options(N=1e4, τ=1e3, Δt=5*KSE.Δt, m=9, T=5*KSE.Δt, verbose=true, history=true)

# ## Compute the LE and Dky for all models 
# @info "Computing the LE and Dky for training..."
# compute_LE_oneIC!(RES, :fom, ["train_LE", "train_dky"], KSE, 
#         DATA["op_fom_tr"], DATA["IC_train"][1], KSE.integrate_FD, options_fom, 1)
# compute_LE_allIC!(RES, :int, ["train_LE", "train_dky"], KSE, 
#         OPS["op_int"], DATA["IC_train"], DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)
# compute_LE_allIC!(RES, :LS, ["train_LE", "train_dky"], KSE, 
#         OPS["op_LS"], DATA["IC_train"], DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)
# compute_LE_allIC!(RES, :ephec, ["train_LE", "train_dky"], KSE, 
#         OPS["op_ephec"], DATA["IC_train"], DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)

# ## Save data
# save(resultfile, "RES", RES)

# ## Testing
# GC.gc()

# TEST_RES = load(testresultfile)
# TEST_IC = load(testICfile)

# ## Organize the initial conditions into a matrix 
# TEST1_ICs = [u0(a,b) for (a,b) in zip(TEST_IC["ic_a_rand_in"], TEST_IC["ic_b_rand_in"])]
# TEST2_ICs = [u0(a,b) for (a,b) in zip(TEST_IC["ic_a_rand_out"], TEST_IC["ic_b_rand_out"])]

# num_IC = length(TEST_IC["ic_a_rand_in"])

# TEST_RES["test1_LE"] = Dict(
#     :int   => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
#     :LS    => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
#     :ephec => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
#     :fom   => Array{Array{Float64}}(undef, KSE.Pdim)
# )
# TEST_RES["test2_LE"] = Dict(
#     :int   => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
#     :LS    => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
#     :ephec => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
#     :fom   => Array{Array{Float64}}(undef, KSE.Pdim)
# )
# TEST_RES["test1_dky"] = Dict(
#     :int   => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
#     :LS    => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
#     :ephec => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
#     :fom   => 0.0
# )
# TEST_RES["test2_dky"] = Dict(
#     :int   => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
#     :LS    => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
#     :ephec => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
#     :fom   => 0.0
# )


# ## Test 1
# @info "Computing the LE and Dky for test 1..."
# compute_LE_oneIC!(TEST_RES, :fom, ["test1_LE", "test1_dky"], KSE, 
#     DATA["op_fom_tr"], TEST1_ICs[1], KSE.integrate_FD, options_fom, 1)
# compute_LE_allIC!(TEST_RES, :int, ["test1_LE", "test1_dky"], KSE, 
#     OPS["op_int"], TEST1_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)
# compute_LE_allIC!(TEST_RES, :LS, ["test1_LE", "test1_dky"], KSE, 
#     OPS["op_LS"], TEST1_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)
# compute_LE_allIC!(TEST_RES, :ephec, ["test1_LE", "test1_dky"], KSE, 
#     OPS["op_ephec"], TEST1_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)

# ## Test 2
# @debug "Computing the LE and Dky for test 2..."
# compute_LE_oneIC!(TEST_RES, :fom, ["test2_LE", "test2_dky"], KSE, 
#     DATA["op_fom_tr"], TEST2_ICs[1], KSE.integrate_FD, options_fom, 1)
# compute_LE_allIC!(TEST_RES, :int, ["test2_LE", "test2_dky"], KSE, 
#     OPS["op_int"], TEST2_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)
# compute_LE_allIC!(TEST_RES, :LS, ["test2_LE", "test2_dky"], KSE, 
#     OPS["op_LS"], TEST2_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)
# compute_LE_allIC!(TEST_RES, :ephec, ["test2_LE", "test2_dky"], KSE, 
#     OPS["op_ephec"], TEST2_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options_rom)

# ## Save data
# save(testresultfile, "TEST_RES", TEST_RES)

# ## Plotting Results 

# ## Plot histograms with bell curve
# ridx = 7
# mdl = :ephec
# plot_dky_distribution(RES["train_dky"][mdl], RES["train_dky"][:fom], ridx, "Train: EP-OpInf"; annote_loc=(3.3, 0.7))
# plot_dky_distribution(TEST_RES["test1_dky"][mdl], TEST_RES["test1_dky"][:fom], ridx, "Test 1: EP-OpInf"; annote_loc=(3.3, 0.7))
# plot_dky_distribution(TEST_RES["test2_dky"][mdl], TEST_RES["test2_dky"][:fom], ridx, "Test 2: EP-OpInf"; annote_loc=(3.3, 0.7))

# ## Plot LEmax distribution
# ridx = 2
# mdl = :ephec
# plot_LEmax_distribution(RES["train_LE"][mdl], RES["train_LE"][:fom], ridx, "Train: EP-OpInf"; annote_loc=(-0.1, 10))
# plot_LEmax_distribution(TEST_RES["test1_LE"][mdl], TEST_RES["test1_LE"][:fom], ridx, "Test 1: EP-OpInf"; annote_loc=(-0.1, 10))
# plot_LEmax_distribution(TEST_RES["test2_LE"][mdl], TEST_RES["test2_LE"][:fom], ridx, "Test 2: EP-OpInf"; annote_loc=(-0.1, 10))

# ## Plot LE convergence
# ridx = 2
# mdl = :ephec
# plot_LE_convergence(RES["train_LE"][mdl], ridx, 1, 200, "Train: EP-OpInf"; ylimits=(1e-5, 1e3))
# plot_LE_convergence(TEST_RES["test1_LE"][mdl], ridx, 1, 200, "Test 1: EP-OpInf"; ylimits=(1e-5, 1e3))
# plot_LE_convergence(TEST_RES["test2_LE"][mdl], ridx, 1, 200, "Test 2: EP-OpInf"; ylimits=(1e-5, 1e3))

# ## Plot LEmax errors
# plot_LEmax_error(RES["train_LE"], 0.043, DATA["ro"], "Train: Max LE Error")
# plot_LEmax_error(TEST_RES["test1_LE"], 0.043, DATA["ro"], "Test 1: Max LE Error")
# plot_LEmax_error(TEST_RES["test2_LE"], 0.043, DATA["ro"], "Test 2: Max LE Error")

# ## Plot Dky errors
# plot_dky_error(RES["train_dky"], 4.2381, DATA["ro"], "Train: "*L"D_{ky}"*" Error")
# plot_dky_error(TEST_RES["test1_dky"], 4.2381, DATA["ro"], "Test 1: "*L"D_{ky}"*" Error")
# plot_dky_error(TEST_RES["test2_dky"], 4.2381, DATA["ro"], "Test 2: "*L"D_{ky}"*" Error")