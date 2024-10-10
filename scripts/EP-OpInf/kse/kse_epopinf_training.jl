"""
Kuramoto–Sivashinsky equation training for EP-OpInf
"""

#================#
## Load packages
#================#
using ChaosGizmo
using FileIO
using JLD2
using LinearAlgebra
using ProgressMeter
using StatsBase
using Statistics
using UniqueKronecker

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

#===================#
## Import functions
#===================#
include(joinpath(FILEPATH, "utils/kse_analyze.jl"))

#======================#
## Load all data files
#======================#
KSE = load(joinpath(FILEPATH, "data/kse_epopinf_model_setting.jld2"), "KSE")
OPS = load(joinpath(FILEPATH, "data/kse_epopinf_ops.jld2"), "OPS")
REDUCTION_INFO = load(joinpath(FILEPATH, "data/kse_epopinf_reduction_info.jld2"), "REDUCTION_INFO")
DS = 100

#=========================#
## Get all data filenames
#=========================#
training_data_files = readdir(joinpath(FILEPATH, "data/kse_train"), join=true)

#==========================#
## Prepare to save results
#==========================#
RES = Dict{String, Any}()
RES["proj_err"] = Array{Float64}(undef, length(REDUCTION_INFO["ro"]), KSE.param_dim)

#=================#
## Project errors
#=================#
@info "Compute projection errors"
RES["proj_err"] = kse_analyze_proj_err(KSE, training_data_files, REDUCTION_INFO["Vr"], 
                                             REDUCTION_INFO["ro"], REDUCTION_INFO["num_ic_params"])
@info "Done."

#========================#
## Relative state errors
#========================#
@info "Compute relative state errors"
RES["state_err"] = Dict(
    :int => Array{Float64}(undef, length(REDUCTION_INFO["ro"]), KSE.param_dim),
    :LS => Array{Float64}(undef, length(REDUCTION_INFO["ro"]), KSE.param_dim),
    :ephec => Array{Float64}(undef, length(REDUCTION_INFO["ro"]), KSE.param_dim),
    # :epsic => Array{Float64}(undef, length(REDUCTION_INFO["ro"]), KSE.param_dim),
    # :epp => Array{Float64}(undef, length(REDUCTION_INFO["ro"]), KSE.param_dim),
)

# Standard OpInf
RES["state_err"][:LS] = kse_analyze_rse(
    OPS["op_LS"], KSE, training_data_files, REDUCTION_INFO["Vr"], REDUCTION_INFO["num_ic_params"], REDUCTION_INFO["ro"], KSE.integrate_model
    )
# Intrusive
RES["state_err"][:int] = kse_analyze_rse(
    OPS["op_int"], KSE, training_data_files, REDUCTION_INFO["Vr"], REDUCTION_INFO["num_ic_params"], REDUCTION_INFO["ro"], KSE.integrate_model
    )
# EPHEC
RES["state_err"][:ephec] = kse_analyze_rse(
    OPS["op_ephec"], KSE, training_data_files, REDUCTION_INFO["Vr"], REDUCTION_INFO["num_ic_params"], REDUCTION_INFO["ro"], KSE.integrate_model
    )
# # EPSIC
# RES["state_err"][:epsic] = kse_analyze_rse(
#     OPS["op_epsic"], KSE, training_data_files, REDUCTION_INFO["Vr"], REDUCTION_INFO["num_ic_params"], REDUCTION_INFO["ro"], KSE.integrate_model
#     )
# # EPP
# RES["state_err"][:epp] = kse_analyze_rse(
#     OPS["op_epp"], KSE, training_data_files, REDUCTION_INFO["Vr"], REDUCTION_INFO["num_ic_params"], REDUCTION_INFO["ro"], KSE.integrate_model
#     )
@info "Done."

#=======================#
## Constraint residuals
#=======================#
@info "Compute constraint residuals"
RES["CR"] = Dict(
    :int => Array{Float64}(undef, length(REDUCTION_INFO["ro"]), KSE.param_dim),
    :LS => Array{Float64}(undef, length(REDUCTION_INFO["ro"]), KSE.param_dim),
    :ephec => Array{Float64}(undef, length(REDUCTION_INFO["ro"]), KSE.param_dim),
    # :epsic => Array{Float64}(undef, length(REDUCTION_INFO["ro"]), KSE.param_dim),
    # :epp => Array{Float64}(undef, length(REDUCTION_INFO["ro"]), KSE.param_dim),
    :fom => Array{Float64}(undef, KSE.param_dim)
)

# Compute CR of full order model
RES["CR"][:fom] = kse_fom_CR(REDUCTION_INFO["op_fom_tr"], KSE)
# Standard OpInf
RES["CR"][:LS] = kse_analyze_cr(OPS["op_LS"], KSE, REDUCTION_INFO["num_ic_params"], REDUCTION_INFO["ro"])
# Intrusive
RES["CR"][:int] = kse_analyze_cr(OPS["op_int"], KSE, REDUCTION_INFO["num_ic_params"], REDUCTION_INFO["ro"])
# EPHEC
RES["CR"][:ephec] = kse_analyze_cr(OPS["op_ephec"], KSE, REDUCTION_INFO["num_ic_params"], REDUCTION_INFO["ro"])
# # EPSIC
# RES["CR"][:epsic] = kse_analyze_cr(OPS["op_epsic"], KSE, REDUCTION_INFO["num_ic_params"], REDUCTION_INFO["ro"])
# # EPP
# RES["CR"][:epp] = kse_analyze_cr(OPS["op_epp"], KSE, REDUCTION_INFO["num_ic_params"], REDUCTION_INFO["ro"])
@info "Done."

#=======================================#
## Normalized Autocorrelation functions
#=======================================#
@info "Compute normalized autocorrelation functions"
# Time lag
lags = 0:DS:(KSE.time_dim ÷ 2)

RES["AC_lags"] = lags
RES["AC"] = Dict(
    :int   => Array{Float64}(undef, length(lags), length(REDUCTION_INFO["ro"])),
    :LS    => Array{Float64}(undef, length(lags), length(REDUCTION_INFO["ro"])),
    :ephec => Array{Float64}(undef, length(lags), length(REDUCTION_INFO["ro"])),
    # :epsic => Array{Float64}(undef, length(lags), length(REDUCTION_INFO["ro"])),
    # :epp   => Array{Float64}(undef, length(lags), length(REDUCTION_INFO["ro"])),
    :fom   => Array{Float64}(undef, length(lags))
)
RES["AC_ERR"] = Dict(
    :int   => Array{Float64}(undef, length(REDUCTION_INFO["ro"]), KSE.param_dim),
    :LS    => Array{Float64}(undef, length(REDUCTION_INFO["ro"]), KSE.param_dim),
    :ephec => Array{Float64}(undef, length(REDUCTION_INFO["ro"]), KSE.param_dim),
    # :epsic => Array{Float64}(undef, length(REDUCTION_INFO["ro"]), KSE.param_dim),
    # :epp   => Array{Float64}(undef, length(REDUCTION_INFO["ro"]), KSE.param_dim),
)

# Compute autocorrelation functions
ac_fom = zeros(length(lags))
ac_LS = zeros(length(lags), length(REDUCTION_INFO["ro"]))
ac_LS_err = zeros(length(REDUCTION_INFO["ro"]))
ac_int = zeros(length(lags), length(REDUCTION_INFO["ro"]))
ac_int_err = zeros(length(REDUCTION_INFO["ro"]))
ac_ephec = zeros(length(lags), length(REDUCTION_INFO["ro"]))
ac_ephec_err = zeros(length(REDUCTION_INFO["ro"]))
# ac_epsic = zeros(length(lags), length(REDUCTION_INFO["ro"]))
# ac_epsic_err = zeros(length(REDUCTION_INFO["ro"]))
# ac_epp = zeros(length(lags), length(REDUCTION_INFO["ro"]))
# ac_epp_err = zeros(length(REDUCTION_INFO["ro"]))

bar = REDUCTION_INFO["num_ic_params"]
prog = Progress(bar)

Threads.@threads for data_file in training_data_files
    jldopen(data_file, "r") do file
        IC = file["IC"]
        X = file["X"]

        ac_fom_tmp = kse_analyze_autocorr(KSE, X, lags)[1]
        ac_fom .+= ac_fom_tmp

        ac_LS_tmp = kse_analyze_autocorr(OPS["op_LS"], KSE, REDUCTION_INFO["Vr"], IC, REDUCTION_INFO["ro"], KSE.integrate_model, lags)
        ac_int_tmp = kse_analyze_autocorr(OPS["op_int"], KSE, REDUCTION_INFO["Vr"], IC, REDUCTION_INFO["ro"], KSE.integrate_model, lags)
        ac_ephec_tmp = kse_analyze_autocorr(OPS["op_ephec"], KSE, REDUCTION_INFO["Vr"], IC, REDUCTION_INFO["ro"], KSE.integrate_model, lags)
        # ac_epsic_tmp = kse_analyze_autocorr(OPS["op_epsic"], KSE, REDUCTION_INFO["Vr"], IC, REDUCTION_INFO["ro"], KSE.integrate_model, lags)
        # ac_epp_tmp = kse_analyze_autocorr(OPS["op_epp"], KSE, REDUCTION_INFO["Vr"], IC, REDUCTION_INFO["ro"], KSE.integrate_model, lags)
        for r in eachindex(REDUCTION_INFO["ro"])
            ac_LS[:,r] .+= ac_LS_tmp[r,1]
            ac_LS_err[r] += norm(ac_LS_tmp[r,1] - ac_fom_tmp, 2) / norm(ac_fom_tmp, 2)

            ac_int[:,r] .+= ac_int_tmp[r,1]
            ac_int_err[r] += norm(ac_int_tmp[r,1] - ac_fom_tmp, 2) / norm(ac_fom_tmp, 2)

            ac_ephec[:,r] .+= ac_ephec_tmp[r,1]
            ac_ephec_err[r] += norm(ac_ephec_tmp[r,1] - ac_fom_tmp, 2) / norm(ac_fom_tmp, 2)

            # ac_epsic[:,r] .+= ac_epsic_tmp[r,1]
            # ac_epsic_err[r] += norm(ac_epsic_tmp[r,1] - ac_fom_tmp, 2) / norm(ac_fom_tmp, 2)

            # ac_epp[:,r] .+= ac_epp_tmp[r,1]
            # ac_epp_err[r] += norm(ac_epp_tmp[r,1] - ac_fom_tmp, 2) / norm(ac_fom_tmp, 2)
        end
    end
    next!(prog)
end
# save the mean normalized autocorrelation
RES["AC"][:fom] = ac_fom ./ bar
for r in eachindex(REDUCTION_INFO["ro"])
    RES["AC"][:LS][:,r] = ac_LS[:,r] ./ bar
    RES["AC"][:int][:,r] = ac_int[:,r] ./ bar
    RES["AC"][:ephec][:,r] = ac_ephec[:,r] ./ bar
    # RES["AC"][:epsic][:,r] = ac_epsic[:,r] ./ bar
    # RES["AC"][:epp][:,r] = ac_epp[:,r] ./ bar
end
# average over all initial conditions
ac_LS_err ./= bar  
ac_int_err ./= bar
ac_ephec_err ./= bar
ac_epsic_err ./= bar
ac_epp_err ./= bar
RES["AC_ERR"][:LS] = reshape(ac_LS_err, length(ac_LS_err), 1)
RES["AC_ERR"][:int] = reshape(ac_int_err, length(ac_int_err), 1)
RES["AC_ERR"][:ephec] = reshape(ac_ephec_err, length(ac_ephec_err), 1)
# RES["AC_ERR"][:epsic] = reshape(ac_epsic_err, length(ac_epsic_err), 1)
# RES["AC_ERR"][:epp] = reshape(ac_epp_err, length(ac_epp_err), 1)
@info "Done."

#================================================#
## Lyapunov exponents and Kaplan-Yorke dimension
#================================================#
@info "Compute the Lyapunov exponents and the Kaplan-Yorke dimensions"

# Lyapunov exponent Settings
max_num_of_LE = 3
LEOption = ChaosGizmo.LyapunovExponentOptions(
    m=max_num_of_LE, τ=2e+3, T=1, Δt=0.01, N=1e+4, ϵ=1e-6, verbose=false, jacobian=true,
)

RES["LE"] = Dict(
    :int   => Array{Float64}(undef, max_num_of_LE, length(REDUCTION_INFO["ro"])),
    :LS    => Array{Float64}(undef, max_num_of_LE, length(REDUCTION_INFO["ro"])),
    :ephec => Array{Float64}(undef, max_num_of_LE, length(REDUCTION_INFO["ro"])),
    # :epsic => Array{Float64}(undef, max_num_of_LE, length(REDUCTION_INFO["ro"])),
    # :epp   => Array{Float64}(undef, max_num_of_LE, length(REDUCTION_INFO["ro"])),
    # :fom   => Array{Float64}(undef, max_num_of_LE)
)
RES["KY"] = Dict(
    :int   => Array{Float64}(undef, length(REDUCTION_INFO["ro"])),
    :LS    => Array{Float64}(undef, length(REDUCTION_INFO["ro"])),
    :ephec => Array{Float64}(undef, length(REDUCTION_INFO["ro"])),
    # :epsic => Array{Float64}(undef, length(REDUCTION_INFO["ro"])),
    # :epp   => Array{Float64}(undef, length(REDUCTION_INFO["ro"])),
    # :fom   => 0.0
)

# Compute Lypuanov exponents
# le_fom = zeros(10)
le_LS = zeros(max_num_of_LE, length(REDUCTION_INFO["ro"]))
le_int = zeros(max_num_of_LE, length(REDUCTION_INFO["ro"]))
le_ephec = zeros(max_num_of_LE, length(REDUCTION_INFO["ro"]))
# le_epsic = zeros(max_num_of_LE, length(REDUCTION_INFO["ro"]))
# le_epp = zeros(max_num_of_LE, length(REDUCTION_INFO["ro"]))

# Compute Kaplan-Yorke dimensions
# ky_fom = 0.0
ky_LS = zeros(length(REDUCTION_INFO["ro"]))
ky_int = zeros(length(REDUCTION_INFO["ro"]))
ky_ephec = zeros(length(REDUCTION_INFO["ro"]))
# ky_epsic = zeros(length(REDUCTION_INFO["ro"]))
# ky_epp = zeros(length(REDUCTION_INFO["ro"]))

bar = REDUCTION_INFO["num_ic_params"]
prog = Progress(bar)

Threads.@threads for data_file in training_data_files
    jldopen(data_file, "r") do file
        IC = file["IC"]

        # Lyapunov exponents
        # le_fom_tmp = kse_lypuanov_exponent(REDUCTION_INFO["op_fom_tr"], KSE, IC, KSE.integrate_model, LEOption)
        # le_fom .+= le_fom_tmp

        le_LS_tmp = kse_lyapunov_exponent(OPS["op_LS"], KSE, REDUCTION_INFO["Vr"], IC, REDUCTION_INFO["ro"], KSE.integrate_model, LEOption; jacobian=KSE.jacobian)
        le_int_tmp = kse_lyapunov_exponent(OPS["op_int"], KSE, REDUCTION_INFO["Vr"], IC, REDUCTION_INFO["ro"], KSE.integrate_model, LEOption; jacobian=KSE.jacobian)
        le_ephec_tmp = kse_lyapunov_exponent(OPS["op_ephec"], KSE, REDUCTION_INFO["Vr"], IC, REDUCTION_INFO["ro"], KSE.integrate_model, LEOption; jacobian=KSE.jacobian)
        # le_epsic_tmp = kse_lyapunov_exponent(OPS["op_epsic"], KSE, REDUCTION_INFO["Vr"], IC, REDUCTION_INFO["ro"], KSE.integrate_model, LEOption; jacobian=KSE.jacobian)
        # le_epp_tmp = kse_lyapunov_exponent(OPS["op_epp"], KSE, REDUCTION_INFO["Vr"], IC, REDUCTION_INFO["ro"], KSE.integrate_model, LEOption; jacobian=KSE.jacobian)

        # Kaplan-Yorke dimensions
        # ky_fom_tmp = ChaosGizmo.kaplan_yorke_dim(le_fom_tmp)
        # ky_fom += ky_fom_tmp

        ky_LS_tmp = [ChaosGizmo.kaplan_yorke_dim(le_LS_tmp[r,1]) for r in eachindex(REDUCTION_INFO["ro"])]
        ky_int_tmp = [ChaosGizmo.kaplan_yorke_dim(le_int_tmp[r,1]) for r in eachindex(REDUCTION_INFO["ro"])]
        ky_ephec_tmp = [ChaosGizmo.kaplan_yorke_dim(le_ephec_tmp[r,1]) for r in eachindex(REDUCTION_INFO["ro"])]
        # ky_epsic_tmp = [ChaosGizmo.kaplan_yorke_dim(le_epsic_tmp[r,1]) for r in eachindex(REDUCTION_INFO["ro"])]
        # ky_epp_tmp = [ChaosGizmo.kaplan_yorke_dim(le_epp_tmp[r,1]) for r in eachindex(REDUCTION_INFO["ro"])]

        for r in eachindex(REDUCTION_INFO["ro"])
            le_LS[:,r] .+= le_LS_tmp[r,1]
            le_int[:,r] .+= le_int_tmp[r,1]
            le_ephec[:,r] .+= le_ephec_tmp[r,1]
            # le_epsic[:,r] .+= le_epsic_tmp[r,1]
            # le_epp[:,r] .+= le_epp_tmp[r,1]

            ky_LS[r] += ky_LS_tmp[r]
            ky_int[r] += ky_int_tmp[r]
            ky_ephec[r] += ky_ephec_tmp[r]
            # ky_epsic[r] += ky_epsic_tmp[r]
            # ky_epp[r] += ky_epp_tmp[r]
        end
    end
    next!(prog)
end
# save the mean normalized autocorrelation
# RES["train_LE"][:fom] = le_fom ./ bar
for r in eachindex(REDUCTION_INFO["ro"])
    RES["LE"][:LS][:,r] = le_LS[:,r] ./ bar
    RES["LE"][:int][:,r] = le_int[:,r] ./ bar
    RES["LE"][:ephec][:,r] = le_ephec[:,r] ./ bar
    # RES["LE"][:epsic][:,r] = le_epsic[:,r] ./ bar
    # RES["LE"][:epp][:,r] = le_epp[:,r] ./ bar

    RES["KY"][:LS][r] = ky_LS[r] / bar
    RES["KY"][:int][r] = ky_int[r] / bar
    RES["KY"][:ephec][r] = ky_ephec[r] / bar
    # RES["KY"][:epsic][r] = ky_epsic[r] / bar
    # RES["KY"][:epp][r] = ky_epp[r] / bar
end
@info "Done."

#===================#
## Save the results
#===================#
tmp = joinpath(FILEPATH, "data/kse_epopinf_training_results.jld2")
@info "Save the results to $(tmp)"
save(tmp, RES)