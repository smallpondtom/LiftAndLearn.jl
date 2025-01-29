"""
Kuramoto–Sivashinsky equation flow statistics analysis for testing data
"""

#================#
## Load packages
#================#
using ChaosGizmo
using FileIO
using JLD2
using LinearAlgebra
using NaNStatistics: nanmedian, nanmean
using ProgressMeter
using StatsBase
using Statistics
using UniqueKronecker
using PolynomialModelReductionDataset: KuramotoSivashinskyModel, AbstractModel
using LiftAndLearn
const LnL = LiftAndLearn

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"Two-Pass_Streaming-OpInf/kse") : joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/kse")

#===================#
## Import functions
#===================#
include(joinpath(FILEPATH, "../utilities/kse_analyze.jl"))

#======================================#
## Obtain all the saved testing files
#======================================#
testing_data_files = readdir(joinpath(FILEPATH, "data/testing"), join=true)
basis_file = joinpath(FILEPATH, "data/streaming/basis.jld2")
setup_file = joinpath(FILEPATH, "data/setup.jld2")

#===================#
## Load the options
#===================#
setup = load(setup_file)
options = setup["options"]
kse = setup["kse"]
DS = setup["DS"]
rrange = setup["rrange"]

#=================#
## Load the bases 
#=================#
basis_data = load(basis_file)
Vrmax = basis_data["batch"].Vr
iVrmax = basis_data["baker"].iVr  # choose Baker's iSVD basis
rmax = size(iVrmax,2)

#=================#
## Load the model
#=================#
model_files = readdir(joinpath(FILEPATH, "data/models"), join=true)
model_idx = findfirst(x -> occursin("mu1.0",x), model_files)
ops = load(model_files[model_idx]) 

#==========================#
## Prepare to save results
#==========================#
RES = Dict{String, Any}()

#=======================================#
## Normalized Autocorrelation functions
#=======================================#
# Time lag
lags = 0:DS:(kse.time_dim ÷ 2)

RES["AC_lags"] = lags
RES["AC"] = Dict(
    :pod           => Array{Float64}(undef, length(lags), length(rrange)),
    :opinf         => Array{Float64}(undef, length(lags), length(rrange)),
    :tropinf       => Array{Float64}(undef, length(lags), length(rrange)),
    :stream_rls    => Array{Float64}(undef, length(lags), length(rrange)),
    :stream_iqrrls => Array{Float64}(undef, length(lags), length(rrange)),
    :stream_qrrls  => Array{Float64}(undef, length(lags), length(rrange)),
    :fom           => Array{Float64}(undef, length(lags))
)
RES["AC_ERR"] = Dict(
    :pod           => Array{Float64}(undef, length(rrange)),
    :opinf         => Array{Float64}(undef, length(rrange)),
    :tropinf       => Array{Float64}(undef, length(rrange)),
    :stream_rls    => Array{Float64}(undef, length(rrange)),
    :stream_iqrrls => Array{Float64}(undef, length(rrange)),
    :stream_qrrls  => Array{Float64}(undef, length(rrange)),
)

# Compute autocorrelation functions
ac_fom          = zeros(length(lags))
ac_pod          = zeros(length(lags), length(rrange))
ac_pod_err      = zeros(length(rrange))
ac_opinf        = zeros(length(lags), length(rrange))
ac_opinf_err    = zeros(length(rrange))
ac_tropinf      = zeros(length(lags), length(rrange))
ac_tropinf_err  = zeros(length(rrange))
ac_rls          = zeros(length(lags), length(rrange))
ac_rls_err      = zeros(length(rrange))
ac_iqrrls       = zeros(length(lags), length(rrange))
ac_iqrrls_err   = zeros(length(rrange))
ac_qrrls        = zeros(length(lags), length(rrange))
ac_qrrls_err    = zeros(length(rrange))

##
num_of_testing = length(testing_data_files)

@showprogress Threads.@threads for data_file in testing_data_files
    jldopen(data_file, "r") do file
        IC = file["IC"]
        X  = file["X"]

        ac_fom_tmp = kse_analyze_autocorr(kse, X, lags)[1]
        ac_fom .+= ac_fom_tmp

        ac_pod_tmp     = kse_analyze_autocorr(ops["pod"],           kse, iVrmax, IC, rrange, kse.integrate_model, lags)
        ac_opinf_tmp   = kse_analyze_autocorr(ops["opinf"],         kse, iVrmax, IC, rrange, kse.integrate_model, lags)
        ac_tropinf_tmp = kse_analyze_autocorr(ops["tropinf"],       kse, iVrmax, IC, rrange, kse.integrate_model, lags)
        ac_rls_tmp     = kse_analyze_autocorr(ops["stream_rls"],    kse, iVrmax, IC, rrange, kse.integrate_model, lags)
        ac_iqrrls_tmp  = kse_analyze_autocorr(ops["stream_iqrrls"], kse, iVrmax, IC, rrange, kse.integrate_model, lags)
        ac_qrrls_tmp   = kse_analyze_autocorr(ops["stream_qrrls"],  kse, iVrmax, IC, rrange, kse.integrate_model, lags)

        for r in eachindex(rrange)
            ac_pod[:,r]  .+= ac_pod_tmp[r,1]
            ac_pod_err[r] += norm(ac_pod_tmp[r,1] - ac_fom_tmp, 2) / norm(ac_fom_tmp, 2)

            ac_opinf[:,r]  .+= ac_opinf_tmp[r,1]
            ac_opinf_err[r] += norm(ac_opinf_tmp[r,1] - ac_fom_tmp, 2) / norm(ac_fom_tmp, 2)

            ac_tropinf[:,r]  .+= ac_tropinf_tmp[r,1]
            ac_tropinf_err[r] += norm(ac_tropinf_tmp[r,1] - ac_fom_tmp, 2) / norm(ac_fom_tmp, 2)

            ac_rls[:,r]  .+= ac_rls_tmp[r,1]
            ac_rls_err[r] += norm(ac_rls_tmp[r,1] - ac_fom_tmp, 2) / norm(ac_fom_tmp, 2)

            ac_qrrls[:,r]  .+= ac_qrrls_tmp[r,1]
            ac_qrrls_err[r] += norm(ac_qrrls_tmp[r,1] - ac_fom_tmp, 2) / norm(ac_fom_tmp, 2)

            ac_iqrrls[:,r]  .+= ac_iqrrls_tmp[r,1]
            ac_iqrrls_err[r] += norm(ac_iqrrls_tmp[r,1] - ac_fom_tmp, 2) / norm(ac_fom_tmp, 2)
        end
    end
end

# save the mean normalized autocorrelation
RES["AC"][:fom] = ac_fom ./ num_of_testing
for r in eachindex(rrange)
    RES["AC"][:pod][:,r]           = ac_pod[:,r] ./ num_of_testing
    RES["AC"][:opinf][:,r]         = ac_opinf[:,r] ./ num_of_testing
    RES["AC"][:tropinf][:,r]       = ac_tropinf[:,r] ./ num_of_testing
    RES["AC"][:stream_rls][:,r]    = ac_rls[:,r] ./ num_of_testing
    RES["AC"][:stream_iqrrls][:,r] = ac_iqrrls[:,r] ./ num_of_testing
    RES["AC"][:stream_qrrls][:,r]  = ac_qrrls[:,r] ./ num_of_testing
end

# Reshape into column vector
RES["AC_ERR"][:pod]           = ac_pod_err ./ num_of_testing
RES["AC_ERR"][:opinf]         = ac_opinf_err./ num_of_testing
RES["AC_ERR"][:tropinf]       = ac_tropinf_err ./ num_of_testing
RES["AC_ERR"][:stream_rls]    = ac_rls_err ./ num_of_testing
RES["AC_ERR"][:stream_iqrrls] = ac_iqrrls_err ./ num_of_testing
RES["AC_ERR"][:stream_qrrls]  = ac_qrrls_err ./ num_of_testing

#================================================#
## Lyapunov exponents and Kaplan-Yorke dimension
#================================================#
# Lyapunov exponent Settings
max_num_of_LE = 10
LEOption = ChaosGizmo.LyapunovExponentOptions(
    m=max_num_of_LE, τ=2e+3, T=0.001, Δt=0.001, N=1e+5, ϵ=1e-6, verbose=false, jacobian=true,
)

RES["LE"] = Dict(
    :pod           => Array{Float64}(undef, max_num_of_LE, length(rrange)),
    :opinf         => Array{Float64}(undef, max_num_of_LE, length(rrange)),
    :tropinf       => Array{Float64}(undef, max_num_of_LE, length(rrange)),
    :stream_rls    => Array{Float64}(undef, max_num_of_LE, length(rrange)),
    :stream_iqrrls => Array{Float64}(undef, max_num_of_LE, length(rrange)),
    :stream_qrrls  => Array{Float64}(undef, max_num_of_LE, length(rrange)),
)

RES["KY"] = Dict(
    :pod           => Array{Float64}(undef, length(rrange)),
    :opinf         => Array{Float64}(undef, length(rrange)),
    :tropinf       => Array{Float64}(undef, length(rrange)),
    :stream_rls    => Array{Float64}(undef, length(rrange)),
    :stream_iqrrls => Array{Float64}(undef, length(rrange)),
    :stream_qrrls  => Array{Float64}(undef, length(rrange)),
)

num_of_testing = length(testing_data_files)

# Compute Lypuanov exponents
le_pod     = zeros(max_num_of_LE, length(rrange), num_of_testing)
le_opinf   = zeros(max_num_of_LE, length(rrange), num_of_testing)
le_tropinf = zeros(max_num_of_LE, length(rrange), num_of_testing)
le_rls     = zeros(max_num_of_LE, length(rrange), num_of_testing)
le_iqrrls  = zeros(max_num_of_LE, length(rrange), num_of_testing)
le_qrrls   = zeros(max_num_of_LE, length(rrange), num_of_testing)

# Compute Kaplan-Yorke dimensions
ky_pod     = zeros(length(rrange), num_of_testing)
ky_opinf   = zeros(length(rrange), num_of_testing)
ky_tropinf = zeros(length(rrange), num_of_testing)
ky_rls     = zeros(length(rrange), num_of_testing)
ky_iqrrls  = zeros(length(rrange), num_of_testing)
ky_qrrls   = zeros(length(rrange), num_of_testing)

##
@showprogress Threads.@threads for (idx, data_file) in collect(enumerate(testing_data_files))
    jldopen(data_file, "r") do file
        IC = file["IC"]

        # Lyapunov exponents
        le_pod_tmp     = kse_lyapunov_exponent(ops["pod"],           kse, iVrmax, IC, rrange, kse.integrate_model, LEOption; jacobian=kse.jacobian)
        le_opinf_tmp   = kse_lyapunov_exponent(ops["opinf"],         kse, iVrmax, IC, rrange, kse.integrate_model, LEOption; jacobian=kse.jacobian)
        le_tropinf_tmp = kse_lyapunov_exponent(ops["tropinf"],       kse, iVrmax, IC, rrange, kse.integrate_model, LEOption; jacobian=kse.jacobian)
        le_rls_tmp     = kse_lyapunov_exponent(ops["stream_rls"],    kse, iVrmax, IC, rrange, kse.integrate_model, LEOption; jacobian=kse.jacobian)
        le_iqrrls_tmp  = kse_lyapunov_exponent(ops["stream_iqrrls"], kse, iVrmax, IC, rrange, kse.integrate_model, LEOption; jacobian=kse.jacobian)
        le_qrrls_tmp   = kse_lyapunov_exponent(ops["stream_qrrls"],  kse, iVrmax, IC, rrange, kse.integrate_model, LEOption; jacobian=kse.jacobian)

        # Kaplan-Yorke dimensions
        ky_pod_tmp     = [ChaosGizmo.kaplan_yorke_dim(le_pod_tmp[r,1])     for r in eachindex(rrange)]
        ky_opinf_tmp   = [ChaosGizmo.kaplan_yorke_dim(le_opinf_tmp[r,1])   for r in eachindex(rrange)]
        ky_tropinf_tmp = [ChaosGizmo.kaplan_yorke_dim(le_tropinf_tmp[r,1]) for r in eachindex(rrange)]
        ky_rls_tmp     = [ChaosGizmo.kaplan_yorke_dim(le_rls_tmp[r,1])     for r in eachindex(rrange)]
        ky_iqrrls_tmp  = [ChaosGizmo.kaplan_yorke_dim(le_iqrrls_tmp[r,1])  for r in eachindex(rrange)]
        ky_qrrls_tmp   = [ChaosGizmo.kaplan_yorke_dim(le_qrrls_tmp[r,1])   for r in eachindex(rrange)]

        for r in eachindex(rrange)
            le_pod[:,r,idx]     = le_pod_tmp[r]
            le_opinf[:,r,idx]   = le_opinf_tmp[r]
            le_tropinf[:,r,idx] = le_tropinf_tmp[r]
            le_rls[:,r,idx]     = le_rls_tmp[r]
            le_iqrrls[:,r,idx]  = le_iqrrls_tmp[r]
            le_qrrls[:,r,idx]   = le_qrrls_tmp[r]
        end

        ky_pod[:,idx]     = ky_pod_tmp
        ky_opinf[:,idx]   = ky_opinf_tmp
        ky_tropinf[:,idx] = ky_tropinf_tmp
        ky_rls[:,idx]     = ky_rls_tmp
        ky_iqrrls[:,idx]  = ky_iqrrls_tmp
        ky_qrrls[:,idx]   = ky_qrrls_tmp
    end
end

# save the mean normalized autocorrelation
RES["LE"][:pod]           .= nanmean(le_pod;     dims=3)
RES["LE"][:opinf]         .= nanmean(le_opinf;   dims=3)
RES["LE"][:tropinf]       .= nanmean(le_tropinf; dims=3)
RES["LE"][:stream_rls]    .= nanmean(le_rls;     dims=3)
RES["LE"][:stream_rls]    .= nanmean(le_iqrrls;  dims=3)
RES["LE"][:stream_iqrrls] .= nanmean(le_qrrls;   dims=3)

RES["KY"][:pod]           .= nanmean(ky_pod;     dims=2)
RES["KY"][:opinf]         .= nanmean(ky_opinf;   dims=2)
RES["KY"][:tropinf]       .= nanmean(ky_tropinf; dims=2)
RES["KY"][:stream_rls]    .= nanmean(ky_rls;     dims=2)
RES["KY"][:stream_iqrrls] .= nanmean(ky_iqrrls;  dims=2)
RES["KY"][:stream_qrrls]  .= nanmean(ky_qrrls;   dims=2)

#===================#
## Save the results
#===================#
tmp = joinpath(FILEPATH, "data/testing_statistics.jld2")
@info "Save the results to $(tmp)"
save(tmp, RES)
