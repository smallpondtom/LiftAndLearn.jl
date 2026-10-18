"""
Kuramoto-Sivashinsky equation flow statistics analysis for training data
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
FILEPATH = @__DIR__

#===================#
## Import functions
#===================#
include(joinpath(FILEPATH, "../utilities/kse_analyze.jl"))

#======================================#
## Obtain all the saved training files
#======================================#
training_data_files = readdir(joinpath(FILEPATH, "data/training"), join=true)
basis_file = joinpath(FILEPATH, "data/streaming/basis.jld2")
setup_file = joinpath(FILEPATH, "data/setup.jld2")

#===================#
## Load the options
#===================#
setup = load(setup_file)
options = setup["options"]
kse = setup["kse"]
DS = setup["DS"]
setup["rrange"] = rrange

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

#================================================#
## Lyapunov exponents and Kaplan-Yorke dimension
#================================================#
@info "Compute Lyapunov exponents and Kaplan-Yorke dimensions for training data"
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
)

RES["KY"] = Dict(
    :pod           => Array{Float64}(undef, length(rrange)),
    :opinf         => Array{Float64}(undef, length(rrange)),
    :tropinf       => Array{Float64}(undef, length(rrange)),
    :stream_rls    => Array{Float64}(undef, length(rrange)),
    :stream_iqrrls => Array{Float64}(undef, length(rrange)),
)

num_of_training = length(training_data_files)

# Compute Lypuanov exponents
le_pod     = zeros(max_num_of_LE, length(rrange), num_of_training)
le_opinf   = zeros(max_num_of_LE, length(rrange), num_of_training)
le_tropinf = zeros(max_num_of_LE, length(rrange), num_of_training)
le_rls     = zeros(max_num_of_LE, length(rrange), num_of_training)
le_iqrrls  = zeros(max_num_of_LE, length(rrange), num_of_training)

# Compute Kaplan-Yorke dimensions
ky_pod     = zeros(length(rrange), num_of_training)
ky_opinf   = zeros(length(rrange), num_of_training)
ky_tropinf = zeros(length(rrange), num_of_training)
ky_rls     = zeros(length(rrange), num_of_training)
ky_iqrrls  = zeros(length(rrange), num_of_training)

#
@showprogress Threads.@threads for (idx, data_file) in collect(enumerate(training_data_files))
    jldopen(data_file, "r") do file
        IC = file["IC"]

        # Lyapunov exponents
        le_pod_tmp     = kse_lyapunov_exponent(ops["pod"],           kse, iVrmax, IC, rrange, kse.integrate_model, LEOption; jacobian=kse.jacobian)
        le_opinf_tmp   = kse_lyapunov_exponent(ops["opinf"],         kse, iVrmax, IC, rrange, kse.integrate_model, LEOption; jacobian=kse.jacobian)
        le_tropinf_tmp = kse_lyapunov_exponent(ops["tropinf"],       kse, iVrmax, IC, rrange, kse.integrate_model, LEOption; jacobian=kse.jacobian)
        le_rls_tmp     = kse_lyapunov_exponent(ops["stream_rls"],    kse, iVrmax, IC, rrange, kse.integrate_model, LEOption; jacobian=kse.jacobian)
        le_iqrrls_tmp  = kse_lyapunov_exponent(ops["stream_iqrrls"], kse, iVrmax, IC, rrange, kse.integrate_model, LEOption; jacobian=kse.jacobian)

        # Kaplan-Yorke dimensions
        ky_pod_tmp     = [ChaosGizmo.kaplan_yorke_dim(le_pod_tmp[r,1])     for r in eachindex(rrange)]
        ky_opinf_tmp   = [ChaosGizmo.kaplan_yorke_dim(le_opinf_tmp[r,1])   for r in eachindex(rrange)]
        ky_tropinf_tmp = [ChaosGizmo.kaplan_yorke_dim(le_tropinf_tmp[r,1]) for r in eachindex(rrange)]
        ky_rls_tmp     = [ChaosGizmo.kaplan_yorke_dim(le_rls_tmp[r,1])     for r in eachindex(rrange)]
        ky_iqrrls_tmp  = [ChaosGizmo.kaplan_yorke_dim(le_iqrrls_tmp[r,1])  for r in eachindex(rrange)]

        for r in eachindex(rrange)
            le_pod[:,r,idx]     = le_pod_tmp[r]
            le_opinf[:,r,idx]   = le_opinf_tmp[r]
            le_tropinf[:,r,idx] = le_tropinf_tmp[r]
            le_rls[:,r,idx]     = le_rls_tmp[r]
            le_iqrrls[:,r,idx]  = le_iqrrls_tmp[r]
        end

        ky_pod[:,idx]     = ky_pod_tmp
        ky_opinf[:,idx]   = ky_opinf_tmp
        ky_tropinf[:,idx] = ky_tropinf_tmp
        ky_rls[:,idx]     = ky_rls_tmp
        ky_iqrrls[:,idx]  = ky_iqrrls_tmp
    end
end

## save the mean normalized autocorrelation
RES["LE"][:pod]           .= nanmean(le_pod;     dims=3)
RES["LE"][:opinf]         .= nanmean(le_opinf;   dims=3)
RES["LE"][:tropinf]       .= nanmean(le_tropinf; dims=3)
RES["LE"][:stream_rls]    .= nanmean(le_rls;     dims=3)
RES["LE"][:stream_iqrrls] .= nanmean(le_iqrrls;  dims=3)

RES["KY"][:pod]           .= nanmean(ky_pod;     dims=2)
RES["KY"][:opinf]         .= nanmean(ky_opinf;   dims=2)
RES["KY"][:tropinf]       .= nanmean(ky_tropinf; dims=2)
RES["KY"][:stream_rls]    .= nanmean(ky_rls;     dims=2)
RES["KY"][:stream_iqrrls] .= nanmean(ky_iqrrls;  dims=2)

@info "Lyapunov exponents and Kaplan-Yorke dimensions computed."

#===================#
## Save the results
#===================#
tmp = joinpath(FILEPATH, "data/training_statistics.jld2")
@info "Save the results to $(tmp)"
save(tmp, RES)