"""
2D heat equation: testing models
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using LinearAlgebra
using ProgressMeter
using PolynomialModelReductionDataset: Heat2DModel
using Printf
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"Streaming-OpInf/heat2d") : joinpath(pwd(), "scripts/Streaming-OpInf/heat2d")

#======================================#
## Obtain all the saved training files
#======================================#
testing_data_files = readdir(joinpath(FILEPATH, "data/testing"), join=true)
model_files = readdir(joinpath(FILEPATH, "data/models"), join=true)
basis_file = joinpath(FILEPATH, "data/streaming/basis.jld2")
setup_file = joinpath(FILEPATH, "data/setup.jld2")

#===================#
## Load the options
#===================#
setup = load(setup_file)
options = setup["options"]
heat2d = setup["heat2d"]

#=================#
## Load the bases 
#=================#
basis_data = load(basis_file)
Vrmax = basis_data["batch"].Vr
iVrmax = basis_data["baker"].iVr  # choose Baker's iSVD basis
rmax = size(Vrmax, 2)

#=================#
## Load the models
#=================#
ops = Dict(
    "pod" => Dict(:A => [], :B => []),
    "opinf" => Dict(:A => [], :B => []),
    "tropinf" => Dict(:A => [], :B => []),
    "stream_rls" => Dict(:A => [], :B => []),
    "stream_iqrrls" => Dict(:A => [], :B => []),
    "stream_qrrls" => Dict(:A => [], :B => [])
)
for model_file in model_files
    model = load(model_file)
    for key in keys(model)
        if key == "mu"
            continue
        end
        push!(ops[key][:A], model[key].A)
        push!(ops[key][:B], model[key].B)
    end
end

#=====================#
## Testing the models
#=====================#
num_tests = length(testing_data_files)

# Error analysis 
test_errors = Dict(
    "pod" => zeros(rmax,1),
    "opinf" => zeros(rmax,1),
    "tropinf" => zeros(rmax,1),
    "stream_rls" => zeros(rmax,1),
    "stream_iqrrls" => zeros(rmax,1),
    "stream_qrrls" => zeros(rmax,1)
)

param_region = collect(heat2d.diffusion_coeffs)

##
@showprogress for (file_idx, test_file) in enumerate(testing_data_files)
    jldopen(test_file, "r") do data
        # Load the data
        X = data["X"]
        U = data["U"]
        μ = data["mu"]  # test parameter

        # Interpolate model operators
        # POD model
        Aint = LnL.interpolate_matrix_elements(param_region, ops["pod"][:A], μ; order=3)
        Bint = LnL.interpolate_matrix_elements(param_region, ops["pod"][:B], μ; order=3)
        # OpInf model
        Ainf = LnL.interpolate_matrix_elements(param_region, ops["opinf"][:A], μ; order=3)
        Binf = LnL.interpolate_matrix_elements(param_region, ops["opinf"][:B], μ; order=3)
        # TR-OpInf model
        Atrinf = LnL.interpolate_matrix_elements(param_region, ops["tropinf"][:A], μ; order=3)
        Btrinf = LnL.interpolate_matrix_elements(param_region, ops["tropinf"][:B], μ; order=3)
        # RLS-Streaming model
        Astream_rls = LnL.interpolate_matrix_elements(param_region, ops["stream_rls"][:A], μ; order=3)
        Bstream_rls = LnL.interpolate_matrix_elements(param_region, ops["stream_rls"][:B], μ; order=3)
        # iQRRLS-Streaming model
        Astream_iqrrls = LnL.interpolate_matrix_elements(param_region, ops["stream_iqrrls"][:A], μ; order=3)
        Bstream_iqrrls = LnL.interpolate_matrix_elements(param_region, ops["stream_iqrrls"][:B], μ; order=3)
        # QRRLS-Streaming model
        Astream_qrrls = LnL.interpolate_matrix_elements(param_region, ops["stream_qrrls"][:A], μ; order=3)
        Bstream_qrrls = LnL.interpolate_matrix_elements(param_region, ops["stream_qrrls"][:B], μ; order=3)

        op_tmp = Dict(
            "pod" => (A=Aint, B=Bint),
            "opinf" => (A=Ainf, B=Binf),
            "tropinf" => (A=Atrinf, B=Btrinf),
            "stream_rls" => (A=Astream_rls, B=Bstream_rls),
            "stream_iqrrls" => (A=Astream_iqrrls, B=Bstream_iqrrls),
            "stream_qrrls" => (A=Astream_qrrls, B=Bstream_qrrls),
        )

        op_keys = [key for key in keys(op_tmp)]
        Threads.@threads for i in eachindex(op_keys)
            key = op_keys[i]
            for (i,r) = enumerate(1:rmax)
                if occursin(r"stream", key)
                    Vr = iVrmax[:, 1:r]
                else
                    Vr = Vrmax[:, 1:r]
                end

                # Integrate the model
                Xrecon = heat2d.integrate_model(
                    heat2d.tspan, Vr' * heat2d.IC, U,
                    linear_matrix=op_tmp[key].A[1:r, 1:r], control_matrix=op_tmp[key].B[1:r,:],
                    system_input=true, integrator_type=:BackwardEuler
                )

                # Compute relative state error (averaged over parameters)
                test_errors[key][i] += norm(X - Vr * Xrecon) / norm(X) / num_tests
            end
        end
    end
end

# Save the errors
save(joinpath(FILEPATH, "data/testing_errors.jld2"), test_errors)