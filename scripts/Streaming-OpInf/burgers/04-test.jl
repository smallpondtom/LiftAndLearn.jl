"""
1D Viscous Burgers equation: testing models
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using LinearAlgebra
using ProgressMeter
using PolynomialModelReductionDataset: BurgersModel
using Printf
import UniqueKronecker
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
FILEPATH = @__DIR__

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
burgers = setup["burgers"]

#=================#
## Load the bases 
#=================#
basis_data = load(basis_file)
Vrmax = basis_data["batch"].Vr
iVrmax = basis_data["sketchy"].iVr  # choose Baker's iSVD basis
rmax = size(Vrmax, 2)

#=================#
## Load the models
#=================#
ops = Dict(
    "pod" => Dict(:A => [], :B => [], :F => []),
    "opinf" => Dict(:A => [], :B => [], :F => []),
    "tropinf" => Dict(:A => [], :B => [], :F => []),
    "stream_rls" => Dict(:A => [], :B => [], :F => []),
    "stream_iqrrls" => Dict(:A => [], :B => [], :F => []),
    "stream_qrrls" => Dict(:A => [], :B => [], :F => []),
    "stream" => Dict(:A => [], :B => [], :F => []),
)
for model_file in model_files
    model = load(model_file)
    for key in keys(model)
        if key == "mu"
            continue
        end
        push!(ops[key][:A], model[key].A)
        push!(ops[key][:B], model[key].B)
        push!(ops[key][:F], model[key].A2u)
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
    "stream_qrrls" => zeros(rmax,1),
    "stream" => zeros(rmax,1),
)

param_region = collect(burgers.diffusion_coeffs)

@info "Testing the trained models..."

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
        Fint = LnL.interpolate_matrix_elements(param_region, ops["pod"][:F], μ; order=3)
        # OpInf model
        Ainf = LnL.interpolate_matrix_elements(param_region, ops["opinf"][:A], μ; order=3)
        Binf = LnL.interpolate_matrix_elements(param_region, ops["opinf"][:B], μ; order=3)
        Finf = LnL.interpolate_matrix_elements(param_region, ops["opinf"][:F], μ; order=3)
        # TR-OpInf model
        Atrinf = LnL.interpolate_matrix_elements(param_region, ops["tropinf"][:A], μ; order=3)
        Btrinf = LnL.interpolate_matrix_elements(param_region, ops["tropinf"][:B], μ; order=3)
        Ftrinf = LnL.interpolate_matrix_elements(param_region, ops["tropinf"][:F], μ; order=3)
        # RLS-Streaming model
        Astream_rls = LnL.interpolate_matrix_elements(param_region, ops["stream_rls"][:A], μ; order=3)
        Bstream_rls = LnL.interpolate_matrix_elements(param_region, ops["stream_rls"][:B], μ; order=3)
        Fstream_rls = LnL.interpolate_matrix_elements(param_region, ops["stream_rls"][:F], μ; order=3)
        # iQRRLS-Streaming model
        Astream_iqrrls = LnL.interpolate_matrix_elements(param_region, ops["stream_iqrrls"][:A], μ; order=3)
        Bstream_iqrrls = LnL.interpolate_matrix_elements(param_region, ops["stream_iqrrls"][:B], μ; order=3)
        Fstream_iqrrls = LnL.interpolate_matrix_elements(param_region, ops["stream_iqrrls"][:F], μ; order=3)
        # One-Pass Streaming model
        A_stream = LnL.interpolate_matrix_elements(param_region, ops["stream"][:A], μ; order=3)
        B_stream = LnL.interpolate_matrix_elements(param_region, ops["stream"][:B], μ; order=3)
        Fstream = LnL.interpolate_matrix_elements(param_region, ops["stream"][:F], μ; order=3)

        op_tmp = Dict(
            "pod" => (A=Aint, B=Bint, A2u=Fint),
            "opinf" => (A=Ainf, B=Binf, A2u=Finf),
            "tropinf" => (A=Atrinf, B=Btrinf, A2u=Ftrinf),
            "stream_rls" => (A=Astream_rls, B=Bstream_rls, A2u=Fstream_rls),
            "stream_iqrrls" => (A=Astream_iqrrls, B=Bstream_iqrrls, A2u=Fstream_iqrrls),
            "stream" => (A=A_stream, B=B_stream, A2u=Fstream)
        )

        # Load the trained models
        op_keys = [key for key in keys(op_tmp)]
        Threads.@threads for i in eachindex(op_keys)
            key = op_keys[i]
            Vrmax = iVrmax
            for (i,r) = enumerate(1:rmax)
                Vr = Vrmax[:, 1:r]

                # Integrate the model
                Fextract = UniqueKronecker.extractF(op_tmp[key].A2u, r)
                Xrecon = burgers.integrate_model(
                    burgers.tspan, Vr' * burgers.IC, U,
                    linear_matrix=op_tmp[key].A[1:r,1:r], control_matrix=op_tmp[key].B[1:r,:], 
                    quadratic_matrix=Fextract, system_input=true
                )

                # Compute relative state error (averaged over parameters)
                test_errors[key][i] += norm(X - Vr * Xrecon) / norm(X) / num_tests
            end
        end
    end
end

# Save the errors
save(joinpath(FILEPATH, "data/testing_errors.jld2"), test_errors)

@info "Testing completed."