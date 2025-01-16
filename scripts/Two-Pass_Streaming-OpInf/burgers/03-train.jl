"""
1D Viscous Burgers equation: training models
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
using Random
import UniqueKronecker
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"Two-Pass_Streaming-OpInf/burgers") : joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/burgers")

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
burgers = setup["burgers"]

#=================#
## Load the bases 
#=================#
basis_data = load(basis_file)
Vrmax = basis_data["batch"].Vr
iVrmax = basis_data["baker"].iVr  # choose Baker's iSVD basis

#=======================#
## Additional functions
#=======================#
include(joinpath(FILEPATH, "../utilities/extract_operators.jl"))

#====================#
## Train batch models
#====================#
model_files = []
@showprogress for (i, data_file) in enumerate(training_data_files)
    jldopen(data_file, "r") do data
        # Load the data
        X = data["X"]
        U = data["U"]
        Xdot = data["Xdot"]

        # Load the operators
        A = data["A"]
        B = data["B"]
        F = data["F"]
        μ = data["mu"]

        # Train the models 
        # POD-Galerkin 
        tmp = LnL.Operators(A=A, B=B, A2u=F)
        op_pod = LnL.pod(tmp, Vrmax, options.system)
        # OpInf
        options.with_reg = false
        op_inf = LnL.opinf(X, Vrmax, options; U=U, Xdot=Xdot)
        # Tikhonov Regularized OpInf
        options.with_reg = true
        options.λ = LnL.TikhonovParameter(
            A = 1e-6,
            B = 1e-6,
            A2 = 1e-6
        )
        op_trinf = LnL.opinf(X, Vrmax, options; U=U, Xdot=Xdot)
        # Save the model
        mu_str = @sprintf("%1.4f", μ)
        ops = Dict("pod" => op_pod, "opinf" => op_inf, "tropinf" => op_trinf, "mu" => μ)
        filename = joinpath(FILEPATH, "data/models", "op_mu$(mu_str).jld2")
        push!(model_files, filename)
        save(filename, ops)
    end
end

#====================================#
## Generate Two-Pass Streaming-OpInf 
#====================================#
# Placeholders
rmax = size(Vrmax,2)
num_of_streams = 5000 # Number of streams !!! CHANGE THIS MANUALLY !!!
tmp_res = (
    true_stream_err = zeros(rmax, num_of_streams),
    stream_err      = zeros(rmax, num_of_streams),
    rse             = zeros(rmax, num_of_streams),
    post_err        = zeros(num_of_streams),
    conv_factor     = zeros(num_of_streams),
)
# Dict to store all streaming results for different algorithms 
stream_res = Dict(
    :rls => deepcopy(tmp_res),
    :iqrrls => deepcopy(tmp_res),
    :qrrls => deepcopy(tmp_res)
)
model_files = readdir(joinpath(FILEPATH, "data/models"), join=true)
for (file_idx, data_file) in enumerate(training_data_files)
    jldopen(data_file, "r") do data
        # Load the data
        X = data["X"]
        U = data["U"]
        U = reshape(U, size(U,1), :)
        Xdot = data["Xdot"]
        Xref = data["Xref"]
        Uref = data["Uref"]

        ## Streamify the data based on the selected streamsizes
        streamsize = 1
        X_stream = LnL.streamify(iVrmax' * X, streamsize)
        U_stream = LnL.streamify(U, streamsize)
        Xdot_stream = LnL.streamify(iVrmax' * Xdot, streamsize)
        num_of_streams = length(X_stream)

        ## Initialize the stream
        Γs = 1e-9
        # standard RLS
        rls_stream  = LnL.StreamingOpInf(options=options, n=rmax, m=1, algorithm=:RLS, Γs=Γs) 
        # inverse-QR RLS
        iqrrls_stream = LnL.StreamingOpInf(options=options, n=rmax, m=1, algorithm=:iQRRLS, Γs=Γs)
        # QR RLS
        qrrls_stream = LnL.StreamingOpInf(options=options, n=rmax, m=1, algorithm=:QRRLS, Γs=Γs)

        # Initialize the error factor
        Eps = nothing

        # Load the batch solution 
        mu_str = @sprintf("%1.4f", data["mu"])
        model_idx = findfirst(x -> occursin("mu$(mu_str)", x), model_files)
        op_inf = load(model_files[model_idx], "opinf")
        O_inf = vcat(op_inf.A', op_inf.B', op_inf.A2u')

        ## Stream one-by-one and collect data
        @showprogress for i in 1:num_of_streams
            # Stream, update, and get data matrix for the state system
            D = LnL.stream!(rls_stream, X_stream[i], Xdot_stream[i]; U=U_stream[i], final_step=true)
            _ = LnL.stream!(iqrrls_stream, X_stream[i], Xdot_stream[i]; U=U_stream[i], final_step=true)
            _ = LnL.stream!(qrrls_stream, X_stream[i], Xdot_stream[i]; U=U_stream[i], final_step=true)

            # Unpack operators
            # RLS
            op_rls_tmp = LnL.Operators()
            LnL.unpack_operators!(
                op_rls_tmp, rls_stream.cache.O', 
                rls_stream.termination_settings[:dims], rls_stream.termination_settings[:syms]
            )
            # iQRRLS
            op_iqrrls_tmp = LnL.Operators()
            LnL.unpack_operators!(
                op_iqrrls_tmp, iqrrls_stream.cache.O', 
                iqrrls_stream.termination_settings[:dims], iqrrls_stream.termination_settings[:syms]
            )
            # QRRLS
            op_qrrls_tmp = LnL.Operators()
            LnL.unpack_operators!(
                op_qrrls_tmp, qrrls_stream.cache.O', 
                qrrls_stream.termination_settings[:dims], qrrls_stream.termination_settings[:syms]
            )
            # Collect all the operators into a dictionary
            op_tmp = Dict(
                :rls => op_rls_tmp,
                :iqrrls => op_iqrrls_tmp,
                :qrrls => op_qrrls_tmp
            )

            # Error factors
            error_factors = Dict(
                :rls => 1.0I - rls_stream.cache.K * D,
                :iqrrls => 1.0I - iqrrls_stream.cache.K * D,
                :qrrls => 1.0I - qrrls_stream.cache.K * D
            )
            Eps_true = Dict(
                :rls => O_inf - rls_stream.cache.O,
                :iqrrls => O_inf - iqrrls_stream.cache.O,
                :qrrls => O_inf - qrrls_stream.cache.O
            )

            # Initialize the streaming errors
            if i == 1
                Eps = Eps_true
            end
            
            # Update the streaming errors
            for key in keys(Eps)
                Eps[key] = error_factors[key] * Eps[key]
            end

            algo_keys = [key for key in keys(op_tmp)]
            Threads.@threads for k in eachindex(algo_keys)  # Loop through each algorithm to compute the error quantities 
                key = algo_keys[k]
                # Loop through each reduced dimension
                for (j, ri) in enumerate(1:rmax)
                    # Numerical integrate the reference solution with reference input
                    F_extract = UniqueKronecker.extractF(op_tmp[key].A2u, ri)
                    Xtmp = burgers.integrate_model(
                        burgers.tspan, iVrmax[:,1:ri]' * burgers.IC, Uref; linear_matrix=op_tmp[key].A[1:ri, 1:ri],
                        control_matrix=op_tmp[key].B[1:ri,:], quadratic_matrix=F_extract, system_input=true
                    )
                    stream_res[key].rse[j, i] += LnL.rel_state_error(Xref, Xtmp, iVrmax[:,1:ri])

                    # Index for streaming errors
                    idx = extract_indices(rls_stream, rmax, ri, options.system)

                    # Streaming errors
                    O_norm = norm(O_inf[idx,1:ri], 2)
                    Eps_true_sub = Eps_true[key][idx,1:ri]
                    Eps_sub = Eps[key][idx,1:ri]
                    
                    # Errors
                    stream_res[key].true_stream_err[j, i] += norm(Eps_true_sub, 2) / O_norm 
                    stream_res[key].stream_err[j,i] += norm(Eps_sub,2) / O_norm
                end
            end

            # A posteriori error and conversion factors 
            # RLS
            stream_res[:rls].post_err[i] += norm(rls_stream.cache.ξpost,2)
            stream_res[:rls].conv_factor[i] += rls_stream.cache.C[1] 
            # iQRRLS
            stream_res[:iqrrls].post_err[i] += norm(iqrrls_stream.cache.ξpost,2)
            stream_res[:iqrrls].conv_factor[i] += iqrrls_stream.cache.C[1]
            # QRRLS
            stream_res[:qrrls].post_err[i] += norm(qrrls_stream.cache.ξpost,2)
            stream_res[:qrrls].conv_factor[i] += qrrls_stream.cache.C[1]
        end

        # Terminate the streaming operators
        op_stream_rls = LnL.terminate_stream(rls_stream)
        op_stream_iqrrls = LnL.terminate_stream(iqrrls_stream)
        op_stream_qrrls = LnL.terminate_stream(qrrls_stream)

        # Save the streaming-based operators
        jldopen(model_files[model_idx], "a+") do model
            model["stream_rls"] = op_stream_rls
            model["stream_iqrrls"] = op_stream_iqrrls
            model["stream_qrrls"] = op_stream_qrrls
        end 
    end
    @info "Streaming for model $(file_idx) out of $(length(training_data_files)) is completed"
end

## Average over the number of parameters
for key in keys(stream_res)
    stream_res[key].rse ./= burgers.param_dim
    stream_res[key].true_stream_err ./= burgers.param_dim
    stream_res[key].stream_err ./= burgers.param_dim
    stream_res[key].post_err ./= burgers.param_dim
    stream_res[key].conv_factor ./= burgers.param_dim
end

## Save the streaming results
filename = joinpath(FILEPATH, "data/streaming", "stream_results.jld2")
save(filename, "stream_res", stream_res)

#====================================#
## Compute the relative state errors 
#====================================#
num_train = length(training_data_files)

# Error analysis 
train_errors = Dict(
    :pod => zeros(rmax,1),
    :opinf => zeros(rmax,1),
    :tropinf => zeros(rmax,1),
    :stream_rls => zeros(rmax,1),
    :stream_iqrrls => zeros(rmax,1),
    :stream_qrrls => zeros(rmax,1)
)

##
@showprogress for (file_idx, train_file) in enumerate(training_data_files)
    jldopen(train_file, "r") do data
        # Load the data
        Xref = data["Xref"]
        Uref = data["Uref"]

        # Load the trained models
        mu_str = @sprintf("%1.4f", data["mu"])
        model_idx = findfirst(x -> occursin("mu$(mu_str)", x), model_files)
        ops = load(model_files[model_idx]) 

        op_keys = [key for key in keys(train_errors)]
        Threads.@threads for i in eachindex(op_keys)
            key = op_keys[i]
            for (i,r) = enumerate(1:rmax)
                if occursin(r"stream", string(key))
                    Vr = iVrmax[:, 1:r]
                else
                    Vr = Vrmax[:, 1:r]
                end

                # Integrate the model
                F_extract = UniqueKronecker.extractF(ops[string(key)].A2u, r)
                Xrecon = burgers.integrate_model(
                    burgers.tspan, Vr' * burgers.IC, Uref,
                    linear_matrix=ops[string(key)].A[1:r, 1:r], control_matrix=ops[string(key)].B[1:r,:],
                    quadratic_matrix=F_extract, system_input=true
                )

                # Compute relative state error (averaged over parameters)
                train_errors[key][i] += norm(Xref - Vr * Xrecon) / norm(Xref) / num_train
            end
        end
    end
end

# Save the errors
save(joinpath(FILEPATH, "data/training_errors.jld2"), train_errors)