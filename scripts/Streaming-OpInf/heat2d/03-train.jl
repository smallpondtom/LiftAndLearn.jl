"""
2D heat equation: training models
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
using Random
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"Streaming-OpInf/heat2d") : joinpath(pwd(), "scripts/Streaming-OpInf/heat2d")

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
heat2d = setup["heat2d"]

#=================#
## Load the bases 
#=================#
basis_data = load(basis_file)
Vrmax = basis_data["batch"].Vr
iVrmax = basis_data["baker"].iVr  # choose Baker's iSVD basis

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
        # Y = data["Y"]

        # Load the operators
        A = data["A"]
        B = data["B"]
        # C = data["C"]
        μ = data["mu"]

        # Train the models 
        # POD-Galerkin 
        # tmp = LnL.Operators(A=A, B=B, C=C)
        tmp = LnL.Operators(A=A, B=B)
        op_pod = LnL.pod(tmp, Vrmax, options.system)
        # OpInf
        options.with_reg = false
        # op_inf = LnL.opinf(X, Vr, options; U=U, Y=Y, Xdot=Xdot)
        op_inf = LnL.opinf(X, Vrmax, options; U=U, Xdot=Xdot)
        # Tikhonov Regularized OpInf
        options.with_reg = true
        options.λ = LnL.TikhonovParameter(
            A = 1e-6,
            B = 1e-6,
            # C = 1e-6
        )
        # op_trinf = LnL.opinf(X, Vr, options; U=U, Y=Y, Xdot=Xdot)
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
num_of_streams = heat2d.time_dim-1  # -1 for the derivative and -1 for the removed last state
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
        Xdot = data["Xdot"]
        Xref = data["Xref"]
        Uref = data["Uref"]

        # # Obtain derivative data and adjust data
        # Xdot = (Xfull[:, 2:end] - Xfull[:, 1:end-1]) / heat2d.Δt
        # idx = 2:heat2d.time_dim-1
        # X = Xfull[:, idx]  
        # U = Ufull[:, idx]
        
        ## Streamify the data based on the selected streamsizes
        streamsize = 1
        X_stream = LnL.streamify(iVrmax' * X, streamsize)
        U_stream = LnL.streamify(U, streamsize)
        Xdot_stream = LnL.streamify(iVrmax' * Xdot, streamsize)
        num_of_streams = length(X_stream)

        ## Initialize the stream
        Γs = 1e-9
        # standard RLS
        rls_stream  = LnL.StreamingOpInf(options=options, n=rmax, m=4, algorithm=:RLS, Γs=Γs) 
        # inverse-QR RLS
        iqrrls_stream = LnL.StreamingOpInf(options=options, n=rmax, m=4, algorithm=:iQRRLS, Γs=Γs)
        # QR RLS
        qrrls_stream = LnL.StreamingOpInf(options=options, n=rmax, m=4, algorithm=:QRRLS, Γs=Γs)

        # Initialize the error factor
        Eps = nothing

        # Load the batch solution 
        mu_str = @sprintf("%1.4f", data["mu"])
        model_idx = findfirst(x -> occursin("mu$(mu_str)", x), model_files)
        op_inf = load(model_files[model_idx], "opinf")
        O_inf = vcat(op_inf.A', op_inf.B')

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
                    # Relative state and output errors
                    Xtmp = heat2d.integrate_model(
                        heat2d.tspan, iVrmax[:,1:ri]' * heat2d.IC, Uref; linear_matrix=op_tmp[key].A[1:ri,1:ri],
                        control_matrix=op_tmp[key].B[1:ri,:], system_input=true, integrator_type=:BackwardEuler
                    )
                    stream_res[key].rse[j, i] = LnL.rel_state_error(Xref, Xtmp, iVrmax[:,1:ri])

                    # Index for streaming errors
                    idx = vcat(collect(1:ri),collect(rmax+1:rmax+4))

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

# Average over the number of parameters
for key in keys(stream_res)
    stream_res[key].true_stream_err ./= heat2d.param_dim
    stream_res[key].stream_err ./= heat2d.param_dim
    stream_res[key].post_err ./= heat2d.param_dim
    stream_res[key].conv_factor ./= heat2d.param_dim
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
                Xrecon = heat2d.integrate_model(
                    heat2d.tspan, Vr' * heat2d.IC, Uref,
                    linear_matrix=ops[string(key)].A[1:r, 1:r], control_matrix=ops[string(key)].B[1:r,:],
                    system_input=true, integrator_type=:BackwardEuler
                )

                # Compute relative state error (averaged over parameters)
                train_errors[key][i] += norm(Xref - Vr * Xrecon) / norm(Xref) / num_train
            end
        end
    end
end

# Save the errors
save(joinpath(FILEPATH, "data/training_errors.jld2"), train_errors)

##
# #==============================#
# ## Generate RLS-Streaming-OpInf 
# #==============================#
# # Placeholders
# r = basis_data["r"]
# num_of_streams = heat2d.time_dim
# state_stream_res = (
#     true_stream_err = zeros(r, num_of_streams),
#     stream_err      = zeros(r, num_of_streams),
#     rse             = zeros(r, num_of_streams),
#     post_err        = zeros(num_of_streams),
#     conv_factor     = zeros(num_of_streams),
# )
# # output_stream_res = (
# #     true_stream_err = zeros(r, num_of_streams),
# #     stream_err      = zeros(r, num_of_streams),
# #     rse             = zeros(r, num_of_streams),
# #     post_err        = zeros(num_of_streams),
# #     conv_factor     = zeros(num_of_streams),
# # )
# # Dict to store all streaming results for different algorithms 
# stream_res = Dict(
#     "rls" => Dict(
#         "state" => state_stream_res,
#         "output" => output_stream_res
#     ),
#     "iqrrls" => Dict(
#         "state" => state_stream_res,
#         "output" => output_stream_res
#     ),
#     "qrrls" => Dict(
#         "state" => state_stream_res,
#         "output" => output_stream_res
#     )
# )
# model_files = readdir(joinpath(FILEPATH, "data/models"), join=true)
# for (i, data_file) in enumerate(training_data_files)
#     jldopen(data_file, "r") do data
#         # Load the data
#         Xfull = data["X"]
#         Ufull = data["U"]
#         Yfull = data["Y"]

#         # Obtain derivative data and adjust data
#         Xdot = (Xfull[:, 2:end] - Xfull[:, 1:end-1]) / heat2d.Δt
#         idx = 2:heat2d.time_dim
#         X = Xfull[:, idx]  
#         U = Ufull[:, idx]
#         Y = Yfull[:, idx] 
        
#         ## Streamify the data based on the selected streamsizes
#         streamsize = 1
#         X_stream = LnL.streamify(iVr' * X, streamsize)
#         U_stream = LnL.streamify(U, streamsize)
#         Y_stream = LnL.streamify(Y, streamsize)
#         Xdot_stream = LnL.streamify(iVr' * Xdot, streamsize)
#         num_of_streams = length(X_stream)

#         ## Initialize the stream
#         Γs = 1e-15
#         Γo = 1e-15
#         # standard RLS
#         rls_state_stream, rls_output_stream = LnL.StreamingOpInf(options=options, n=r, m=4, l=1, algorithm=:RLS, Γs=Γs, Γo=Γo) 
#         # inverse-QR RLS
#         rls_state_stream, rls_output_stream = LnL.StreamingOpInf(options=options, n=r, m=4, l=1, algorithm=:iQRRLS, Γs=Γs, Γo=Γo)
#         # QR RLS
#         rls_state_stream, rls_output_stream = LnL.StreamingOpInf(options=options, n=r, m=4, l=1, algorithm=:QRRLS, Γs=Γs, Γo=Γo)

#         Es = nothing
#         Eo = nothing

#         # Load the batch solution 
#         mu_str = @sprintf("%1.4f", data["mu"])
#         op_inf = load(model_files[findfirst(x -> occursin("mu$(mu_str)", x), model_files)], "opinf")
#         O_inf = vcat(op_inf.A', op_inf.B')

#         ## Stream one-by-one and collect data
#         @showprogress for i in 1:num_of_streams
#             # Stream, update, and get data matrix for the state system
#             D_rls = LnL.stream!(rls_state_stream, X_stream[i], Xdot_stream[i]; U=U_stream[i], final_step=true)
#             D_iqrrls = LnL.stream!(iqrrls_state_stream, X_stream[i], Xdot_stream[i]; U=U_stream[i], final_step=true)
#             D_qrrls = LnL.stream!(qrrls_state_stream, X_stream[i], Xdot_stream[i]; U=U_stream[i], final_step=true)

#             # Stream and update the output system
#             LnL.stream_output!(rls_output_stream, X_stream[i], Y_stream[i])
#             LnL.stream_output!(iqrrls_output_stream, X_stream[i], Y_stream[i])
#             LnL.stream_output!(qrrls_output_stream, X_stream[i], Y_stream[i])

#             # Unpack operators
#             op_rls_tmp = LnL.Operators()
#             LnL.unpack_operators!(op_rls_tmp, rls_state_stream.cache.O', rls_state_stream.termination_settings[:dims], rls_state_stream.termination_settings[:syms])
#             op_rls_tmp.C = rls_output_stream.cache.O'    

#             op_iqrrls_tmp = LnL.Operators()
#             LnL.unpack_operators!(op_iqrrls_tmp, iqrrls_state_stream.cache.O', iqrrls_state_stream.termination_settings[:dims], iqrrls_state_stream.termination_settings[:syms])
#             op_iqrrls_tmp.C = iqrrls_output_stream.cache.O'    

#             op_qrrls_tmp = LnL.Operators()
#             LnL.unpack_operators!(op_qrrls_tmp, qrrls_state_stream.cache.O', qrrls_state_stream.termination_settings[:dims], qrrls_state_stream.termination_settings[:syms])
#             op_qrrls_tmp.C = qrrls_output_stream.cache.O'    

#             # Error factors
#             state_err_fact = 1.0I - state_stream.cache.K * D
#             output_err_fact = 1.0I - output_stream.cache.K * X_stream[i]'
#             Es_true = O_inf - state_stream.cache.O
#             Eo_true = op_inf.C - output_stream.cache.O'

#             # Initialize the error factors
#             if i == 1
#                 Es = Es_true
#                 Eo = Eo_true'
#             end
            
#             # Update the error factors
#             Es = state_err_fact * Es
#             Eo = output_err_fact * Eo

#             # Loop through each reduced dimension
#             for (j, ri) in enumerate(1:r)
#                 # Relative state and output errors
#                 Xtmp = heat2d.integrate_model(
#                     heat2d.tspan, iVr[:,1:ri]' * heat2d.IC, U; linear_matrix=tmp.A[1:ri,1:ri], control_matrix=tmp.B[1:ri,:], 
#                     system_input=true, integrator_type=:BackwardEuler
#                 )
#                 Ytmp = tmp.C[:,1:ri] * Xtmp
#                 state_stream_res.rse[j, i] = LnL.rel_state_error(Xfull, Xtmp, iVr[:,1:ri])
#                 output_stream_res.rse[j, i] = LnL.rel_output_error(Yfull, Ytmp)

#                 # Index for streaming errors
#                 idx = vcat(collect(1:ri),collect(r+1:r+4))

#                 # Streaming errors
#                 O_norm = norm(O_inf[idx,1:ri], 2)
#                 C_norm = norm(op_inf.C[:,1:ri], 2)
#                 Es_true_sub = Es_true[idx,1:ri]
#                 Eo_true_sub = Eo_true[1:ri]
#                 Es_sub = Es[idx,1:ri]
#                 Eo_sub = Eo[1:ri]
                
#                 # Errors
#                 state_stream_res.true_stream_err[j, i] += norm(Es_true_sub, 2) / O_norm 
#                 state_stream_res.stream_err[j,i] += norm(Es_sub,2) / O_norm
#                 output_stream_res.true_stream_err[j,i] += norm(Eo_true_sub, 2) / C_norm 
#                 output_stream_res.stream_err[j,i] += norm(Eo_sub,2) / C_norm
#             end

#             # A posteriori error and conversion factors 
#             state_stream_res.post_err[i] += norm(state_stream.cache.ξpost,2)
#             state_stream_res.conv_factor[i] += state_stream.cache.C[1] 
#             output_stream_res.post_err[i] += norm(output_stream.cache.ξpost,2) 
#             output_stream_res.conv_factor[i] += output_stream.cache.C[1]
#         end

#         op_stream = LnL.terminate_stream(state_stream, output_stream)

#         # Save the streaming-based operators
#         jldopen(model_files[i], "a+") do model
#             model["opstream"] = op_stream
#         end 
#     end
#     @info "Streaming for model $(i) out of $(length(training_data_files)) is completed"
# end

# # Average over the number of parameters
# state_stream_res.true_stream_err ./= heat2d.param_dim
# state_stream_res.stream_err ./= heat2d.param_dim
# state_stream_res.post_err ./= heat2d.param_dim
# state_stream_res.conv_factor ./= heat2d.param_dim
# output_stream_res.true_stream_err ./= heat2d.param_dim
# output_stream_res.stream_err ./= heat2d.param_dim
# output_stream_res.post_err ./= heat2d.param_dim
# output_stream_res.conv_factor ./= heat2d.param_dim

# ## Save the streaming results
# filename = joinpath(FILEPATH, "data/streaming", "rls_stream_results.jld2")
# save(
#     filename,
#     "state_stream_res", state_stream_res, "output_stream_res", output_stream_res
# )
