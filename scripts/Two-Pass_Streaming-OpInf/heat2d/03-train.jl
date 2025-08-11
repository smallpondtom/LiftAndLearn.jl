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
using Revise
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"Two-Pass_Streaming-OpInf/heat2d") : 
                                        joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/heat2d")

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
rmax = size(iVrmax,2)

#=======================#
## Additional functions
#=======================#
include(joinpath(FILEPATH, "../utilities/extract_operators.jl"))

#===================================#
## Train batch and streaming models
#===================================#
Γ = 1e-9  # Regularization parameter

num_of_streams = heat2d.time_dim-1  # -1 for the derivative and -1 for the removed last state
tmp_res = (
    true_stream_err = zeros(rmax, num_of_streams),
    stream_err      = zeros(rmax, num_of_streams),
    rse             = zeros(rmax, num_of_streams),
    post_err        = zeros(num_of_streams),
    conv_factor     = zeros(num_of_streams),
    cost            = zeros(num_of_streams),
)

# Dict to store all streaming results for different algorithms 
stream_res = Dict(
    :rls    => deepcopy(tmp_res),
    :iqrrls => deepcopy(tmp_res),
    :qrrls  => deepcopy(tmp_res)
)

for (file_idx, data_file) in enumerate(training_data_files)
    jldopen(data_file, "r") do data
        """
        Data 
        """
        # Load the data
        X = data["X"]
        U = data["U"]
        Xdot = data["Xdot"]
        Xref = data["Xref"]
        Uref = data["Uref"]

        # Load the operators
        A = data["A"]
        B = data["B"]
        μ = data["mu"]

        # Obtain the reduced data
        Xhat = iVrmax' * X
        Xhatdot = iVrmax' * Xdot

        """
        Train the batch models 
        """
        # POD-Galerkin 
        tmp = LnL.Operators(A=A, B=B)
        op_pod = LnL.pod(tmp, iVrmax, options.system)

        # OpInf
        options.with_reg = false
        op_inf = LnL.opinf(X, iVrmax, options; U=U, Xdot=Xdot)

        # Tikhonov Regularized OpInf
        options.with_reg = true
        options.λ = LnL.TikhonovParameter(A=Γ, B=Γ)
        op_trinf = LnL.opinf(X, iVrmax, options; U=U, Xdot=Xdot)

        # Keep the reference batch model to compare with the streaming models
        Ostar = op_trinf.O'

        """
        Train and analyze the streaming models
        """
        # Streamify the data based on the selected streamsizes
        streamsize = 1
        X_stream = LnL.streamify(Xhat, streamsize)
        U_stream = LnL.streamify(U, streamsize)
        Xdot_stream = LnL.streamify(Xhatdot, streamsize)
        foo = length(X_stream)
        @assert foo == num_of_streams "Wrong number of streams"

        # Initialize the streaming OpInfs
        rls_stream  = LnL.TwoPassStreamingOpInf(
            options=options, n=rmax, m=4, algorithm=:RLS, Γs=Γ) 
        iqrrls_stream = LnL.TwoPassStreamingOpInf(
            options=options, n=rmax, m=4, algorithm=:iQRRLS, Γs=Γ, qr_method=:qr, use_gpu=true)
        qrrls_stream = LnL.TwoPassStreamingOpInf(
            options=options, n=rmax, m=4, algorithm=:QRRLS, Γs=Γ, qr_method=:qr)

        # Preallocate a dictionary to store the streaming results
        # error_factors = Dict{Symbol, Matrix{Float64}}(
        #     :rls    => Matrix{Float64}(undef, rls_stream.dims[:d], rls_stream.dims[:d]), 
        #     :iqrrls => Matrix{Float64}(undef, rls_stream.dims[:d], rls_stream.dims[:d]), 
        #     :qrrls  => Matrix{Float64}(undef, rls_stream.dims[:d], rls_stream.dims[:d])
        # )
        Eps_true = Dict{Symbol, Matrix{Float64}}(
            :rls    => Matrix{Float64}(undef, rls_stream.dims[:d], rmax), 
            :iqrrls => Matrix{Float64}(undef, rls_stream.dims[:d], rmax), 
            :qrrls  => Matrix{Float64}(undef, rls_stream.dims[:d], rmax)
        )
        Eps = deepcopy(Eps_true)  # Initialize the error factor

        # Stream one-by-one and collect data
        @showprogress for i in 1:num_of_streams
            # The stream of data
            x_i    = X_stream[i]
            xdot_i = Xdot_stream[i]
            u_i    = U_stream[i]

            # Stream, update, and get data matrix for the state system
            LnL.stream!(rls_stream, x_i, xdot_i; U=u_i)     # RLS
            LnL.stream!(iqrrls_stream, x_i, xdot_i; U=u_i, use_gpu=true)  # iQRRLS
            LnL.stream!(qrrls_stream, x_i, xdot_i; U=u_i)   # QRRLS

            # Compute the error factor 
            # error_factors[:rls]    = 1.0I - rls_stream.cache.K * d
            # error_factors[:iqrrls] = 1.0I - iqrrls_stream.cache.K * d
            # error_factors[:qrrls]  = 1.0I - qrrls_stream.cache.K * d

            # Compute the true streaming error
            Eps_true[:rls]    .= Ostar - rls_stream.cache.O
            Eps_true[:iqrrls] .= Ostar - Array(iqrrls_stream.cache.O)
            Eps_true[:qrrls]  .= Ostar - qrrls_stream.cache.O

            # Streaming error for first update
            if i == 1
                Eps[:rls]    = copy(Eps_true[:rls])
                Eps[:iqrrls] = copy(Eps_true[:iqrrls])
                Eps[:qrrls]  = copy(Eps_true[:qrrls])
            else
                # Eps[:rls]    .= error_factors[:rls] * Eps[:rls]
                # Eps[:iqrrls] .= error_factors[:iqrrls] * Eps[:iqrrls]
                # Eps[:qrrls]  .= error_factors[:qrrls] * Eps[:qrrls]
                Eps[:rls]    .= Eps[:rls] - rls_stream.cache.K * rls_stream.cache.ξpre
                Eps[:iqrrls] .= Eps[:iqrrls] - iqrrls_stream.cache.K * iqrrls_stream.cache.ξpre
                Eps[:qrrls]  .= Eps[:qrrls] - qrrls_stream.cache.K * qrrls_stream.cache.ξpre
            end

            # Unpack operators
            # RLS
            op_rls = LnL.Operators()
            LnL.unpack_operators!(
                op_rls, rls_stream.cache.O', 
                rls_stream.termination_settings[:dims], 
                rls_stream.termination_settings[:syms]
            )

            # iQRRLS
            op_iqrrls = LnL.Operators()
            LnL.unpack_operators!(
                op_iqrrls, iqrrls_stream.cache.O', 
                iqrrls_stream.termination_settings[:dims], 
                iqrrls_stream.termination_settings[:syms]
            )

            # QRRLS
            op_qrrls = LnL.Operators()
            LnL.unpack_operators!(
                op_qrrls, qrrls_stream.cache.O', 
                qrrls_stream.termination_settings[:dims], 
                qrrls_stream.termination_settings[:syms]
            )

            # Collect all the (temporary) operators into a dictionary
            op_tmp = Dict(:rls => op_rls, :iqrrls => op_iqrrls, :qrrls => op_qrrls)

            algo_keys = [key for key in keys(op_tmp)]       # Get the keys of the operators
            Threads.@threads for k in eachindex(algo_keys)  # Loop through each algorithm
                key = algo_keys[k]
                # iterate through the reduced dimensions
                for (j, rj) in enumerate(1:rmax)
                    # Integrate to reconstruct the state
                    Xtmp = heat2d.integrate_model(
                        heat2d.tspan, iVrmax[:,1:rj]' * heat2d.IC, Uref; 
                        linear_matrix=op_tmp[key].A[1:rj,1:rj],
                        control_matrix=op_tmp[key].B[1:rj,:], 
                        system_input=true, integrator_type=:BackwardEuler
                    )

                    # Compute the relative state error or reconstruction error
                    stream_res[key].rse[j,i] += LnL.rel_state_error(Xref, Xtmp, iVrmax[:,1:rj])

                    # Index to extract for lower dimensions
                    idx = extract_indices(rls_stream, rmax, rj, options.system)

                    # Extract for lower dimensions
                    Ostar_norm = norm(Ostar[idx,1:rj], 2)
                    @views Eps_true_sub = Eps_true[key][idx,1:rj]
                    @views Eps_sub = Eps[key][idx,1:rj]

                    # Streaming errors
                    stream_res[key].true_stream_err[j, i] += norm(Eps_true_sub, 2) / Ostar_norm 
                    stream_res[key].stream_err[j,i] += norm(Eps_sub,2) / Ostar_norm
                end
            end

            # A posteriori error, conversion factors, and costs
            # RLS
            stream_res[:rls].post_err[i] += norm(rls_stream.cache.ξpost,2)
            stream_res[:rls].conv_factor[i] += rls_stream.cache.C[1] 
            stream_res[:rls].cost[i] += rls_stream.cache.J[1]
            # iQRRLS
            stream_res[:iqrrls].post_err[i] += norm(iqrrls_stream.cache.ξpost,2)
            stream_res[:iqrrls].conv_factor[i] += iqrrls_stream.cache.C[1]
            stream_res[:iqrrls].cost[i] += iqrrls_stream.cache.J[1]
            # QRRLS
            stream_res[:qrrls].post_err[i] += norm(qrrls_stream.cache.ξpost,2)
            stream_res[:qrrls].conv_factor[i] += qrrls_stream.cache.C[1]
            stream_res[:qrrls].cost[i] += qrrls_stream.cache.J[1]
        end

        # Terminate the streaming operators
        op_stream_rls    = LnL.terminate_stream(rls_stream)
        op_stream_iqrrls = LnL.terminate_stream(iqrrls_stream)
        op_stream_qrrls  = LnL.terminate_stream(qrrls_stream)

        # Compute final streaming errors for debugging
        @printf("(RLS)    ||O - Ostar||_F / ||Ostar||_F = %.5e\n", 
            norm(op_stream_rls.O - Ostar, 2)/norm(Ostar, 2))
        @printf("(iQRRLS) ||O - Ostar||_F / ||Ostar||_F = %.5e\n", 
            norm(op_stream_iqrrls.O - Ostar, 2)/norm(Ostar, 2))
        @printf("(QRRLS)  ||O - Ostar||_F / ||Ostar||_F = %.5e\n", 
            norm(op_stream_qrrls.O - Ostar, 2)/norm(Ostar, 2))

        # Save the model
        mu_str = @sprintf("%1.4f", μ)
        ops = Dict(
            "pod" => op_pod, "opinf" => op_inf, "tropinf" => op_trinf, 
            "stream_rls" => op_stream_rls, "stream_iqrrls" => op_stream_iqrrls, 
            "stream_qrrls" => op_stream_qrrls, "mu" => μ
        )
        filename = joinpath(FILEPATH, "data/models", "op_mu$(mu_str).jld2")
        save(filename, ops)
    end
    @info "Streaming for model $(file_idx) out of $(length(training_data_files)) is completed"
end

## Average over the number of parameters
for key in keys(stream_res)
    stream_res[key].rse ./= heat2d.param_dim
    stream_res[key].true_stream_err ./= heat2d.param_dim
    stream_res[key].stream_err ./= heat2d.param_dim
    stream_res[key].post_err ./= heat2d.param_dim
    stream_res[key].conv_factor ./= heat2d.param_dim
    stream_res[key].cost ./= heat2d.param_dim
end

## Save the streaming results
filename = joinpath(FILEPATH, "data/streaming", "stream_results.jld2")
save(filename, "stream_res", stream_res)

#====================================#
## Compute the relative state errors 
#====================================#
model_files = readdir(joinpath(FILEPATH, "data/models"), join=true)
num_train = length(training_data_files)

# Error analysis 
train_errors = Dict(
    :pod           => zeros(rmax,1),
    :opinf         => zeros(rmax,1),
    :tropinf       => zeros(rmax,1),
    :stream_rls    => zeros(rmax,1),
    :stream_iqrrls => zeros(rmax,1),
    :stream_qrrls  => zeros(rmax,1)
)

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

                # if occursin(r"stream", string(key))
                #     Vr = iVrmax[:, 1:r]
                # else
                #     Vr = Vrmax[:, 1:r]
                # end

                Vr = iVrmax[:, 1:r]

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