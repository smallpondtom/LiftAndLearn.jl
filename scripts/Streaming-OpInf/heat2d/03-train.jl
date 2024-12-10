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
using LiftAndLearn
const LnL = LiftAndLearn

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"Streaming-OpInf/heat2d") : joinpath(pwd(), "scripts/Streaming-OpInf/heat2d")

#======================================#
## Obtain all the saved training files
#======================================#
training_data_files = readdir(joinpath(FILEPATH, "data/training"), join=true)
basis_file = joinpath(FILEPATH, "data/basis.jld2")
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
Vr = basis_data["Vr"]
iVr = basis_data["iVr"]

#====================#
## Train batch models
#====================#
model_files = []
@showprogress for (i, data_file) in enumerate(training_data_files)
    jldopen(data_file, "r") do data
        # Load the data
        X = data["X"]
        U = data["U"]
        Y = data["Y"]

        # Load the operators
        A = data["A"]
        B = data["B"]
        C = data["C"]
        μ = data["mu"]

        # Train the models 
        # POD-Galerkin 
        tmp = LnL.Operators(A=A, B=B, C=C)
        op_pod = LnL.pod(tmp, Vr, options.system)
        # OpInf
        options.with_reg = false
        op_inf = LnL.opinf(X, Vr, options; U=U, Y=Y)
        # Tikhonov Regularized OpInf
        options.with_reg = true
        options.λ = LnL.TikhonovParameter(
            A = 1e-6,
            B = 1e-6,
            C = 1e-6
        )
        op_trinf = LnL.opinf(X, Vr, options; U=U, Y=Y)
        # Save the model
        mu_str = @sprintf("%1.4f", μ)
        ops = Dict("pod" => op_pod, "opinf" => op_inf, "tropinf" => op_trinf, "mu" => μ)
        filename = joinpath(FILEPATH, "data/models", "op_mu$(mu_str).jld2")
        push!(model_files, filename)
        save(filename, ops)
    end
end

#===========================#
## Generate Streaming-OpInf 
#===========================#
# Placeholders
r = basis_data["r"]
num_of_streams = heat2d.time_dim
state_stream_res = (
    true_stream_err = zeros(r, num_of_streams),
    stream_err      = zeros(r, num_of_streams),
    rse             = zeros(r, num_of_streams),
    post_err        = zeros(num_of_streams),
    conv_factor     = zeros(num_of_streams),
)
output_stream_res = (
    true_stream_err = zeros(r, num_of_streams),
    stream_err      = zeros(r, num_of_streams),
    rse             = zeros(r, num_of_streams),
    post_err        = zeros(num_of_streams),
    conv_factor     = zeros(num_of_streams),
)
model_files = readdir(joinpath(FILEPATH, "data/models"), join=true)
for (i, data_file) in enumerate(training_data_files)
    jldopen(data_file, "r") do data
        # Load the data
        Xfull = data["X"]
        Ufull = data["U"]
        Yfull = data["Y"]

        # Obtain derivative data and adjust data
        Xdot = (Xfull[:, 2:end] - Xfull[:, 1:end-1]) / heat2d.Δt
        idx = 2:heat2d.time_dim
        X = Xfull[:, idx]  
        U = Ufull[:, idx]
        Y = Yfull[:, idx] 
        
        ## Streamify the data based on the selected streamsizes
        streamsize = 1
        X_stream = LnL.streamify(iVr' * X, streamsize)
        U_stream = LnL.streamify(U, streamsize)
        Y_stream = LnL.streamify(Y, streamsize)
        Xdot_stream = LnL.streamify(iVr' * Xdot, streamsize)
        num_of_streams = length(X_stream)

        ## Initialize the stream
        γs = 1e-15
        γo = 1e-15
        # γs = 1e-9
        # γo = 1e-9
        state_stream, output_stream = LnL.StreamingOpInf(options=options, n=r, m=4, l=1, algorithm=:iQRRLS, γs=γs, γo=γo)

        Es = nothing
        Eo = nothing

        # Load the batch solution 
        mu_str = @sprintf("%1.4f", data["mu"])
        op_inf = load(model_files[findfirst(x -> occursin("mu$(mu_str)", x), model_files)], "opinf")
        O_inf = vcat(op_inf.A', op_inf.B')

        ## Stream one-by-one and collect data
        @showprogress for i in 1:num_of_streams
            # Stream, update, and get data matrix for the state system
            D = LnL.stream!(state_stream, X_stream[i], Xdot_stream[i]; U=U_stream[i], final_step=true)

            # Stream and update the output system
            LnL.stream_output!(output_stream, X_stream[i], Y_stream[i])

            # Unpack operators
            tmp = LnL.Operators()
            LnL.unpack_operators!(tmp, state_stream.cache.O', state_stream.termination_settings[:dims], state_stream.termination_settings[:syms])
            tmp.C = output_stream.cache.O'    

            # Error factors
            state_err_fact = 1.0I - state_stream.cache.K * D
            output_err_fact = 1.0I - output_stream.cache.K * X_stream[i]'
            Es_true = O_inf - state_stream.cache.O
            Eo_true = op_inf.C - output_stream.cache.O'

            # Initialize the error factors
            if i == 1
                Es = Es_true
                Eo = Eo_true'
            end
            
            # Update the error factors
            Es = state_err_fact * Es
            Eo = output_err_fact * Eo

            # Loop through each reduced dimension
            for (j, ri) in enumerate(1:r)
                # Relative state and output errors
                Xtmp = heat2d.integrate_model(
                    heat2d.tspan, iVr[:,1:ri]' * heat2d.IC, U; linear_matrix=tmp.A[1:ri,1:ri], control_matrix=tmp.B[1:ri,:], 
                    system_input=true, integrator_type=:BackwardEuler
                )
                Ytmp = tmp.C[:,1:ri] * Xtmp
                state_stream_res.rse[j, i] = LnL.rel_state_error(Xfull, Xtmp, iVr[:,1:ri])
                output_stream_res.rse[j, i] = LnL.rel_output_error(Yfull, Ytmp)

                # Index for streaming errors
                idx = vcat(collect(1:ri),collect(r+1:r+4))

                # Streaming errors
                O_norm = norm(O_inf[idx,1:ri], 2)
                Es_true_sub = Es_true[idx,1:ri]
                Eo_true_sub = Eo_true[1:ri]
                Es_sub = Es[idx,1:ri]
                Eo_sub = Eo[1:ri]
                
                # Errors
                state_stream_res.true_stream_err[j, i] += norm(Es_true_sub, 2) / O_norm 
                state_stream_res.stream_err[j,i] += norm(Es_sub,2) / O_norm
                output_stream_res.true_stream_err[j,i] += norm(Eo_true_sub, 2) / O_norm 
                output_stream_res.stream_err[j,i] += norm(Eo_sub,2) / O_norm
            end

            # A posteriori error and conversion factors 
            state_stream_res.post_err[i] += norm(state_stream.cache.ξpost,2)
            state_stream_res.conv_factor[i] += state_stream.cache.C[1] 
            output_stream_res.post_err[i] += norm(output_stream.cache.ξpost,2) 
            output_stream_res.conv_factor[i] += output_stream.cache.C[1]
        end

        op_stream = LnL.terminate_stream(state_stream, output_stream)

        # Save the streaming-based operators
        jldopen(model_files[i], "a+") do model
            model["opstream"] = op_stream
        end 
    end
    @info "Streaming for model $(i) out of $(length(training_data_files)) is completed"
end

# Average over the number of parameters
state_stream_res.true_stream_err ./= heat2d.param_dim
state_stream_res.stream_err ./= heat2d.param_dim
state_stream_res.post_err ./= heat2d.param_dim
state_stream_res.conv_factor ./= heat2d.param_dim
output_stream_res.true_stream_err ./= heat2d.param_dim
output_stream_res.stream_err ./= heat2d.param_dim
output_stream_res.post_err ./= heat2d.param_dim
output_stream_res.conv_factor ./= heat2d.param_dim

## Save the streaming results
filename = joinpath(FILEPATH, "data/streaming", "stream_results.jld2")
save(
    filename,
    "state_stream_res", state_stream_res, "output_stream_res", output_stream_res
)