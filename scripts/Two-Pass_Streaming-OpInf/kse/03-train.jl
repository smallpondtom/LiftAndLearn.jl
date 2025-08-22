"""
Kuramoto–Sivashinsky equation: training models
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using LinearAlgebra
using ProgressMeter
using PolynomialModelReductionDataset: KuramotoSivashinskyModel
using Printf
using Random
using UniqueKronecker
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"Two-Pass_Streaming-OpInf/kse") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/kse")

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

#=================#
## Load the bases 
#=================#
basis_data = load(basis_file)
Vrmax = basis_data["batch"].Vr
iVrmax = basis_data["baker"].iVr  # choose Baker's iSVD basis
rmax = size(iVrmax,2)

#=====================================================#
## Load all the data and concatenate into data matrix
#=====================================================#
X = Vector{Matrix{Float64}}(undef, length(training_data_files))
Xdot = Vector{Matrix{Float64}}(undef, length(training_data_files))
Xref = Vector{Matrix{Float64}}(undef, length(training_data_files))
for (file_idx, data_file) in enumerate(training_data_files)
    jldopen(data_file, "r") do data
        X[file_idx] = data["X"]
        Xdot[file_idx] = data["Xdot"]
        Xref[file_idx] = data["Xref"]
    end
end
X = reduce(hcat, X)
Xdot = reduce(hcat, Xdot) 
Xref = reduce(hcat, Xref)

#=======================#
## Additional functions
#=======================#
include(joinpath(FILEPATH, "../utilities/extract_operators.jl"))
include(joinpath(FILEPATH, "../utilities/interpolate.jl"))

#===================================#
## Train batch and streaming models
#===================================#
Γ = 1e-9  # Regularization parameter
rmin = 9
rrange = rmin:3:rmax
rlength = length(rrange)
setup["rrange"] = rrange

num_of_streams = Int(setup["num_of_streams"])
tmp_res = (
    true_stream_err = zeros(rlength, num_of_streams),
    stream_err      = zeros(rlength, num_of_streams),
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

# Load the FOM operators
FOM = setup["FOM"]
A = FOM["A"]
F = FOM["F"]

begin 
    """
    Data 
    """
    # Obtain the reduced data
    Xhat = iVrmax' * X
    Xhatdot = iVrmax' * Xdot

    """
    Train the batch models 
    """
    # POD-Galerkin 
    tmp = LnL.Operators(A=A, A2u=F)
    op_pod = LnL.pod(tmp, iVrmax, options.system)

    # OpInf
    options.with_reg = false
    op_inf = LnL.opinf(X, iVrmax, options; Xdot=Xdot)

    # Tikhonov Regularized OpInf
    options.with_reg = true
    options.λ = LnL.TikhonovParameter(A=Γ, A2=Γ)
    op_trinf = LnL.opinf(X, iVrmax, options; Xdot=Xdot)

    # Keep the reference batch model to compare with the streaming models
    Ostar = op_trinf.O'

    """
    Train and analyze the streaming models
    """
    # Streamify the data based on the selected streamsizes
    streamsize = 1
    X_stream = LnL.streamify(Xhat, streamsize)
    Xdot_stream = LnL.streamify(Xhatdot, streamsize)
    foo = length(X_stream)
    @assert foo == num_of_streams "Wrong number of streams"

    # Initialize the streaming OpInfs
    rls_stream  = LnL.TwoPassStreamingOpInf(options=options, n=rmax, algorithm=:RLS, Γs=Γ) 
    iqrrls_stream = LnL.TwoPassStreamingOpInf(options=options, n=rmax, algorithm=:iQRRLS, Γs=Γ)
    qrrls_stream = LnL.TwoPassStreamingOpInf(options=options, n=rmax, algorithm=:QRRLS, Γs=Γ)

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

        # Stream, update, and get data matrix for the state system
        LnL.stream!(rls_stream, x_i, xdot_i)  # RLS
        LnL.stream!(iqrrls_stream, x_i, xdot_i)   # iQRRLS
        LnL.stream!(qrrls_stream, x_i, xdot_i)    # QRRLS

        # Compute the true streaming error
        Eps_true[:rls]    .= Ostar - rls_stream.cache.O
        Eps_true[:iqrrls] .= Ostar - iqrrls_stream.cache.O
        Eps_true[:qrrls]  .= Ostar - qrrls_stream.cache.O

        # Streaming error for first update
        if i == 1
            Eps[:rls]    = copy(Eps_true[:rls])
            Eps[:iqrrls] = copy(Eps_true[:iqrrls])
            Eps[:qrrls]  = copy(Eps_true[:qrrls])
        else
            Eps[:rls]    .= Eps[:rls] - rls_stream.cache.K * rls_stream.cache.ξpre
            Eps[:iqrrls] .= Eps[:iqrrls] - iqrrls_stream.cache.K * iqrrls_stream.cache.ξpre
            Eps[:qrrls]  .= Eps[:qrrls] - qrrls_stream.cache.K * qrrls_stream.cache.ξpre
        end

        if (i-1) % 2 == 0 || i ∈ num_of_streams-10:num_of_streams
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
                for (j, rj) in enumerate(rrange)
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
    ops = Dict(
        "pod" => op_pod, "opinf" => op_inf, "tropinf" => op_trinf, 
        "stream_rls" => op_stream_rls, "stream_iqrrls" => op_stream_iqrrls, 
        "stream_qrrls" => op_stream_qrrls,
    )
    fn = joinpath(FILEPATH, "data/models", "op_mu1.0.jld2")
    save(fn, ops)
end

## Interpolate some of the results
for key in keys(stream_res)
    interpolate_zero_columns!(stream_res[key].true_stream_err)
    interpolate_zero_columns!(stream_res[key].stream_err)
end

## Save the streaming results
fn = joinpath(FILEPATH, "data/streaming", "stream_results.jld2")
save(fn, "stream_res", stream_res)

#====================================#
## Compute the relative state errors 
#====================================#
model_files = readdir(joinpath(FILEPATH, "data/models"), join=true)
num_train = length(training_data_files)

# Error analysis 
train_errors = Dict(
    :pod           => zeros(rlength,1),
    :opinf         => zeros(rlength,1),
    :tropinf       => zeros(rlength,1),
    :stream_rls    => zeros(rlength,1),
    :stream_iqrrls => zeros(rlength,1),
    :stream_qrrls  => zeros(rlength,1)
)
##
@showprogress for (file_idx, train_file) in enumerate(training_data_files)
    jldopen(train_file, "r") do data
        # Load the data
        Xref = data["Xref"]
        IC = data["IC"]

        # Load the trained models
        model_idx = findfirst(x -> occursin("mu1.0",x), model_files)
        ops = load(model_files[model_idx]) 

        op_keys = [key for key in keys(train_errors)]
        Threads.@threads for i in eachindex(op_keys)
            key = op_keys[i]
            for (i,r) = enumerate(rrange)

                # if occursin(r"stream", string(key))
                #     Vr = iVrmax[:, 1:r]
                # else
                #     Vr = Vrmax[:, 1:r]
                # end

                Vr = iVrmax[:, 1:r]

                # Integrate the model
                F_extract = UniqueKronecker.extractF(ops[string(key)].A2u, r)
                Xrecon = kse.integrate_model(
                    kse.tspan, Vr' * IC,
                    linear_matrix=ops[string(key)].A[1:r, 1:r],
                    quadratic_matrix=F_extract,
                    system_input=false, const_stepsize=true
                )

                # Compute relative state error (averaged over parameters)
                train_errors[key][i] += norm(Xref - Vr * Xrecon) / norm(Xref) / num_train
            end
        end
    end
end

# Save the errors
save(joinpath(FILEPATH, "data/training_errors.jld2"), train_errors)