"""
3D Channel flow: training models
"""

#================#
## Load Packages
#================#
using FileIO
using JLD2
using LinearAlgebra
using ProgressMeter
using Printf
using Random
using UniqueKronecker
import LiftAndLearn as LnL

#================================#
## Configure filepath for saving
#================================#
DATAPATH = "../../../../../DATA/NREL/3D_CHANNEL"
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"Two-Pass_Streaming-OpInf/3d_channel") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/3d_channel")
fn = "channel_5200_data_0_10000.h5"
datafile = joinpath(DATAPATH, fn)

#==========================================#
## Load struct to read data in HDF5 format 
#==========================================#
include(joinpath(FILEPATH, "datasource.jl"))

#========================#
## Additional functions ##
#========================#
include(joinpath(FILEPATH, "derivative.jl"))
include(joinpath(FILEPATH, "../utilities/extract_operators.jl"))
include(joinpath(FILEPATH, "../utilities/interpolate.jl"))

#=============================#
## Load the training dataset
#=============================#
ds = ChannelDataSource(datafile, ["z", "y", "x", "fields", "times"])
Nz, Ny, Nx, n_fields, n = ds.dims

#===============#
## Input data  ##
#===============#
U = - 0.001722 * ones(1,n)  

#===================#
## Setup the options
#===================#
# Some options for operator inference
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=[1,2],
        control=1,
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)
save(joinpath(FILEPATH, "data/setup.jld2"), 
    "options", options,
    "3dchannel", Dict(
        "Nx" => Nx, "Ny" => Ny, "Nz" => Nz, "n" => n, 
        "xspan" => ds["x"][:], 
        "yspan" => ds["y"][:],
        "zspan" => ds["z"][:],
        "tspan" => ds["times"][:]
    )
)

#=================#
## Load the bases 
#=================#
basis_file = joinpath(FILEPATH, "data/streaming/basis.jld2")
basis_data = load(basis_file)
iVrmax = basis_data["baker"].iVr  # choose Baker's iSVD basis
rmax = size(iVrmax,2)

#=========================#
## Train streaming model
#=========================#
Γ = 1e-8  # Regularization parameter

# The reduced dimensions to evaluate on
rspan = [rmax ÷ 2, rmax]

# The time span to integrate the reduced model
tspan = 1:0.01:n

num_of_streams = n  
tmp_res = (
    stream_err_fac  = zeros(length(rspan), num_of_streams),
    rse             = zeros(length(rspan), num_of_streams),
    post_err        = zeros(num_of_streams),
    conv_factor     = zeros(num_of_streams),
    cost            = zeros(num_of_streams),
)

# Dict to store all streaming results for different algorithms 
stream_res = Dict(
    :rls    => deepcopy(tmp_res),
    :iqrrls => deepcopy(tmp_res),
    :qrrls  => deepcopy(tmp_res),
)

## Initialize the streaming OpInfs
rls_stream  = LnL.TwoPassStreamingOpInf(options=options, n=rmax, m=1, algorithm=:RLS, Γs=Γ) 
# iqrrls_stream = LnL.TwoPassStreamingOpInf(options=options, n=rmax, m=1, algorithm=:iQRRLS, Γs=Γ)
# qrrls_stream = LnL.TwoPassStreamingOpInf(options=options, n=rmax, m=1, algorithm=:QRRLS, Γs=Γ)

## Preallocate a dictionary to store the streaming results
Eps = Dict{Symbol, Matrix{Float64}}(
    :rls    => Matrix{Float64}(undef, rls_stream.dims[:d], rmax), 
    :iqrrls => Matrix{Float64}(undef, rls_stream.dims[:d], rmax), 
    :qrrls  => Matrix{Float64}(undef, rls_stream.dims[:d], rmax)
)

## Stream one-by-one and collect data
@showprogress for i in 1:num_of_streams
    # Get the i-th snapshot
    x_i = ds[i]  
    xhat_i = iVrmax' * x_i  # Project the snapshot onto the basis
    
    # Compute the i-th time derivative 
    if i == 1
        # First snapshot, use forward finite difference
        dt = ds["times"][5] - ds["times"][1]
        xdot_i = fwd4(ds[i:i+4], dt/4, true)
    elseif i == 2
        # Second snapshot, use forward finite difference but with adjusted stencil
        dt = ds["times"][5] - ds["times"][1]
        xdot_i = fwd4(ds[i-1:i+3], dt/4, false)
    elseif i == n-1
        # Second to last snapshot, use backward finite difference
        dt = ds["times"][n] - ds["times"][n-4]
        xdot_i = bwd4(ds[i-3:i+1], dt/4, false)
    elseif i == n 
        # Last snapshot, use backward finite difference with adjusted stencil
        dt = ds["times"][n] - ds["times"][n-4]
        xdot_i = bwd4(ds[i-4:i], dt/4, true)
    else
        # For all other snapshots, use central finite difference
        dt = ds["times"][i+2] - ds["times"][i-2]
        xdot_i = ctd4(ds[i-2:i+2], dt/4)
    end
    xhatdot_i = iVrmax' * xdot_i  # Project the time derivative onto the basis

    # Get the i-th input 
    u_i = U[i]

    # Stream, update, and get data matrix for the state system
    LnL.stream!(rls_stream, xhat_i, xhatdot_i, U=[u_i])      # RLS
    # LnL.stream!(iqrrls_stream, xhat_i, xhatdot_i, U=[u_i])   # iQRRLS
    # LnL.stream!(qrrls_stream, xhat_i, xhatdot_i, U=[u_i])    # QRRLS

    # # Streaming errors (cannot be computed since we don't have Ostar)
    # if i == 1
    #     Eps[:rls]    = Ostar - rls_stream.cache.O ≈ 1.0
    #     Eps[:iqrrls] = Ostar - iqrrls_stream.cache.O ≈ 1.0
    #     Eps[:qrrls]  = Ostar - qrrls_stream.cache.O ≈ 1.0
    # else
    #     Eps[:rls]    .= Eps[:rls] - rls_stream.cache.K * rls_stream.cache.ξpre
    #     Eps[:iqrrls] .= Eps[:iqrrls] - iqrrls_stream.cache.K * iqrrls_stream.cache.ξpre
    #     Eps[:qrrls]  .= Eps[:qrrls] - qrrls_stream.cache.K * qrrls_stream.cache.ξpre
    # end

    # Streaming error factors
    Eps[:rls]    = rls_stream.cache.K * rls_stream.cache.ξpre
    # Eps[:iqrrls] = iqrrls_stream.cache.K * iqrrls_stream.cache.ξpre
    # Eps[:qrrls]  = qrrls_stream.cache.K * qrrls_stream.cache.ξpre

    stream_skip = num_of_streams ÷ 5
    if (i-1) % stream_skip == 0 || i ∈ num_of_streams-2:num_of_streams
        # Unpack operators
        # # RLS
        # op_rls = LnL.Operators()
        # LnL.unpack_operators!(
        #     op_rls, rls_stream.cache.O', 
        #     rls_stream.termination_settings[:dims], 
        #     rls_stream.termination_settings[:syms]
        # )

        # # iQRRLS
        # op_iqrrls = LnL.Operators()
        # LnL.unpack_operators!(
        #     op_iqrrls, iqrrls_stream.cache.O', 
        #     iqrrls_stream.termination_settings[:dims], 
        #     iqrrls_stream.termination_settings[:syms]
        # )

        # # QRRLS
        # op_qrrls = LnL.Operators()
        # LnL.unpack_operators!(
        #     op_qrrls, qrrls_stream.cache.O', 
        #     qrrls_stream.termination_settings[:dims], 
        #     qrrls_stream.termination_settings[:syms]
        # )

        # # Collect all the (temporary) operators into a dictionary
        # op_tmp = Dict(:rls => op_rls, :iqrrls => op_iqrrls, :qrrls => op_qrrls)
        # algo_keys = [key for key in keys(op_tmp)]       # Get the keys of the operators

        algo_keys = [:rls]

        Threads.@threads for k in eachindex(algo_keys)  # Loop through each algorithm
            key = algo_keys[k]
            # iterate through the reduced dimensions
            Threads.@threads for (j, rj) in collect(enumerate(rspan))
                #################
                ## INFO: Relative state error isn't really a good measure here
                #################
                # # Extract the quadratic matrix for lower dimensions
                # F_extract = UniqueKronecker.extractF(op_tmp[key].A2u, rj)
                # # Integrate to reconstruct the state
                # Xtmp = zeros(rj, n)
                # Xtmp[:,1] = iVrmax[:,1:rj]' * ds[1]
                # for k in 1:n-1
                #     dt = tspan[k+1] - tspan[k]
                #     Xtmp[:,k+1] = rk4_step(
                #         Xtmp[:,k], U[k], dt, 
                #         op_tmp[key].A[1:rj, 1:rj], F_extract, 
                #         op_tmp[key].B[1:rj,:]
                #     )
                # end

                # # Compute the relative state error or reconstruction error
                # rse_tmp = 0.0
                # tot_tmp = 0.0
                # for k in 1:n
                #     rse_tmp += norm(ds[k] - iVrmax[:,1:rj] * Xtmp[:,k], 2)
                #     tot_tmp += norm(ds[k], 2)
                # end
                # stream_res[key].rse[j,i] += rse_tmp / tot_tmp

                # Index to extract for lower dimensions
                idx = extract_indices(rls_stream, rmax, rj, options.system)

                # Extract for lower dimensions
                @views Eps_sub = Eps[key][idx,1:rj]

                # Streaming error factors
                stream_res[key].stream_err_fac[j,i] += norm(Eps_sub,2)

                @info "Done: Stream $i / $num_of_streams, Algorithm: $key, Reduced dimension: $rj"
            end
        end
    end

    # A posteriori error, conversion factors, and costs
    # RLS
    stream_res[:rls].post_err[i] += norm(rls_stream.cache.ξpost,2)
    stream_res[:rls].conv_factor[i] += rls_stream.cache.C[1] 
    stream_res[:rls].cost[i] += rls_stream.cache.J[1]
    # # iQRRLS
    # stream_res[:iqrrls].post_err[i] += norm(iqrrls_stream.cache.ξpost,2)
    # stream_res[:iqrrls].conv_factor[i] += iqrrls_stream.cache.C[1]
    # stream_res[:iqrrls].cost[i] += iqrrls_stream.cache.J[1]
    # # QRRLS
    # stream_res[:qrrls].post_err[i] += norm(qrrls_stream.cache.ξpost,2)
    # stream_res[:qrrls].conv_factor[i] += qrrls_stream.cache.C[1]
    # stream_res[:qrrls].cost[i] += qrrls_stream.cache.J[1]

    @info "Stream $i / $num_of_streams completed"
end

## Terminate the streaming operators
op_stream_rls    = LnL.terminate_stream(rls_stream)
# op_stream_iqrrls = LnL.terminate_stream(iqrrls_stream)
# op_stream_qrrls  = LnL.terminate_stream(qrrls_stream)

# ## Print the final relative state errors
# @printf("(RLS)    ||Xtrue - Xrecon||_F / ||Xtrue||_F = %.5e\n", 
#     stream_res[:rls].rse[end,end])
# @printf("(iQRRLS) ||Xtrue - Xrecon||_F / ||Xtrue||_F = %.5e\n", 
#     stream_res[:iqrrls].rse[end,end])
# @printf("(QRRLS)  ||Xtrue - Xrecon||_F / ||Xtrue||_F = %.5e\n",
#     stream_res[:qrrls].rse[end,end])

## Save the model
ops = Dict(
    # "opinf" => op_inf, "tropinf" => op_trinf, 
    "stream_rls" => op_stream_rls, 
    # "stream_iqrrls" => op_stream_iqrrls, 
    # "stream_qrrls" => op_stream_qrrls, 
    "rspan" => rspan
)
filename = joinpath(FILEPATH, "data/models", "operators.jld2")
save(filename, ops)

## Interpolate some of the results
# for key in keys(stream_res)
#     interpolate_zero_columns!(stream_res[key].rse)
#     interpolate_zero_columns!(stream_res[key].stream_err)
# end
interpolate_zero_columns!(stream_res[:rls].rse)
interpolate_zero_columns!(stream_res[:rls].stream_err)

## Save the streaming results
filename = joinpath(FILEPATH, "data/streaming", "stream_results.jld2")
save(filename, "stream_res", stream_res, "rspan", rspan)

#====================================#
## Compute the relative state errors 
#====================================#
# Error analysis 
rspan = [25, 50, 100]
train_errors = Dict(
    # :pod           => zeros(length(rspan),1),
    # :opinf         => zeros(length(rspan),1),
    # :tropinf       => zeros(length(rspan),1),
    :stream_rls    => zeros(length(rspan),1),
    :stream_iqrrls => zeros(length(rspan),1),
    :stream_qrrls  => zeros(length(rspan),1)
)

ops = load(joinpath(FILEPATH, "data/streamwise/models/operators.jld2"))
op_keys = [key for key in keys(train_errors)]
Threads.@threads for i in eachindex(op_keys)
    key = op_keys[i]
    Threads.@threads for (i,r) = collect(enumerate(rspan))

        Vr = iVrmax[:, 1:r]

        # Integrate the model
        tspan_rk4 = 0:0.01:tspan[end]
        F_extract = UniqueKronecker.extractF(ops[string(key)].A2u, r)
        Xrecon = zeros(r, length(tspan_rk4))
        Xrecon[:,1] = Vr' * ds[1]
        for k in 1:length(tspan_rk4)-1
            dt = tspan_rk4[k+1] - tspan_rk4[k]
            Xrecon[:,k+1] = rk4_step(
                Xrecon[:,k], U[1], dt, 
                ops[string(key)].A[1:r, 1:r], F_extract, 
                ops[string(key)].B[1:r,:]
            )
        end

        # Compute relative state error (averaged over parameters)
        X_interp = cubic_interpolate_matrix(X, tspan, tspan_rk4)
        train_errors[key][i] += norm(X_interp - Vr * Xrecon) / norm(X_interp)

        @info "Done: Algorithm: $key, Reduced dimension: $r"
    end
end

# Save the errors
save(joinpath(FILEPATH, "data/streamwise/training_errors.jld2"), 
    "train_errors", train_errors, 
    "rspan", rspan
)