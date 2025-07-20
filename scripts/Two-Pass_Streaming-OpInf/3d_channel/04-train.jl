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
using UniqueKronecker
using BlockDiagonals
using SparseArrays
using Statistics
using Revise
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
dim_per_field = Nz * Ny * Nx


# ##
# function linear_to_4d_index(linear_idx, Nx, Ny, Nz, num_fields)
#     # Convert to 0-based index for easier calculation
#     idx = linear_idx - 1
    
#     # Calculate dimensions
#     spatial_size = Nx * Ny * Nz
    
#     # Extract field index
#     f = (idx ÷ spatial_size) + 1
    
#     # Extract spatial index within the field
#     spatial_idx = idx % spatial_size
    
#     # Extract x, y, z coordinates from spatial index
#     # Assuming column-major ordering: idx = a + b*Nx + c*Nx*Ny
#     a = (spatial_idx % Nx) + 1
#     b = ((spatial_idx ÷ Nx) % Ny) + 1  
#     c = (spatial_idx ÷ (Nx * Ny)) + 1
    
#     return (a, b, c, f)
# end

# function get_4d_coordinates(ds::ChannelDataSource, linear_idx)
#     Nz, Ny, Nx, num_fields, _ = ds.dims
#     return linear_to_4d_index(linear_idx, Nx, Ny, Nz, num_fields)
# end

# function verify_index_mapping(ds::ChannelDataSource, time_idx, linear_idx)
#     # Get 4D coordinates
#     a, b, c, f = get_4d_coordinates(ds, linear_idx)
    
#     # Get values using both methods
#     value1 = ds[time_idx][linear_idx]
#     value2 = ds[a, b, c, f, time_idx]
    
#     println("Linear index $linear_idx maps to (c=$c, b=$b, a=$a, f=$f)")
#     println("ds[$time_idx][$linear_idx] = $value1")
#     println("ds[$c, $b, $a, $f, $time_idx] = $value2")
#     println("Match: $(value1 == value2)")
    
#     return value1 == value2
# end

# ## Test with your example
# verify_index_mapping(ds, rand(1:n), rand(1:(Nx*Ny*Nz*4)))

# ##
# h = (i) -> 2*(i-1) + 1

# h.(1:16)

#==========================#
## Load the mean velocity
#==========================#
xbar = load(joinpath(FILEPATH, "data/streaming/mean.jld2"))["xbar"]
# minmax = load(joinpath(FILEPATH, "data/streaming/minmax.jld2"))["minmax"]
# xbar = minmax["xbar"]
# scale_factors = minmax["scale_factors"]
# minmax = nothing

dPdx = 0.001722
# scale_factors = [sqrt(dPdx), sqrt(dPdx), sqrt(dPdx), dPdx]
# scale_factors = [1.0, 0.1, 0.01]
scale_factors = [1.0, 0.01, 0.01, dPdx]

##
# ubar = sum(abs, xbar[1:dim_per_field]) / dim_per_field
# vbar = sum(abs, xbar[dim_per_field+1:dim_per_field*2]) / dim_per_field
# wbar = sum(abs, xbar[dim_per_field*2+1:dim_per_field*3]) / dim_per_field
# pbar = sum(abs, xbar[dim_per_field*3+1:dim_per_field*4]) / dim_per_field
# scale_factors = [
#     1.0, vbar / ubar, wbar / ubar, pbar / ubar
# ]

#===================#
## Setup the options
#===================#
# Some options for operator inference
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=[1,2],
        control=0,
        constant=1,
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
    use_backslash=true,
    # use_svd_truncation=false,
    # tolerance=1e-22
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
rmax = 500

#=================#
## Load the bases 
#=================#
# Standard basis
basis_file = joinpath(FILEPATH, "data/streaming/basis.jld2")
basis_data = load(basis_file)
iVrmax = basis_data["bases"]["baker"].iVr[:, 1:rmax]

# Fieldwise basis
# basis_file = joinpath(FILEPATH, "data/streaming/basis_fieldwise.jld2")
# field_results = load(basis_file)["field_results"]
# rmax = 50
# iVr_u = field_results[1].Q[:, 1:rmax]  # u-component basis
# iVr_v = field_results[2].Q[:, 1:rmax]  # v-component basis
# iVr_w = field_results[3].Q[:, 1:rmax]  # w-component basis
# iVr_p = field_results[4].Q[:, 1:rmax]  # p-component basis
# field_results = nothing  # Free memory
# GC.gc()
# iVrmax = sparse(BlockDiagonal([iVr_u, iVr_v, iVr_w, iVr_p]))
# iVr_u, iVr_v, iVr_w, iVr_p = nothing, nothing, nothing, nothing  
# GC.gc()


#=========================#
## Load reduced data
#=========================#
Xhat = load(joinpath(FILEPATH, "data/streaming/reduced_data_discrete_r$(rmax).jld2"))["Xhat"]
# Xhatdot = load(joinpath(FILEPATH, "data/streaming/reduced_data.jld2"))["Xhatdot"]
Xhat2 = Xhat[:, 2:end]
Xhat1 = Xhat[:, 1:end-1]  
# U = load(joinpath(FILEPATH, "data/streaming/reduced_data.jld2"))["U"]
# U = ones(size(Xhat1,2))

size(Xhat1,1) != rmax && @warn "Xhat has a different number of \
    rows than the basis. This might lead to unexpected results."

#=========================#
## Train Batch model
#=========================#
# OpInf
options.with_reg = false
op_inf = LnL.opinf(Xhat1, options; Xhatdot=Xhat2)

##
# ops = Dict("opinf" => op_inf)
# save(joinpath(FILEPATH, "data/models", "operators.jld2"), ops)
# ops = load(joinpath(FILEPATH, "data/models", "operators.jld2"))

## Tikhonov Regularized OpInf
options.with_reg = true
options.λ = LnL.TikhonovParameter(A=1e-12, A2=1e-6, K=1e-8)
op_inf = LnL.opinf(Xhat1, options; Xhatdot=Xhat2)

##
save(joinpath(FILEPATH, "data/models", "operators_tol1e-12.jld2"), 
    "opinf", op_inf)

## Manual Computation for memory restrictions
rmax2 = Int(rmax*(rmax+1)/2)
nt = n - 1
D = [Xhat1', (Xhat1 ⨸ Xhat1)', ones(nt,1)]
D = reduce(hcat, D)
Γsq = BlockDiagonal(
    [sqrt(1e-6) * I(rmax), sqrt(1e-4) * I(rmax2), sqrt(1e-6) * I(1)]
)  # Change these regularization parameters as needed
D = vcat(D, Γsq)
R = vcat(Xhat2', zeros(size(Γsq, 1), rmax))

##
Γsq = nothing
GC.gc()  # Free memory

##

O = D \ R  # Solve the linear system
op_inf = LnL.Operators(
    A=O[1:rmax, :],
    A2u=O[rmax+1:rmax+rmax2, :],
    K=O[rmax+rmax2+1:end, :]
)


# ops["tropinf"] = op_trinf
# save(joinpath(FILEPATH, "data/models", "operators.jld2"), ops)

## Keep the reference batch model to compare with the streaming models
Ostar = op_trinf.O'
# Ostar = ops["tropinf"].O'


#========================================#
## Grid Search Regularization Parameters
#========================================#
function solve_opinf_difference_model(init_cond, n_steps, reduced_model)
    Qhat = zeros(length(init_cond), n_steps)
    contains_nan = false
    Qhat[:, 1] = init_cond
    final_idx = 0
    for i in 2:n_steps
        Qhat[:, i] = reduced_model(Qhat[:, i-1])
        if any(isnan.(Qhat[:, i]))
            contains_nan = true
            final_idx = i - 1
            break
        end
    end
    return contains_nan, Qhat, final_idx
end

function find_best_opinf_model(
    reg_pairs, Xhat, Xhat1, Xhat2, 
    n_time, n_time_pred, max_growth, opinf_options)

    @assert options.with_reg == true "Regularization must be enabled in options."
    
    best_train_err = 1e20
    best_beta1, best_beta2 = nothing, nothing
    best_final_idx = 0
    Xtilde_opt = nothing
    eval_time_opt = nothing

    mean_Xhat = mean(Xhat, dims=2)
    max_diff_Xhat = maximum(abs.(Xhat .- mean_Xhat), dims=2)
    
    # Loop over all regularization pairs
    @showprogress for (beta1, beta2) in reg_pairs
        
        # Construct a regularizer that penalizes the linear and constant reduced
        # operators using beta1 and the quadratic operator using beta2
        reg = LnL.TikhonovParameter(A=beta1, A2=beta2, K=beta1)
        opinf_options.λ = reg
        
        # Solve the regularized OpInf problem
        ops = LnL.opinf(Xhat1, opinf_options; Xhatdot=Xhat2)
        
        # Define the OpInf reduced model
        opinf_reduced_model = x -> ops.A * x + ops.A2u * (x ⊘ x) + ops.K

        # Extract the reduced initial condition from Qhat_1
        xhat0 = Xhat1[:,1]
        
        # Compute the reduced solution over the trial time horizon
        start_eval_time = time()
        contains_nans, Xtilde, fidx = solve_opinf_difference_model(
            xhat0, n_time_pred, opinf_reduced_model)
        end_eval_time = time()
        time_opinf_eval = end_eval_time - start_eval_time
        
        # If the model produced an unstable solution, move on to the next
        # regularization candidates
        if contains_nans
            ops = nothing
            GC.gc() 
            continue
        end
        
        # If the ratio of the maximum coefficient growth exceeds the allowed
        # threshold, move on to the next regularization candidates
        max_diff_Xhat_trial = maximum(abs.(Xtilde .- mean_Xhat), dims=2)
        max_growth_trial = maximum(max_diff_Xhat_trial) / maximum(max_diff_Xhat)
        if max_growth_trial > max_growth
            ops = nothing
            GC.gc() 
            continue
        end
        
        # At this point we know the model produced a stable solution without too
        # much growth. Compute the training error and, if it's better than the
        # current best error, save the regularization, reduced solution, and
        # the learning times
        train_err = norm(
                Xhat[:, 1:n_time] - Xtilde[:, 1:n_time]
            )^2 / norm(Xhat[:, 1:n_time])^2
        if train_err < best_train_err
            best_beta1 = beta1
            best_beta2 = beta2
            best_train_err = train_err
            Xtilde_opt = Xtilde
            eval_time_opt = time_opinf_eval
        end

        if best_final_idx < fidx
            best_final_idx = fidx
        end

        ops = nothing
        GC.gc() 
    end

    if isnothing(Xtilde_opt)
        @error "No suitable OpInf model found with the given regularization pairs."
    else
        @info "Best OpInf model found with β1 = $best_beta1, β2 = $best_beta2, \
               training error = $best_train_err, evaluation time = $eval_time_opt"
    end

    return best_beta1, best_beta2, best_train_err, Xtilde_opt, eval_time_opt, best_final_idx
end

##
B1 = 10.0 .^ range(-24.0, -20.0, length=8)
B2 = 10.0 .^ range(-20.0, -8.0, length=8)
reg_pairs_global = vec([(b1, b2) for b1 in B1, b2 in B2])
n_reg_global = length(reg_pairs_global)
max_growth = 1.2
options.with_reg = true
best_beta1, best_beta2, best_train_err, op_trinf, eval_time, fidx = 
    find_best_opinf_model(reg_pairs_global, Xhat, Xhat1, Xhat2,
                          n, Int(n+(n // 10)), max_growth, options)


#=========================#
## Train streaming model
#=========================#
rmax2 = Int(rmax*(rmax+1)/2)
options.with_reg = true
options.λ = LnL.TikhonovParameter(A=1e-6, A2=1e-4, K=1e-6)
rls_stream  = LnL.TwoPassStreamingOpInf(
    options=options, n=rmax, m=0, algorithm=:RLS, qr_method=:givens) 
start_time = time()
for i in 1:n-1
    xhat1_i = @views Xhat1[:,i]
    xhat2_i = @views Xhat2[:,i]  
    LnL.stream!(rls_stream, xhat1_i, xhat2_i)

    @info "Processed $i snapshots in $(time() - start_time) seconds"
    start_time = time()
end
op_stream_rls = LnL.terminate_stream(rls_stream)

#=========================#
## Train streaming model
#=========================#
# The reduced dimensions to evaluate on
rspan = [rmax ÷ 2, rmax]

num_of_streams = n - 1
tmp_res = (
    stream_err  = zeros(length(rspan), num_of_streams),
    rse         = zeros(length(rspan), num_of_streams),
    post_err    = zeros(num_of_streams),
    conv_factor = zeros(num_of_streams),
    cost        = zeros(num_of_streams),
)

# Dict to store all streaming results for different algorithms 
stream_res = Dict(
    :rls    => deepcopy(tmp_res),
    :iqrrls => deepcopy(tmp_res),
    :qrrls  => deepcopy(tmp_res),
)

## Initialize the streaming OpInfs
rls_stream  = LnL.TwoPassStreamingOpInf(
    options=options, n=rmax, m=0, algorithm=:RLS, Γs=Γsq) 
# iqrrls_stream = LnL.TwoPassStreamingOpInf(
#     options=options, n=rmax, m=0, algorithm=:iQRRLS, Γs=Γsq)
# qrrls_stream = LnL.TwoPassStreamingOpInf(
#     options=options, n=rmax, m=0, algorithm=:QRRLS, Γs=Γsq)

## Preallocate a dictionary to store the streaming results
Eps = Dict{Symbol, Matrix{Float64}}(
    :rls    => Matrix{Float64}(undef, rls_stream.dims[:d], rmax), 
    :iqrrls => Matrix{Float64}(undef, rls_stream.dims[:d], rmax), 
    :qrrls  => Matrix{Float64}(undef, rls_stream.dims[:d], rmax)
)

## Stream one-by-one and collect data
@showprogress for i in 1:num_of_streams
    # Get the i-th snapshot
    # x_i = ds[i]  
    # xhat_i = iVrmax' * x_i  # Project the snapshot onto the basis
    xhat1_i = Xhat1[:,i]
    
    # # Compute the i-th time derivative 
    # if i == 1
    #     # First snapshot, use forward finite difference
    #     dt = ds["times"][5] - ds["times"][1]
    #     xdot_i = fwd4(ds[i:i+4], dt/4, true)
    # elseif i == 2
    #     # Second snapshot, use forward finite difference but with adjusted stencil
    #     dt = ds["times"][5] - ds["times"][1]
    #     xdot_i = fwd4(ds[i-1:i+3], dt/4, false)
    # elseif i == n-1
    #     # Second to last snapshot, use backward finite difference
    #     dt = ds["times"][n] - ds["times"][n-4]
    #     xdot_i = bwd4(ds[i-3:i+1], dt/4, false)
    # elseif i == n 
    #     # Last snapshot, use backward finite difference with adjusted stencil
    #     dt = ds["times"][n] - ds["times"][n-4]
    #     xdot_i = bwd4(ds[i-4:i], dt/4, true)
    # else
    #     # For all other snapshots, use central finite difference
    #     dt = ds["times"][i+2] - ds["times"][i-2]
    #     xdot_i = ctd4(ds[i-2:i+2], dt/4)
    # end
    # xhatdot_i = iVrmax' * xdot_i  # Project the time derivative onto the basis

    xhat2_i = Xhat2[:,i]  # Use the precomputed time derivative

    # Stream, update, and get data matrix for the state system
    LnL.stream!(rls_stream, xhat1_i, xhat2_i)      # RLS
    # LnL.stream!(iqrrls_stream, xhat_i, xhatdot_i)   # iQRRLS
    # LnL.stream!(qrrls_stream, xhat_i, xhatdot_i)    # QRRLS

    # Streaming errors (cannot be computed since we don't have Ostar)
    if i == 1
        Eps[:rls]    = Ostar - rls_stream.cache.O 
        Eps[:iqrrls] = Ostar - iqrrls_stream.cache.O
        Eps[:qrrls]  = Ostar - qrrls_stream.cache.O
    else
        Eps[:rls]    .= Eps[:rls] - rls_stream.cache.K * rls_stream.cache.ξpre
        Eps[:iqrrls] .= Eps[:iqrrls] - iqrrls_stream.cache.K * iqrrls_stream.cache.ξpre
        Eps[:qrrls]  .= Eps[:qrrls] - qrrls_stream.cache.K * qrrls_stream.cache.ξpre
    end

    # # Streaming error factors
    # Eps[:rls]    = rls_stream.cache.K * rls_stream.cache.ξpre
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
                stream_res[key].stream_err[j,i] += norm(Eps_sub,2)

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
    "opinf" => op_inf, "tropinf" => op_trinf, 
    # "stream_rls" => op_stream_rls, 
    # "stream_iqrrls" => op_stream_iqrrls, 
    # "stream_qrrls" => op_stream_qrrls, 
    # "rspan" => rspan
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
    :opinf         => zeros(length(rspan),1),
    :tropinf       => zeros(length(rspan),1),
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


#====================================#
## Simulate the reduced model
#====================================#
reduced_model = (x) -> op_inf.A * x + op_inf.A2u * (x ⊘ x) + op_inf.K

##
tspan = ds["times"][:] .- ds["times"][1]
states = zeros(rmax, length(tspan))
states[:,1] = Xhat1[:,1]
for i in 2:length(tspan)
    states[:,i] = reduced_model(states[:,i-1])
    if any(isnan.(states[:, i]))
        @info "Reduced model produced NaN at time step $i"
        break
    end
end

## Plot the states 
using CairoMakie
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1200, 900))
    # Get midpoint index for z-direction
    z_mid = Nz ÷ 2

    # Select 3 time steps (beginning, middle, end)
    time_indices = [50, 1000, length(tspan)÷2, length(tspan)]

    # Pre-calculate all data for colorbar scaling
    all_full_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_rom_data = Vector{Matrix{Float64}}(undef, length(time_indices))
    all_error_data = Vector{Matrix{Float64}}(undef, length(time_indices))

    # Collect all data first
    for (i, t_idx) in enumerate(time_indices)
        # Get full data
        u_full_field = reshape(ds[t_idx][1:dim_per_field], Nx, Ny, Nz)
        all_full_data[i] = u_full_field[:, :, z_mid]
        
        # Get ROM data
        x_rom_t = iVrmax * states[:, t_idx]
        x_rom_t = unscale(x_rom_t, dim_per_field, scale_factors) + xbar
        u_rom_field = reshape(x_rom_t[1:dim_per_field], Nx, Ny, Nz)
        all_rom_data[i] = u_rom_field[:, :, z_mid]

        # Compute error
        all_error_data[i] = abs.(all_full_data[i] - all_rom_data[i]) 
    end

    # Calculate global min/max for each row type
    full_min, full_max = extrema(vcat(all_full_data...))
    rom_min, rom_max = extrema(vcat(all_rom_data...))
    error_min, error_max = extrema(vcat(all_error_data...))

    # Align the color ranges for first and second rows (full and ROM)
    common_min = min(full_min, rom_min)
    common_max = max(full_max, rom_max)

    # Create axes and heatmaps
    hm_full = nothing
    hm_rom = nothing
    hm_error = nothing

    for (i, t_idx) in enumerate(time_indices)
        # Create axes
        ax_full = Axis(fig[1, i], 
            title = L"$t$ = %$(round(tspan[t_idx], digits=2))",
            ylabel = i == 1 ? L"$y$" : "", 
            xlabelsize=30, ylabelsize=30, 
            xticklabelsize=25, yticklabelsize=25,
            titlesize=30,
        )
        ax_rom = Axis(fig[2, i], 
            ylabel = i == 1 ? L"$y$" : "", 
            xlabelsize=30, ylabelsize=30, 
            xticklabelsize=25, yticklabelsize=25,
        )
        ax_error = Axis(fig[3, i], 
            ylabel = i == 1 ? L"$y$" : "", 
            xlabel = L"$x$",
            xlabelsize=30, ylabelsize=30, 
            xticklabelsize=25, yticklabelsize=25,
        )

        # Create heatmaps with aligned color ranges
        hm_full = heatmap!(ax_full, ds["x"][:], ds["y"][:], all_full_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_rom = heatmap!(ax_rom, ds["x"][:], ds["y"][:], all_rom_data[i], 
            colormap = :viridis, colorrange = (common_min, common_max))
        hm_error = heatmap!(ax_error, ds["x"][:], ds["y"][:], all_error_data[i], 
            colormap = :matter, colorrange = (error_min, error_max))
    end
    
    # Add colorbars at the end of each row
    Colorbar(fig[1, length(time_indices) + 1], hm_full, label="Full", labelsize=20)
    Colorbar(fig[2, length(time_indices) + 1], hm_rom, label="ROM", labelsize=20)
    Colorbar(fig[3, length(time_indices) + 1], hm_error, label="Abs. Error", labelsize=20)
    
    display(fig)
end

## Plot one reconstructed state over time for each field 
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1200, 800))
    
    # Pick the first spatial point for each field
    idx_u = 1  # First point in u field
    idx_v = dim_per_field + 1  # First point in v field  
    idx_w = 2*dim_per_field + 1  # First point in w field
    idx_p = 3*dim_per_field + 1  # First point in p field
    
    field_indices = [idx_u, idx_v, idx_w, idx_p]
    field_names = ["u", "v", "w", "p"]
    field_colors = [:orange, :orange, :orange, :orange]
    
    # Pre-extract basis rows for each field (much more efficient)
    basis_rows = [iVrmax[idx, :] for idx in field_indices]
    
    for (i, (idx, name, color)) in enumerate(zip(field_indices, field_names, field_colors))
        ax = Axis(fig[i, 1], 
            ylabel = L"%$(name)", 
            xlabel = i == 4 ? "Time" : "",
            xlabelsize = 20, 
            ylabelsize = 20,
            xticklabelsize = 15, 
            yticklabelsize = 15,
            title = i == 1 ? "Reconstructed vs True States" : "",
            titlesize = 20
        )
        
        # Pre-allocate arrays
        true_field = zeros(length(tspan))
        rom_field = zeros(length(tspan))
        
        # Extract basis row once for this field
        basis_row = basis_rows[i]
        
        # Determine which field we're in (0-indexed)
        field_idx = (idx - 1) ÷ dim_per_field
        scale_factor = scale_factors[field_idx + 1]
        mean_val = xbar[idx]
        
        # Vectorized operations for efficiency
        tmp = ds[name][]

        for t in 1:length(tspan)
            # True state (direct indexing)
            true_field[t] = ds[t][idx]
            
            # ROM state (efficient dot product + scaling)
            rom_val = dot(basis_row, states[:, t])
            rom_field[t] = rom_val * scale_factor + mean_val
        end
        
        # Plot true vs reconstructed
        lines!(ax, tspan, true_field, color=:black, linewidth=2, 
               labelsize=25, label="True")
        lines!(ax, tspan, rom_field, color=color, linewidth=2, linestyle=:dash,
               labelsize=25, label="ROM")
        
        # Add legend only to the top subplot
        if i == 1
            axislegend(ax, position=:rt)
        end
    end
    
    display(fig)
end


## Compute the relative state error
rse = zeros(length(tspan))
for i in 1:length(tspan)
    x_full_i = iVrmax * states[:, i]
    x_full_i = unscale(x_full_i, dim_per_field, scale_factors) + xbar
    # x_full_i = x_full_i[1:Nz*Ny*Nx*3] 
    rse[i] = norm(x_full_i - ds[i], 2)
end
rse_tot = sum(rse) / length(rse)

## Compute the relative state error efficiently with parallelization
@info "Computing relative state error with $(Threads.nthreads()) threads..."

@time begin
    rse = zeros(length(tspan))
    den = zeros(length(tspan))
    
    # Pre-allocate thread-local temporary arrays to avoid allocations in the loop
    temp_arrays = [zeros(size(iVrmax, 1)) for _ in 1:Threads.nthreads()]
    true_states = [zeros(size(iVrmax, 1)) for _ in 1:Threads.nthreads()]
    
    # Parallelize the RSE computation across time steps
    Threads.@threads for i in 1:length(tspan)
        tid = Threads.threadid()
        temp_full = temp_arrays[tid]
        true_state = true_states[tid]
        
        # Reconstruct full state for time step i (reuse pre-allocated array)
        mul!(temp_full, iVrmax, states[:, i])  # More efficient matrix-vector multiplication
        
        # Apply scaling and mean (in-place operations)
        temp_full .= unscale(temp_full, dim_per_field, scale_factors) .+ xbar
        
        # Load true state (reuse pre-allocated array)
        copyto!(true_state, ds[i])
        
        # Compute relative state error using efficient norm computation
        temp_full .-= true_state  # Compute difference in-place
        rse[i] = norm(temp_full) 
        den[i] = norm(true_state)
    end
    
    # Compute total relative state error
    rse_tot = sum(rse) / sum(den)
    
    @info "Relative state error computation completed"
    @info "Average RSE: $(rse_tot)"
    @info "Max RSE: $(maximum(rse))"
    @info "Min RSE: $(minimum(rse))"
end