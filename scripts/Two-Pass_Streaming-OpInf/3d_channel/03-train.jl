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

#==========================#
## Load the mean velocity
#==========================#
xbar = load(joinpath(FILEPATH, "data/streaming/mean.jld2"))["xbar"]

#=============================#
## Load the training dataset
#=============================#
ds = ChannelDataSource(datafile, ["z", "y", "x", "fields", "times"])
Nz, Ny, Nx, n_fields, n = ds.dims
dim_per_field = Nz * Ny * Nx
dPdx = 0.001722
# scale_factors = [sqrt(dPdx), sqrt(dPdx), sqrt(dPdx), dPdx]
# scale_factors = [1.0, 0.1, 0.01]
scale_factors = [1.0, 0.01, 0.01, dPdx]

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
basis_file = joinpath(FILEPATH, "data/streaming/basis_fieldwise.jld2")
# basis_data = load(basis_file)
field_results = load(basis_file)["field_results"]
# rmax = 300
# iVrmax = basis_data["bases"]["baker"].iVr[:,1:rmax]  # choose Baker's iSVD basis
# iVrmax = basis_data["merged_basis"].iVr[:,1:rmax]  # choose Baker's iSVD basis
# iVrmax = basis_data["bases"]["baker_fieldwise"].iVr[:,1:rmax]  # choose Baker's iSVD basis

rmax = 100
iVr_u = field_results[1].Q[:, 1:rmax]  # u-component basis
iVr_v = field_results[2].Q[:, 1:rmax]  # v-component basis
iVr_w = field_results[3].Q[:, 1:rmax]  # w-component basis
iVr_p = field_results[4].Q[:, 1:rmax]  # p-component basis
field_results = nothing  # Free memory
iVrmax = sparse(BlockDiagonal([iVr_u, iVr_v, iVr_w, iVr_p]))

#=========================#
## Load reduced data
#=========================#
Xhat = load(joinpath(FILEPATH, "data/streaming/reduced_data_discrete_r$(rmax).jld2"))["Xhat"]
# Xhatdot = load(joinpath(FILEPATH, "data/streaming/reduced_data.jld2"))["Xhatdot"]
Xhatdot = Xhat[:, 2:end]
Xhat = Xhat[:, 1:end-1]  
# U = load(joinpath(FILEPATH, "data/streaming/reduced_data.jld2"))["U"]
# U = dPdx * ones(size(Xhat,2))

size(Xhat,1) != rmax && @warn "Xhat has a different number of \
    rows than the basis. This might lead to unexpected results."

#=========================#
## Train Batch model
#=========================#
# OpInf
options.with_reg = false
op_inf = LnL.opinf(Xhat, options; Xhatdot=Xhatdot)

##
# ops = Dict("opinf" => op_inf)
# save(joinpath(FILEPATH, "data/models", "operators.jld2"), ops)
# ops = load(joinpath(FILEPATH, "data/models", "operators.jld2"))

## Tikhonov Regularized OpInf
options.with_reg = true
options.λ = LnL.TikhonovParameter(A=1e-15, A2=1e-8, K=1e-15)
op_trinf = LnL.opinf(Xhat, options; Xhatdot=Xhatdot)

# ops["tropinf"] = op_trinf
# save(joinpath(FILEPATH, "data/models", "operators.jld2"), ops)

## Keep the reference batch model to compare with the streaming models
Ostar = op_trinf.O'
# Ostar = ops["tropinf"].O'

#=========================#
## Train streaming model
#=========================#
# The reduced dimensions to evaluate on
rspan = [rmax ÷ 2, rmax]

# The time span to integrate the reduced model
tspan = 1:0.01:n

num_of_streams = n  
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
rls_stream  = LnL.TwoPassStreamingOpInf(options=options, n=rmax, m=1, algorithm=:RLS) 
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
    # x_i = ds[i]  
    # xhat_i = iVrmax' * x_i  # Project the snapshot onto the basis
    xhat_i = Xhat[:,i]
    
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

    xhatdot_i = Xhatdot[:,i]  # Use the precomputed time derivative

    # Get the i-th input 
    u_i = U[i]

    # Stream, update, and get data matrix for the state system
    LnL.stream!(rls_stream, xhat_i, xhatdot_i, U=[u_i])      # RLS
    # LnL.stream!(iqrrls_stream, xhat_i, xhatdot_i, U=[u_i])   # iQRRLS
    # LnL.stream!(qrrls_stream, xhat_i, xhatdot_i, U=[u_i])    # QRRLS

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

##
reduced_model = (x) -> op_inf.A * x + op_inf.A2u * (x ⊘ x) + op_inf.K 

##
tspan = ds["times"][:] .- ds["times"][1]
states = zeros(4*rmax, length(tspan))
states[:,1] = Xhat[:,1]
for i in 2:length(tspan)
    states[:,i] = reduced_model(states[:,i-1])
end

## Plot the states 
using CairoMakie
with_theme(theme_latexfonts()) do 
    fig = Figure(size=(1200, 900))
    # Get midpoint index for z-direction
    z_mid = Nz ÷ 2

    # Select 3 time steps (beginning, middle, end)
    time_indices = [50, 1000, length(tspan)÷2, length(tspan)]

    all_full_data = nothing
    all_rom_data = nothing
    all_error_data = nothing

    # Reconstruct full states from reduced states
    # Create 1x3 subplot layout
    for (i, t_idx) in enumerate(time_indices)
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
            ylabel = i == 1 ? L"$x$" : "", 
            xlabel = L"$x$",
            xlabelsize=30, ylabelsize=30, 
            xticklabelsize=25, yticklabelsize=25,
        )

        # Get data 
        u_full_field = reshape(ds[t_idx][1:dim_per_field], Nx, Ny, Nz)
        u_full_slice = u_full_field[:, :, z_mid]
        
        # Reconstruct rom full state for this time step only
        x_rom_t = iVrmax * states[:, t_idx]
        x_rom_t = unscale(x_rom_t, dim_per_field, scale_factors) + xbar
        
        # Extract velocity field at z_mid for time t_idx
        # Assuming first field is u-velocity
        u_rom_field = reshape(x_rom_t[1:dim_per_field], Nx, Ny, Nz)
        u_rom_slice = u_rom_field[:, :, z_mid]

        # Compute the error between full and ROM states
        u_error = u_full_slice - u_rom_slice
        
        # Collect data for colorbar scaling
        if i == 1
            all_full_data = [u_full_slice]
            all_rom_data = [u_rom_slice]
            all_error_data = [u_error]
        else
            push!(all_full_data, u_full_slice)
            push!(all_rom_data, u_rom_slice)
            push!(all_error_data, u_error)
        end
        
        # Create heatmaps with consistent color limits
        if i == length(time_indices)
            # Calculate global min/max for each row
            full_min, full_max = extrema(vcat(all_full_data...))
            rom_min, rom_max = extrema(vcat(all_rom_data...))
            error_min, error_max = extrema(vcat(all_error_data...))

            hm_full = nothing
            hm_rom = nothing
            hm_error = nothing
            
            # Create heatmaps with consistent color limits
            for (j, t_idx_j) in enumerate(time_indices)
                # Recalculate data for each time step
                u_full_j = reshape(ds[t_idx_j][1:dim_per_field], Nx, Ny, Nz)[:, :, z_mid]
                x_rom_j = iVrmax * states[:, t_idx_j]
                x_rom_j = unscale(x_rom_j, dim_per_field, scale_factors) + xbar
                u_rom_j = reshape(x_rom_j[1:dim_per_field], Nx, Ny, Nz)[:, :, z_mid]
                u_error_j = u_full_j - u_rom_j
                
                hm_full = heatmap!(fig[1, j], ds["x"][:], ds["y"][:], u_full_j, 
                    colormap = :viridis, colorrange = (full_min, full_max))
                hm_rom = heatmap!(fig[2, j], ds["x"][:], ds["y"][:], u_rom_j, 
                    colormap = :viridis, colorrange = (rom_min, rom_max))
                hm_error = heatmap!(fig[3, j], ds["x"][:], ds["y"][:], u_error_j, 
                    colormap = :thermal, colorrange = (error_min, error_max))
            end
            
            # Add colorbars at the end of each row
            Colorbar(fig[1, length(time_indices) + 1], hm_full)
            Colorbar(fig[2, length(time_indices) + 1], hm_rom)
            Colorbar(fig[3, length(time_indices) + 1], hm_error)
        end
    end
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
    field_colors = [:blue, :red, :green, :orange]
    
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
        for t in 1:length(tspan)
            # True state (direct indexing)
            true_field[t] = ds[t][idx]
            
            # ROM state (efficient dot product + scaling)
            rom_val = dot(basis_row, states[:, t])
            rom_field[t] = rom_val * scale_factor + mean_val
        end
        
        # Plot true vs reconstructed
        lines!(ax, tspan, true_field, color=:black, linewidth=2, label="True")
        lines!(ax, tspan, rom_field, color=color, linewidth=2, linestyle=:dash, label="ROM")
        
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