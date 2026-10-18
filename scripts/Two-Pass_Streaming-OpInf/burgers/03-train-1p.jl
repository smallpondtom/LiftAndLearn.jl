"""
1D Viscous Burgers' equation: Training with One-Pass Streaming-OpInf
"""

#=================#
## Load Packages ##
#=================#
using FileIO
using JLD2
using LinearAlgebra
using ProgressMeter
using BlockDiagonals
using Printf
using Revise
using UniqueKronecker
import LiftAndLearn as LnL
using PolynomialModelReductionDataset: BurgersModel

#=================================#
## Configure filepath for saving ##
#=================================#
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"Two-Pass_Streaming-OpInf/burgers") : 
           joinpath(pwd(), "scripts/Two-Pass_Streaming-OpInf/burgers")

#=======================================#
## Obtain all the saved training files ##
#=======================================#
training_data_files = readdir(joinpath(FILEPATH, "data/training"), join=true)
n_files = length(training_data_files)

#==================#
## Load the setup ##
#==================#
setup_file = joinpath(FILEPATH, "data/setup.jld2")
setup = load(setup_file)
burgers = setup["burgers"]
options = setup["options"]
n_t = burgers.time_dim

#=========================================================#
## Generate the POD basis using iSVD using all algorithms
#=========================================================#
rmax = 14
num_of_streams = n_t * 10 * n_files
method = :sketchy  # :baker or :sketchy
## Obtaining the SVD incrementally by going through all the data
@info "Training Streaming-OpInf with $(n_files) files"
@info "Processing file 1 out of $(n_files)"
X = load(training_data_files[1])["X"]
stream = LnL.OnePassStreamingOpInf(
    method == :baker ? X[:,1,1] : [0.0];
    options=options, 
    n_state=size(X,1), n_input=1, n_snapshots=num_of_streams,
    rank=rmax, finite_diff=true, isvd_method=method
)
@showprogress for i in axes(X,3)
    if method == :baker
        for xi in eachcol(i == 1 ? X[:,2:end,i] : X[:,:,i])
            LnL.stream!(stream, xi)
        end
    else
        for xi in eachcol(X[:,:,i])
            LnL.stream!(stream, xi)
        end
    end
end
# The rest of the training data
for (i,data_file) in enumerate(training_data_files[2:end])
    @info "Processing file $(i+1) out of $(n_files)"
    jldopen(data_file, "r") do data
        # Load the data
        X = data["X"]
        # Train One-Pass Streaming-OpInf
        @showprogress for Xi in eachslice(X, dims=3)
            for xi in eachcol(Xi)
                LnL.stream!(stream, xi)
            end
        end
    end
end

## Compute the SVD
LnL.compute_svd_sketchy(stream)

## Save the streaming object
save(
    joinpath(FILEPATH, "data/streaming", "stream1p.jld2"), 
    "stream", stream
)

## Compute the operators
# Setup regularization
options.with_reg = true
options.λ = LnL.TikhonovParameter(A=1e-9, B=1e-9, A2=1e-9)
options.use_backslash = false
options.use_svd_truncation = true

# Operator storage file 
model_files = readdir(joinpath(FILEPATH, "data/models"), join=true)
shift = 0


n_inputs = 10
for i in 1:n_files
    @info "Processing file $(i) out of $(n_files) for Xhat and Xhatdot"
    data = load(training_data_files[i])
    X = data["X"]
    U = data["U"]
    μ = data["mu"]
    Xhat_save = Array{Float64,3}(undef, rmax, size(X,2)-1, n_inputs)
    Xhatdot_save = Array{Float64,3}(undef, rmax, size(X,2)-1, n_inputs)
    U_save = Array{Float64,2}(undef, size(U,1)-1, n_inputs)
    for j in 1:n_inputs 
        @info "  Processing input $(j) out of $(n_inputs)"
        idx_start = (i-1) * n_t * n_inputs + (j-1) * n_t + 1
        idx_end = idx_start + n_t - 1
        Xhat = Diagonal(stream.Σ) * stream.W[idx_start:idx_end, 1:rmax]'
        E, Δidx = LnL.finite_diff_matrix(
            options.data.deriv_type, n_t, options.data.Δt
        )
        Xhatdot = Xhat * E
        Xhat = Xhat[:, Δidx]

        # Store values
        Xhat_save[:, :, j] = Xhat
        Xhatdot_save[:, :, j] = Xhatdot
        U_save[:, j] = U[Δidx, j]
    end
    Xhat = reshape(Xhat_save, rmax, :)
    Xhatdot = reshape(Xhatdot_save, rmax, :)
    U = reshape(U_save, :, 1)

    op_stream = LnL.opinf(Xhat, options; U=U, Xhatdot=Xhatdot)

    # Save the model
    mu_str = @sprintf("%1.4f", μ)
    model_idx = findfirst(x -> occursin("mu$(mu_str)", x), model_files)
    model_file = model_files[model_idx]
    ops = load(model_file) 
    ops["stream"] = op_stream
    save(model_file, ops)
end

##
# # Assemble all input data into one 
# Uall = Array[]
# for data_file in training_data_files
#     data = load(data_file)
#     for j in 1:data["num_inputs"]
#         push!(Uall, data["U"][:,j])
#     end
# end
# Uall = reduce(vcat, Uall)
for (i,data_file) in enumerate(training_data_files)
    @info "Learning model for parameter $(i) out of $(n_files)"
    jldopen(data_file, "r") do data
        # Load the data
        U = data["U"]
        μ = data["mu"]
        n_inputs = data["num_inputs"]

        # Store the finite difference matrices and indices
        Es = AbstractMatrix{Float64}[]
        Δidxs = Array[]
        global shift

        # Train One-Pass Streaming-OpInf
        for j in 1:n_inputs
            E, Δidx = LnL.finite_diff_matrix(
                options.data.deriv_type, n_t, options.data.Δt
            )
            push!(Es, E)
            push!(Δidxs, Δidx)
            shift += n_t
        end

        # Compute the operators
        Δidxs = reduce(vcat, Δidxs)
        @info "  Computing operators"
        @info "    Indices: ($(Δidxs[1]), $(Δidxs[end]))"
        op_stream = LnL.compute_stream_operators(
            stream, BlockDiagonal(Es), Δidxs_cat, U=Uall
        )

        # Save the model
        mu_str = @sprintf("%1.4f", μ)
        model_idx = findfirst(x -> occursin("mu$(mu_str)", x), model_files)
        model_file = model_files[model_idx]
        ops = load(model_file) 
        ops["stream"] = op_stream
        save(model_file, ops)
    end
end

#====================================#
## Compute the relative state errors 
#====================================#
model_files = readdir(joinpath(FILEPATH, "data/models"), join=true)
stream_files = readdir(joinpath(FILEPATH, "data/streaming"), join=true)
num_train = length(training_data_files)

# Error analysis 
train_errors = load(joinpath(FILEPATH, "data/training_errors.jld2"))
train_errors["stream"] = zeros(rmax,1)
@showprogress for (file_idx, train_file) in enumerate(training_data_files)
    jldopen(train_file, "r") do data
        # Load the data
        Xref = data["Xref"]
        Uref = data["Uref"]

        # Load the trained models
        mu_str = @sprintf("%1.4f", data["mu"])
        model_idx = findfirst(x -> occursin("mu$(mu_str)", x), model_files)
        ops = load(model_files[model_idx])["stream"]

        for (i,r) = enumerate(1:rmax)
            Vr = stream.V[:, 1:r]

            # Integrate the model
            F_extract = UniqueKronecker.extractF(ops.A2u, r)
            Xrecon = burgers.integrate_model(
                burgers.tspan, Vr' * burgers.IC, Uref,
                linear_matrix=ops.A[1:r, 1:r], 
                control_matrix=ops.B[1:r,:],
                quadratic_matrix=F_extract, system_input=true
            )

            # Compute relative state error (averaged over parameters)
            train_errors["stream"][i] += norm(Xref - Vr * Xrecon) / norm(Xref) / num_train
        end
        @info "Training error r = $(rmax): $(train_errors["stream"][end])"
    end
end

# Save the errors
save(joinpath(FILEPATH, "data/training_errors.jld2"), "train_errors", train_errors)
