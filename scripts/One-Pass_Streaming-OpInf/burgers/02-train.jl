"""
1D Viscous Burgers' equation: Training with One-Pass Streaming-OpInf
"""

#================#
## Load Packages
#================#
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

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"One-Pass_Streaming-OpInf/burgers") : 
           joinpath(pwd(), "scripts/One-Pass_Streaming-OpInf/burgers")

#======================================#
## Obtain all the saved training files
#======================================#
training_data_files = readdir(joinpath(FILEPATH, "data/training"), join=true)

#=================#
## Load the setup
#=================#
setup_file = joinpath(FILEPATH, "data/setup.jld2")
setup = load(setup_file)
burgers = setup["burgers"]
options = setup["options"]

#=========================================================#
## Generate the POD basis using iSVD using all algorithms
#=========================================================#
rmax = 14

# Setup regularization
options.with_reg = true
options.λ = LnL.TikhonovParameter(A=1e-9, B=1e-9, A2=1e-9)

for (i,data_file) in enumerate(training_data_files)
    @info "Processing file $(i) out of $(length(training_data_files))"
    jldopen(data_file, "r") do data
        # Load the data
        X = data["X"]
        U = data["U"]
        μ = data["mu"]
        n_inputs = data["num_inputs"]

        # Initialize the One-Pass Streaming-OpInf object
        stream = LnL.OnePassStreamingOpInf(
            X[:,1,1]; options=options, n=size(X,1), m=1, 
            rank=rmax, finite_diff=true
        )

        # Store the finite difference matrices and indices
        Es = AbstractMatrix{Float64}[]
        Δidxs = Array[]
        shift = 0

        # Train One-Pass Streaming-OpInf
        @showprogress for i in axes(X,3)
            Xi = view(X, :, :, i)
            E, Δidx = LnL.finite_diff_matrix(
                options.data.deriv_type, size(Xi,2), options.data.Δt
            )
            push!(Es, E)
            push!(Δidxs, Δidx .+ shift)
            shift += size(Xi,2)
            for xi in eachcol(i == 1 ? Xi[:,2:end] : Xi)
                LnL.stream!(stream, xi, tol=1e-7)
            end
        end

        # Compute the operators
        op_stream = LnL.compute_stream_operators(
            stream, BlockDiagonal(Es), reduce(vcat, Δidxs), 
            U=reshape(U, :, 1)
        )

        # Save the model
        mu_str = @sprintf("%1.4f", μ)
        ops = Dict("stream" => op_stream, "mu" => μ)
        filename = joinpath(FILEPATH, "data/models", "op_mu$(mu_str).jld2")
        save(filename, ops)
        filename = joinpath(FILEPATH, "data/streaming", "stream_mu$(mu_str).jld2")
        save(filename, "stream", stream)
    end
end

#====================================#
## Compute the relative state errors 
#====================================#
model_files = readdir(joinpath(FILEPATH, "data/models"), join=true)
stream_files = readdir(joinpath(FILEPATH, "data/streaming"), join=true)
num_train = length(training_data_files)

# Error analysis 
train_errors = zeros(rmax)

##
@showprogress for (file_idx, train_file) in enumerate(training_data_files)
    jldopen(train_file, "r") do data
        # Load the data
        Xref = data["Xref"]
        Uref = data["Uref"]

        # Load the trained models
        mu_str = @sprintf("%1.4f", data["mu"])
        model_idx = findfirst(x -> occursin("mu$(mu_str)", x), model_files)
        ops = load(model_files[model_idx])["stream"]
        stream_idx = findfirst(x -> occursin("mu$(mu_str)", x), stream_files)
        stream = load(stream_files[stream_idx])["stream"]

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
            train_errors[i] += norm(Xref - Vr * Xrecon) / norm(Xref) / num_train
        end
        @info "Training error r = $(rmax): $(train_errors[end])"
    end
end

# Save the errors
save(joinpath(FILEPATH, "data/training_errors.jld2"), "train_errors", train_errors)
