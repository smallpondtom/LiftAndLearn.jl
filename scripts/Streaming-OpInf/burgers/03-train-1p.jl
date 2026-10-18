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
FILEPATH = @__DIR__

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
rmax = 14

#=========================================================#
## Generate the POD basis using iSVD using all algorithms
#=========================================================#
# Setup regularization
options.with_reg = true
options.λ = LnL.TikhonovParameter(A=1e-9, B=1e-9, A2=1e-9)
options.use_backslash = false
options.use_svd_truncation = true
num_of_streams = (n_t - 1) * 10 

@info "Training iSVD-LS models."
for (i,data_file) in enumerate(training_data_files)
    @info "Processing file $(i) out of $(length(training_data_files))"
    jldopen(data_file, "r") do data
        # Load the data
        μ = data["mu"]
        X = data["X"]
        U = data["U"]
        Xhat = data["Xhat"]
        Xhatdot = data["Xhatdot"]

        # Flatten the reduced data
        Xhat = reshape(Xhat, rmax, :)
        Xhatdot = reshape(Xhatdot, rmax, :)

        # Organize the input data 
        U = U[2:end, :]
        U = reshape(U, num_of_streams, :)

        println("Sizes: Xhat = ", size(Xhat), ", Xhatdot = ", size(Xhatdot), ", U = ", size(U))

        # Compute the operators
        op_stream = LnL.opinf(Xhat, options; U=U, Xhatdot=Xhatdot)

        # Save the model
        mu_str = @sprintf("%1.4f", μ)
        filename = joinpath(FILEPATH, "data/models", "op_mu$(mu_str).jld2")
        ops = load(filename)
        ops["stream"] = op_stream
        save(filename, ops)
    end
end
@info "Done."

#====================================#
## Compute the relative state errors 
#====================================#
@info "Computing training errors for iSVD-LS models."
model_files = readdir(joinpath(FILEPATH, "data/models"), join=true)
stream_files = readdir(joinpath(FILEPATH, "data/streaming"), join=true)
num_train = length(training_data_files)

# Error analysis 
train_errors = load(joinpath(FILEPATH, "data/training_errors.jld2"))
train_errors["stream"] = zeros(rmax,1)

# Load the basis
basis_file = joinpath(FILEPATH, "data/streaming/basis.jld2")
basis_data = load(basis_file)
iVrmax = basis_data["sketchy"].iVr 

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
            Vr = iVrmax[:, 1:r]

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
@info "Done."