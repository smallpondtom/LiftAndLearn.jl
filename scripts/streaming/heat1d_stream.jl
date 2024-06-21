"""
    Streaming-OpInf example of the 1D heat equation.
"""

#############
## Packages
#############
using CairoMakie
using LinearAlgebra
using ProgressMeter


###############
## My modules
###############
using LiftAndLearn
const LnL = LiftAndLearn


###################
## Global Settings
###################
CONST_STREAM = true
SAVEFIG = true


###############################
## Include functions and files
###############################
include("utilities/plot_theme.jl")
include("utilities/analysis.jl")
include("utilities/plotting.jl")


##########################
## 1D Heat equation setup
##########################
Nx = 2^7; dt = 1e-3
heat1d = LnL.Heat1DModel(  # define the model
    spatial_domain=(0.0, 1.0), time_domain=(0.0, 2.0), diffusion_coeffs=0.1,
    Δx=1/Nx, Δt=1e-3, BC=:dirichlet
)
foo = zeros(heat1d.spatial_dim)
foo[(Nx÷2+1):end] .= 1
heat1d.IC = foo .* (0.5 * sin.(2π * heat1d.xspan))  # change IC
U = ones(heat1d.time_dim)  # boundary condition → control input

# OpInf options
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        is_lin=true,
        has_control=true,
        has_output=true,
    ),
    vars=LnL.VariableStructure(
        N=1,  # number of state variables
    ),
    data=LnL.DataStructure(
        Δt=dt, # time step
        deriv_type="BE"  # backward Euler
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,  # show the optimization process
    ),
)


#################
## Generate Data
#################
# Construct full model
μ = heat1d.diffusion_coeffs
A, B = heat1d.finite_diff_model(heat1d, μ)
C = ones(1, heat1d.spatial_dim) / heat1d.spatial_dim
op_heat = LnL.Operators(A=A, B=B, C=C)

# Compute the state snapshot data with backward Euler
X = LnL.backwardEuler(A, B, U, heat1d.tspan, heat1d.IC)

# Compute the SVD for the POD basis
r = 15  # order of the reduced form
Vr = svd(X).U[:, 1:r]

# Compute the output of the system
Y = C * X

# Copy the data for later analysis
Xfull = copy(X)
Yfull = copy(Y)
Ufull = copy(U)


######################
## Plot Data to Check
######################
with_theme(theme_latexfonts()) do
    fig0 = Figure(fontsize=20, size=(1300,500), backgroundcolor="#FFFFFF")
    ax1 = Axis3(fig0[1, 1], xlabel="x", ylabel="t", zlabel="u(x,t)")
    surface!(ax1, heat1d.xspan, heat1d.tspan, X)
    ax2 = Axis(fig0[1, 2], xlabel="x", ylabel="t")
    hm = heatmap!(ax2, heat1d.xspan, heat1d.tspan, X)
    Colorbar(fig0[1, 3], hm)
    display(fig0)
end


#############
## Intrusive
#############
op_int = LnL.pod(op_heat, Vr, options)


######################
## Operator Inference
######################
# Obtain derivative data
Xdot = (X[:, 2:end] - X[:, 1:end-1]) / heat1d.Δt
idx = 2:heat1d.time_dim
X = X[:, idx]  # fix the index of states
U = U[idx, :]  # fix the index of inputs
Y = Y[:, idx]  # fix the index of outputs
op_inf = LnL.opinf(X, Vr, options; U=U, Y=Y, Xdot=Xdot)


##############################
## Tikhonov Regularized OpInf
##############################
options.with_reg = true
options.λ = LnL.TikhonovParameter(
    lin = 1e-13,
    ctrl = 1e-13,
    output = 1e-10
)
op_inf_reg = LnL.opinf(X, Vr, options; U=U, Y=Y, Xdot=Xdot)


###################
## Streaming-OpInf
###################
# Construct batches of the training data
if CONST_STREAM  # using a single constant batchsize
    global streamsize = 1
else  # initial batch updated with smaller batches
    init_streamsize = 1
    update_size = 1
    global streamsize = vcat([init_streamsize], [update_size for _ in 1:((size(X,2)-init_streamsize)÷update_size)])
end

# Streamify the data based on the selected streamsizes
# INFO: Remember to make data matrices a tall matrix except X matrix
Xhat_stream = LnL.streamify(Vr' * X, streamsize)
U_stream = LnL.streamify(U, streamsize)
Y_stream = LnL.streamify(Y', streamsize)
R_stream = LnL.streamify((Vr' * Xdot)', streamsize)
num_of_streams = length(Xhat_stream)

# Initialize the stream
# TR-Streaming-OpInf
γs = 1e-10
γo = 7.6e-9
# iQR/QR-Streaming-OpInf
# γs = 1e-13
# γo = 1e-10
algo = :RLS
stream = LnL.StreamingOpInf(options, r, size(U,2), size(Y,1); γs_k=γs, γo_k=γo, algorithm=algo)

# Stream all at once
stream.stream!(stream, Xhat_stream, R_stream; U_k=U_stream)
stream.stream_output!(stream, Xhat_stream, Y_stream)

# Unpack solution operators
op_stream = stream.unpack_operators(stream)


###############################
## (Analysis 1) Relative Error 
###############################
# Collect all operators into a dictionary
op_dict = Dict(
    "POD" => op_int,
    "OpInf" => op_inf,
    "TR-OpInf" => op_inf_reg,
    "TR-Streaming-OpInf" => op_stream
    # "Streaming-OpInf" => op_stream
)
rse, roe = analysis_1(op_dict, heat1d, Vr, Xfull, Ufull, Yfull, [:A, :B], LnL.backwardEuler)

## Plot
fig1 = plot_rse(rse, roe, r, ace_light; provided_keys=["POD", "OpInf", "TR-OpInf", "TR-Streaming-OpInf"])
display(fig1)


##################################################
## (Analysis 2) Per stream quantities of interest
##################################################
r_select = 1:r
analysis_results = analysis_2(
    Xhat_stream, U_stream, Y_stream, R_stream, num_of_streams, 
    op_inf_reg, Xfull, Vr, Ufull, Yfull, heat1d, r_select, options, 
    [:A, :B], LnL.backwardEuler; VR=false, α=γs, β=γo, algo=algo
)

## Plot
fig2 = plot_rse_per_stream(analysis_results["rse_stream"], analysis_results["roe_stream"], 
                           analysis_results["streaming_error"], analysis_results["streaming_error_output"], 
                           [5,10,15], num_of_streams; ylimits=([1e-7,1.3e1], [1e-9,1e1]))
display(fig2)
##
fig3 = plot_errorfactor_condition(analysis_results["cond_state_EF"], analysis_results["cond_output_EF"], 
                                  r_select, num_of_streams, ace_light)
display(fig3)
##
fig4 = plot_streaming_error(analysis_results["streaming_error"], analysis_results["streaming_error_output"], 
                            analysis_results["true_streaming_error"], analysis_results["true_streaming_error_output"],
                            r_select, num_of_streams, ace_light)
display(fig4)


##############################################
## (Analysis 3) Initial error over streamsize
##############################################
streamsizes = 1:num_of_streams
init_rse, init_roe = analysis_3(streamsizes, Vr, X, U, Y, Vr' * Xdot, op_inf_reg, 1:15, options; 
                                tol=nothing, α=γs, β=γo, algo=algo)

## Plot
fig5 = plot_initial_error(streamsizes, init_rse, init_roe, ace_light, 1:15)
display(fig5)


################
## Save figures
################
if SAVEFIG
    save("scripts/streaming/plots/heat/rse_over_dim.png", fig1)
    save("scripts/streaming/plots/heat/rse_and_streaming_error.png", fig2)
    save("scripts/streaming/plots/heat/cond_streaming_error_factor.png", fig3)
    save("scripts/streaming/plots/heat/verify_streaming_error.png", fig4)
    save("scripts/streaming/plots/heat/initial_error.png", fig5)
end


###################################################
## Trying to compute heuristic regularization term
###################################################
# using JuMP
# using Ipopt

# ##
# function hanke_raus(D, R, α_km1)
#     model = JuMP.Model(Ipopt.Optimizer; add_bridges = false)
#     JuMP.set_optimizer_attribute(model, "max_iter", 100)
#     JuMP.@variable(model, 1 >= α >= 0)
#     JuMP.set_start_value(α, α_km1)
#     # JuMP.set_silent(model)
#     d = size(D, 2)
#     r = size(R, 2)

#     U, S, V = svd(D)
#     m = length(S)
#     println(size(U))
#     println(size(S))
#     println(size(V))

#     x0 = Matrix{JuMP.NonlinearExpr}(undef, d, r)
#     for i in 1:r
#         x0[:,i] .= sum(S[j] / (S[j]^2 + α) * (U[:,j]' * R[:,i]) * V[:,j] for j in 1:m)
#     end
#     R0 = R - D * x0
#     x1 = Matrix{JuMP.NonlinearExpr}(undef, d, r)
#     for i in 1:r
#         x1[:,i] .= x0[:,i] + sum(S[j] / (S[j]^2 + α) * (U[:,j]' * R0[:,i]) * V[:,j] for j in 1:m)
#     end
#     R1 = R - D * x1

#     # JuMP.@objective(model, Min, sqrt(1 + 1/α) * sqrt(sum((R1[:,i]' * R0[:,i]) for i in 1:r)))
#     JuMP.@objective(model, Min, (1 + 1/α) * ((sum ∘ diag)(R1' * R0)))
#     JuMP.optimize!(model)
#     return JuMP.value(α)
# end

# ##
# idx = rand(1:10)
# D = hcat(Xhat_batch[idx]', U_batch[idx])
# R = R_batch[idx]

# ## 
# α_star = hanke_raus(D, R, 1e-3)