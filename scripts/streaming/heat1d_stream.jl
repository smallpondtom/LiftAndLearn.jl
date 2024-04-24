"""
    Streaming example of the 1D heat equation.
"""

#############
## Packages
#############
using CairoMakie
using LinearAlgebra
using Statistics: mean


###############
## My modules
###############
using LiftAndLearn
const LnL = LiftAndLearn


###################
## Global Settings
###################
CONST_BATCH = true


###############################
## Include functions and files
###############################
include("utilities/plot_theme.jl")
include("utilities/analysis.jl")
include("utilities/plotting.jl")


##########################
## 1D Heat equation setup
##########################
heat1d = LnL.heat1d(
    [0.0, 1.0], [0.0, 2.0], [0.1, 0.1],
    2^(-7), 1e-3, 1
)
foo = zeros(heat1d.Xdim)
foo[65:end] .= 1
heat1d.IC = Diagonal(foo) * 0.5 * sin.(2π * heat1d.x)  # change IC
U = heat1d.Ubc

# OpInf options
options = LnL.LS_options(
    system=LnL.sys_struct(
        is_lin=true,
        has_control=true,
        has_output=true,
    ),
    vars=LnL.vars(
        N=1,
    ),
    data=LnL.data(
        Δt=1e-3,
        deriv_type="BE"
    ),
    optim=LnL.opt_settings(
        verbose=true,
    ),
)


#################
## Generate Data
#################
# Construct full model
μ = heat1d.μs[1]
A, B = heat1d.generateABmatrix(heat1d.Xdim, μ, heat1d.Δx)
C = ones(1, heat1d.Xdim) / heat1d.Xdim
op_heat = LnL.operators(A=A, B=B, C=C)

## Compute the state snapshot data with backward Euler
X = LnL.backwardEuler(A, B, U, heat1d.t, heat1d.IC)

## Compute the SVD for the POD basis
r = 15  # order of the reduced form
Vr = svd(X).U[:, 1:r]

# Compute the output of the system
Y = C * X

# Copy the data for later analysis
Xf = copy(X)
Yf = copy(Y)
Uf = copy(U)


######################
## Plot Data to Check
######################
with_theme(theme_latexfonts()) do
    fig1 = Figure(fontsize=20, size=(1300,500), backgroundcolor="#F2F2F2")
    ax1 = Axis3(fig1[1, 1], xlabel="x", ylabel="t", zlabel="u(x,t)")
    surface!(ax1, heat1d.x, heat1d.t, X)
    ax2 = Axis(fig1[1, 2], xlabel="x", ylabel="t")
    hm = heatmap!(ax2, heat1d.x, heat1d.t, X)
    Colorbar(fig1[1, 3], hm)
    display(fig1)
end


#############
## Intrusive
#############
op_int = LnL.intrusiveMR(op_heat, Vr, options)


######################
## Operator Inference
######################
# Obtain derivative data
Xdot = (X[:, 2:end] - X[:, 1:end-1]) / heat1d.Δt
idx = 2:heat1d.Tdim
X = X[:, idx]  # fix the index of states
U = U[idx, :]  # fix the index of inputs
Y = Y[:, idx]  # fix the index of outputs
op_inf = LnL.inferOp(X, U, Y, Vr, Xdot, options)


###################
## Streaming-OpInf
###################
# Construct batches of the training data
if CONST_BATCH
    global batchsize = 100
else
    # Large initial batch updated with smaller batches
    init_batchsize = 100
    update_size = 1
    global batchsize = vcat([init_batchsize], [update_size for _ in 1:((size(X,2)-init_batchsize)÷update_size)])
end

## Shuffle the data
shuffle_idx = Random.shuffle(1:size(X, 2))
X_shuffle = X[:, shuffle_idx]
Xdot_shuffle = Xdot[:, shuffle_idx]
Y_shuffle = Matrix(Y[shuffle_idx]')

##
# Xhat_batch = LnL.batchify(Vr' * X, batchsize)
# U_batch = LnL.batchify(U, batchsize)
# Y_batch = LnL.batchify(Y', batchsize)
# R_batch = LnL.batchify((Vr' * Xdot)', batchsize)

Xhat_batch = LnL.batchify(Vr' * X_shuffle, batchsize)
U_batch = LnL.batchify(U, batchsize)
Y_batch = LnL.batchify(Y_shuffle', batchsize)
R_batch = LnL.batchify((Vr' * Xdot_shuffle)', batchsize)

## Initialize the stream
# INFO: Remember to make data matrices a tall matrix except X matrix
stream = LnL.Streaming_InferOp(options; tol=nothing)
D_k = stream.init!(stream, Xhat_batch[1], U_batch[1], Y_batch[1], R_batch[1])

## Stream all at once
stream.tol = nothing
stream.stream!(stream, Xhat_batch[2:end], U_batch[2:end], R_batch[2:end])
stream.stream_output!(stream, Xhat_batch[2:end], Y_batch[2:end])

## Unpack solution operators
op_stream = stream.unpack_operators(stream)


##################
## Error Analysis
##################
# RSE: Relative State Error
# ROE: Relative Output Error
rse, roe = compute_rse(op_int, op_inf, op_stream, heat1d, Vr, Xf, Uf, Yf)

## Plot
fig1 = plot_rse(rse, roe, r, ace_light)
display(fig1)

###########################
## Error per stream update
###########################
# SEF: State Error Factor
# OEF: Output Error Factor
r_select = [5, 10, 15]
rse_stream, roe_stream, sef_stream, oef_stream, sef_cond, oef_cond = compute_rse_per_stream(Xhat_batch, U_batch, Y_batch, R_batch, batchsize, 
                                                                            r_select, options; CONST_BATCH=CONST_BATCH)

## Plot
fig2 = plot_rse_per_stream(rse_stream, roe_stream, r_select, ace_light; CONST_BATCH=CONST_BATCH)
fig3 = plot_error_acc_per_stream(sef_stream, oef_stream, ace_light; CONST_BATCH=CONST_BATCH)
display(fig2)
display(fig3)

## Plot the condition number of the error Factor
fig4 = plot_error_condition(sef_cond, oef_cond, ace_light; CONST_BATCH=CONST_BATCH)
display(fig4)

################################
## Initial error over batchsize
################################
batchsizes = 100:50:1000
init_rse, init_roe = compute_inital_stream_error(batchsizes, Vr, X, U, Y, Vr' * Xdot, op_inf, options)

## Plot
fig5 = plot_initial_error(batchsizes, init_rse, init_roe, ace_light)
display(fig5)


