"""
    Streaming-OpInf example of the 1D heat equation.
"""

#############
## Packages
#############
using CairoMakie
using LinearAlgebra
using Random
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
heat1d = LnL.heat1d(  # define the model
    [0.0, 1.0], [0.0, 2.0], [0.1, 0.1],
    2^(-7), 1e-3, 1
)
foo = zeros(heat1d.Xdim)
foo[65:end] .= 1
heat1d.IC = Diagonal(foo) * 0.5 * sin.(2π * heat1d.x)  # change IC
U = heat1d.Ubc  # boundary condition → control input

# OpInf options
options = LnL.LS_options(
    system=LnL.sys_struct(
        is_lin=true,
        has_control=true,
        has_output=true,
    ),
    vars=LnL.vars(
        N=1,  # number of state variables
    ),
    data=LnL.data(
        Δt=1e-3, # time step
        deriv_type="BE"  # backward Euler
    ),
    optim=LnL.opt_settings(
        verbose=true,  # show the optimization process
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

# Compute the state snapshot data with backward Euler
X = LnL.backwardEuler(A, B, U, heat1d.t, heat1d.IC)

# Compute the SVD for the POD basis
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
    fig1 = Figure(fontsize=20, size=(1300,500), backgroundcolor="#FFFFFF")
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
op_int = LnL.pod(op_heat, Vr, options)


######################
## Operator Inference
######################
# Obtain derivative data
Xdot = (X[:, 2:end] - X[:, 1:end-1]) / heat1d.Δt
idx = 2:heat1d.Tdim
X = X[:, idx]  # fix the index of states
U = U[idx, :]  # fix the index of inputs
Y = Y[:, idx]  # fix the index of outputs
op_inf = LnL.opinf(X, Vr, options; U=U, Y=Y, Xdot=Xdot)


##############################
## Tikhonov Regularized OpInf
##############################
options.with_reg = true
options.λ = LnL.λtik(
    lin = 1e-8,
    ctrl = 1e-8,
    output = 1e-5
)
op_inf_reg = LnL.opinf(X, Vr, options; U=U, Y=Y, Xdot=Xdot)


###################
## Streaming-OpInf
###################
# Construct batches of the training data
if CONST_BATCH  # using a single constant batchsize
    global batchsize = 1
else  # initial batch updated with smaller batches
    init_batchsize = 1
    update_size = 1
    global batchsize = vcat([init_batchsize], [update_size for _ in 1:((size(X,2)-init_batchsize)÷update_size)])
end

# Batchify the data based on the selected batchsizes
# INFO: Remember to make data matrices a tall matrix except X matrix
Xhat_batch = LnL.batchify(Vr' * X, batchsize)
U_batch = LnL.batchify(U, batchsize)
Y_batch = LnL.batchify(Y', batchsize)
R_batch = LnL.batchify((Vr' * Xdot)', batchsize)
num_of_batches = length(Xhat_batch)

# Initialize the stream
# tol = [1e-12, 1e-15]  # tolerance for pinv 
tol = nothing
α = 1e-8
β = 1e-5

# Initialize the stream
stream = LnL.StreamingOpInf(options; variable_regularize=false, tol=tol)
D_k = stream.init!(stream, Xhat_batch[1], R_batch[1]; U_k=U_batch[1], Y_k=Y_batch[1], α_k=α, β_k=β)

# Stream all at once
stream.stream!(stream, Xhat_batch[2:end], R_batch[2:end]; U_kp1=U_batch[2:end])
stream.stream_output!(stream, Xhat_batch[2:end], Y_batch[2:end])

# Unpack solution operators
op_stream = stream.unpack_operators(stream)


##################
## Error Analysis
##################
# RSE: Relative State Error
# ROE: Relative Output Error
rse, roe = compute_rse(op_int, op_inf, op_stream, heat1d, Vr, Xf, Uf, Yf; op_inf_reg=op_inf_reg)

## Plot
fig1 = plot_rse(rse, roe, r, ace_light)
display(fig1)

###########################
## Error per stream update
###########################
# SEF: State Error Factor
# OEF: Output Error Factor
r_select = [5, 10, 15]
rse_stream, roe_stream, sef_stream, oef_stream, sef_cond, oef_cond, trace_cov = compute_rse_per_stream(Xhat_batch, U_batch, Y_batch, R_batch, batchsize, 
                                                                            r_select, options; tol=tol, CONST_BATCH=CONST_BATCH,
                                                                            VR=false, α=α, β=β)

## Plot
fig2 = plot_rse_per_stream(rse_stream, roe_stream, r_select, ace_light, num_of_batches)
fig3 = plot_error_acc_per_stream(sef_stream, oef_stream, ace_light, num_of_batches)
display(fig2)
display(fig3)

##
fig6 = Figure()
ax = Axis(fig6[1,1], yscale=log10)
scatter!(ax, (16 ./ trace_cov)[2:end])
display(fig6)


## Plot the condition number of the error Factor
fig4 = plot_error_condition(sef_cond, oef_cond, ace_light; CONST_BATCH=CONST_BATCH)
display(fig4)

################################
## Initial error over batchsize
################################
batchsizes = 1:50:2000
init_rse, init_roe = compute_inital_stream_error(batchsizes, Vr, X, U, Y, Vr' * Xdot, op_inf, options, tol, α=1e-8, β=1e-5, orders=1:15)

## Plot
fig5 = plot_initial_error(batchsizes, init_rse, init_roe, ace_light, 1:15)
display(fig5)



###################################################
## Trying to compute heuristic regularization term
###################################################
using JuMP
using Ipopt

##
function hanke_raus(D, R, α_km1)
    model = JuMP.Model(Ipopt.Optimizer; add_bridges = false)
    JuMP.set_optimizer_attribute(model, "max_iter", 100)
    JuMP.@variable(model, 1 >= α >= 0)
    JuMP.set_start_value(α, α_km1)
    # JuMP.set_silent(model)
    d = size(D, 2)
    r = size(R, 2)

    U, S, V = svd(D)
    m = length(S)
    println(size(U))
    println(size(S))
    println(size(V))

    x0 = Matrix{JuMP.NonlinearExpr}(undef, d, r)
    for i in 1:r
        x0[:,i] .= sum(S[j] / (S[j]^2 + α) * (U[:,j]' * R[:,i]) * V[:,j] for j in 1:m)
    end
    R0 = R - D * x0
    x1 = Matrix{JuMP.NonlinearExpr}(undef, d, r)
    for i in 1:r
        x1[:,i] .= x0[:,i] + sum(S[j] / (S[j]^2 + α) * (U[:,j]' * R0[:,i]) * V[:,j] for j in 1:m)
    end
    R1 = R - D * x1

    # JuMP.@objective(model, Min, sqrt(1 + 1/α) * sqrt(sum((R1[:,i]' * R0[:,i]) for i in 1:r)))
    JuMP.@objective(model, Min, (1 + 1/α) * ((sum ∘ diag)(R1' * R0)))
    JuMP.optimize!(model)
    return JuMP.value(α)
end

##
idx = rand(1:10)
D = hcat(Xhat_batch[idx]', U_batch[idx])
R = R_batch[idx]

## 
α_star = hanke_raus(D, R, 1e-3)