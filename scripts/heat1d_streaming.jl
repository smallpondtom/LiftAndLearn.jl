#############
## Packages
#############
using CairoMakie
using LinearAlgebra
using Random: randn, seed!
using Statistics: cov

###############
## My modules
###############
using LiftAndLearn
const LnL = LiftAndLearn

##################
## Plotting Theme
##################
ace_light = Merge(Theme(
    fontsize = 20,
    backgroundcolor = "#F2F2F2",
    Axis = (
        backgroundcolor = "#F2F2F2",
        xlabelsize = 20, xlabelpaddingg=-5,
        xgridstyle = :dash, ygridstyle = :dash,
        xtickalign = 1, ytickalign = 1,
        xticksize = 10, yticksize = 10,
        rightspinevisible = false, topspinevisible = false,
    ),
), theme_latexfonts())

#######################
## Additional function
#######################
function batchify(X, batchsize)
    m, n = size(X)
    if m > n
        return map(Iterators.partition(axes(X,1), batchsize)) do cols
            X[cols, :]
        end
    else
        return map(Iterators.partition(axes(X,2), batchsize)) do cols
            X[:, cols]
        end
    end
end

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
fig1 = Figure(fontsize=20, size=(1300,500))
ax1 = Axis3(fig1[1, 1], xlabel="x", ylabel="t", zlabel="u(x,t)")
surface!(ax1, heat1d.x, heat1d.t, X)
ax2 = Axis(fig1[1, 2], xlabel="x", ylabel="t")
hm = heatmap!(ax2, heat1d.x, heat1d.t, X)
Colorbar(fig1[1, 3], hm)
display(fig1)

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
R = Vr' * Xdot  # Reduced form of the derivative data
op_inf = LnL.inferOp(X, U, Y, Vr, R, options)

###################
## Streaming-OpInf
###################
# Construct batches of the training data
batchsize = 250
Xhat_batch = batchify(Vr' * X, batchsize)
U_batch = batchify(U, batchsize)
Y_batch = batchify(Y', batchsize)
R_batch = batchify(R', batchsize)

## Initialize the stream
# INFO: Remember to make data matrices a tall matrix except X matrix
stream = LnL.Streaming_InferOp(options)
D_k = stream.init!(stream, Xhat_batch[1], U_batch[1], Y_batch[1], R_batch[1])

## Stream all at once
stream.stream!(stream, Xhat_batch[2:end], U_batch[2:end], R_batch[2:end])
stream.stream_output!(stream, Xhat_batch[2:end], Y_batch[2:end])

##
op_stream = stream.unpack_operators(stream)

##################
## Error Analysis
##################
rel_state_err = zeros(r, 3)
rel_output_err = zeros(r, 3)

for i = 1:r
    Vrr = Vr[:, 1:i]

    # Integrate the intrusive model
    Xint = LnL.backwardEuler(op_int.A[1:i, 1:i], op_int.B[1:i, :], Uf, heat1d.t, Vrr' * heat1d.IC)
    Yint = op_int.C[1:1, 1:i] * Xint

    # Integrate the inferred model
    Xinf = LnL.backwardEuler(op_inf.A[1:i, 1:i], op_inf.B[1:i, :], Uf, heat1d.t, Vrr' * heat1d.IC)
    Yinf = op_inf.C[1:1, 1:i] * Xinf

    # Integrate the streaming inferred model
    Xstr = LnL.backwardEuler(op_stream.A[1:i, 1:i], op_stream.B[1:i, :], Uf, heat1d.t, Vrr' * heat1d.IC)
    Ystr = op_stream.C[1:1, 1:i] * Xstr

    # Compute relative state errors
    rel_state_err[i, 1] = LnL.compStateError(Xf, Xint, Vrr)
    rel_state_err[i, 2] = LnL.compStateError(Xf, Xinf, Vrr)
    rel_state_err[i, 3] = LnL.compStateError(Xf, Xstr, Vrr)

    # Compute relative output errors
    rel_output_err[i, 1] = LnL.compOutputError(Yf, Yint)
    rel_output_err[i, 2] = LnL.compOutputError(Yf, Yinf)
    rel_output_err[i, 3] = LnL.compOutputError(Yf, Ystr)
end

## Plot
fig2 = Figure(fontsize=20, size=(1200,600))
ax1 = Axis(fig2[1, 1], xlabel="r", ylabel="Relative Error", title="Relative State Error", yscale=log10)
scatterlines!(ax1, 1:r, rel_state_err[:, 1], label="Intrusive")
scatterlines!(ax1, 1:r, rel_state_err[:, 2], label="OpInf")
scatterlines!(ax1, 1:r, rel_state_err[:, 3], label="Streaming-OpInf")
ax2 = Axis(fig2[1, 2], xlabel="r", ylabel="Relative Error", title="Relative Output Error", yscale=log10)
l1 = scatterlines!(ax2, 1:r, rel_output_err[:, 1], label="Intrusive")
l2 = scatterlines!(ax2, 1:r, rel_output_err[:, 2], label="OpInf")
l3 = scatterlines!(ax2, 1:r, rel_output_err[:, 3], label="Streaming-OpInf")
Legend(fig2[2, 1:2], [l1, l2, l3], ["Intrusive", "OpInf", "Streaming-OpInf"],
        orientation=:horizontal, halign=:center, tellwidth=false, tellheight=true)
display(fig2)


###########################
## Error per stream update
###########################
# Initialize the stream
M = size(X,2)÷batchsize
r_select = [5, 10, 15]
rel_state_err_stream = zeros(M, length(r_select))
rel_output_err_stream = zeros(M, length(r_select))

stream = LnL.Streaming_InferOp(options)
for (j,ri) in enumerate(r_select)
    for i in 1:M
        if i == 1
            D_k = stream.init!(stream, Xhat_batch[i], U_batch[i], Y_batch[i], R_batch[i])
        else 
            stream.stream!(stream, Xhat_batch[i], U_batch[i], R_batch[i])
            stream.stream_output!(stream, Xhat_batch[i], Y_batch[i])
        end
        op_stream = stream.unpack_operators(stream)

        # Integrate the streaming inferred model
        Xstr = LnL.backwardEuler(op_stream.A[1:ri, 1:ri], op_stream.B[1:ri, :], Uf, heat1d.t, Vr[:,1:ri]' * heat1d.IC)
        Ystr = op_stream.C[1:1, 1:ri] * Xstr

        # Compute relative state errors
        rel_state_err_stream[i,j] = LnL.compStateError(Xf, Xstr, Vr[:,1:ri])

        # Compute relative output errors
        rel_output_err_stream[i,j] = LnL.compOutputError(Yf, Ystr)
    end
end
## Plot
fig3 = Figure(fontsize=20, size=(1200,600))
ax1 = Axis(fig3[1, 1], xlabel="Stream Update", ylabel="Relative Error", title="Relative State Error", yscale=log10, xticks=1:M)
ax2 = Axis(fig3[1, 2], xlabel="Stream Update", ylabel="Relative Error", title="Relative Output Error", yscale=log10, xticks=1:M)
lines_ = []
labels_ = []
for (j,ri) in enumerate(r_select)
    scatterlines!(ax1, 1:M, rel_state_err_stream[:,j])
    l = scatterlines!(ax2, 1:M, rel_output_err_stream[:,j])
    push!(lines_, l)
    push!(labels_, "r = $ri")
end
Legend(fig3[2, 1:2], lines_, labels_, orientation=:horizontal, halign=:center, tellwidth=false, tellheight=true)
display(fig3)


##############################
## Streaming-OpInf with noise
##############################
# Construct batches of the training data
batchsize = 500
Xhat_batch = batchify(Vr' * X, batchsize)
U_batch = batchify(U, batchsize)
Y_batch = batchify(Y', batchsize)
R_batch = batchify(R', batchsize)

# Define the noise with zero mean and variance
σ = 1e-2
Random.seed!(1234)
WN(v,sz) = sqrt(v) * Random.randn(sz...)
covmat(e) = cov(e * e')

## Initialize the stream
stream = LnL.Streaming_InferOp(options)
noise = WN(σ, size(R_batch[1]))
noise_output = WN(σ, size(Y_batch[1]))
# WARNING: Don't forget to add noise to your data matrix
D_k = stream.init!(stream, Xhat_batch[1], U_batch[1], Y_batch[1] + noise_output, R_batch[1] + noise, 
                    covmat(noise), covmat(noise_output))

## Stream all at once
# WARNING: Don't forget to add noise to your data matrix
noise = [WN(σ, size(R_batch[i])) for i in 2:length(Xhat_batch)]
noise_output = [WN(σ, size(Y_batch[i])) for i in 2:length(Y_batch)]
R_batch_noisy = [R_batch[i] + noise[i-1] for i in 2:length(Xhat_batch)]
Y_batch_noisy = [Y_batch[i] + noise_output[i-1] for i in 2:length(Y_batch)]

stream.stream!(stream, Xhat_batch[2:end], U_batch[2:end], R_batch_noisy, [covmat(n) for n in noise])
stream.stream_output!(stream, Xhat_batch[2:end], Y_batch_noisy, [covmat(n) for n in noise_output])

##
op_stream = stream.unpack_operators(stream)

##################
## Error Analysis
##################
rel_state_err = zeros(r, 3)
rel_output_err = zeros(r, 3)

for i = 1:r
    Vrr = Vr[:, 1:i]

    # Integrate the intrusive model
    Xint = LnL.backwardEuler(op_int.A[1:i, 1:i], op_int.B[1:i, :], Uf, heat1d.t, Vrr' * heat1d.IC)
    Yint = op_int.C[1:1, 1:i] * Xint

    # Integrate the inferred model
    Xinf = LnL.backwardEuler(op_inf.A[1:i, 1:i], op_inf.B[1:i, :], Uf, heat1d.t, Vrr' * heat1d.IC)
    Yinf = op_inf.C[1:1, 1:i] * Xinf

    # Integrate the streaming inferred model
    Xstr = LnL.backwardEuler(op_stream.A[1:i, 1:i], op_stream.B[1:i, :], Uf, heat1d.t, Vrr' * heat1d.IC)
    Ystr = op_stream.C[1:1, 1:i] * Xstr

    # Compute relative state errors
    rel_state_err[i, 1] = LnL.compStateError(Xf, Xint, Vrr)
    rel_state_err[i, 2] = LnL.compStateError(Xf, Xinf, Vrr)
    rel_state_err[i, 3] = LnL.compStateError(Xf, Xstr, Vrr)

    # Compute relative output errors
    rel_output_err[i, 1] = LnL.compOutputError(Yf, Yint)
    rel_output_err[i, 2] = LnL.compOutputError(Yf, Yinf)
    rel_output_err[i, 3] = LnL.compOutputError(Yf, Ystr)
end

## Plot
fig4 = Figure(fontsize=20, size=(1200,600))
ax1 = Axis(fig4[1, 1], xlabel="r", ylabel="Relative Error", title="Relative State Error with Noise", yscale=log10)
scatterlines!(ax1, 1:r, rel_state_err[:, 1], label="Intrusive")
scatterlines!(ax1, 1:r, rel_state_err[:, 2], label="OpInf")
scatterlines!(ax1, 1:r, rel_state_err[:, 3], label="Streaming-OpInf")
ax2 = Axis(fig4[1, 2], xlabel="r", ylabel="Relative Error", title="Relative Output Error with Noise", yscale=log10)
l1 = scatterlines!(ax2, 1:r, rel_output_err[:, 1], label="Intrusive")
l2 = scatterlines!(ax2, 1:r, rel_output_err[:, 2], label="OpInf")
l3 = scatterlines!(ax2, 1:r, rel_output_err[:, 3], label="Streaming-OpInf")
Legend(fig4[2, 1:2], [l1, l2, l3], ["Intrusive", "OpInf", "Streaming-OpInf"],
        orientation=:horizontal, halign=:center, tellwidth=false, tellheight=true)
display(fig4)


###########################
## Error per stream update
###########################
# Initialize the stream
M = size(X,2)÷batchsize
r_select = [5, 10, 15]
rel_state_err_stream = zeros(M, length(r_select))
rel_output_err_stream = zeros(M, length(r_select))

stream = LnL.Streaming_InferOp(options)
for (j,ri) in enumerate(r_select)
    Qs = covmat(WN(σ, batchsize))
    Qo = covmat(WN(σ, batchsize))
    for i in 1:M
        if i == 1
            D_k = stream.init!(stream, Xhat_batch[i], U_batch[i], Y_batch[i], R_batch[i], Qs, Qo)
        else 
            stream.stream!(stream, Xhat_batch[i], U_batch[i], R_batch[i], Qs)
            stream.stream_output!(stream, Xhat_batch[i], Y_batch[i], Qo)
        end
        op_stream = stream.unpack_operators(stream)

        # Integrate the streaming inferred model
        Xstr = LnL.backwardEuler(op_stream.A[1:ri, 1:ri], op_stream.B[1:ri, :], Uf, heat1d.t, Vr[:,1:ri]' * heat1d.IC)
        Ystr = op_stream.C[1:1, 1:ri] * Xstr

        # Compute relative state errors
        rel_state_err_stream[i,j] = LnL.compStateError(Xf, Xstr, Vr[:,1:ri])

        # Compute relative output errors
        rel_output_err_stream[i,j] = LnL.compOutputError(Yf, Ystr)
    end
end
## Plot
fig5 = Figure(fontsize=20, size=(1200,600))
ax1 = Axis(fig5[1, 1], xlabel="Stream Update", ylabel="Relative Error", title="Relative State Error", yscale=log10, xticks=1:M)
ax2 = Axis(fig5[1, 2], xlabel="Stream Update", ylabel="Relative Error", title="Relative Output Error", yscale=log10, xticks=1:M)
lines_ = []
labels_ = []
for (j,ri) in enumerate(r_select)
    scatterlines!(ax1, 1:M, rel_state_err_stream[:,j])
    l = scatterlines!(ax2, 1:M, rel_output_err_stream[:,j])
    push!(lines_, l)
    push!(labels_, "r = $ri")
end
Legend(fig5[2, 1:2], lines_, labels_, orientation=:horizontal, halign=:center, tellwidth=false, tellheight=true)
display(fig5)