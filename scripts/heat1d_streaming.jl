#############
## Packages
#############
using CairoMakie
using LinearAlgebra

###############
## My modules
###############
using LiftAndLearn
const LnL = LiftAndLearn


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
    [0.0, 1.0], [0.0, 1.0], [0.1, 0.1],
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
X = LnL.backwardEuler(A, B, heat1d.Ubc, heat1d.t, heat1d.IC)

# Compute the SVD for the POD basis
r = 15  # order of the reduced form
Vr = svd(X).U[:, 1:r]

# Compute the output of the system
Y = C * X

######################
## Plot Data to Check
######################
# fig = Figure(fontsize=20, size=(1300,500))
# ax1 = Axis3(fig[1, 1], xlabel="x", ylabel="t", zlabel="u(x,t)")
# surface!(ax1, heat1d.x, heat1d.t, X)
# ax2 = Axis(fig[1, 2], xlabel="x", ylabel="t")
# hm = heatmap!(ax2, heat1d.x, heat1d.t, X)
# Colorbar(fig[1, 3], hm)
# display(fig)

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
batchsize = 200
Xhat_batch = batchify(Vr' * X, batchsize)
U_batch = batchify(U, batchsize)
Y_batch = batchify(Y', batchsize)
R_batch = batchify(R', batchsize)

## Initialize the stream
# INFO: Remember to make data matrices a tall matrix except X matrix
stream = LnL.Streaming_InferOp(options)
D_k = stream.init!(stream, Xhat_batch[1], U_batch[1], Y_batch[1], R_batch[1])

## Stream the data (second batch)
stream.stream!(stream, Xhat_batch[2], U_batch[2], R_batch[2])
stream.stream_output!(stream, Xhat_batch[2], Y_batch[2])

## Stream all at once
stream.stream!(stream, Xhat_batch[2:end], U_batch[2:end], R_batch[2:end])