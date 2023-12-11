"""
Fitzhugh-Nagumo Equation test case using Lift & Learn.
"""

## Setups 
using BlockDiagonals
using LinearAlgebra
using NaNStatistics
using Plots
using ProgressMeter
using Random
using SparseArrays
using Statistics

using LiftAndLearn
const LnL = LiftAndLearn

## Generate models
start = time()
# First order Burger's equation setup
fhn = LnL.fhn(
    [0.0, 1.0], [0.0, 4.0], [500, 50000], [10, 15], 2^(-9), 1e-4
)

# Some options for operator inference
options = LnL.LS_options(
    system=LnL.sys_struct(
        is_lin=true,
        is_quad=true,
        is_bilin=true,
        has_control=true,
        has_const=true,
        has_funcOp=true,
        is_lifted=true,
    ),
    vars=LnL.vars(
        N=2,
        N_lift=3,
    ),
    data=LnL.data(
        Δt=1e-4,
        DS=100,
    ),
    optim=LnL.opt_settings(
        verbose=true,
        which_quad_term="H",
        reproject=true,
    ),
)

# grid points
gp = Int(1 / fhn.Δx)

# Downsampling
DS = options.data.DS

# Get the full-order model operators for intrusive model
tmp = fhn.generateFHNmatrices(gp, fhn.Ω[2])
fomLinOps = LnL.operators(
    A=tmp[1],
    B=tmp[2][:, :],  # Make sure its matrix
    C=tmp[3][:, :],
    H=tmp[4],
    F=LnL.H2F(tmp[4]),
    N=tmp[5],
    K=tmp[6]
)

@info "(t=$(time()-start)) Complete generating intrusive model operators"

# Generic function for FOM
tmp = fhn.FOM(gp, fhn.Ω[2])  # much efficient to calculate for FOM
fomOps = LnL.operators(
    A=tmp[1],
    B=tmp[2][:, :],  # Make sure its a matrix
    C=tmp[3][:, :],
    K=tmp[4],
    f=tmp[5]
)
fom_state(x, u) = fomOps.A * x + fomOps.B * u + fomOps.f(x) + fomOps.K

@info "(t=$(time()-start)) Complete generating full-order model operators"

## Generate training data
# parameters for the training data
α_train = vec([500 5000 50000] .* ones(3))
β_train = vec([10 12.5 15]' .* ones(3)')

###################### (I) Create training data ###########################
Xtrain = Vector{Matrix{Float64}}(undef, length(α_train))
Utrain = Vector{Matrix{Float64}}(undef, length(α_train))
Utrain_all = Vector{Matrix{Float64}}(undef, length(α_train))

@showprogress for i in axes(α_train, 1)
    α, β = α_train[i], β_train[i]
    genU(t) = α * t^3 * exp(-β * t)  # generic function for input

    ## training data for inferred dynamical models
    X = LnL.forwardEuler(fom_state, genU, fhn.t, fhn.ICx)
    Xtrain[i] = X[:, 1:DS:end]  # make sure to only record every 0.01s
    U = genU.(fhn.t)
    Utrain_all[i] = U'
    Utrain[i] = U[1:DS:end]'
end
Xtr = reduce(hcat, Xtrain)
Utr = reshape(reduce(hcat, Utrain), :, 1)  # make sure that the U matrix is a tall matrix (x-row & t-col)
Ytr = fomOps.C * Xtr


## Visualize training data
p = plot(fhn.t[1:DS:end], Xtrain[1][1, :], legend=false, show=true)
plot!(p, fhn.t[1:DS:end], Xtrain[1][gp+1, :], legend=false)
for i in 2:8
    plot!(p, fhn.t[1:DS:end], Xtrain[i][1, :], legend=false)
    plot!(p, fhn.t[1:DS:end], Xtrain[i][gp+1, :], legend=false)
end
plot!(p, fhn.t[1:DS:end], Xtrain[9][1, :], legend=false)
plot!(p, fhn.t[1:DS:end], Xtrain[9][gp+1, :], legend=false)
display(p)

# Contour plot
p1 = contourf(fhn.t[1:DS:end],fhn.x,Xtrain[1][1:gp, :],lw=0, xlabel="t", ylabel="x", title="x1")
p2 = contourf(fhn.t[1:DS:end],fhn.x,Xtrain[1][gp+1:end, :],lw=0, xlabel="t", ylabel="x", title="x2")

l = @layout [a b]
p = plot(p1, p2, layout=l, size=(900, 400), show=true)
display(p)


## Create test 1 data
N_tests = 16

# Parameters for the testing data (1)
@info "Creating test data 1"
α_test1 = vec(rand(N_tests, 1) .* (fhn.αD[2] - fhn.αD[1]) .+ fhn.αD[1])
β_test1 = vec(rand(N_tests, 1) .* (fhn.βD[2] - fhn.βD[1]) .+ fhn.βD[1])

Xtest1 = Vector{Matrix{Float64}}(undef, N_tests)
Utest1 = Vector{Matrix{Float64}}(undef, N_tests)
Utest1_all = Vector{Matrix{Float64}}(undef, N_tests)
@showprogress for i in axes(α_test1, 1)
    α, β = α_test1[i], β_test1[i]
    genU(t) = α * t^3 * exp(-β * t)  # generic function for input

    @inbounds X = LnL.forwardEuler(fom_state, genU, fhn.t, fhn.ICx)
    Xtest1[i] = X[:, 1:DS:end]  # make sure to only record every 0.01s
    U = genU.(fhn.t)
    Utest1_all[i] = U'
    Utest1[i] = U[1:DS:end]'
end

## Create test 2 data
@info "Creating test data 2"
N_test_sqrt = Int(sqrt(N_tests))
# Parameters for testing data (2)
α_test2 = vec((5 * 10 .^ range(start=4, stop=6, length=N_test_sqrt))' .* ones(N_test_sqrt))
β_test2 = vec(range(start=15, stop=20, length=N_test_sqrt) .* ones(N_test_sqrt)')

Xtest2 = Vector{Matrix{Float64}}(undef, N_tests)
Utest2 = Vector{Matrix{Float64}}(undef, N_tests)
Utest2_all = Vector{Matrix{Float64}}(undef, N_tests)
@showprogress for i in axes(α_test2, 1)
    α, β = α_test2[i], β_test2[i]
    genU(t) = α * t^3 * exp(-β * t)  # generic function for input

    @inbounds X = LnL.forwardEuler(fom_state, genU, fhn.t, fhn.ICx)
    Xtest2[i] = X[:, 1:DS:end]  # make sure to only record every 0.01s
    U = genU.(fhn.t)
    Utest2_all[i] = U'
    Utest2[i] = U[1:DS:end]'
end

## Analyze the training and tests 
@info "Analyzing the training and test data"
mode_req = [1 1 1; 2 1 3; 3 3 4; 5 4 5]  # Required number of modes for each lifted variables

# Data lifting
Xsep = [Xtr[1:gp, :], Xtr[gp+1:end, :]]
lifter = LnL.lifting(options.vars.N, options.vars.N_lift, [x -> x[1] .^ 2])
Wtr = lifter.map(Xsep)


# Take the SVD for each variable
W1 = svd(Wtr[1:gp, :])
W2 = svd(Wtr[gp+1:2*gp, :])
W3 = svd(Wtr[2*gp+1:end, :])

# dictionary with intrusive and LnL errors as matrices (9-by-4)
train_err = Dict(
    :intrusive => zeros(length(α_train), size(mode_req, 1)),
    :inferred => zeros(length(α_train), size(mode_req, 1))
)
test1_err = Dict(
    :intrusive => zeros(N_tests, size(mode_req, 1)),
    :inferred => zeros(N_tests, size(mode_req, 1))
)
test2_err = Dict(
    :intrusive => zeros(N_tests, size(mode_req, 1)),
    :inferred => zeros(N_tests, size(mode_req, 1))
)

@showprogress for (i, row) in enumerate(eachrow(mode_req))
    r1, r2, r3 = row
    Vr1 = W1.U[:, 1:r1]
    Vr2 = W2.U[:, 1:r2]
    Vr3 = W3.U[:, 1:r3]
    Vr = BlockDiagonal([Vr1, Vr2, Vr3])

    # infOps = inferOp(Wtr, Utr, Ytr, Vr, fhn.Δt, 0, options)
    infOps = LnL.inferOp(Wtr, Utr, Ytr, Vr, lifter, fomOps, options)
    intruOps = LnL.intrusiveMR(fomLinOps, Vr, options)
    # At = Vr' * fomLinOps.A * Vr
    # Bt = Vr' * fomLinOps.B
    # Kt = Vr' * fomLinOps.K
    # Ht = Vr' * fomLinOps.H * kron(Vr, Vr)
    # Nt = Vr' * fomLinOps.N * Vr

    # fint(x, u) = fF(intruOps.A, intruOps.B, intruOps.F, intruOps.N, intruOps.K, x, u)
    finf = (x, u) -> infOps.A * x + infOps.B * u + infOps.H * kron(x, x) + (infOps.N * x) * u + infOps.K

    # fint(x, u) = fF(intruOps.A, intruOps.B, intruOps.F, intruOps.N, intruOps.K, x, u)
    fint = (x, u) -> intruOps.A * x  + intruOps.B * u + intruOps.H * kron(x, x) + (intruOps.N*x)*u + intruOps.K
    # fint = (x, u) -> At * x + Bt * u + Ht * kron(x, x) + (Nt * x) * u + Kt

    k, l = 0, 0
    for (X, U) in zip(Xtrain, Utrain_all)
        Xint = LnL.forwardEuler(fint, U, fhn.t, Vr' * fhn.ICw)
        Xinf = LnL.forwardEuler(finf, U, fhn.t, Vr' * fhn.ICw)

        # Down sample 
        Xint = Xint[:, 1:DS:end]
        Xinf = Xinf[:, 1:DS:end]

        train_err[:intrusive][k+=1, i] = LnL.compStateError(X, Xint, Vr)
        train_err[:inferred][l+=1, i] = LnL.compStateError(X, Xinf, Vr)
    end

    k, l = 0, 0
    for (X, U) in zip(Xtest1, Utest1_all)
        Xint = LnL.forwardEuler(fint, U, fhn.t, Vr' * fhn.ICw)
        Xinf = LnL.forwardEuler(finf, U, fhn.t, Vr' * fhn.ICw)

        # Down sample
        Xint = Xint[:, 1:DS:end]
        Xinf = Xinf[:, 1:DS:end]

        test1_err[:intrusive][k+=1, i] = LnL.compStateError(X, Xint, Vr)
        test1_err[:inferred][l+=1, i] = LnL.compStateError(X, Xinf, Vr)
    end

    k, l = 0, 0
    for (X, U) in zip(Xtest2, Utest2_all)
        Xint = LnL.forwardEuler(fint, U, fhn.t, Vr' * fhn.ICw)
        Xinf = LnL.forwardEuler(finf, U, fhn.t, Vr' * fhn.ICw)

        # Down sample
        Xint = Xint[:, 1:DS:end]
        Xinf = Xinf[:, 1:DS:end]

        test2_err[:intrusive][k+=1, i] = LnL.compStateError(X, Xint, Vr)
        test2_err[:inferred][l+=1, i] = LnL.compStateError(X, Xinf, Vr)
    end
end
dims = sum(mode_req, dims=2)

## Plot results
err_intru = vec(median(train_err[:intrusive], dims=1))
err_infer = vec(median(train_err[:inferred], dims=1))

t1err_intru = vec(median(test1_err[:intrusive], dims=1))
t1err_infer = vec(median(test1_err[:inferred], dims=1))

t2err_intru = vec(nanmedian(test2_err[:intrusive], dims=1))
t2err_infer = vec(nanmedian(test2_err[:inferred], dims=1));

# Training
p = plot(dims, err_intru, marker=(:cross, 10), label="intru")
plot!(dims, err_infer, marker=(:circle), ls=:dash, label="opinf")
plot!(yscale=:log10, majorgrid=true, minorgrid=true)
tmp = log10.(err_infer)
yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
xticks!(vec(dims))
xlabel!("dimension n")
ylabel!("Relative State Error")
title!("Median Error over Training Trajectories")
display(p)

# Test 1
p = plot(dims, t1err_intru, marker=(:cross, 10), label="intru")
plot!(dims, t1err_infer, marker=(:circle), ls=:dash, label="opinf")
plot!(yscale=:log10, majorgrid=true, minorgrid=true)
tmp = log10.(t1err_infer)
yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
xticks!(vec(dims))
xlabel!("dimension n")
ylabel!("Relative State Error")
title!("Median Test1 Error over New Trajectories")
display(p)

# Test 2
p = plot(dims, t2err_intru, marker=(:cross, 10), label="intru")
plot!(dims, t2err_infer, marker=(:circle), ls=:dash, label="opinf")
plot!(yscale=:log10, majorgrid=true, minorgrid=true)
tmp = log10.(t2err_infer)
yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
xticks!(vec(dims))
xlabel!("dimension n")
ylabel!("Relative State Error")
title!("Median Test2 Error over New Trajectories")
display(p)