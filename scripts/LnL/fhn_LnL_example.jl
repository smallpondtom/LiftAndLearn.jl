"""
Fitzhugh-Nagumo Equation test case using Lift & Learn.
"""

##########
## Setups 
##########
using BlockDiagonals
using Kronecker
using LinearAlgebra
using NaNStatistics
using Plots
using ProgressMeter
using Random
using SparseArrays
using Statistics
using UniqueKronecker
using PolynomialModelReductionDataset
const Pomoreda = PolynomialModelReductionDataset

using LiftAndLearn
const LnL = LiftAndLearn
const SAVE_FIGURE = true

##################
## Generate model
##################
start = time()
Ω = (0.0, 1.0); dt = 1e-4; Nx = 2^9
fhn = Pomoreda.FitzHughNagumoModel(
    spatial_domain=Ω, time_domain=(0.0,4.0), Δx=(Ω[2] - 1/Nx)/Nx, Δt=dt,
    alpha_input_params=[500, 50000], beta_input_params=[10, 15]
)

# Some options for operator inference
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=[1, 2],
        control=1,
        output=1,
        coupled_input=1,
        constant=1,
        # lifted=true,
    ),
    vars=LnL.VariableStructure(
        N=2,
        N_lift=3,
    ),
    data=LnL.DataStructure(
        Δt=dt,
        DS=100,
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
        nonredundant_operators=true,
        reproject=true,
    ),
)

# grid points
gp = fhn.spatial_dim

# Downsampling
DS = options.data.DS

##########################################################
## Get the full-order model operators for intrusive model
##########################################################
tmp = fhn.lifted_finite_diff_model(gp, fhn.spatial_domain[2])
fomLinOps = LnL.Operators(
    A=tmp[1],
    B=tmp[2][:, :],  # Make sure its matrix
    C=tmp[3][:, :],
    A2=tmp[4],
    A2u=LnL.eliminate(tmp[4],2),  # takes too long
    N=tmp[5],
    K=tmp[6]
)

@info "(t=$(time()-start)) Complete generating intrusive model operators"

# Generic function for FOM
tmp = fhn.full_order_model(gp, fhn.spatial_domain[2])  # much efficient to calculate for FOM
fomOps = LnL.Operators(
    A=tmp[1],
    B=tmp[2][:, :],  # Make sure its a matrix
    C=tmp[3][:, :],
    K=tmp[4],
    f=tmp[5]
)
fom_state(x, u) = fomOps.A * x + fomOps.B * u + fomOps.f(x,u) + fomOps.K

@info "(t=$(time()-start)) Complete generating full-order model operators"

##########################
## Generate training data
##########################
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
    X = fhn.integrate_model(fhn.tspan, fhn.IC, genU; functional=fom_state)
    Xtrain[i] = X[:, 1:DS:end]  # make sure to only record every 0.01s
    U = genU.(fhn.tspan)
    Utrain_all[i] = U'
    Utrain[i] = U[1:DS:end]'
end
Xtr = reduce(hcat, Xtrain)
Utr = reshape(reduce(hcat, Utrain), :, 1)  # make sure that the U matrix is a tall matrix (x-row & t-col)
Ytr = fomOps.C * Xtr


## Visualize training data
p = plot(fhn.tspan[1:DS:end], Xtrain[1][1, :], legend=false, show=true)
plot!(p, fhn.tspan[1:DS:end], Xtrain[1][gp+1, :], legend=false)
for i in 2:8
    plot!(p, fhn.tspan[1:DS:end], Xtrain[i][1, :], legend=false)
    plot!(p, fhn.tspan[1:DS:end], Xtrain[i][gp+1, :], legend=false)
end
plot!(p, fhn.tspan[1:DS:end], Xtrain[9][1, :], legend=false)
plot!(p, fhn.tspan[1:DS:end], Xtrain[9][gp+1, :], legend=false)
display(p)

# Contour plot
p1 = contourf(fhn.tspan[1:DS:end],fhn.xspan,Xtrain[1][1:gp, :],lw=0, xlabel="t", ylabel="x", title="x1")
p2 = contourf(fhn.tspan[1:DS:end],fhn.xspan,Xtrain[1][gp+1:end, :],lw=0, xlabel="t", ylabel="x", title="x2")

l = @layout [a b]
p = plot(p1, p2, layout=l, size=(900, 400), show=true)
display(p)


## Create test 1 data
N_tests = 16

# Parameters for the testing data (1)
@info "Creating test data 1"
α_test1 = vec(rand(N_tests, 1) .* (fhn.alpha_input_params[2] - fhn.alpha_input_params[1]) .+ fhn.alpha_input_params[1])
β_test1 = vec(rand(N_tests, 1) .* (fhn.beta_input_params[2] - fhn.beta_input_params[1]) .+ fhn.beta_input_params[1])

Xtest1 = Vector{Matrix{Float64}}(undef, N_tests)
Utest1 = Vector{Matrix{Float64}}(undef, N_tests)
Utest1_all = Vector{Matrix{Float64}}(undef, N_tests)
@showprogress for i in axes(α_test1, 1)
    α, β = α_test1[i], β_test1[i]
    genU(t) = α * t^3 * exp(-β * t)  # generic function for input

    @inbounds X = fhn.integrate_model(fhn.tspan, fhn.IC, genU; functional=fom_state)
    Xtest1[i] = X[:, 1:DS:end]  # make sure to only record every 0.01s
    U = genU.(fhn.tspan)
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

    # @inbounds X = LnL.forwardEuler(fom_state, genU, fhn.tspan, fhn.IC)
    @inbounds X = fhn.integrate_model(fhn.tspan, fhn.IC, genU,  functional=fom_state)
    Xtest2[i] = X[:, 1:DS:end]  # make sure to only record every 0.01s
    U = genU.(fhn.tspan)
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

    infOps = LnL.opinf(Wtr, Vr, lifter, fomOps, options; U=Utr, Y=Ytr)
    infOps.A2 = duplicate(infOps.A2u, 2)
    intruOps = LnL.pod(fomLinOps, Vr, options.system)
    # At = Vr' * fomLinOps.A * Vr
    # Bt = Vr' * fomLinOps.B
    # Kt = Vr' * fomLinOps.K
    # Ht = Vr' * fomLinOps.H * (Vr ⊗ Vr)
    # Nt = Vr' * fomLinOps.N * Vr

    # fint(x, u) = fF(intruOps.A, intruOps.B, intruOps.F, intruOps.N, intruOps.K, x, u)
    finf = (x, u) -> infOps.A * x + infOps.B * u + infOps.A2 * (x ⊗ x) + (infOps.N * x) * u[1] + infOps.K

    # fint(x, u) = fF(intruOps.A, intruOps.B, intruOps.F, intruOps.N, intruOps.K, x, u)
    fint = (x, u) -> intruOps.A * x  + intruOps.B * u + intruOps.A2 * (x ⊗ x) + (intruOps.N*x)*u[1] + intruOps.K
    # fint = (x, u) -> At * x + Bt * u + Ht * (x ⊗ x) + (Nt * x) * u + Kt

    k, l = 0, 0
    for (X, U) in zip(Xtrain, Utrain_all)
        Xint = fhn.integrate_model(fhn.tspan, Vr' * fhn.IC_lift, U; functional=fint)
        Xinf = fhn.integrate_model(fhn.tspan, Vr' * fhn.IC_lift, U; functional=finf)

        # Down sample 
        Xint = Xint[:, 1:DS:end]
        Xinf = Xinf[:, 1:DS:end]

        train_err[:intrusive][k+=1, i] = LnL.compStateError(X, Xint, Vr)
        train_err[:inferred][l+=1, i] = LnL.compStateError(X, Xinf, Vr)
    end

    k, l = 0, 0
    for (X, U) in zip(Xtest1, Utest1_all)
        Xint = fhn.integrate_model(fhn.tspan, Vr' * fhn.IC_lift, U; functional=fint)
        Xinf = fhn.integrate_model(fhn.tspan, Vr' * fhn.IC_lift, U; functional=finf)

        # Down sample
        Xint = Xint[:, 1:DS:end]
        Xinf = Xinf[:, 1:DS:end]

        test1_err[:intrusive][k+=1, i] = LnL.compStateError(X, Xint, Vr)
        test1_err[:inferred][l+=1, i] = LnL.compStateError(X, Xinf, Vr)
    end

    k, l = 0, 0
    for (X, U) in zip(Xtest2, Utest2_all)
        Xint = fhn.integrate_model(fhn.tspan, Vr' * fhn.IC_lift, U; functional=fint)
        Xinf = fhn.integrate_model(fhn.tspan, Vr' * fhn.IC_lift, U; functional=finf)

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
p1 = plot(dims, err_intru, marker=(:cross, 10), label="intru")
plot!(dims, err_infer, marker=(:circle), ls=:dash, label="opinf")
plot!(yscale=:log10, majorgrid=true, minorgrid=true)
tmp = log10.(err_infer)
yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
xticks!(vec(dims))
xlabel!("dimension n")
ylabel!("Relative State Error")
title!("Median Error over Training Trajectories")
display(p1)

# Test 1
p2 = plot(dims, t1err_intru, marker=(:cross, 10), label="intru")
plot!(dims, t1err_infer, marker=(:circle), ls=:dash, label="opinf")
plot!(yscale=:log10, majorgrid=true, minorgrid=true)
tmp = log10.(t1err_infer)
yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
xticks!(vec(dims))
xlabel!("dimension n")
ylabel!("Relative State Error")
title!("Median Test1 Error over New Trajectories")
display(p2)

# Test 2
p3 = plot(dims, t2err_intru, marker=(:cross, 10), label="intru")
plot!(dims, t2err_infer, marker=(:circle), ls=:dash, label="opinf")
plot!(yscale=:log10, majorgrid=true, minorgrid=true)
tmp = log10.(t2err_infer)
yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
xticks!(vec(dims))
xlabel!("dimension n")
ylabel!("Relative State Error")
title!("Median Test2 Error over New Trajectories")
display(p3)

if SAVE_FIGURE
    savefig(p1, "scripts/LnL/plots/fhn_LnL_train_error.png")
    savefig(p2, "scripts/LnL/plots/fhn_LnL_test1_error.png")
    savefig(p3, "scripts/LnL/plots/fhn_LnL_test2_error.png")
end