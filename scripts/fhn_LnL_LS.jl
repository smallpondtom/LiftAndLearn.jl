"""
Fitzhugh-Nagumo Equation test case using Lift & Learn.
"""

using BlockDiagonals
using CSV
using DelimitedFiles
using FileIO
using JLD2
using LinearAlgebra
using Plots
using ProgressMeter
using Random
using Statistics
using Tables


include("../src/model/FHN.jl")
include("../src/LiftAndLearn.jl")
const LnL = LiftAndLearn


start = time()

## First order Burger's equation setup
fhn = FHN(
    [0.0, 1.0], [0.0, 4.0], [500, 50000], [10, 15], 2^(-9), 1e-4
)

# Some options for operator inference
options = LnL.OpInf_options(
    is_quad=true,
    is_bilin=true,
    is_lifted=true,
    has_const=true,
    has_funcOp=true,
    reproject=true,
    DS=100,
    N=2,
    N_lift=3,
    Δt=1e-4,
    which_quad_term="H"
)

# grid points
gp = Int(1 / fhn.Δx)

# Downsampling
DS = options.DS

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

println("[INFO] ($(time()-start)) Complete generating intrusive model operators")

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

println("[INFO] ($(time()-start)) Complete generating full-order model operators")

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
    @inbounds X = LnL.forwardEuler(fom_state, genU, fhn.t, fhn.ICx)
    Xtrain[i] = X[:, 1:DS:end]  # make sure to only record every 0.01s
    U = genU.(fhn.t)
    Utrain_all[i] = U'
    Utrain[i] = U[1:DS:end]'
end
Xtr = reduce(hcat, Xtrain)
Utr = reshape(reduce(hcat, Utrain), :, 1)  # make sure that the U matrix is a tall matrix (x-row & t-col)
Ytr = fomOps.C * Xtr

println("[INFO] ($(time()-start)) Created training data")


################### (II) Create testing data 1 ############################
N_tests = 100

# Parameters for the testing data (1)
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

println("[INFO] ($(time()-start)) Created test 1 data")

################### (III) Create testing data 2 ###########################
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

println("[INFO] ($(time()-start)) Created test 2 data")

##################### (IV~VII) Learn & Analysis ###########################
mode_req = [1 1 1; 2 1 3; 3 3 4; 5 4 5]  # Required number of modes for each lifted variables
# mode_req = [1 1 1; 2 1 3; 3 3 4]  # Required number of modes for each lifted variables


# Data lifting
# Wtr = vcat(Xtr, Xtr[1:gp, :] .^ 2)  # lifted training states
Xsep = [Xtr[1:gp, :], Xtr[gp+1:end, :]]
lifter = LnL.lifting(options.N, options.N_lift, [x -> x[1] .^ 2])
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

for (i, row) in enumerate(eachrow(mode_req))
    r1, r2, r3 = row
    Vr1 = W1.U[:, 1:r1]
    Vr2 = W2.U[:, 1:r2]
    Vr3 = W3.U[:, 1:r3]
    Vr = BlockDiagonal([Vr1, Vr2, Vr3])

    ####################### (IV) Infer and Intrusive ######################
    # infOps = inferOp(Wtr, Utr, Ytr, Vr, fhn.Δt, 0, options)
    infOps = LnL.inferOp(Wtr, Utr, Ytr, Vr, lifter, fomOps, options)
    # intruOps = LnL.intrusiveMR(fomLinOps, Vr, options)
    At = Vr' * fomLinOps.A * Vr
    Bt = Vr' * fomLinOps.B
    Kt = Vr' * fomLinOps.K
    Ht = Vr' * fomLinOps.H * kron(Vr, Vr)
    Nt = Vr' * fomLinOps.N * Vr

    # fint(x, u) = fF(intruOps.A, intruOps.B, intruOps.F, intruOps.N, intruOps.K, x, u)
    finf = (x, u) -> infOps.A * x + infOps.B * u + infOps.H * kron(x, x) + (infOps.N * x) * u + infOps.K

    # fint(x, u) = fF(intruOps.A, intruOps.B, intruOps.F, intruOps.N, intruOps.K, x, u)
    # fint = (x,u) -> intruOps.A * x  + intruOps.B * u + intruOps.H * kron(x, x) + (intruOps.N*x)*u + intruOps.K
    fint = (x, u) -> At * x + Bt * u + Ht * kron(x, x) + (Nt * x) * u + Kt

    ################ (V) Compute errors for Training Data #################
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

    ############### (VI) Compute errors for Test1 Data ####################
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

    ############### (VII) Compute errors for Test2 Data ###################
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

    println("[INFO] ($(time()-start)) r = $(sum(row)) test done.")
end
dims = sum(mode_req, dims=2)
println("[INFO] ($(time()-start)) Complete testing and error analysis.")


println("[INFO] ($(time()-start)) Exporting data.")
@save "scripts/data/fhn/fhn_err.jld2" dims train_err test1_err test2_err  # save data

CSV.write("scripts/data/fhn/train_err_infer.csv", Tables.table(vcat(dims', train_err[:inferred])), writeheader=false)
CSV.write("scripts/data/fhn/train_err_intru.csv", Tables.table(vcat(dims', train_err[:intrusive])), writeheader=false)
CSV.write("scripts/data/fhn/test1_err_infer.csv", Tables.table(vcat(dims', test1_err[:inferred])), writeheader=false)
CSV.write("scripts/data/fhn/test1_err_intru.csv", Tables.table(vcat(dims', test1_err[:intrusive])), writeheader=false)
CSV.write("scripts/data/fhn/test2_err_infer.csv", Tables.table(vcat(dims', test2_err[:inferred])), writeheader=false)
CSV.write("scripts/data/fhn/test2_err_intru.csv", Tables.table(vcat(dims', test2_err[:intrusive])), writeheader=false)

println("[INFO] ($(time()-start)) Done.")
