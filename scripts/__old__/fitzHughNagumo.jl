module fitzHughNagumo

include("fhn_helper.jl")
include("integrator.jl")
include("intrusiveROM.jl")
include("lift.jl")
include("opinf.jl")
include("utils.jl")

using BlockDiagonals
using CSV
using DataFrames
using LinearAlgebra 
using Plots
using ProgressMeter
using Random
using SparseArrays
using Statistics


function test(dx::Real=2^(-9), dt::Real=1e-4)
    @show dx, dt
    start = time()
    ## First order Burger's equation setup
    FHN = FHN_params(
        [0.0, 1.0], [0.0, 4.0], [500, 50000], [10, 15], dx, dt
    )

    # Some options for operator inference
    options = opInf_options(
        is_quad=true,
        is_bilin=true,
        is_lifted=true,
        has_const=true,
        has_funcOp=true,
        DS=100,
        N=2,
        N_lift=3,
        Δt=dt
    )

    # grid points
    gp = Int(1 / FHN.Δx)

    # Downsampling
    DS = options.DS

    # Get the full-order model operators for intrusive model
    fomLinOps = generateFHNmatrices(gp, FHN.Ω[2])
    println("[INFO] ($(time()-start)) Complete generating intrusive model operators")

    # Generic function for FOM
    fomOps = FOM(gp, FHN.Ω[2])  # much efficient to calculate for FOM
    fom_state(x, u) = fomOps.A * x + fomOps.B * u + fomOps.f(x) + fomOps.K
    println("[INFO] ($(time()-start)) Complete generating full-order model operators")

    # parameters for the training data
    α_train = vec([500 5000 50000] .* ones(3))
    β_train = vec([10 12.5 15]' .* ones(3)')

    ###################### (I) Create training data ###########################
    Xtrain = Vector{Matrix{Float64}}(undef, length(α_train))
    Utrain = Vector{Matrix{Float64}}(undef, length(α_train))
    @showprogress for i in axes(α_train, 1)
        α, β = α_train[i], β_train[i]
        genU(t) = α * t^3 * exp(-β * t)  # generic function for input

        ## training data for inferred dynamical models
        @inbounds X = forwardEuler(fom_state, genU, FHN.t, FHN.ICx)
        Xtrain[i] = X[:, 1:DS:end]  # make sure to only record every 0.01s
        Utrain[i] = transpose(genU.(FHN.t[1:DS:end]))
    end
    Xtr = reduce(hcat, Xtrain)
    Utr = reduce(hcat, Utrain)
    Ytr = fomOps.C * Xtr

    # DEBUG:
    println(size(Xtr), size(Utr), size(Ytr))

    println("[INFO] ($(time()-start)) Created training data")

    # Parameters for the testing data (1)
    α_test1 = vec(rand(100, 1) .* (FHN.αD[2] - FHN.αD[1]) .+ FHN.αD[1])
    β_test1 = vec(rand(100, 1) .* (FHN.βD[2] - FHN.βD[1]) .+ FHN.βD[1])

    ################### (II) Create testing data 1 ############################
    Xtest1 = Vector{Matrix{Float64}}(undef, 100)
    Utest1 = Vector{Matrix{Float64}}(undef, 100)
    @showprogress for i in axes(α_test1, 1)
        α, β = α_test1[i], β_test1[i]
        genU(t) = α * t^3 * exp(-β * t)  # generic function for input

        @inbounds X = forwardEuler(fom_state, genU, FHN.t, FHN.ICx)
        Xtest1[i] = X[:, 1:DS:end]  # make sure to only record every 0.01s
        Utest1[i] = transpose(genU.(FHN.t[1:DS:end]))
    end
    # Xts1 = reduce(hcat, Xtest1)
    # Uts1 = reduce(hcat, Utest1)
    # Yts1 = fomOps.C * Xts1

    println("[INFO] ($(time()-start)) Created test 1 data")

    # Parameters for testing data (2)
    α_test2 = vec((5 * 10 .^ range(start=4, stop=6, length=10))' .* ones(10))
    β_test2 = vec(range(start=15, stop=20, length=10) .* ones(10)')

    ################### (III) Create testing data 2 ###########################
    Xtest2 = Vector{Matrix{Float64}}(undef, 100)
    Utest2 = Vector{Matrix{Float64}}(undef, 100)
    @showprogress for i in axes(α_test2, 1)
        α, β = α_test2[i], β_test2[i]
        genU(t) = α * t^3 * exp(-β * t)  # generic function for input

        @inbounds X = forwardEuler(fom_state, genU, FHN.t, FHN.ICx)
        Xtest2[i] = X[:, 1:DS:end]  # make sure to only record every 0.01s
        Utest2[i] = transpose(genU.(FHN.t[1:DS:end]))
    end
    # Xts2 = reduce(hcat, Xtest2)
    # Uts2 = reduce(hcat, Utest2)
    # Yts2 = Cf * Xts2

    println("[INFO] ($(time()-start)) Created test 2 data")

    ##################### (IV~VII) Learn & Analysis ###########################
    mode_req = [1 1 1; 2 1 3; 3 3 4; 5 4 5]  # Required number of modes for each lifted variables


    Wtr = vcat(Xtr, Xtr[1:gp, :] .^ 2)  # lifted training states

    # NOTE: Trying out the generalized lifting module for later usage
    # Xsep = [Xtr[1:gp, :], Xtr[gp+1:end, :]]
    # Wtr = Lift.lifting(options.N, options.N_lift, [x -> x[1].^2])


    # DEBUG:
    println(size(Wtr))

    # Take the SVD for each variable
    W1 = svd(Wtr[1:gp, :])
    W2 = svd(Wtr[gp+1:2*gp, :])
    W3 = svd(Wtr[2*gp+1:end, :])

    # Compute the inferred and intrusive operators and store them for error analysis
    infOps = Vector{operators}(undef, 4)
    intruOps = Vector{operators}(undef, 4)
    fH(A, B, H, N, C, x, u) = A * x + H * kron(x, x) + (N * x) * u + B * u + C
    fF(A, B, F, N, C, x, u) = A * x + F * vech(x * x') + (N * x) * u + B * u + C

    # dictionary with intrusive and LnL errors as matrices (9-by-4)
    train_err = Dict(
        :intrusive => zeros(length(α_train), size(mode_req, 1)),
        :inferred => zeros(length(α_train), size(mode_req, 1))
    )
    test1_err = Dict(
        :intrusive => zeros(100, size(mode_req, 1)),
        :inferred => zeros(100, size(mode_req, 1))
    )
    test2_err = Dict(
        :intrusive => zeros(100, size(mode_req, 1)),
        :inferred => zeros(100, size(mode_req, 1))
    )

    for (i, row) in enumerate(eachrow(mode_req))
        r1, r2, r3 = row
        Vr1 = W1.U[:, 1:r1]
        Vr2 = W2.U[:, 1:r2]
        Vr3 = W3.U[:, 1:r3]
        Vr = BlockDiagonal([Vr1, Vr2, Vr3])

        # infOps[i] = inferOp(Xtr, Utr, Ytr, Vr, FHN.Δt, 0, params)
        # intruOps[i] = intrusiveMR(fomLinOps.A, fomLinOps.B, fomLinOps.C, Vr, 0, fomLinOps.H, fomLinOps.N)

        ####################### (IV) Infer and Intrusive ######################
        infOps = inferOp(Wtr, Utr, Ytr, Vr, FHN.Δt, 0, options)
        intruOps = intrusiveMR(fomLinOps.A, fomLinOps.B, fomLinOps.C, Vr, fomLinOps.F, fomLinOps.N)

        fint(x, u) = fF(intruOps.A, intruOps.B, intruOps.F, intruOps.N, x, u)
        finf(x, u) = fH(infOps.A, infOps.B, infOps.H, infOps.N, x, u)

        ################ (V) Compute errors for Training Data #################
        k = 0
        for (X, U) in zip(Xtrain, Utrain)
            Xint = forwardEuler(fint, U, FHN.t, FHN.ICw)
            Xinf = forwardEuler(finf, U, FHN.t, FHN.ICw)
            train_err[:intrusive][k+=1, i] = compStateError(X, Xint, Vr)
            train_err[:inferred][k+=1, i] = compStateError(X, Xinf, Vr)
        end

        ############### (VI) Compute errors for Test1 Data ####################
        k = 0
        for (X, U) in zip(Xtest1, Utest1)
            Xint = forwardEuler(fint, U, FHN.t, FHN.ICw)
            Xinf = forwardEuler(finf, U, FHN.t, FHN.ICw)
            test1_err[:intrusive][k+=1, i] = compStateError(X, Xint, Vr)
            test1_err[:inferred][k+=1, i] = compStateError(X, Xinf, Vr)
        end

        ############### (VII) Compute errors for Test2 Data ###################
        k = 0
        for (X, U) in zip(Xtest1, Utest1)
            Xint = forwardEuler(fint, U, FHN.t, FHN.ICw)
            Xinf = forwardEuler(finf, U, FHN.t, FHN.ICw)
            test1_err[:intrusive][k+=1, i] = compStateError(X, Xint, Vr)
            test1_err[:inferred][k+=1, i] = compStateError(X, Xinf, Vr)
        end
    end

    ####################### (VIII) Plot Results ###############################
    dims = sum(mode_req, 2)
    # Training trajectories
    p1 = plot()
    p1 = errBnds(p1, dims, train_err[:intrusive])
    plot!(p1, dims, median(train_err[:inferred], dims=1), ms=10, mc=:red, shape=:circle, ls=:dash, lc=:red, label="LnL")
    plot!(p1, yscale=:log10, majorgrid=true, minorgrid=true)
    tmp = log10.(median(train_err[:intrusive], dims=1))
    yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
    xticks!(dims)
    xlabel!("dimension n")
    ylabel!("Relative state error")
    title!("Error over training trajectories")
    # savefig("src/plots/fhn_train_err.pdf")

    # Test trajectories
    p2 = plot()
    p2 = errBnds(p2, dims, test1_err[:intrusive])
    plot!(p2, dims, median(test1_err[:inferred], dims=1), ms=10, mc=:red, shape=:circle, ls=:dash, lc=:red, label="LnL")
    p2 = errBnds(p2, dims, test2_err[:intrusive])
    plot!(p2, dims, median(test2_err[:inferred], dims=1), ms=10, mc=:red, shape=:circle, ls=:dash, lc=:red, label="LnL")
    plot!(p2, yscale=:log10, majorgrid=true, minorgrid=true)
    tmp = log10.(median(test1_err[:intrusive], dims=1))
    yticks!([10.0^i for i in floor(minimum(tmp))-2:ceil(maximum(tmp))+2])
    xticks!(dims)
    xlabel!("dimension n")
    ylabel!("Relative state error")
    title!("Test errors over new trajectories")
    # savefig("src/plots/fhn_test_err.pdf")
end


end



