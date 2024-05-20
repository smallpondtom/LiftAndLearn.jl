## Setups 
using BlockDiagonals
using LinearAlgebra

using LiftAndLearn
const LnL = LiftAndLearn

@testset "FHN test" begin
    ## Generate models
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


    ## Generate training data
    # parameters for the training data
    α_train = vec([500 5000 50000] .* ones(3))
    β_train = vec([10 12.5 15]' .* ones(3)')

    ###################### (I) Create training data ###########################
    Xtrain = Vector{Matrix{Float64}}(undef, length(α_train))
    Utrain = Vector{Matrix{Float64}}(undef, length(α_train))
    Utrain_all = Vector{Matrix{Float64}}(undef, length(α_train))

    for i in axes(α_train, 1)
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

    ## Create test 1 data
    N_tests = 16

    ## Analyze the training and tests 
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

    for (i, row) in enumerate(eachrow(mode_req))
        r1, r2, r3 = row
        Vr1 = W1.U[:, 1:r1]
        Vr2 = W2.U[:, 1:r2]
        Vr3 = W3.U[:, 1:r3]
        Vr = BlockDiagonal([Vr1, Vr2, Vr3])

        infOps = LnL.inferOp(Wtr, Vr, lifter, fomOps, options; U=Utr, Y=Ytr)
        intruOps = LnL.pod(fomLinOps, Vr, options)

        finf = (x, u) -> infOps.A * x + infOps.B * u + infOps.H * kron(x, x) + (infOps.N * x) * u + infOps.K
        fint = (x, u) -> intruOps.A * x  + intruOps.B * u + intruOps.H * kron(x, x) + (intruOps.N*x)*u + intruOps.K

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
    end
    dims = sum(mode_req, dims=2)

    test_mat1 = [
        0.00413855  0.0259613  0.000903311  0.000188205
        0.00416805  0.0269879  0.000911207  0.000226952
        0.00418182  0.0272766  0.000929233  0.000220063
        0.0109724   0.018806   0.00247275   0.000439961
        0.00712047  0.023284   0.00145107   0.000484011
        0.00490811  0.0257148  0.00103099   0.000375223
        0.0588744   0.0666531  0.00978946   0.00247522
        0.0496467   0.0417722  0.010852     0.00314228
        0.0282867   0.0231516  0.00593505   0.00214965
    ]
    test_mat2 = [
        0.00406325  0.0260274  0.000938877  9.08778e-5
        0.00411861  0.027046   0.00095353   0.000113272
        0.00414396  0.0273343  0.000973439  0.000118377
        0.0108276   0.018781   0.00247572   0.000479533
        0.00701003  0.0233261  0.00144485   0.000194012
        0.00481827  0.0257618  0.00105289   0.000161501
        0.0637862   0.0602712  0.00979959   0.00234719
        0.0502939   0.0413096  0.0110324    0.00182619
        0.0283079   0.022959   0.00601366   0.00106416
    ]

    @test all(isapprox.(test_mat1, train_err[:intrusive], atol=1e-5))
    @test all(isapprox.(test_mat2, train_err[:inferred], atol=1e-5))
end