## Setups 
@testset "FHN test" begin
    ## Generate models
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
        ),
        vars=LnL.VariableStructure(
            N=2,
            N_lift=3,
        ),
        data=LnL.DataStructure(
            Δt=1e-4,
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

    # Get the full-order model operators for intrusive model
    tmp = fhn.lifted_finite_diff_model(gp, fhn.spatial_domain[2])
    fomLinOps = LnL.Operators(
        A=tmp[1],
        B=tmp[2][:, :],  # Make sure its matrix
        C=tmp[3][:, :],
        A2=tmp[4],
        A2u=LnL.eliminate(tmp[4],2),
        N=tmp[5],
        K=tmp[6]
    )

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
        X = fhn.integrate_model(fhn.tspan, fhn.IC, genU; functional=fom_state)
        Xtrain[i] = X[:, 1:DS:end]  # make sure to only record every 0.01s
        U = genU.(fhn.tspan)
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

        infOps = LnL.opinf(Wtr, Vr, lifter, fomOps, options; U=Utr, Y=Ytr)
        infOps.A2 = duplicate(infOps.A2u, 2)
        intruOps = LnL.pod(fomLinOps, Vr, options.system)

        finf = (x, u) -> infOps.A * x + infOps.B * u + infOps.A2 * kron(x, x) + (infOps.N * x) * u[1] + infOps.K
        fint = (x, u) -> intruOps.A * x  + intruOps.B * u + intruOps.A2 * kron(x, x) + (intruOps.N * x) * u[1] + intruOps.K

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


function full_order_model(k, l)
    h = l / (k - 1)
    Alift = spzeros(3 * k, 3 * k)
    Blift = spzeros(3 * k, 2)
    gamma = 0.015
    R = 0.5
    c = 2
    g = 0.05

    E = sparse(1.0I, 3 * k, 3 * k)
    for i in 1:k
        E[i, i] = gamma
        E[i+2*k, i+2*k] = gamma
    end

    # Left boundary
    Alift[1, 1] = -gamma^2 / h^2 - 0.1
    Alift[1, 2] = gamma^2 / h^2
    Alift[1, k+1] = -1
    # Alift[1,2*k+1] = 1.1

    Alift[k+1, 1] = R
    Alift[k+1, k+1] = -c

    Alift[2*k+1, 1] = 2 * g
    Alift[2*k+1, 2*k+1] = -2 * gamma^2 / h^2 - 0.2

    # Right boundary 
    Alift[k, k-1] = gamma^2 / h^2
    Alift[k, k] = -gamma^2 / h^2 - 0.1
    Alift[k, 2*k] = -1
    # Alift[k,3*k] = 1.1

    Alift[2*k, k] = R
    Alift[2*k, 2*k] = -c

    Alift[3*k, 3*k] = -2 * gamma^2 / h^2 - 0.2
    Alift[3*k, k] = 2 * g

    # Inner points
    for i in 2:k-1
        Alift[i, i-1] = gamma^2 / h^2
        Alift[i, i] = -2 * gamma^2 / h^2 - 0.1
        Alift[i, i+1] = gamma^2 / h^2
        Alift[i, i+k] = -1

        Alift[i+k, i] = R
        Alift[i+k, i+k] = -c

        Alift[i+2*k, i] = 2 * g
        Alift[i+2*k, 1+2*k] = -4 * gamma^2 / h^2 - 0.2
    end

    # B matrix
    Blift[1, 1] = gamma^2 / h
    # Blift[:,2] = g
    # NOTE: The second column of the input matrix B corresponds to the constant
    # terms of the FHN PDE.
    Blift[1:2*k, 2] .= g

    Atmp = E \ Alift
    Btmp = E \ Blift

    # A and B matrix
    A = Atmp[1:2*k, 1:2*k]
    B = Btmp[1:2*k, 1]
    K = Btmp[1:2*k, 2]

    # C matrix
    Clift = spzeros(2, 3 * k)
    Clift[1, 1] = 1
    Clift[2, 1+k] = 1
    C = Clift[:, 1:2*k]

    f = (x,u) -> [-x[1:k, :] .^ 3 + 1.1 * x[1:k, :] .^ 2; spzeros(k, size(x, 2))] / gamma

    return A, B, C, K, f
end

function lifted_finite_diff_model(k, l)
    h = l / (k - 1)
    E = sparse(1.0I, 3 * k, 3 * k)
    A = spzeros(3 * k, 3 * k)
    H = spzeros(3 * k, 9 * k^2)
    N = spzeros(3 * k, 3 * k)
    B = spzeros(3 * k, 2)

    gamma = 0.015
    R = 0.5
    c = 2
    g = 0.05

    for i in 1:k
        E[i, i] = gamma
        E[i+2*k, i+2*k] = gamma
    end

    # Left boundary
    A[1, 1] = -gamma^2 / h^2 - 0.1
    A[1, 2] = gamma^2 / h^2
    A[1, k+1] = -1

    A[k+1, 1] = R
    A[k+1, k+1] = -c

    A[2*k+1, 1] = 2 * g
    A[2*k+1, 2*k+1] = -2 * gamma^2 / h^2 - 0.2

    # Right Boundary
    A[k, k-1] = gamma^2 / h^2
    A[k, k] = -gamma^2 / h^2 - 0.1
    A[k, 2*k] = -1

    A[2*k, k] = R
    A[2*k, 2*k] = -c

    A[3*k, 3*k] = -2 * gamma^2 / h^2 - 0.2
    A[3*k, k] = 2 * g

    # inner points
    for i = 2:k-1
        A[i, i-1] = gamma^2 / h^2
        A[i, i] = -2 * gamma^2 / h^2 - 0.1
        A[i, i+1] = gamma^2 / h^2
        A[i, i+k] = -1

        A[i+k, i] = R
        A[i+k, i+k] = -c

        A[i+2*k, i] = 2 * g
        A[i+2*k, i+2*k] = -4 * gamma^2 / h^2 - 0.2
    end

    # left boundary
    H[1, 1] = 1.1
    H[1, 2*k+1] = -0.5
    H[1, 2*k*3*k+1] = -0.5

    H[2*k+1, 2] = gamma^2 / h^2
    H[2*k+1, 3*k+1] = gamma^2 / h^2
    H[2*k+1, 2*k+1] = 1.1
    H[2*k+1, 2*k*3*k+1] = 1.1
    H[2*k+1, 2*k*3*k+2*k+1] = -2
    H[2*k+1, k+1] = -1
    H[2*k+1, k*3*k+1] = -1

    # right boundary
    H[k, 3*k*(k-1)+k] = 1.1

    H[k, (k-1)*3*k+3*k] = -0.5
    H[k, (3*k-1)*3*k+k] = -0.5

    H[3*k, (k-2)*3*k+k] = gamma^2 / h^2
    H[3*k, (k-1)*3*k+k-1] = gamma^2 / h^2
    H[3*k, (k-1)*3*k+3*k] = 1.1
    H[3*k, (3*k-1)*3*k+k] = 1.1
    H[3*k, 9*k^2] = -2
    H[3*k, (k-1)*3*k+2*k] = -1
    H[3*k, (2*k-1)*3*k+k] = -1

    # inner points
    for i = 2:k-1
        H[i, (i-1)*3*k+i] = 1.1
        H[i, (i-1)*3*k+2*k+i] = -0.5
        H[i, (i+2*k-1)*3*k+i] = -0.5

        H[i+2*k, (i-1)*3*k+i+1] = gamma^2 / h^2
        H[i+2*k, i*3*k+i] = gamma^2 / h^2
        H[i+2*k, (i-1)*3*k+i-1] = gamma^2 / h^2
        H[i+2*k, (i-2)*3*k+i] = gamma^2 / h^2
        H[i+2*k, (i-1)*3*k+i+2*k] = 1.1
        H[i+2*k, (i-1+2*k)*3*k+i] = 1.1
        H[i+2*k, (i+2*k-1)*3*k+i+2*k] = -2
        H[i+2*k, (i-1)*3*k+k+i] = -1
        H[i+2*k, (i+k-1)*3*k+i] = -1
    end

    N[2*k+1, 1] = 2 * gamma^2 / h

    B[1, 1] = gamma^2 / h
    # B[:,2] = g
    B[1:2*k, 2] .= g

    C = spzeros(2, 3 * k)
    C[1, 1] = 1
    C[2, 1+k] = 1

    A = E \ A
    tmp = E \ B
    B = tmp[:,1]
    K = tmp[:,2]
    H = E \ H
    N = E \ N

    # NOTE: (BELOW) making the sparse H matrix symmetric
    # n = 3 * k
    # Htensor = ndSparse(H, n)
    # H2 = reshape(permutedims(Htensor, [2, 1, 3]), (n, n^2))
    # H3 = reshape(permutedims(Htensor, [3, 1, 2]), (n, n^2))
    # symH2 = 0.5 * (H2 + H3)
    # symHtensor2 = reshape(symH2, (n, n, n))
    # symHtensor = permutedims(symHtensor2, [2 1 3])
    # symH = reshape(symHtensor, (n, n^2))
    # H = sparse(symH)

    return A, B, C, H, N, K
end