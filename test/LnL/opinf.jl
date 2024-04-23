## Load packages
using LinearAlgebra
using DataFrames

using LiftAndLearn
const LnL = LiftAndLearn

@testset "1D heat equation test" begin
    ## Set some options
    provide_R = false

    # 1D Heat equation setup
    heat1d = LnL.heat1d(
        [0.0, 1.0], [0.0, 1.0], [0.1, 10],
        2^(-7), 1e-3, 10
    )
    heat1d.x = heat1d.x[2:end-1]

    # Some options for operator inference
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

    Xfull = Vector{Matrix{Float64}}(undef, heat1d.Pdim)
    Yfull = Vector{Matrix{Float64}}(undef, heat1d.Pdim)
    pod_bases = Vector{Matrix{Float64}}(undef, heat1d.Pdim)

    A_intru = Vector{Matrix{Float64}}(undef, heat1d.Pdim)
    B_intru = Vector{Matrix{Float64}}(undef, heat1d.Pdim)
    C_intru = Vector{Matrix{Float64}}(undef, heat1d.Pdim)

    A_opinf = Vector{Matrix{Float64}}(undef, heat1d.Pdim)
    B_opinf = Vector{Matrix{Float64}}(undef, heat1d.Pdim)
    C_opinf = Vector{Matrix{Float64}}(undef, heat1d.Pdim)

    ## Generate operators
    r = 15  # order of the reduced form

    for (idx, μ) in enumerate(heat1d.μs)
        A, B = heat1d.generateABmatrix(heat1d.Xdim, μ, heat1d.Δx)
        C = ones(1, heat1d.Xdim) / heat1d.Xdim

        op_heat = LnL.operators(A=A, B=B, C=C)

        # Compute the states with backward Euler
        X = LnL.backwardEuler(A, B, heat1d.Ubc, heat1d.t, heat1d.IC)
        Xfull[idx] = X

        # Compute the SVD for the POD basis
        F = svd(X)
        Vr = F.U[:, 1:r]
        pod_bases[idx] = Vr

        # Compute the output of the system
        Y = C * X
        Yfull[idx] = Y

        # Compute the values for the intrusive model
        op_heat_new = LnL.intrusiveMR(op_heat, Vr, options)
        A_intru[idx] = op_heat_new.A
        B_intru[idx] = op_heat_new.B
        C_intru[idx] = op_heat_new.C

        # Compute the RHS for the operator inference based on the intrusive operators
        if provide_R
            jj = 2:heat1d.Tdim
            Xn = X[:, jj]
            Un = heat1d.Ubc[jj, :]
            Yn = Y[:, jj]
            Xdot = A * Xn + B * Un'
            op_infer = LnL.inferOp(Xn, Un, Yn, Vr, Xdot, options)
        else
            op_infer = LnL.inferOp(X, heat1d.Ubc, Y, Vr, options)
        end

        A_opinf[idx] = op_infer.A
        B_opinf[idx] = op_infer.B
        C_opinf[idx] = op_infer.C
    end

    ## Analyze
    # Error analysis 
    intru_state_err = zeros(r, 1)
    opinf_state_err = zeros(r, 1)
    intru_output_err = zeros(r, 1)
    opinf_output_err = zeros(r, 1)
    proj_err = zeros(r, 1)

    for i = 1:r, j = 1:heat1d.Pdim
        Xf = Xfull[j]  # full order model states
        Yf = Yfull[j]  # full order model outputs
        Vr = pod_bases[j][:, 1:i]  # basis

        # Unpack intrusive operators
        Aint = A_intru[j]
        Bint = B_intru[j]
        Cint = C_intru[j]

        # Unpack inferred operators
        Ainf = A_opinf[j]
        Binf = B_opinf[j]
        Cinf = C_opinf[j]

        # Integrate the intrusive model
        Xint = LnL.backwardEuler(Aint[1:i, 1:i], Bint[1:i, :], heat1d.Ubc, heat1d.t, Vr' * heat1d.IC)
        Yint = Cint[1:1, 1:i] * Xint

        # Integrate the inferred model
        Xinf = LnL.backwardEuler(Ainf[1:i, 1:i], Binf[1:i, :], heat1d.Ubc, heat1d.t, Vr' * heat1d.IC)
        Yinf = Cinf[1:1, 1:i] * Xinf

        # Compute errors
        PE, ISE, IOE, OSE, OOE = LnL.compError(Xf, Yf, Xint, Yint, Xinf, Yinf, Vr)

        # Sum of error values
        proj_err[i] += PE / heat1d.Pdim
        intru_state_err[i] += ISE / heat1d.Pdim
        intru_output_err[i] += IOE / heat1d.Pdim
        opinf_state_err[i] += OSE / heat1d.Pdim
        opinf_output_err[i] += OOE / heat1d.Pdim
    end

    df = DataFrame(
        :order => 1:r,
        :projection_err => vec(proj_err),
        :intrusive_state_err => vec(intru_state_err),
        :intrusive_output_err => vec(intru_output_err),
        :inferred_state_err => vec(opinf_state_err),
        :inferred_output_err => vec(opinf_output_err)
    )

    test_mat1 = [
        0.11400966096979595
        0.014577198982558135
        0.003350366784985043
        0.0009412025090338486
        0.0003030139217119008
        0.00010509077458174854
        3.703267417835386e-5
        1.2789832569698638e-5
        4.250639716297076e-6
        1.347436491598498e-6
        4.0560100247960616e-7
        1.1564636168653036e-7
        3.117866668227968e-8
        7.937271521410912e-9
        1.905649098227372e-9
    ]
    test_mat2 = [
        0.11370450283041714
        0.014536114704524566
        0.003344116661396778
        0.0009402254940992438
        0.00030287105863172656
        0.00010507107883088188
        3.702952347707602e-5
        1.278884582152224e-5
        4.249994139920112e-6
        1.3469010417652767e-6
        4.051420667457527e-7
        1.1525337987583825e-7
        3.0844104072481994e-8
        7.656856258051758e-9
        1.7098849087349572e-9
    ]

    @test all(isapprox.(test_mat1, df.intrusive_state_err, atol=1e-6))
    @test all(isapprox.(test_mat2, df.inferred_state_err, atol=1e-6))
end


@testset "Standard OpInf as Optimization" begin
    burger = LnL.burgers(
        [0.0, 1.0], [0.0, 1.0], [0.1, 0.2],
        2^(-7), 1e-4, 2, "dirichlet"
    );

    options = LnL.LS_options(
        system=LnL.sys_struct(
            is_lin=true,
            is_quad=true,
            has_control=true,
            has_output=true,
        ),
        vars=LnL.vars(
            N=1,
        ),
        data=LnL.data(
            Δt=1e-4,
            deriv_type="SI"
        ),
        optim=LnL.opt_settings(
            verbose=true,
        ),
        with_tol=true,
        with_reg=true,
        pinv_tol=1e-6,
        λ=λtik(
            lin=1.0,
            quad=1e-3,
            ctrl=1e-2,
            bilin=0.0
        )
    )
    Utest = ones(burger.Tdim - 1, 1);  # Reference input/boundary condition for OpInf testing 

    # Error Values 
    num_inputs = 3
    rmax = 5
    k = 3
    proj_err = zeros(rmax - k, burger.Pdim)
    intru_state_err = zeros(rmax - k, burger.Pdim)
    opinf_state_err = zeros(rmax - k, burger.Pdim)
    intru_output_err = zeros(rmax - k, burger.Pdim)
    opinf_output_err = zeros(rmax - k, burger.Pdim)
    Σr = Vector{Vector{Float64}}(undef, burger.Pdim)  # singular values 

    # Add 5 extra parameters drawn randomly from the uniform distribution of range [0, 1]
    μs = vcat(burger.μs)

    for i in 1:length(μs)
        μ = burger.μs[i]

        ## Create testing data
        A, B, F = burger.generateABFmatrix(burger, μ)
        C = ones(1, burger.Xdim) / burger.Xdim
        Xtest = LnL.semiImplicitEuler(A, B, F, Utest, burger.t, burger.IC)
        Ytest = C * Xtest

        op_burger = LnL.operators(A=A, B=B, C=C, F=F)

        ## training data for inferred dynamical models
        Urand = rand(burger.Tdim - 1, num_inputs)
        Xall = Vector{Matrix{Float64}}(undef, num_inputs)
        Xdotall = Vector{Matrix{Float64}}(undef, num_inputs)
        for j in 1:num_inputs
            states = burger.semiImplicitEuler(A, B, F, Urand[:, j], burger.t, burger.IC)
            Xall[j] = states[:, 2:end]
            Xdotall[j] = (states[:, 2:end] - states[:, 1:end-1]) / burger.Δt
        end
        X = reduce(hcat, Xall)
        R = reduce(hcat, Xdotall)
        U = reshape(Urand, (burger.Tdim - 1) * num_inputs, 1)
        Y = C * X

        # compute the POD basis from the training data
        tmp = svd(X)
        Vrmax = tmp.U[:, 1:rmax]
        Σr[i] = tmp.S

        # Compute the values for the intrusive model from the basis of the training data
        op_int = LnL.intrusiveMR(op_burger, Vrmax, options)

        # Compute the inferred operators from the training data
        if options.optim.reproject || i == 1
            op_inf = LnL.inferOp(X, U, Y, Vrmax, op_burger, options)  # Using Reprojection
        else
            op_inf = LnL.inferOp(X, U, Y, Vrmax, R, options)
        end

        for j = 1+k:rmax
            Vr = Vrmax[:, 1:j]  # basis

            # Integrate the intrusive model
            Fint_extract = LnL.extractF(op_int.F, j)
            Xint = burger.semiImplicitEuler(op_int.A[1:j, 1:j], op_int.B[1:j, :], Fint_extract, Utest, burger.t, Vr' * burger.IC) # <- use F
            Yint = op_int.C[1:1, 1:j] * Xint

            # Integrate the inferred model
            Finf_extract = LnL.extractF(op_inf.F, j)
            Xinf = burger.semiImplicitEuler(op_inf.A[1:j, 1:j], op_inf.B[1:j, :], Finf_extract, Utest, burger.t, Vr' * burger.IC)  # <- use F
            Yinf = op_inf.C[1:1, 1:j] * Xinf

            # Compute errors
            PE, ISE, IOE, OSE, OOE = LnL.compError(Xtest, Ytest, Xint, Yint, Xinf, Yinf, Vr)

            # Sum of error values
            proj_err[j-k, i] = PE
            intru_state_err[j-k, i] = ISE
            intru_output_err[j-k, i] = IOE
            opinf_state_err[j-k, i] = OSE
            opinf_output_err[j-k, i] = OOE
        end
    end

    for i in 1:length(burger.μs)
        _, _ = LnL.choose_ro(Σr[i]; en_low=-12)
    end

    @test true  # run without any errors
    
end