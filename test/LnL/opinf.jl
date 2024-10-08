@testset "1D heat equation test" begin
    ## Set some options
    PROVIDE_DERIVATIVE = false

    # # 1D Heat equation setup
    Nx = 2^7; dt = 1e-3
    heat1d = Pomoreda.Heat1DModel(
        spatial_domain=(0.0, 1.0), time_domain=(0.0, 1.0), Δx=1/Nx, Δt=dt, 
        diffusion_coeffs=range(0.1, 10, 10)
    )
    Ubc = ones(heat1d.time_dim)
    # Some options for operator inference
    options = LnL.LSOpInfOption(
        system=LnL.SystemStructure(
            state=1,
            control=1,
            output=1,
        ),
        vars=LnL.VariableStructure(
            N=1,
        ),
        data=LnL.DataStructure(
            Δt=dt,
            deriv_type="BE"
        ),
        optim=LnL.OptimizationSetting(
            verbose=true,
        ),
    )

    Xfull = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
    Yfull = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
    pod_bases = Vector{Matrix{Float64}}(undef, heat1d.param_dim)

    A_intru = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
    B_intru = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
    C_intru = Vector{Matrix{Float64}}(undef, heat1d.param_dim)

    A_opinf = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
    B_opinf = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
    C_opinf = Vector{Matrix{Float64}}(undef, heat1d.param_dim)

    ## Generate operators
    r = 15  # order of the reduced form

    for (idx, μ) in enumerate(heat1d.diffusion_coeffs)
        A, B = heat1d.finite_diff_model(heat1d, μ)
        C = ones(1, heat1d.spatial_dim) / heat1d.spatial_dim
        op_heat = LnL.Operators(A=A, B=B, C=C)

        # Compute the states with backward Euler
        X = heat1d.integrate_model(heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B, 
                                   system_input=true, integrator_type=:BackwardEuler)
        Xfull[idx] = X

        # Compute the SVD for the POD basis
        F = svd(X)
        Vr = F.U[:, 1:r]
        pod_bases[idx] = Vr

        # Compute the output of the system
        Y = C * X
        Yfull[idx] = Y

        # Compute the values for the intrusive model
        op_heat_new = LnL.pod(op_heat, Vr, options.system)
        A_intru[idx] = op_heat_new.A
        B_intru[idx] = op_heat_new.B
        C_intru[idx] = op_heat_new.C

        # Compute the RHS for the operator inference based on the intrusive operators
        if PROVIDE_DERIVATIVE
            jj = 2:heat1d.time_dim
            Xn = X[:, jj]
            Un = Ubc[jj, :]
            Yn = Y[:, jj]
            Xdot = A * Xn + B * Un'
            op_infer = LnL.opinf(Xn, Vr, options; U=Un, Y=Yn, Xdot=Xdot)
        else
            op_infer = LnL.opinf(X, Vr, options; U=Ubc, Y=Y)
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

    for i = 1:r, j = 1:heat1d.param_dim
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
        Xint = heat1d.integrate_model(
            heat1d.tspan, Vr' * heat1d.IC, Ubc,
            linear_matrix=Aint[1:i, 1:i], control_matrix=Bint[1:i,:],
            system_input=true, integrator_type=:BackwardEuler
        )
        Yint = Cint[1:1, 1:i] * Xint

        # Integrate the inferred model
        Xinf = heat1d.integrate_model(
            heat1d.tspan, Vr' * heat1d.IC, Ubc,
            linear_matrix=Ainf[1:i, 1:i], control_matrix=Binf[1:i,:],
            system_input=true, integrator_type=:BackwardEuler
        )
        Yinf = Cinf[1:1, 1:i] * Xinf

        # Compute errors
        PE, ISE, IOE, OSE, OOE = LnL.compute_all_errors(Xf, Yf, Xint, Yint, Xinf, Yinf, Vr)

        # Sum of error values
        proj_err[i] += PE / heat1d.param_dim
        intru_state_err[i] += ISE / heat1d.param_dim
        intru_output_err[i] += IOE / heat1d.param_dim
        opinf_state_err[i] += OSE / heat1d.param_dim
        opinf_output_err[i] += OOE / heat1d.param_dim
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