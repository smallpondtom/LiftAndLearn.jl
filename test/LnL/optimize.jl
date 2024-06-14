using LiftAndLearn
using LinearAlgebra
using Statistics
using Test

const LnL = LiftAndLearn

@testset "Burgers standard NC-OpInf" begin
    # First order Burger's equation setup
    Ω = (0.0, 1.0)
    Nx = 2^7; dt = 1e-4
    burger = LnL.BurgersModel(
        spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
        diffusion_coeffs=range(0.1, 1.0, length=2), BC=:dirichlet,
    )

    num_inputs = 3
    rmax = 5

    options = LnL.NCOpInfOption(
        system=LnL.SystemStructure(
            is_lin=true,
            is_quad=true,
            has_control=true,
            has_output=true,
        ),
        vars=LnL.VariableStructure(
            N=1,
        ),
        data=LnL.DataStructure(
            Δt=1e-4,
            deriv_type="SI",
            DS=100,
        ),
        optim=LnL.OptimizationSetting(
            verbose=false,
            initial_guess=false,
        ),
    )

    Utest = ones(burger.time_dim - 1, 1);  # Reference input/boundary condition for OpInf testing 

    # Error Values 
    k = 3

    # Downsampling rate
    DS = options.data.DS

    for i in 1:length(burger.diffusion_coeffs)
        μ = burger.diffusion_coeffs[i]

        ## Create testing data
        A, B, F = burger.finite_diff_model(burger, μ)
        C = ones(1, burger.spatial_dim) / burger.spatial_dim
        Xtest = LnL.semiImplicitEuler(A, B, F, Utest, burger.tspan, burger.IC)
        Ytest = C * Xtest

        op_burger = LnL.Operators(A=A, B=B, C=C, F=F)

        ## training data for inferred dynamical models
        Urand = rand(burger.time_dim - 1, num_inputs)
        Xall = Vector{Matrix{Float64}}(undef, num_inputs)
        Xdotall = Vector{Matrix{Float64}}(undef, num_inputs)
        for j in 1:num_inputs
            states = burger.integrate_model(A, B, F, Urand[:, j], burger.tspan, burger.IC)
            tmp = states[:, 2:end]
            Xall[j] = tmp[:, 1:DS:end]  # downsample data
            tmp = (states[:, 2:end] - states[:, 1:end-1]) / burger.Δt
            Xdotall[j] = tmp[:, 1:DS:end]  # downsample data
        end
        X = reduce(hcat, Xall)
        R = reduce(hcat, Xdotall)
        Urand = Urand[1:DS:end, :]  # downsample data
        U = vec(Urand)[:,:]  # vectorize
        Y = C * X

        # compute the POD basis from the training data
        tmp = svd(X)
        Vrmax = tmp.U[:, 1:rmax]

        # Compute the values for the intrusive model from the basis of the training data
        op_int = LnL.pod(op_burger, Vrmax, options)

        # Compute the inferred operators from the training data
        if options.optim.reproject
            op_inf = LnL.opinf(X, Vrmax, op_burger, options; U=U, Y=Y, IG=op_int)  # Using Reprojection
        else
            op_inf = LnL.opinf(X, Vrmax, options; Xdot=R, U=U, Y=Y, IG=op_int)  # without reprojection
        end

        for j = 1+k:rmax
            Vr = Vrmax[:, 1:j]  # basis
            
            # Integrate the intrusive model
            Fint_extract = LnL.extractF(op_int.F, j)
            Xint = burger.integrate_model(op_int.A[1:j, 1:j], op_int.B[1:j, :], Fint_extract, Utest, burger.tspan, Vr' * burger.IC)  # use F
            Yint = op_int.C[1:1, 1:j] * Xint
            
            # Integrate the inferred model
            Finf_extract = LnL.extractF(op_inf.F, j)
            Xinf = burger.integrate_model(op_inf.A[1:j, 1:j], op_inf.B[1:j, :], Finf_extract, Utest, burger.tspan, Vr' * burger.IC)  # use F
            Yinf = op_inf.C[1:1, 1:j] * Xinf

            # Compute errors
            PE, ISE, IOE, OSE, OOE = LnL.compError(Xtest, Ytest, Xint, Yint, Xinf, Yinf, Vr)
        end
    end
    @test true  # dummy test to make sure the testset runs
end


@testset "KSE EP-OpInf" begin
    # Settings for the KS equation
    Ω = (0.0, 22.0)
    dt = 0.01
    N = 256
    KSE = LnL.KuramotoSivashinskyModel(
        spatial_domain=Ω, time_domain=(0.0, 300.0), diffusion_coeffs=1.0,
        Δx=(Ω[2] - 1/N)/N, Δt=dt, conservation_type=:EP
    )

    # Settings for Operator Inference
    KSE_system = LnL.SystemStructure(
        is_lin=true,
        is_quad=true,
    )
    KSE_VariableStructure = LnL.VariableStructure(
        N=1,
    )
    KSE_data = LnL.DataStructure(
        Δt=KSE.Δt,
        DS=100,
    )
    KSE_optim = LnL.OptimizationSetting(
        verbose=true,
        initial_guess=false,
        max_iter=1000,
        reproject=false,
        SIGE=false,
        with_bnds=true,
    )

    options = LnL.LSOpInfOption(
        system=KSE_system,
        vars=KSE_VariableStructure,
        data=KSE_data,
        optim=KSE_optim,
    )

    # Downsampling rate
    DS = KSE_data.DS

    # Down-sampled dimension of the time data
    Tdim_ds = size(1:DS:KSE.time_dim, 1)  # downsampled time dimension

    # Number of random test inputs
    num_test_ic = 50

    # Prune data to get only the chaotic region
    prune_data = false
    prune_idx = KSE.time_dim ÷ 2
    t_prune = KSE.tspan[prune_idx-1:end]
    
    # Parameters of the initial condition
    ic_a = [0.8]
    ic_b = [0.2]

    num_ic_params = Int(length(ic_a) * length(ic_b))
    L = KSE.spatial_domain[2] - KSE.spatial_domain[1]  # length of the domain

    # Parameterized function for the initial condition
    u0 = (a,b) -> a * cos.((2*π*KSE.xspan)/L) .+ b * cos.((4*π*KSE.xspan)/L)  # initial condition

    # Store values
    Xtr = Vector{Matrix{Float64}}(undef, KSE.param_dim)  # training state data 
    Rtr = Vector{Matrix{Float64}}(undef, KSE.param_dim)  # training derivative data
    Vr = Vector{Matrix{Float64}}(undef, KSE.param_dim)  # POD basis
    Σr = Vector{Vector{Float64}}(undef, KSE.param_dim)  # singular values 

    for i in eachindex(KSE.diffusion_coeffs)
        μ = KSE.diffusion_coeffs[i]

        # Generate the FOM system matrices (ONLY DEPENDS ON μ)
        A, F = KSE.finite_diff_model(KSE, μ)

        # Store the training data 
        Xall = Vector{Matrix{Float64}}(undef, num_ic_params)
        Xdotall = Vector{Matrix{Float64}}(undef, num_ic_params)
        
        # Generate the data for all combinations of the initial condition parameters
        ic_combos = collect(Iterators.product(ic_a, ic_b))
        for (j, ic) in collect(enumerate(ic_combos))
            a, b = ic

            states = KSE.integrate_model(A, F, KSE.tspan, u0(a,b))
            if prune_data
                tmp = states[:, prune_idx:end]
                Xall[j] = tmp[:, 1:DS:end]  # downsample data
                tmp = (states[:, prune_idx:end] - states[:, prune_idx-1:end-1]) / KSE.Δt
                Xdotall[j] = tmp[:, 1:DS:end]  # downsample data
            else
                tmp = states[:, 2:end]
                Xall[j] = tmp[:, 1:DS:end]  # downsample data
                tmp = (states[:, 2:end] - states[:, 1:end-1]) / KSE.Δt
                Xdotall[j] = tmp[:, 1:DS:end]  # downsample data
            end
        end
        # Combine all initial condition data to form on big training data matrix
        Xtr[i] = reduce(hcat, Xall) 
        Rtr[i] = reduce(hcat, Xdotall)
        
        # Compute the POD basis from the training data
        tmp = svd(Xtr[i])
        Vr[i] = tmp.U
        Σr[i] = tmp.S
    end

    nice_orders_all = Vector{Vector{Int}}(undef, KSE.param_dim)
    for i in eachindex(KSE.diffusion_coeffs)
        nice_orders_all[i], _ = LnL.choose_ro(Σr[i]; en_low=-12)
    end
    nice_orders = Int.(round.(mean(nice_orders_all)))
    ro = nice_orders[2:5]

    options = LnL.EPHECOpInfOption(
        system=KSE_system,
        vars=KSE_VariableStructure,
        data=KSE_data,
        optim=KSE_optim,
        A_bnds=(-1000.0, 1000.0),
        ForH_bnds=(-100.0, 100.0),
    )
    op_ephec =  Array{LnL.Operators}(undef, KSE.param_dim)
    for i in eachindex(KSE.diffusion_coeffs)
        op_ephec[i] = LnL.opinf(Xtr[i], Vr[i][:, 1:ro[end]], options; Xdot=Rtr[i])
    end
    @test true # dummy test to make sure the testset runs

    options = LnL.EPSICOpInfOption(
        system=KSE_system,
        vars=KSE_VariableStructure,
        data=KSE_data,
        optim=KSE_optim,
        ϵ=1e-3,
        A_bnds=(-1000.0, 1000.0),
        ForH_bnds=(-100.0, 100.0),
    )
    op_epsic = Array{LnL.Operators}(undef, KSE.param_dim)
    for i in eachindex(KSE.diffusion_coeffs)
        op_epsic[i] = LnL.opinf(Xtr[i], Vr[i][:, 1:ro[end]], options; Xdot=Rtr[i])
    end
    @test true # dummy test to make sure the testset runs

    options = LnL.EPPOpInfOption(
        system=KSE_system,
        vars=KSE_VariableStructure,
        data=KSE_data,
        optim=KSE_optim,
        α=1e6,
        A_bnds=(-1000.0, 1000.0),
        ForH_bnds=(-100.0, 100.0),
    )
    op_epp =  Array{LnL.Operators}(undef, KSE.param_dim)
    for i in eachindex(KSE.diffusion_coeffs)
        op_epp[i] = LnL.opinf(Xtr[i], Vr[i][:, 1:ro[end]], options; Xdot=Rtr[i])
    end
    Fextract = LnL.extractF(op_epp[1].F, ro[2])
    _ =  LnL.EPConstraintResidual(Fextract, ro[2], "F"; with_mmt=false)
    _, _ =  LnL.EPConstraintResidual(Fextract, ro[2], "F"; with_mmt=true)

    Hextract = LnL.F2Hs(Fextract)
    _ =  LnL.EPConstraintResidual(Hextract, ro[2], "H"; with_mmt=false)
    _, _ =  LnL.EPConstraintResidual(Hextract, ro[2], "H"; with_mmt=true)
    @test true # dummy test to make sure the testset runs
end