"""
Fitzhugh-Nagumo Equation test case using Lift & Learn.
"""

#=========#
## Setups 
#=========#
using BlockDiagonals
using Kronecker
using LinearAlgebra
using NaNStatistics
using CairoMakie
using ProgressMeter
using Random
using SparseArrays
using Statistics
using UniqueKronecker
using PolynomialModelReductionDataset: FitzHughNagumoModel
using IncrementalSVD

using LiftAndLearn
const LnL = LiftAndLearn

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"Streaming-OpInf/") : joinpath(pwd(), "scripts/Streaming-OpInf/")

#================#
## Generate model
#================#
start = time()
Ω = (0.0, 1.0); dt = 1e-4; Nx = 2^9
fhn = FitzHughNagumoModel(
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

#========================================================#
## Get the full-order model operators for intrusive model
#========================================================#
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

#========================#
## Generate training data
#========================#
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
with_theme(theme_latexfonts()) do 
    tspan_sampled = fhn.tspan[1:DS:end]
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="t", ylabel="Value", title="Time Series",
              xlabelsize=25, ylabelsize=25, titlesize=25, xticklabelsize=20, yticklabelsize=20)
    # Plot initial data
    lines!(ax, tspan_sampled, Xtrain[1][1, :], label = "")
    lines!(ax, tspan_sampled, Xtrain[1][gp + 1, :], label = "")
    # Loop to add more lines
    for i in 2:8
        lines!(ax, tspan_sampled, Xtrain[i][1, :], label = "")
        lines!(ax, tspan_sampled, Xtrain[i][gp + 1, :], label = "")
    end
    # Plot final lines
    lines!(ax, tspan_sampled, Xtrain[9][1, :], label = "")
    lines!(ax, tspan_sampled, Xtrain[9][gp + 1, :], label = "")
    display(fig)
    save(joinpath(FILEPATH, "plots/fhn/fhn_training_time_series.png"), fig)
    # Contour plots
    fig2 = Figure(size=(900, 400))
    ax1 = Axis(fig2[1, 1], xlabel = "t", ylabel = "x", title = "x1",
                xlabelsize=25, ylabelsize=25, titlesize=25, xticklabelsize=20, yticklabelsize=20)
    ax2 = Axis(fig2[1, 3], xlabel = "t", ylabel = "x", title = "x2",
                xlabelsize=25, ylabelsize=25, titlesize=25, xticklabelsize=20, yticklabelsize=20)
    # Contour plot for x1
    cf = contourf!(ax1, fhn.xspan, tspan_sampled, Xtrain[1][1:gp, :], colormap = :viridis)
    Colorbar(fig2[1, 2], cf)
    # Contour plot for x2
    cf = contourf!(ax2, fhn.xspan, tspan_sampled, Xtrain[1][gp + 1:end, :], colormap = :viridis)
    Colorbar(fig2[1, 4], cf)
    display(fig2)
    save(joinpath(FILEPATH, "plots/fhn/fhn_training_contour.png"), fig2)
end

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


## Take the SVD for each variable using iSVD
iW1 = iSVD(x1=Wtr[1:gp,1], algo=:baker)
full_increment!(iW1, Wtr[1:gp,2:end], tol=1e-12, verbose=true)

##
iW2 = iSVD(x1=Wtr[gp+1:2*gp,1], algo=:baker)
full_increment!(iW2, Wtr[gp+1:2*gp,2:end], tol=1e-12, verbose=true)

##
iW3 = iSVD(x1=Wtr[2*gp+1:end,1], algo=:baker)
full_increment!(iW3, Wtr[2*gp+1:end,2:end], tol=1e-12, verbose=true)

## Batch SVD
W1 = svd(Wtr[1:gp, :])
W2 = svd(Wtr[gp+1:2*gp, :])
W3 = svd(Wtr[2*gp+1:end, :])

#======================#
## Plot Singular Values
#======================#
with_theme(theme_latexfonts()) do
    r = 20
    fig0 = Figure(fontsize=35, size=(1200,900), backgroundcolor="#FFFFFF")
    ax1 = Axis(fig0[1,1], title="First 20 Singular Values", xlabel="", ylabel="", yscale=log10, 
                xticksvisible=false, xticklabelsvisible=false)
    ax2 = Axis(fig0[2,1], title="", xlabel="", ylabel=L"Singular Value, $\sigma_i$", yscale=log10,
                xticksvisible=false, xticklabelsvisible=false)
    ax3 = Axis(fig0[3,1], title="", xlabel=L"Index, $i$", ylabel="", yscale=log10)
    # Var 1
    scatterlines!(ax1, 1:r, W1.S[1:r], color=:black, linewidth=3, label="SVD")
    scatterlines!(ax1, 1:r, iW1.Σ[1:r], color=:red, linewidth=2, linestyle=:dash, label="iSVD")
    # Var 2
    scatterlines!(ax2, 1:r, W2.S[1:r], color=:black, linewidth=3, label="")
    scatterlines!(ax2, 1:r, iW2.Σ[1:r], color=:red, linewidth=2, linestyle=:dash, label="")
    # Var 3
    scatterlines!(ax3, 1:r, W3.S[1:r], color=:black, linewidth=3, label="")
    scatterlines!(ax3, 1:r, iW3.Σ[1:r], color=:red, linewidth=2, linestyle=:dash, label="")
    axislegend(ax1, labelsize=35, position=:rt)
    display(fig0)
    save(joinpath(FILEPATH, "plots/fhn/singular_values.png"), fig0)
end

## dictionary with intrusive and LnL errors as matrices (9-by-4)
train_err = Dict(
    :intrusive => zeros(length(α_train), size(mode_req, 1)),
    :inferred => zeros(length(α_train), size(mode_req, 1)),
    :streaming => zeros(length(α_train), size(mode_req, 1))
)
test1_err = Dict(
    :intrusive => zeros(N_tests, size(mode_req, 1)),
    :inferred => zeros(N_tests, size(mode_req, 1)),
    :streaming => zeros(N_tests, size(mode_req, 1))
)
test2_err = Dict(
    :intrusive => zeros(N_tests, size(mode_req, 1)),
    :inferred => zeros(N_tests, size(mode_req, 1)),
    :streaming => zeros(N_tests, size(mode_req, 1))
)

@showprogress for (i, row) in enumerate(eachrow(mode_req))
    # Batch SVD
    r1, r2, r3 = row
    Vr1 = W1.U[:, 1:r1]
    Vr2 = W2.U[:, 1:r2]
    Vr3 = W3.U[:, 1:r3]
    Vr = BlockDiagonal([Vr1, Vr2, Vr3])

    # iSVD
    iVr1 = iW1.Q[:, 1:r1]
    iVr2 = iW2.Q[:, 1:r2]
    iVr3 = iW3.Q[:, 1:r3]
    iVr = BlockDiagonal([iVr1, iVr2, iVr3])

    # Streamify the data
    W_stream = LnL.streamify(iVr' * Wtr, 1)
    U_stream = LnL.streamify(Utr, 1)
    Y_stream = LnL.streamify(Ytr, 1)
    Rt = LnL.reproject(iVr' * Wtr, iVr, Utr, lifter, fomOps, options)
    Wdot_stream = LnL.streamify(Rt', 1)
    num_of_streams = length(W_stream)

    infOps = LnL.opinf(Wtr, Vr, lifter, fomOps, options; U=Utr, Y=Ytr)
    infOps.A2 = duplicate(infOps.A2u, 2)
    intruOps = LnL.pod(fomLinOps, Vr, options.system)

    # Compute the streaming-OpInf
    γs = 1e-9
    γo = 1e-12
    state_stream, output_stream = LnL.StreamingOpInf(options=options, n=sum(row), m=size(Utr,2), l=size(Ytr,1), γs=γs, γo=γo, algorithm=:iQRRLS)
    LnL.stream_all!(state_stream, W_stream, Wdot_stream; U=U_stream, verbose=false)
    LnL.stream_output_all!(output_stream, W_stream, Y_stream; verbose=false)
    streamOps = LnL.Operators()
    LnL.unpack_operators!(streamOps, state_stream.cache.O', state_stream.termination_settings[:dims], state_stream.termination_settings[:syms])
    streamOps.C = output_stream.cache.O'
    streamOps.A2 = duplicate(streamOps.A2u, 2)

    finf = (x, u) -> infOps.A * x + infOps.B * u + infOps.A2 * (x ⊗ x) + (infOps.N * x) * u[1] + infOps.K
    fint = (x, u) -> intruOps.A * x  + intruOps.B * u + intruOps.A2 * (x ⊗ x) + (intruOps.N*x)*u[1] + intruOps.K
    fstream = (x, u) -> streamOps.A * x + streamOps.B * u + streamOps.A2 * (x ⊗ x) + (streamOps.N * x) * u[1] + streamOps.K

    k, l, p = 0, 0, 0
    for (X, U) in zip(Xtrain, Utrain_all)
        Xint = fhn.integrate_model(fhn.tspan, Vr' * fhn.IC_lift, U; functional=fint)
        Xinf = fhn.integrate_model(fhn.tspan, Vr' * fhn.IC_lift, U; functional=finf)
        Xstream = fhn.integrate_model(fhn.tspan, iVr' * fhn.IC_lift, U; functional=fstream)

        # Down sample 
        Xint = Xint[:, 1:DS:end]
        Xinf = Xinf[:, 1:DS:end]
        Xstream = Xstream[:, 1:DS:end]

        train_err[:intrusive][k+=1, i] = LnL.rel_state_error(X, Xint, Vr)
        train_err[:inferred][l+=1, i] = LnL.rel_state_error(X, Xinf, Vr)
        train_err[:streaming][p+=1, i] = LnL.rel_state_error(X, Xstream, iVr)
    end

    k, l, p = 0, 0, 0
    for (X, U) in zip(Xtest1, Utest1_all)
        Xint = fhn.integrate_model(fhn.tspan, Vr' * fhn.IC_lift, U; functional=fint)
        Xinf = fhn.integrate_model(fhn.tspan, Vr' * fhn.IC_lift, U; functional=finf)
        Xstream = fhn.integrate_model(fhn.tspan, iVr' * fhn.IC_lift, U; functional=fstream)

        # Down sample
        Xint = Xint[:, 1:DS:end]
        Xinf = Xinf[:, 1:DS:end]
        Xstream = Xstream[:, 1:DS:end]

        test1_err[:intrusive][k+=1, i] = LnL.rel_state_error(X, Xint, Vr)
        test1_err[:inferred][l+=1, i] = LnL.rel_state_error(X, Xinf, Vr)
        test1_err[:streaming][p+=1, i] = LnL.rel_state_error(X, Xstream, iVr)
    end

    k, l, p = 0, 0, 0
    for (X, U) in zip(Xtest2, Utest2_all)
        Xint = fhn.integrate_model(fhn.tspan, Vr' * fhn.IC_lift, U; functional=fint)
        Xinf = fhn.integrate_model(fhn.tspan, Vr' * fhn.IC_lift, U; functional=finf)
        Xstream = fhn.integrate_model(fhn.tspan, iVr' * fhn.IC_lift, U; functional=fstream)

        # Down sample
        Xint = Xint[:, 1:DS:end]
        Xinf = Xinf[:, 1:DS:end]
        Xstream = Xstream[:, 1:DS:end]

        test2_err[:intrusive][k+=1, i] = LnL.rel_state_error(X, Xint, Vr)
        test2_err[:inferred][l+=1, i] = LnL.rel_state_error(X, Xinf, Vr)
        test2_err[:streaming][p+=1, i] = LnL.rel_state_error(X, Xstream, iVr)
    end
end
dims = sum(mode_req, dims=2)

## Plot results
err_intru = vec(median(train_err[:intrusive], dims=1))
err_infer = vec(median(train_err[:inferred], dims=1))
err_stream = vec(median(train_err[:streaming], dims=1))

t1err_intru = vec(median(test1_err[:intrusive], dims=1))
t1err_infer = vec(median(test1_err[:inferred], dims=1))
t1err_stream = vec(median(test1_err[:streaming], dims=1))

t2err_intru = vec(nanmedian(test2_err[:intrusive], dims=1))
t2err_infer = vec(nanmedian(test2_err[:inferred], dims=1))
t2err_stream = vec(nanmedian(test2_err[:streaming], dims=1))

## Training Plot
dims = vec(dims)
with_theme(theme_latexfonts()) do
    fig = Figure(size=(1200, 1000))
    ax1 = Axis(fig[1, 1], title = "Median Error over Training Trajectories", 
            xlabel="", ylabel = "",
            yscale=log10, xticks=vec(dims), titlesize=32, xticklabelsize=20, yticklabelsize=20,
            xlabelsize=30, ylabelsize=30, xticklabelsvisible=false, xticksvisible=false)

    # Plot each line with the appropriate markers and labels
    scatterlines!(ax1, dims, err_intru, marker=:cross, markersize=25, linewidth=6, label = "POD")
    scatterlines!(ax1, dims, err_infer, marker=:circle, markersize=25, linewidth=5, linestyle = :dash, label = "LnL")
    scatterlines!(ax1, dims, err_stream, marker=:diamond, markersize=25, linewidth=5, linestyle = :dot, label = "Streaming-LnL")

    # Set y-axis ticks
    tmp = log10.(err_infer)

    # Test 1 Plot
    ax2 = Axis(fig[2, 1], title = "Median Test1 Error over New Trajectories",
            xlabel="", ylabel = "Relative State Error",
            yscale=log10, xticks=vec(dims), titlesize=32, xticklabelsize=20, yticklabelsize=20,
            xlabelsize=30, ylabelsize=30, xticklabelsvisible=false, xticksvisible=false)

    # Plot each line with the appropriate markers and labels
    scatterlines!(ax2, dims, t1err_intru, marker = :xcross, markersize = 25, label = "POD", linewidth=6)
    scatterlines!(ax2, dims, t1err_infer, marker = :circle, linestyle = :dash, label = "LnL", markersize = 25, linewidth=5)
    scatterlines!(ax2, dims, t1err_stream, marker = :diamond, linestyle = :dot, label = "Streaming-LnL", markersize = 25, linewidth=5)

    # Set y-axis ticks
    tmp = log10.(t1err_infer)
    # axislegend(ax2)

    # Test 2 Plot
    ax3 = Axis(fig[3, 1], title = "Median Test2 Error over New Trajectories",
            xlabel=L"dimension $r$", ylabel = "",
            yscale=log10, xticks=vec(dims), titlesize=32, xticklabelsize=20, yticklabelsize=20,
            xlabelsize=30, ylabelsize=30)

    # Plot each line with the appropriate markers and labels
    scatterlines!(ax3, dims, t2err_intru, marker = :xcross, markersize = 25, label = "POD", linewidth=6)
    scatterlines!(ax3, dims, t2err_infer, marker = :circle, linestyle = :dash, label = "LnL", markersize = 25, linewidth=5)
    scatterlines!(ax3, dims, t2err_stream, marker = :diamond, linestyle = :dot, label = "Streaming-LnL", markersize = 25, linewidth=5)

    # Set y-axis ticks
    tmp = log10.(t2err_infer)
    axislegend(ax3, labelsize=35, position=:lb)

    # Save figures
    display(fig)
    save(joinpath(FILEPATH, "plots/fhn/fhn_LnL_error.png"), fig)
end