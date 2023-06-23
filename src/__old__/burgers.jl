module burgers

using CSV
using DataFrames
using LinearAlgebra
using Plots
using Random
using SparseArrays
using Statistics

include("integrator.jl")
include("intrusiveROM.jl")
include("opinf.jl")
include("utils.jl")

mutable struct burger_params
    Omega::Vector{Float64}  # spatial domain
    T::Vector{Float64}  # temporal domain
    D::Vector{Float64}  # parameter domain
    Δx::Float64  # spatial grid size
    Δt::Float64  # temporal step size
    Ubc::Matrix{Float64}  # boundary condition (input)
    IC::Matrix{Float64}  # initial condition
    x::Vector{Float64}  # spatial grid points
    t::Vector{Float64}  # temporal points
    μs::Vector{Float64}  # parameter vector
    Xdim::Int64  # spatial dimension
    Tdim::Int64  # temporal dimension
    Pdim::Int64  # parameter dimension

    function burger_params(Omega, T, D, Δx, Δt, Pdim)
        x = (Omega[1]:Δx:Omega[2])  # include boundary conditions
        t = T[1]:Δt:T[2]
        μs = Pdim == 1 ? D[1] : range(D[1], D[2], Pdim)
        Xdim = length(x)
        Tdim = length(t)
        Ubc = ones(Tdim, 1)
        IC = zeros(Xdim, 1)
        new(Omega, T, D, Δx, Δt, Ubc, IC, x, t, μs, Xdim, Tdim, Pdim)
    end
end

function generateABFmatrix(N, μ, Δx)
    # Create A matrix
    A = diagm(0 => (-2) * ones(N), 1 => ones(N - 1), -1 => ones(N - 1)) * μ / Δx^2
    A[1, 1:2] = [1, 0]
    A[end, end-1:end] = [0, 1]

    # Create F matrix
    S = Int(N * (N + 1) / 2)
    if N >= 3
        Fval = repeat([1.0, -1.0], outer=N - 2)
        row_i = repeat(2:(N-1), inner=2)
        # seq = [x^2/2+3*x/2 for x in 1:(N-1)]
        seq = Int.([2 + (N + 1) * (x - 1) - x * (x - 1) / 2 for x in 1:(N-1)])
        col_i = vcat(seq[1], repeat(seq[2:end-1], inner=2), seq[end])
        F = sparse(row_i, col_i, Fval, N, S) / 2 / Δx
    else
        F = zeros(N, S)
    end

    # Create B matrix
    B = [1; zeros(N - 2, 1); -1]

    return A, B, F
end

### Test 0 
function test0(rmax)
    @show rmax
    # First order Burger's equation setup
    burger = burger_params(
        [0.0, 1.0], [0.0, 1.0], [0.1, 1.0],
        2^(-7), 1e-4, 10
    )
    burger.Ubc = rand(burger.Tdim - 1, 1)  # input matrix

    # Some parameters for operator inference
    params = Dict("with_R" => false, "NL" => true, "dt_type" => "SI")

    # Error Values
    k = 3
    proj_err = zeros(rmax - k, burger.Pdim)
    intru_state_err = zeros(rmax - k, burger.Pdim)
    opinf_state_err = zeros(rmax - k, burger.Pdim)
    intru_output_err = zeros(rmax - k, burger.Pdim)
    opinf_output_err = zeros(rmax - k, burger.Pdim)

    for (j, μ) in enumerate(burger.μs)
        A, B, F = generateABFmatrix(burger.Xdim, μ, burger.Δx)
        C = ones(1, burger.Xdim) / burger.Xdim  # the output y(t;μ) is defined as the average of the components of x(t;μ)

        # Compute the states with semi-implicit Euler
        state = semiImplicitEuler(A, B, F, burger.Ubc, burger.t, burger.IC)
        X = state[:, 2:end]

        # Compute the SVD for the POD basis
        Z = svd(X)
        Vrmax = Z.U[:, 1:rmax]

        # Compute the output of the system
        Y = C * X
        Ys = C * state

        # Compute the values for the intrusive model
        Aint, Bint, Cint, Fint = intrusiveMR(A, B, C, Vrmax, F)

        # Compute the RHS for the operator inference based on the intrusive operators
        if params["with_R"]
            # idx = 2:burger.Tdim
            # Xn = X[:,idx]
            # Un = burger.Ubc[idx,:]
            # Yn = Y[:,idx]
            # Xhat = Vrmax' * Xn
            # Xhat2 = squareMatStates(Xhat)
            # Xdot = Aint * Xhat + Fint * Xhat2 + Bint * Un'
            Xdot = (state[:, 2:end] - state[:, 1:end-1]) / burger.Δt
            Ainf, Binf, Cinf, Finf = inferOp(X, burger.Ubc, Y, Vrmax, burger.Δt, Vrmax' * Xdot, params)
        else
            Ainf, Binf, Cinf, Finf = inferOp(X, burger.Ubc, Y, Vrmax, burger.Δt, 0, params)
        end

        for i = 1+k:rmax
            Vr = Vrmax[:, 1:i]

            # Integrate the intrusive model
            Fint_extract = extractF(Fint, i)
            Xint = semiImplicitEuler(Aint[1:i, 1:i], Bint[1:i, :], Fint_extract, burger.Ubc, burger.t, Vr' * burger.IC)
            Yint = Cint[1:1, 1:i] * Xint

            # Integrate the inferred model
            Finf_extract = extractF(Finf, i)
            Xinf = semiImplicitEuler(Ainf[1:i, 1:i], Binf[1:i, :], Finf_extract, burger.Ubc, burger.t, Vr' * burger.IC)
            Yinf = Cinf[1:1, 1:i] * Xinf

            # Compute errors
            PE, ISE, IOE, OSE, OOE = compError(state, Ys, Xint, Yint, Xinf, Yinf, Vr)

            # Sum of error values
            proj_err[i-k, j] = PE
            intru_state_err[i-k, j] = ISE
            intru_output_err[i-k, j] = IOE
            opinf_state_err[i-k, j] = OSE
            opinf_output_err[i-k, j] = OOE
        end
    end

    df = DataFrame(
        order=1+k:rmax,
        projection_err=vec(mean(proj_err, dims=2)),
        intrusive_state_err=vec(mean(intru_state_err, dims=2)),
        intrusive_output_err=vec(mean(intru_output_err, dims=2)),
        inferred_state_err=vec(mean(opinf_state_err, dims=2)),
        inferred_output_err=vec(mean(opinf_output_err, dims=2))
    )
    CSV.write("src/data/burger_data.csv", df)  # Write the data just in case

    # Projection error
    plot(df.order, df.projection_err, marker=(:rect))
    plot!(yscale=:log10, majorgrid=true, minorgrid=true, legend=false)
    tmp = log10.(df.projection_err)
    yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
    xticks!(df.order)
    xlabel!("dimension n")
    ylabel!("avg projection error")
    savefig("src/plots/burger_test0_projerr.pdf")

    # State errors
    plot(df.order, df.intrusive_state_err, marker=(:cross, 10), label="intru")
    plot!(df.order, df.inferred_state_err, marker=(:circle), ls=:dash, label="opinf")
    plot!(yscale=:log10, majorgrid=true, minorgrid=true)
    tmp = log10.(df.intrusive_state_err)
    yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
    xticks!(df.order)
    xlabel!("dimension n")
    ylabel!("avg error of states")
    savefig("src/plots/burger_test0_stateerr.pdf")

    # Output errors
    plot(df.order, df.intrusive_output_err, marker=(:cross, 10), label="intru")
    plot!(df.order, df.inferred_output_err, marker=(:circle), ls=:dash, label="opinf")
    plot!(majorgrid=true, minorgrid=true)
    xticks!(df.order)
    xlabel!("dimension n")
    ylabel!("avg error of outputs")
    savefig("src/plots/burger_test0_outputerr.pdf")

end



### Test where we only use one random input (same test as 1D heat)
function test1(r)
    @show r
    # First order Burger's equation setup
    burger = burger_params(
        [0.0, 1.0], [0.0, 1.0], [0.1, 1.0],
        2^(-7), 1e-4, 10
    )

    burger.Ubc = rand(Float64, (burger.Tdim, 1))  # input matrix

    # Full state, output, POD basis
    Xfull = Vector{Matrix{Float64}}(undef, burger.Pdim)
    Yfull = Vector{Matrix{Float64}}(undef, burger.Pdim)
    Afull = Vector{Matrix{Float64}}(undef, burger.Pdim)
    Bfull = Vector{Matrix{Float64}}(undef, burger.Pdim)
    Cfull = Vector{Matrix{Float64}}(undef, burger.Pdim)
    Ffull = Vector{Matrix{Float64}}(undef, burger.Pdim)
    pod_bases = Vector{Matrix{Float64}}(undef, burger.Pdim)

    # Some parameters for operator inference
    params = Dict("with_R" => false, "NL" => true, "dt_type" => "SI")

    for (j, μ) in enumerate(burger.μs)
        A, B, F = generateABFmatrix(burger.Xdim, μ, burger.Δx)
        C = ones(1, burger.Xdim) / burger.Xdim  # the output y(t;μ) is defined as the average of the components of x(t;μ)

        # Compute the states with semi-implicit Euler
        X = semiImplicitEuler(A, B, F, burger.Ubc, burger.t, burger.IC)
        Xfull[j] = X

        # Compute the SVD for the POD basis
        Z = svd(X)
        Vr = Z.U[:, 1:r]
        pod_bases[j] = Vr

        # Compute the output of the system
        Y = C * X
        Yfull[j] = Y

        # Store Operators
        Afull[j] = A
        Bfull[j] = B
        Cfull[j] = C
        Ffull[j] = F
    end

    # Error Values 
    k = 3
    proj_err = zeros(r - k, 1)
    intru_state_err = zeros(r - k, 1)
    opinf_state_err = zeros(r - k, 1)
    intru_output_err = zeros(r - k, 1)
    opinf_output_err = zeros(r - k, 1)

    for i in 1+k:r, j in 1:burger.Pdim
        # Full model operators
        X = Xfull[j]
        Y = Yfull[j]
        A = Afull[j]
        B = Bfull[j]
        C = Cfull[j]
        F = Ffull[j]

        # POD Basis
        Vr = pod_bases[j][:, 1:i]

        # Compute the values for the intrusive model
        Aint, Bint, Cint, Fint = intrusiveMR(A, B, C, Vr, F)

        # Compute the RHS for the operator inference based on the intrusive operators
        if params["with_R"]
            idx = 1:burger.Tdim-1
            Xn = X[:, idx]
            Un = burger.Ubc[idx, :]
            Yn = Y[:, idx]
            Xhat = Vr' * Xn
            Xhat2 = squareMatStates(Xhat)
            Xdot = Aint * Xhat + Fint * Xhat2 + Bint * Un'

            Ainf, Binf, Cinf, Finf = inferOp(Xn, Un, Yn, Vr, burger.Δt, Xdot, params)
        else
            Ainf, Binf, Cinf, Finf = inferOp(X, burger.Ubc, Y, Vr, burger.Δt, 0, params)
        end

        # Integrate the intrusive model
        Xint = semiImplicitEuler(Aint, Bint, Fint, burger.Ubc, burger.t, Vr' * burger.IC)
        Yint = Cint * Xint

        # Integrate the inferred model
        Xinf = semiImplicitEuler(Ainf, Binf, Finf, burger.Ubc, burger.t, Vr' * burger.IC)
        Yinf = Cinf * Xinf

        # Compute errors
        PE, ISE, IOE, OSE, OOE = compError(X, Y, Xint, Yint, Xinf, Yinf, Vr)

        # Sum of error values
        proj_err[i-k] += PE / burger.Pdim
        intru_state_err[i-k] += ISE / burger.Pdim
        intru_output_err[i-k] += IOE / burger.Pdim
        opinf_state_err[i-k] += OSE / burger.Pdim
        opinf_output_err[i-k] += OOE / burger.Pdim
    end
    df = DataFrame(
        order=1+k:r,
        projection_err=vec(proj_err),
        intrusive_state_err=vec(intru_state_err),
        intrusive_output_err=vec(intru_output_err),
        inferred_state_err=vec(opinf_state_err),
        inferred_output_err=vec(opinf_output_err)
    )
    CSV.write("src/data/burger_data.csv", df)  # Write the data just in case

    # Projection error
    plot(df.order, df.projection_err, marker=(:rect))
    plot!(yscale=:log10, majorgrid=true, minorgrid=true, legend=false)
    tmp = log10.(df.projection_err)
    yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
    xticks!(df.order)
    xlabel!("dimension n")
    ylabel!("avg projection error")
    savefig("src/plots/burger_test1_projerr.pdf")

    # State errors
    plot(df.order, df.intrusive_state_err, marker=(:cross, 10), label="intru")
    plot!(df.order, df.inferred_state_err, marker=(:circle), ls=:dash, label="opinf")
    plot!(yscale=:log10, majorgrid=true, minorgrid=true)
    tmp = log10.(df.intrusive_state_err)
    yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
    xticks!(df.order)
    xlabel!("dimension n")
    ylabel!("avg error of states")
    savefig("src/plots/burger_test1_stateerr.pdf")

    # Output errors
    plot(df.order, df.intrusive_output_err, marker=(:cross, 10), label="intru")
    plot!(df.order, df.inferred_output_err, marker=(:circle), ls=:dash, label="opinf")
    plot!(majorgrid=true, minorgrid=true)
    xticks!(df.order)
    xlabel!("dimension n")
    ylabel!("avg error of outputs")
    savefig("src/plots/burger_test1_outputerr.pdf")
end


### Second test function with the exact same simulation from Benjamin's paper
function test2(num_inputs, rmax)
    @show num_inputs rmax
    ## First order Burger's equation setup
    burger = burger_params(
        [0.0, 1.0], [0.0, 1.0], [0.1, 1.0],
        2^(-7), 1e-4, 10
    )
    params = Dict("with_R" => true, "NL" => true, "dt_type" => "SI")  # Some parameters for operator inference
    Utest = ones(burger.Tdim - 1, 1)  # Reference input/boundary condition for OpInf testing 

    # Error Values 
    k = 3
    proj_err = zeros(rmax - k, burger.Pdim)
    intru_state_err = zeros(rmax - k, burger.Pdim)
    opinf_state_err = zeros(rmax - k, burger.Pdim)
    intru_output_err = zeros(rmax - k, burger.Pdim)
    opinf_output_err = zeros(rmax - k, burger.Pdim)

    for (i, μ) in enumerate(burger.μs)
        ## Create testing data
        A, B, F = generateABFmatrix(burger.Xdim, μ, burger.Δx)
        C = ones(1, burger.Xdim) / burger.Xdim
        Xtest = semiImplicitEuler(A, B, F, Utest, burger.t, burger.IC)
        Ytest = C * Xtest

        ## training data for inferred dynamical models
        Urand = rand(burger.Tdim - 1, num_inputs)
        Xall = Vector{Matrix{Float64}}(undef, num_inputs)
        Xdotall = Vector{Matrix{Float64}}(undef, num_inputs)
        for j in 1:num_inputs
            states = semiImplicitEuler(A, B, F, Urand[:, j], burger.t, burger.IC)
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

        # Compute the values for the intrusive model from the basis of the training data
        Aint, Bint, Cint, Fint = intrusiveMR(A, B, C, Vrmax, F)

        # Compute the inferred operators from the training data
        Ainf, Binf, Cinf, Finf = inferOp(X, U, Y, Vrmax, burger.Δt, Vrmax' * R, params)

        for j = 1+k:rmax
            Vr = Vrmax[:, 1:j]  # basis

            # Integrate the intrusive model
            Fint_extract = extractF(Fint, j)
            Xint = semiImplicitEuler(Aint[1:j, 1:j], Bint[1:j, :], Fint_extract, Utest, burger.t, Vr' * burger.IC)
            Yint = Cint[1:1, 1:j] * Xint

            # Integrate the inferred model
            Finf_extract = extractF(Finf, j)
            Xinf = semiImplicitEuler(Ainf[1:j, 1:j], Binf[1:j, :], Finf_extract, Utest, burger.t, Vr' * burger.IC)
            Yinf = Cinf[1:1, 1:j] * Xinf

            # Compute errors
            PE, ISE, IOE, OSE, OOE = compError(Xtest, Ytest, Xint, Yint, Xinf, Yinf, Vr)

            # Sum of error values
            proj_err[j-k, i] = PE
            intru_state_err[j-k, i] = ISE
            intru_output_err[j-k, i] = IOE
            opinf_state_err[j-k, i] = OSE
            opinf_output_err[j-k, i] = OOE
        end
        println("iteration $i done.")
    end

    proj_err = mean(proj_err, dims=2)
    intru_state_err = mean(intru_state_err, dims=2)
    intru_output_err = mean(intru_output_err, dims=2)
    opinf_state_err = mean(opinf_state_err, dims=2)
    opinf_output_err = mean(opinf_output_err, dims=2)

    df = DataFrame(
        order=1+k:rmax,
        projection_err=vec(proj_err),
        intrusive_state_err=vec(intru_state_err),
        intrusive_output_err=vec(intru_output_err),
        inferred_state_err=vec(opinf_state_err),
        inferred_output_err=vec(opinf_output_err)
    )
    CSV.write("src/data/burger_data.csv", df)  # Write the data just in case

    # Projection error
    plot(df.order, df.projection_err, marker=(:rect))
    plot!(yscale=:log10, majorgrid=true, minorgrid=true, legend=false)
    tmp = log10.(df.projection_err)
    yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
    xticks!(df.order)
    xlabel!("dimension n")
    ylabel!("avg projection error")
    savefig("src/plots/burger_test2_projerr.pdf")

    # State errors
    plot(df.order, df.intrusive_state_err, marker=(:cross, 10), label="intru")
    plot!(df.order, df.inferred_state_err, marker=(:circle), ls=:dash, label="opinf")
    plot!(yscale=:log10, majorgrid=true, minorgrid=true)
    tmp = log10.(df.intrusive_state_err)
    yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
    xticks!(df.order)
    xlabel!("dimension n")
    ylabel!("avg error of states")
    savefig("src/plots/burger_test2_stateerr.pdf")

    # Output errors
    plot(df.order, df.intrusive_output_err, marker=(:cross, 10), label="intru")
    plot!(df.order, df.inferred_output_err, marker=(:circle), ls=:dash, label="opinf")
    plot!(majorgrid=true, minorgrid=true)
    xticks!(df.order)
    xlabel!("dimension n")
    ylabel!("avg error of outputs")
    savefig("src/plots/burger_test2_outputerr.pdf")
end

end

