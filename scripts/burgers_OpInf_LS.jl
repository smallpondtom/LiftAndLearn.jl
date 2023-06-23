"""
Burger's equation test case using Operator Inference.
"""


using CSV
using DataFrames
using LinearAlgebra
using Plots
using ProgressMeter
using Random
using SparseArrays
using Statistics

include("../src/model/Burgers.jl")
include("../src/LiftAndLearn.jl")
const LnL = LiftAndLearn


num_inputs = 10
rmax = 14

## First order Burger's equation setup
burger = Burgers(
    [0.0, 1.0], [0.0, 1.0], [0.1, 1.0],
    2^(-7), 1e-4, 10
)
options = LnL.OpInf_options(
    reproject=false,
    is_quad=true,
    N=1,
    Δt=1e-4,
    deriv_type="SI"
)
Utest = ones(burger.Tdim - 1, 1)  # Reference input/boundary condition for OpInf testing 

# Error Values 
k = 3
proj_err = zeros(rmax - k, burger.Pdim)
intru_state_err = zeros(rmax - k, burger.Pdim)
opinf_state_err = zeros(rmax - k, burger.Pdim)
intru_output_err = zeros(rmax - k, burger.Pdim)
opinf_output_err = zeros(rmax - k, burger.Pdim)

println("[INFO] Compute inferred and intrusive operators and calculate the errors")
@showprogress for i in 1:length(burger.μs)
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
        states = LnL.semiImplicitEuler(A, B, F, Urand[:, j], burger.t, burger.IC)
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
    op_int = LnL.intrusiveMR(op_burger, Vrmax, options)

    # Compute the inferred operators from the training data
    op_inf = LnL.inferOp(X, U, Y, Vrmax, Vrmax' * R, options)

    for j = 1+k:rmax
        Vr = Vrmax[:, 1:j]  # basis

        # Integrate the intrusive model
        Fint_extract = LnL.extractF(op_int.F, j)
        Xint = LnL.semiImplicitEuler(op_int.A[1:j, 1:j], op_int.B[1:j, :], Fint_extract, Utest, burger.t, Vr' * burger.IC)
        Yint = op_int.C[1:1, 1:j] * Xint

        # Integrate the inferred model
        Finf_extract = LnL.extractF(op_inf.F, j)
        Xinf = LnL.semiImplicitEuler(op_inf.A[1:j, 1:j], op_inf.B[1:j, :], Finf_extract, Utest, burger.t, Vr' * burger.IC)
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


proj_err = mean(proj_err, dims=2)
intru_state_err = mean(intru_state_err, dims=2)
intru_output_err = mean(intru_output_err, dims=2)
opinf_state_err = mean(opinf_state_err, dims=2)
opinf_output_err = mean(opinf_output_err, dims=2)

println("[INFO] Export data")

df = DataFrame(
    order=1+k:rmax,
    projection_err=vec(proj_err),
    intrusive_state_err=vec(intru_state_err),
    intrusive_output_err=vec(intru_output_err),
    inferred_state_err=vec(opinf_state_err),
    inferred_output_err=vec(opinf_output_err)
)
CSV.write("scripts/data/burger_data.csv", df)  # Write the data just in case

println("[INFO] Plot results")

# Projection error
plot(df.order, df.projection_err, marker=(:rect))
plot!(yscale=:log10, majorgrid=true, minorgrid=true, legend=false)
tmp = log10.(df.projection_err)
yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
xticks!(df.order)
xlabel!("dimension n")
ylabel!("avg projection error")
savefig("scripts/plots/burger_projerr.pdf")

# State errors
plot(df.order, df.intrusive_state_err, marker=(:cross, 10), label="intru")
plot!(df.order, df.inferred_state_err, marker=(:circle), ls=:dash, label="opinf")
plot!(yscale=:log10, majorgrid=true, minorgrid=true)
tmp = log10.(df.intrusive_state_err)
yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
xticks!(df.order)
xlabel!("dimension n")
ylabel!("avg error of states")
savefig("scripts/plots/burger_stateerr.pdf")

# Output errors
plot(df.order, df.intrusive_output_err, marker=(:cross, 10), label="intru")
plot!(df.order, df.inferred_output_err, marker=(:circle), ls=:dash, label="opinf")
plot!(majorgrid=true, minorgrid=true)
xticks!(df.order)
xlabel!("dimension n")
ylabel!("avg error of outputs")
savefig("scripts/plots/burger_outputerr.pdf")

println("[INFO] Done")
