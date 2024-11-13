"""
Burger's equation test case using Operator Inference.
"""

#===============#
## Load Packages
#===============#
using CSV
using DataFrames
using LinearAlgebra
using Plots
using ProgressMeter
using Random
using Statistics
using UniqueKronecker
using PolynomialModelReductionDataset
const Pomoreda = PolynomialModelReductionDataset

#==========#
## Load LnL
#==========#
using LiftAndLearn
const LnL = LiftAndLearn

#=============#
## Some option
#=============#
SAVEFIG = false
SAVEDATA = false

#==================#
## Setup burgers eq
#==================#
Ω = (0.0, 1.0)
Nx = 2^7; dt = 1e-4
burger = Pomoreda.BurgersModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    diffusion_coeffs=range(0.1, 1.0, length=10), BC=:dirichlet,
)
num_inputs = 10

options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=[1,2],
        control=1,
        output=1
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=dt,
        deriv_type="SI"
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
    with_tol=true,
    pinv_tol=1e-2,
)
Utest = ones(burger.time_dim, 1);  # Reference input/boundary condition for OpInf testing 

# Placeholders
Xtrain = Vector{Matrix{Float64}}(undef, burger.param_dim)
Rtrain = Vector{Matrix{Float64}}(undef, burger.param_dim)
Ytrain = Vector{Matrix{Float64}}(undef, burger.param_dim)'
Utrain = Vector{Matrix{Float64}}(undef, burger.param_dim)'

Xtests = Vector{Matrix{Float64}}(undef, burger.param_dim)
Ytests = Vector{Matrix{Float64}}(undef, burger.param_dim)'

A_full = Vector{Matrix{Float64}}(undef, burger.param_dim)
F_full = Vector{Matrix{Float64}}(undef, burger.param_dim)
B_full = Vector{Matrix{Float64}}(undef, burger.param_dim)
C_full = Vector{Matrix{Float64}}(undef, burger.param_dim)

A_intru = Vector{Matrix{Float64}}(undef, burger.param_dim)
F_intru = Vector{Matrix{Float64}}(undef, burger.param_dim)
B_intru = Vector{Matrix{Float64}}(undef, burger.param_dim)
C_intru = Vector{Matrix{Float64}}(undef, burger.param_dim)

A_opinf = Vector{Matrix{Float64}}(undef, burger.param_dim)
F_opinf = Vector{Matrix{Float64}}(undef, burger.param_dim)
B_opinf = Vector{Matrix{Float64}}(undef, burger.param_dim)
C_opinf = Vector{Matrix{Float64}}(undef, burger.param_dim)

#===========================#
## Generate the data
#===========================#
@info "Generate the data"
@showprogress for i in eachindex(burger.diffusion_coeffs)
    μ = burger.diffusion_coeffs[i]

    # Obtain full operators
    A, F, B = burger.finite_diff_model(burger, μ)
    C = ones(1, burger.spatial_dim) / burger.spatial_dim
    A_full[i] = A
    F_full[i] = F
    B_full[i] = B
    C_full[i] = C

    # Integrate to obtain data
    Xtest = burger.integrate_model(burger.tspan, burger.IC, Utest; linear_matrix=A,
                                   control_matrix=B, quadratic_matrix=F, system_input=true)
    Ytest = C * Xtest
    Xtests[i] = Xtest
    Ytests[i] = Ytest

    # training data for inferred dynamical models
    Urand = rand(burger.time_dim, num_inputs)
    Xall = Vector{Matrix{Float64}}(undef, num_inputs)
    Xdotall = Vector{Matrix{Float64}}(undef, num_inputs)
    for j in 1:num_inputs
        states = burger.integrate_model(burger.tspan, burger.IC, Urand[:, j], linear_matrix=A,
                                        control_matrix=B, quadratic_matrix=F, system_input=true) 
        Xall[j] = states[:, 2:end]
        Xdotall[j] = (states[:, 2:end] - states[:, 1:end-1]) / burger.Δt
    end
    X = reduce(hcat, Xall)
    R = reduce(hcat, Xdotall)
    U = reshape(Urand[2:end,:], (burger.time_dim - 1) * num_inputs, 1)
    Y = C * X
    Xtrain[i] = X
    Rtrain[i] = R
    Ytrain[i] = Y
    Utrain[i] = U
end

#===========================#
## Generate the basis
#===========================#
@info "Generate the basis"
rmax = 20
tmp = svd(reduce(hcat, Xtrain))
Vrmax = tmp.U[:, 1:rmax]

#=====================================================#
## Compute reduced operators and calculate the errors
#=====================================================#
# Error Values 
k = 1
proj_err = zeros(rmax - k)
intru_state_err = zeros(rmax - k)
opinf_state_err = zeros(rmax - k)
intru_output_err = zeros(rmax - k)
opinf_output_err = zeros(rmax - k)

@info "Compute inferred and intrusive operators and calculate the errors"
@showprogress for i in eachindex(burger.diffusion_coeffs)
    # Unpack the data and operators
    X = Xtrain[i]
    R = Rtrain[i]
    Y = Ytrain[i]
    U = Utrain[i]
    A = A_full[i]
    F = F_full[i]
    B = B_full[i]
    C = C_full[i]
    op_burger = LnL.Operators(A=A, B=B, C=C, A2u=F)

    # Unpack the reference data
    Xtest = Xtests[i]
    Ytest = Ytests[i]

    # Generate the POD-Galerkin intrusive model
    op_int = LnL.pod(op_burger, Vrmax, options.system)
    A_intru[i] = op_int.A
    F_intru[i] = op_int.A2u
    B_intru[i] = op_int.B
    C_intru[i] = op_int.C

    # Compute the inferred operators from the training data
    if options.optim.reproject 
        op_inf = LnL.opinf(X, Vrmax, op_burger, options; U=U, Y=Y)  # Using Reprojection
    else
        op_inf = LnL.opinf(X, Vrmax, options; U=U, Y=Y, Xdot=R)
    end
    A_opinf[i] = op_inf.A
    F_opinf[i] = op_inf.A2u
    B_opinf[i] = op_inf.B
    C_opinf[i] = op_inf.C

    for j = 1+k:rmax
        Vr = Vrmax[:, 1:j]  # basis

        # Integrate the intrusive model
        Fint_extract = UniqueKronecker.extractF(op_int.A2u, j)
        Xint = burger.integrate_model(burger.tspan, Vr' * burger.IC, Utest; linear_matrix=op_int.A[1:j, 1:j],
                                      control_matrix=op_int.B[1:j, :], quadratic_matrix=Fint_extract, system_input=true) # <- use F
        Yint = op_int.C[1:1, 1:j] * Xint

        # Integrate the inferred model
        Finf_extract = UniqueKronecker.extractF(op_inf.A2u, j)
        Xinf = burger.integrate_model(burger.tspan, Vr' * burger.IC, Utest; linear_matrix=op_inf.A[1:j, 1:j],
                                      control_matrix=op_inf.B[1:j, :], quadratic_matrix=Finf_extract, system_input=true) # <- use F
        Yinf = op_inf.C[1:1, 1:j] * Xinf

        # Compute errors
        PE, ISE, IOE, OSE, OOE = LnL.compute_all_errors(Xtest, Ytest, Xint, Yint, Xinf, Yinf, Vr)

        # Sum of error values
        proj_err[j-k] += PE / burger.param_dim
        intru_state_err[j-k] += ISE / burger.param_dim
        intru_output_err[j-k] += IOE / burger.param_dim
        opinf_state_err[j-k] += OSE / burger.param_dim
        opinf_output_err[j-k] += OOE / burger.param_dim
    end
end

df = DataFrame(
    order=1+k:rmax,
    projection_err=vec(proj_err),
    intrusive_state_err=vec(intru_state_err),
    intrusive_output_err=vec(intru_output_err),
    inferred_state_err=vec(opinf_state_err),
    inferred_output_err=vec(opinf_output_err)
)
if SAVEDATA
    CSV.write("scripts/OpInf/data/burger_data.csv", df)  # Write the data just in case
end

#==========#
## Plotting
#==========#
@info "Plotting results"
cutoff = 1:13
# Projection error
p1 = plot(df.order[cutoff], df.projection_err[cutoff], marker=(:rect))
plot!(yscale=:log10, majorgrid=true, minorgrid=true, legend=false)
tmp = log10.(df.projection_err)
yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
xticks!(df.order)
xlabel!("dimension n")
ylabel!("avg projection error")
display(p1)

# State errors
p2 = plot(df.order[cutoff], df.intrusive_state_err[cutoff], marker=(:cross, 10), label="intru")
plot!(df.order[cutoff], df.inferred_state_err[cutoff], marker=(:circle), ls=:dash, label="opinf")
plot!(yscale=:log10, majorgrid=true, minorgrid=true)
tmp = log10.(df.intrusive_state_err)
yticks!([10.0^i for i in floor(minimum(tmp))-1:ceil(maximum(tmp))+1])
xticks!(df.order)
xlabel!("dimension n")
ylabel!("avg error of states")
display(p2)

# Output errors
p3 = plot(df.order[cutoff], df.intrusive_output_err[cutoff], marker=(:cross, 10), label="intru")
plot!(df.order[cutoff], df.inferred_output_err[cutoff], marker=(:circle), ls=:dash, label="opinf")
plot!(majorgrid=true, minorgrid=true)
xticks!(df.order)
xlabel!("dimension n")
ylabel!("avg error of outputs")
display(p3)

if SAVEFIG
    savefig(p1, "scripts/OpInf/plots/burger_projerr.pdf")
    savefig(p2, "scripts/OpInf/plots/burger_stateerr.pdf")
    savefig(p3, "scripts/OpInf/plots/burger_outputerr.pdf")
end

@info "Done"

#==========#
## Testing
#==========#
@info "Testing by interpolating with the training parameter region."
num_tests = 5

# Error analysis 
intru_state_err = zeros(rmax-k, 1)
opinf_state_err = zeros(rmax-k, 1)
intru_output_err = zeros(rmax-k, 1)
opinf_output_err = zeros(rmax-k, 1)

param_region = collect(burger.diffusion_coeffs)

@showprogress for j = 1:num_tests
    # Generate new parameter
    μ = rand(burger.param_domain[1]+eps():0.01:burger.param_domain[2]-eps())

    # Interpolate model operators
    Aint = LnL.interpolate_matrix_elements(param_region, A_intru, μ; order=3)
    Fint = LnL.interpolate_matrix_elements(param_region, F_intru, μ; order=3)
    Bint = LnL.interpolate_matrix_elements(param_region, B_intru, μ; order=3)
    Cint = LnL.interpolate_matrix_elements(param_region, C_intru, μ; order=3)

    Ainf = LnL.interpolate_matrix_elements(param_region, A_opinf, μ; order=3)
    Finf = LnL.interpolate_matrix_elements(param_region, F_opinf, μ; order=3)
    Binf = LnL.interpolate_matrix_elements(param_region, B_opinf, μ; order=3)
    Cinf = LnL.interpolate_matrix_elements(param_region, C_opinf, μ; order=3)

    # Generate full models for new parameter to get POD-basis
    A, F, B = burger.finite_diff_model(burger, μ)
    C = ones(1, burger.spatial_dim) / burger.spatial_dim

    # Compute the states with backward Euler
    X = burger.integrate_model(burger.tspan, burger.IC, Utest; linear_matrix=A, control_matrix=B,
                               quadratic_matrix=F, system_input=true)
    Y = C * X

    for i = 1+k:rmax
        Vr = Vrmax[:, 1:i]

        # Integrate the intrusive model
        Fint_extract = UniqueKronecker.extractF(Fint, i)
        Xint = burger.integrate_model(burger.tspan, Vr' * burger.IC, Utest; linear_matrix=Aint[1:i, 1:i],
                                      control_matrix=Bint[1:i, :], quadratic_matrix=Fint_extract, system_input=true) # <- use F
        Yint = Cint[1:1, 1:i] * Xint

        # Integrate the inferred model
        Finf_extract = UniqueKronecker.extractF(Finf, i)
        Xinf = burger.integrate_model(burger.tspan, Vr' * burger.IC, Utest; linear_matrix=Ainf[1:i, 1:i],
                                      control_matrix=Binf[1:i, :], quadratic_matrix=Finf_extract, system_input=true) # <- use F
        Yinf = Cinf[1:1, 1:i] * Xinf

        # Compute errors
        _, ISE, IOE, OSE, OOE = LnL.compute_all_errors(X, Y, Xint, Yint, Xinf, Yinf, Vr)

        # Sum of error values
        intru_state_err[i-k] += ISE / num_tests
        intru_output_err[i-k] += IOE / num_tests
        opinf_state_err[i-k] += OSE / num_tests
        opinf_output_err[i-k] += OOE / num_tests
    end
end

df = DataFrame(
    :order => 1+k:rmax,
    :intrusive_state_err => vec(intru_state_err),
    :intrusive_output_err => vec(intru_output_err),
    :inferred_state_err => vec(opinf_state_err),
    :inferred_output_err => vec(opinf_output_err)
)
if SAVEDATA
    CSV.write("scripts/OpInf/data/heat1d_test_data.csv", df)  # Write the data just in case
end

#==============#
## Plot results
#==============#
@info "Plotting results"
cutoff = 1:13
# State error
p2 = plot(df.order[cutoff], df.intrusive_state_err[cutoff], marker=(:cross, 10), label="intru", show=true)
plot!(p2, df.order[cutoff], df.inferred_state_err[cutoff], marker=(:circle), ls=:dash, label="opinf")
plot!(p2, 
    yscale=:log10, 
    majorgrid=true, minorgrid=true,
    yticks=[round(10.0^i, digits=-i) for i in -10:0],
    xticks=1:rmax,
    xlabel="dimension n",
    ylabel="avg error of states",
    legend=:topright,
    show=true
)
display(p2)

# Output error
p3 = plot(df.order[cutoff], df.intrusive_output_err[cutoff], marker=(:cross, 10), label="intru", show=true)
plot!(p3, df.order[cutoff], df.inferred_output_err[cutoff], marker=(:circle), ls=:dash, label="opinf")
plot!(p3, 
    yscale=:log10, 
    majorgrid=true, minorgrid=true,
    yticks=[10.0^i for i in -1:0.1:1],
    xticks=1:rmax,
    xlabel="dimension n",
    ylabel="avg error of outputs",
    legend=:topright,
    show=true
)
display(p3)

if SAVEFIG
    savefig(p2, "scripts/OpInf/plots/burger_test_state_err.pdf")
    savefig(p3, "scripts/OpInf/plots/burger_test_output_err.pdf")
end

@info "Done"