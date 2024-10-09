"""
Burger's equation test case using Operator Inference.
"""

#################
## Load Packages
#################
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

############
## Load LnL
############
using LiftAndLearn
const LnL = LiftAndLearn

###############
## Some option
###############
SAVEFIG = false
SAVEDATA = false

####################
## Setup burgers eq
####################
Ω = (0.0, 1.0)
Nx = 2^7; dt = 1e-4
burger = Pomoreda.BurgersModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    diffusion_coeffs=range(0.1, 1.0, length=10), BC=:dirichlet,
)
num_inputs = 10
rmax = 25

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
)
Utest = ones(burger.time_dim, 1);  # Reference input/boundary condition for OpInf testing 

# Error Values 
k = 3
proj_err = zeros(rmax - k, burger.param_dim)
intru_state_err = zeros(rmax - k, burger.param_dim)
opinf_state_err = zeros(rmax - k, burger.param_dim)
intru_output_err = zeros(rmax - k, burger.param_dim)
opinf_output_err = zeros(rmax - k, burger.param_dim)
Σr = Vector{Vector{Float64}}(undef, burger.param_dim)  # singular values 

# Add 5 extra parameters drawn randomly from the uniform distribution of range [0, 1]
# μs = vcat(burger.μs)

#############################
## Compute reduced operators
#############################
@info "Compute inferred and intrusive operators and calculate the errors"
prog = Progress(length(burger.diffusion_coeffs))
for i in 1:length(burger.diffusion_coeffs)
    μ = burger.diffusion_coeffs[i]

    ## Create testing data
    A, F, B = burger.finite_diff_model(burger, μ)
    C = ones(1, burger.spatial_dim) / burger.spatial_dim
    Xtest = burger.integrate_model(burger.tspan, burger.IC, Utest; linear_matrix=A,
                                   control_matrix=B, quadratic_matrix=F, system_input=true)
    Ytest = C * Xtest

    op_burger = LnL.Operators(A=A, B=B, C=C, A2u=F)

    ## training data for inferred dynamical models
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

    # compute the POD basis from the training data
    tmp = svd(X)
    Vrmax = tmp.U[:, 1:rmax]
    Σr[i] = tmp.S

    # Compute the values for the intrusive model from the basis of the training data
    op_int = LnL.pod(op_burger, Vrmax, options.system)

    # Compute the inferred operators from the training data
    if options.optim.reproject 
        op_inf = LnL.opinf(X, Vrmax, op_burger, options; U=U, Y=Y)  # Using Reprojection
    else
        op_inf = LnL.opinf(X, Vrmax, options; U=U, Y=Y, Xdot=R)
    end

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
        proj_err[j-k, i] = PE
        intru_state_err[j-k, i] = ISE
        intru_output_err[j-k, i] = IOE
        opinf_state_err[j-k, i] = OSE
        opinf_output_err[j-k, i] = OOE
    end
    next!(prog)
end

proj_err = mean(proj_err, dims=2)
intru_state_err = mean(intru_state_err, dims=2)
intru_output_err = mean(intru_output_err, dims=2)
opinf_state_err = mean(opinf_state_err, dims=2)
opinf_output_err = mean(opinf_output_err, dims=2)

cutoff = 1:16

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

############
## Plotting
############
@info "Plotting results"
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