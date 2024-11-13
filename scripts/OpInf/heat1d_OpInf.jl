"""
One-dimensional heat equation test case using Operator Inference.
"""

#===============#
## Load packages
#===============#
using CSV
using DataFrames
using LinearAlgebra
using Plots
using ProgressMeter
using PolynomialModelReductionDataset
const Pomoreda = PolynomialModelReductionDataset

#==========#
## Load LnL
#==========#
using LiftAndLearn
const LnL = LiftAndLearn

#==================#
## Set some options
#==================#
SAVEFIG = false
PROVIDE_DERIVATIVE = false
SAVEDATA = false

#=======================#
## 1D Heat equation setup
#=======================#
Ω = (0.0, 1.0)
Nx = 2^7; dt = 1e-3
heat1d = Pomoreda.Heat1DModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), 
    Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt, 
    diffusion_coeffs=range(0.1, 10, 10),
)

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

A_full = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
B_full = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
C_full = Vector{Matrix{Float64}}(undef, heat1d.param_dim)

A_intru = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
B_intru = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
C_intru = Vector{Matrix{Float64}}(undef, heat1d.param_dim)

A_opinf = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
B_opinf = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
C_opinf = Vector{Matrix{Float64}}(undef, heat1d.param_dim)

#====================#
## Generate data
#====================#
Ubc = ones(heat1d.time_dim)
@info "Generate the data"
p = Progress(length(heat1d.diffusion_coeffs))
for (idx, μ) in enumerate(heat1d.diffusion_coeffs)
    A, B = heat1d.finite_diff_model(heat1d, μ)
    C = ones(1, heat1d.spatial_dim) / heat1d.spatial_dim
    op_heat = LnL.Operators(A=A, B=B, C=C)
    A_full[idx] = A
    B_full[idx] = B
    C_full[idx] = C

    # Compute the states with backward Euler
    X = heat1d.integrate_model(heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
                               system_input=true, integrator_type=:BackwardEuler)
    Xfull[idx] = X

    # Compute the output of the system
    Y = C * X
    Yfull[idx] = Y
    next!(p)
end

#====================#
## Generate basis
#====================#
@info "Generate the POD basis"
r = 8  # order of the reduced form
tmp = svd(reduce(hcat, Xfull))
Vrmax = tmp.U[:, 1:r]

#====================#
## Generate operators
#====================#
@info "Generate intrusive and inferred operators"
p = Progress(length(heat1d.diffusion_coeffs))
for (idx, μ) in enumerate(heat1d.diffusion_coeffs)
    A = A_full[idx]
    B = B_full[idx]
    C = C_full[idx]
    X = Xfull[idx]
    Y = Yfull[idx]

    # Compute the values for the intrusive model
    op_heat = Operators(A=A, B=B, C=C)
    op_heat_new = LnL.pod(op_heat, Vrmax, options.system)
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
        op_infer = LnL.opinf(Xn, Vrmax, options; U=Un, Y=Yn, Xdot=Xdot)
    else
        op_infer = LnL.opinf(X, Vrmax, options; U=Ubc, Y=Y)
    end

    A_opinf[idx] = op_infer.A
    B_opinf[idx] = op_infer.B
    C_opinf[idx] = op_infer.C
    
    next!(p)
end

#=========#
## Analyze
#=========#
@info "Compute errors"

# Error analysis 
intru_state_err = zeros(r, 1)
opinf_state_err = zeros(r, 1)
intru_output_err = zeros(r, 1)
opinf_output_err = zeros(r, 1)
proj_err = zeros(r, 1)

@showprogress for i = 1:r, j = 1:heat1d.param_dim
    Xf = Xfull[j]  # full order model states
    Yf = Yfull[j]  # full order model outputs
    Vr = Vrmax[:, 1:i]

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
if SAVEDATA
    CSV.write("scripts/OpInf/data/heat1d_data.csv", df)  # Write the data just in case
end

#==============#
## Plot results
#==============#
@info "Plotting results"
# Projection error
p1 = plot(1:r, df.projection_err, marker=(:rect),show=true)
plot!(p1, 
    yscale=:log10, 
    majorgrid=true, minorgrid=true, 
    legend=false,
    yticks=[round(10.0^i, digits=-i) for i in -10:0],
    xticks=1:r,
    xlabel="dimension n",
    ylabel="avg projection error",
    show=true
)
display(p1)

# State error
p2 = plot(1:r, df.intrusive_state_err, marker=(:cross, 10), label="intru", show=true)
plot!(p2, 1:r, df.inferred_state_err, marker=(:circle), ls=:dash, label="opinf")
plot!(p2, 
    yscale=:log10, 
    majorgrid=true, minorgrid=true,
    yticks=[round(10.0^i, digits=-i) for i in -10:0],
    xticks=1:r,
    xlabel="dimension n",
    ylabel="avg error of states",
    show=true
)
display(p2)

# Output error
p3 = plot(1:r, df.intrusive_output_err, marker=(:cross, 10), label="intru", show=true)
plot!(p3, 1:r, df.inferred_output_err, marker=(:circle), ls=:dash, label="opinf")
plot!(p3, 
    yscale=:log10, 
    majorgrid=true, minorgrid=true,
    yticks=[round(10.0^i, digits=-i) for i in -10:0],
    xticks=1:r,
    xlabel="dimension n",
    ylabel="avg error of outputs",
    show=true
)
display(p3)

if SAVEFIG
    savefig(p1, "scripts/OpInf/plots/heat1d_proj_err.png")
    savefig(p2, "scripts/OpInf/plots/heat1d_state_err.png")
    savefig(p3, "scripts/OpInf/plots/heat1d_output_err.png")
end

@info "Done"

#==========#
## Testing
#==========#
@info "Testing by interpolating with the training parameter region."
num_tests = 5

# Error analysis 
intru_state_err = zeros(r, 1)
opinf_state_err = zeros(r, 1)
intru_output_err = zeros(r, 1)
opinf_output_err = zeros(r, 1)

param_region = collect(heat1d.diffusion_coeffs)

@showprogress for j = 1:num_tests
    # Generate new parameter
    μ = rand(heat1d.param_domain[1]+eps():0.01:heat1d.param_domain[2]-eps())

    # Interpolate model operators
    Aint = LnL.interpolate_matrix_elements(param_region, A_intru, μ; order=3)
    Bint = LnL.interpolate_matrix_elements(param_region, B_intru, μ; order=3)
    Cint = LnL.interpolate_matrix_elements(param_region, C_intru, μ; order=3)

    Ainf = LnL.interpolate_matrix_elements(param_region, A_opinf, μ; order=3)
    Binf = LnL.interpolate_matrix_elements(param_region, B_opinf, μ; order=3)
    Cinf = LnL.interpolate_matrix_elements(param_region, C_opinf, μ; order=3)

    # Generate full models for new parameter to get POD-basis
    A, B = heat1d.finite_diff_model(heat1d, μ)
    C = ones(1, heat1d.spatial_dim) / heat1d.spatial_dim

    # Compute the states with backward Euler
    X = heat1d.integrate_model(heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
                               system_input=true, integrator_type=:BackwardEuler)
    Y = C * X

    for i = 1:r
        Vr = Vrmax[:, 1:i]

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
        _, ISE, IOE, OSE, OOE = LnL.compute_all_errors(X, Y, Xint, Yint, Xinf, Yinf, Vr)

        # Sum of error values
        intru_state_err[i] += ISE / num_tests
        intru_output_err[i] += IOE / num_tests
        opinf_state_err[i] += OSE / num_tests
        opinf_output_err[i] += OOE / num_tests
    end
end

df = DataFrame(
    :order => 1:r,
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
# State error
p2 = plot(1:r, df.intrusive_state_err, marker=(:cross, 10), label="intru", show=true)
plot!(p2, 1:r, df.inferred_state_err, marker=(:circle), ls=:dash, label="opinf")
plot!(p2, 
    yscale=:log10, 
    majorgrid=true, minorgrid=true,
    yticks=[round(10.0^i, digits=-i) for i in -10:0],
    xticks=1:r,
    xlabel="dimension n",
    ylabel="avg error of states",
    legend=:topright,
    show=true
)
display(p2)

# Output error
p3 = plot(1:r, df.intrusive_output_err, marker=(:cross, 10), label="intru", show=true)
plot!(p3, 1:r, df.inferred_output_err, marker=(:circle), ls=:dash, label="opinf")
plot!(p3, 
    yscale=:log10, 
    majorgrid=true, minorgrid=true,
    yticks=[round(10.0^i, digits=-i) for i in -10:0],
    xticks=1:r,
    xlabel="dimension n",
    ylabel="avg error of outputs",
    legend=:topright,
    show=true
)
display(p3)

if SAVEFIG
    savefig(p2, "scripts/OpInf/plots/heat1d_test_state_err.png")
    savefig(p3, "scripts/OpInf/plots/heat1d_test_output_err.png")
end

@info "Done"