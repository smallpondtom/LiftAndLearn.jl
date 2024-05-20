"""
One-dimensional heat equation test case using Operator Inference.
"""

#################
## Load packages
#################
using CSV
using DataFrames
using LinearAlgebra
using Plots
using ProgressMeter

############
## Load LnL
############
using LiftAndLearn
const LnL = LiftAndLearn

####################
## Set some options
####################
SAVEFIG = false
PROVIDE_DERIVATIVE = false
SAVEDATA = false

#########################
## 1D Heat equation setup
#########################
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


######################
## Generate operators
######################
r = 15  # order of the reduced form

@info "Generate intrusive and inferred operators"
p = Progress(length(heat1d.μs))
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
    op_heat_new = LnL.pod(op_heat, Vr, options)
    A_intru[idx] = op_heat_new.A
    B_intru[idx] = op_heat_new.B
    C_intru[idx] = op_heat_new.C

    # Compute the RHS for the operator inference based on the intrusive operators
    if PROVIDE_DERIVATIVE
        jj = 2:heat1d.Tdim
        Xn = X[:, jj]
        Un = heat1d.Ubc[jj, :]
        Yn = Y[:, jj]
        Xdot = A * Xn + B * Un'
        op_infer = LnL.inferOp(Xn, Vr, options; U=Un, Y=Yn, Xdot=Xdot)
    else
        op_infer = LnL.inferOp(X, Vr, options; U=heat1d.Ubc, Y=Y)
    end

    A_opinf[idx] = op_infer.A
    B_opinf[idx] = op_infer.B
    C_opinf[idx] = op_infer.C
    
    next!(p)
end

###########
## Analyze
###########
@info "Compute errors"

# Error analysis 
intru_state_err = zeros(r, 1)
opinf_state_err = zeros(r, 1)
intru_output_err = zeros(r, 1)
opinf_output_err = zeros(r, 1)
proj_err = zeros(r, 1)

@showprogress for i = 1:r, j = 1:heat1d.Pdim
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
if SAVEDATA
    CSV.write("scripts/OpInf/data/heat1d_data.csv", df)  # Write the data just in case
end

################
## Plot results
################
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
