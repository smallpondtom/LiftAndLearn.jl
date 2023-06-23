using DataFrames
using LinearAlgebra
using Plots
using Random
using SparseArrays
using Statistics
using NaNStatistics
using JLD2

include("../src/model/Burgers.jl")
include("../src/LiftAndLearn.jl")
const LnL = LiftAndLearn

# First order Burger's equation setup
burger = Burgers(
    [0.0, 1.0], [0.0, 1.0], [0.1, 1.0],
    2^(-7), 1e-4, 5
)


num_inputs = 10
rmin = 4
rmax = 20

options = LnL.OpInf_options(
    reproject=false,
    is_quad=true,
    optimization="NC",  #!!! This options changes the problem into an optimization problem
    opt_verbose=false,
    initial_guess_for_opt=false,
    which_quad_term="H",
    N=1,
    Δt=1e-4,
    deriv_type="SI",
    has_output=false  # suppress output
)

# Downsampling rate
DS = 500

# Reference input/boundary condition for OpInf testing 
Utest = ones(burger.Tdim - 1, 1);  

# Error Values 
proj_err = zeros(rmax - (rmin-1), burger.Pdim)
intru_state_err = zeros(rmax - (rmin-1), burger.Pdim)
nc_opinf_state_err = zeros(rmax - (rmin-1), burger.Pdim)

# Non-Constrained
Qnc = Matrix{Vector{Matrix{Float64}}}(undef, rmax-(rmin-1), burger.Pdim)
Hnc = Matrix{Matrix{Float64}}(undef, rmax-(rmin-1), burger.Pdim)

# Intrusive Operators
Qint = Matrix{Vector{Matrix{Float64}}}(undef, rmax-(rmin-1), burger.Pdim)
Hint = Matrix{Matrix{Float64}}(undef, rmax-(rmin-1), burger.Pdim)

# Store values
X = Array{Matrix{Real}}(undef, burger.Pdim)
Xtest = Array{Matrix{Real}}(undef, burger.Pdim)
R = Array{Matrix{Real}}(undef, burger.Pdim)
U = Array{Matrix{Real}}(undef, burger.Pdim)
Y = Array{Matrix{Real}}(undef, burger.Pdim)
Vrmax = Array{Matrix{Real}}(undef, burger.Pdim)
# op_ref = Vector{LnL.operators}(undef, burger.Pdim)

println("[INFO] Compute inferred and intrusive operators and calculate the errors")
for i in 1:length(burger.μs)
    μ = burger.μs[i]

    ## Create testing data
    A, B, F = burger.generateABFmatrix(burger, μ)
    C = ones(1, burger.Xdim) / burger.Xdim
    Xtest_ = LnL.semiImplicitEuler(A, B, F, Utest, burger.t, burger.IC)
    Xtest[i] = Xtest_

    op_burger = LnL.operators(A=A, B=B, C=C, F=F)

    ## training data for inferred dynamical models
    Urand = rand(burger.Tdim - 1, num_inputs)
    Xall = Vector{Matrix{Float64}}(undef, num_inputs)
    Xdotall = Vector{Matrix{Float64}}(undef, num_inputs)
    for j in 1:num_inputs
        states = LnL.semiImplicitEuler(A, B, F, Urand[:, j], burger.t, burger.IC)
        tmp = states[:, 2:end]
        Xall[j] = tmp[:, 1:DS:end]  # downsample data
        tmp = (states[:, 2:end] - states[:, 1:end-1]) / burger.Δt
        Xdotall[j] = tmp[:, 1:DS:end]  # downsample data
    end
    X_ = reduce(hcat, Xall)
    R_ = reduce(hcat, Xdotall)
    Urand = Urand[1:DS:end, :]  # downsample data
    U_ = vec(Urand)[:,:]  # vectorize
    Y_ = C * X_
    
    # compute the POD basis from the training data
    tmp = svd(X_)
    Vrmax_ = tmp.U[:, 1:rmax]
    
    # Store important data values
    X[i] = X_
    R[i] = R_
    Y[i] = Y_
    U[i] = U_[:,:]
    Vrmax[i] = Vrmax_

    # Compute the values for the intrusive model from the basis of the training data
    op_int_ = LnL.intrusiveMR(op_burger, Vrmax_, options)
    
    # Compute non-constrained OpInf
    op_inf_ = LnL.inferOp(X_, U_, Y_, Vrmax_, Vrmax_' * R_, options)
    # op_ref[i] = op_inf_

    for j = rmin:rmax
        Vr = Vrmax_[:, 1:j]  # basis

        # Integrate the intrusive model
        Fint_extract = LnL.extractF(op_int_.F, j)
        Hint_extract = LnL.F2Hs(Fint_extract)
        Xint = LnL.semiImplicitEuler(op_int_.A[1:j, 1:j], op_int_.B[1:j, :], Fint_extract, Utest, burger.t, Vr' * burger.IC)
        
        # Integrate the inferred model
        Finf_extract = LnL.extractF(op_inf_.F, j)
        Hinf_extract = LnL.extractH(op_inf_.H, j)
        Xinf = LnL.semiImplicitEuler(op_inf_.A[1:j, 1:j], op_inf_.B[1:j, :], Finf_extract, Utest, burger.t, Vr' * burger.IC)

        # Compute errors
        PE = LnL.compProjError(Xtest_, Vr)
        ISE = LnL.compStateError(Xtest_, Xint, Vr)
        OSE = LnL.compStateError(Xtest_, Xinf, Vr)

        # Sum of error values
        proj_err[j-(rmin-1), i] = PE
        intru_state_err[j-(rmin-1), i] = ISE
        nc_opinf_state_err[j-(rmin-1), i] = OSE
        
        # Store the H and Q matrices
        Hint[j-(rmin-1), i] = Hint_extract
        Qint[j-(rmin-1), i] = LnL.H2Q(Hint_extract)
        Hnc[j-(rmin-1), i] = Hinf_extract
        Qnc[j-(rmin-1), i] = LnL.H2Q(Hinf_extract)
    end
end

# Switch optimization scheme
options.optimization = "EPHC"
options.initial_guess_for_opt = false

# Values to store
ephc_opinf_state_err = zeros(rmax - (rmin-1), burger.Pdim)
Qephc = Matrix{Vector{Matrix{Float64}}}(undef, rmax-(rmin-1), burger.Pdim)
Hephc = Matrix{Matrix{Float64}}(undef, rmax-(rmin-1), burger.Pdim)

println("[INFO] Compute inferred energy-preserving operators and calculate state error")
for i in 1:length(burger.μs)
    op_ref_ = op_ref[i]
    op_inf = LnL.inferOp(X[i], U[i], Y[i], Vrmax[i], Vrmax[i]' * R[i], options)

    for j = rmin:rmax
        Vr = Vrmax[:, 1:j]  # basis
        
        # Integrate the inferred model
        Finf_extract = LnL.extractF(op_inf.F, j)
        Hinf_extract = LnL.extractH(op_inf.H, j)
        Xinf = LnL.semiImplicitEuler(op_inf.A[1:j, 1:j], op_inf.B[1:j, :], Finf_extract, Utest, burger.t, Vr' * burger.IC)

        # Compute state error
        OSE = LnL.compStateError(Xtest[i], Xinf, Vr)

        # Sum of error value
        ephc_opinf_state_err[j-(rmin-1), i] = OSE
        
        # Store the H and Q matrices
        Hephc[j-(rmin-1), i] = Hinf_extract
        Qephc[j-(rmin-1), i] = LnL.H2Q(Hinf_extract)
    end
end

data_dictionary = Dict(
    :proj_err => proj_err,
    :ncOpInf_state_err => nc_opinf_state_err,
    :ephcOpInf_state_err => ephc_opinf_state_err,
    :Qint => Qint,
    :Hint => Hint,
    :Qnc => Qnc,
    :Hnc => Hnc,
    :Qephc => Qephc,
    :Hephc => Hephc
)
@save "ephc_opinf_data.jld2" data_dictionary