## Setup
using FileIO
using JLD2
using LinearAlgebra

using LiftAndLearn
const LnL = LiftAndLearn
const CG = LiftAndLearn.ChaosGizmo

# Settings for the KS equation
KSE = LnL.ks(
    [0.0, 22.0], [0.0, 300.0], [1.0, 1.0],
    512, 0.001, 1, "ep"
)

# Create file name to save data
datafile = "examples/data/kse_data_L22.jld2"
opfile = "examples/data/kse_operators_L22.jld2"
resultfile = "examples/data/kse_results_L22.jld2"
testresultfile = "examples/data/kse_test_results_L22.jld2"
testICfile = "examples/data/kse_test_ics_L22.jld2"

# Settings for Operator Inference
KSE_system = LnL.sys_struct(
    is_lin=true,
    is_quad=true,
)
KSE_vars = LnL.vars(
    N=1,
)
KSE_data = LnL.data(
    Δt=KSE.Δt,
    DS=100,
)
KSE_optim = LnL.opt_settings(
    verbose=true,
    initial_guess=false,
    max_iter=1000,
    reproject=false,
    SIGE=false,
    with_bnds=true,
    linear_solver="ma86",
)

options = LnL.LS_options(
    system=KSE_system,
    vars=KSE_vars,
    data=KSE_data,
    optim=KSE_optim,
)

# # Downsampling rate
# DS = KSE_data.DS

# # Down-sampled dimension of the time data
# Tdim_ds = size(1:DS:KSE.Tdim, 1)  # downsampled time dimension

# # Number of random test inputs
# num_test_ic = 50

# # Prune data to get only the chaotic region
# prune_data = false
# prune_idx = KSE.Tdim ÷ 2
# t_prune = KSE.t[prune_idx-1:end]

# # Parameters of the initial condition
# ic_a = [0.8, 1.0, 1.2]
# ic_b = [0.2, 0.4, 0.6]

# num_ic_params = Int(length(ic_a) * length(ic_b))
L = KSE.Omega[2] - KSE.Omega[1]  # length of the domain

# # Parameterized function for the initial condition
u0 = (a,b) -> a * cos.((2*π*KSE.x)/L) .+ b * cos.((4*π*KSE.x)/L)  # initial condition

## Load data
DATA = load(datafile)
OPS = load(opfile)
RES = load(resultfile)
rmax = size(OPS["op_LS"][1].A, 1)
ro = DATA["ro"]

## Training: Compute the Lyapunov exponent and Kaplan-Yorke dimension
num_IC = length(DATA["IC_train"])
RES["train_LE"] = Dict(
    :int   => Array{Array{Float64}}(undef, length(ro), KSE.Pdim, num_IC),
    :LS    => Array{Array{Float64}}(undef, length(ro), KSE.Pdim, num_IC),
    :ephec => Array{Array{Float64}}(undef, length(ro), KSE.Pdim, num_IC),
    :fom   => Array{Array{Float64}}(undef, KSE.Pdim)
)
RES["train_dky"] = Dict(
    :int   => Array{Float64}(undef, length(ro), KSE.Pdim, num_IC),
    :LS    => Array{Float64}(undef, length(ro), KSE.Pdim, num_IC),
    :ephec => Array{Float64}(undef, length(ro), KSE.Pdim, num_IC),
    :fom   => Array{Float64}(undef, KSE.Pdim)
)


## Function to compute the Lyapunov exponent and Kaplan-Yorke dimension for one initial condition
function compute_LE_oneIC!(RES, type, keys, model, op, IC, Vr, ro, integrator, jacobian, options, idx)
    for i in eachindex(model.μs)
        for (j, r) in collect(enumerate(ro))
            Ar = op[i].A[1:r,1:r]
            Hr = LnL.extractH(op[i].H, r)
            Fr = LnL.extractF(op[i].F, r)
            op_tmp = LnL.operators(A=Ar, H=Hr, F=Fr)
            if options.history
                _, foo = CG.lyapunovExponentJacobian(op_tmp, integrator, jacobian, Vr[i][:,1:r]' * IC, options)
                RES[keys[1]][type][j,i,idx] = foo
                RES[keys[2]][type][j,i,idx] = CG.kaplanYorkeDim(foo[:,end]; sorted=false)
            else
                foo = lyapunovExponentJacobian(op_tmp, integrator, jacobian, Vr[i][:,1:r]' * IC, options)
                RES[keys[1]][type][j,i,idx] = foo[:,end]
                RES[keys[2]][type][j,i,idx] = CG.kaplanYorkeDim(foo; sorted=false)
            end
            @debug "Reduced order of $(r) completed..."
        end
        @debug "Loop $(i) out of $(model.Pdim) completed..."
    end
end

function compute_LE_allIC!(RES, type, keys, model, op, ICs, Vr, ro, integrator, jacobian, options)
    for (idx, IC) in collect(enumerate(ICs))
        compute_LE_oneIC!(RES, type, keys, model, op, IC, Vr, ro, integrator, jacobian, options, idx)
        @info "Initial condition $(idx) out of $(length(ICs)) completed..."
    end
end

# FOM dispatch
function compute_LE_oneIC!(RES, type, keys, model, op, IC, integrator, jacobian, options, idx)
    for i in eachindex(model.μs)    
        if options.history
            _, foo = CG.lyapunovExponentJacobian(op[i], integrator, jacobian, IC, options)
            RES[keys[1]][type][i,idx] = foo
            RES[keys[2]][type] = CG.kaplanYorkeDim(foo[:,end]; sorted=false)
        else
            foo = lyapunovExponentJacobian(op[i], integrator, jacobian, IC, options)
            RES[keys[1]][type][i,idx] = foo
            RES[keys[2]][type] = CG.kaplanYorkeDim(foo; sorted=false)
        end
        @debug "Loop $(i) out of $(model.Pdim) completed..."

    end
end

function compute_LE_allIC!(RES, type, keys, model, op, ICs, integrator, jacobian, options)
    for (idx, IC) in collect(enumerate(ICs))
        compute_LE_oneIC!(RES, type, keys, model, op, IC, integrator, jacobian, options, idx)
        @info "Initial condition $(idx) out of $(length(ICs)) completed..."
    end
end

## Compute the LE and Dky for all models 
options = CG.LE_options(N=1e4, τ=1e3, Δt=10*KSE.Δt, m=9, T=10*KSE.Δt, verbose=true, history=true)

##
compute_LE_oneIC!(RES, :fom, ["train_LE", "train_dky"], KSE, 
        DATA["op_fom_tr"], DATA["IC_train"][1], KSE.integrate_FD, KSE.jacob, options, 1)
compute_LE_allIC!(RES, :int, ["train_LE", "train_dky"], KSE, 
        OPS["op_int"], DATA["IC_train"], DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options)
compute_LE_allIC!(RES, :LS, ["train_LE", "train_dky"], KSE, 
        OPS["op_LS"], DATA["IC_train"], DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options)
compute_LE_allIC!(RES, :ephec, ["train_LE", "train_dky"], KSE, 
        OPS["op_ephec"], DATA["IC_train"], DATA["Vr"], ro, KSE.integrator_FD, KSE.jacob, options)

## Save data
save(resultfile, "RES", RES)

## Testing
GC.gc()

TEST_RES = load(testresultfile)
TEST_IC = load(testICfile)

## Organize the initial conditions into a matrix 
TEST1_ICs = [u0(a,b) for (a,b) in zip(TEST_IC["ic_a_rand_in"], TEST_IC["ic_b_rand_in"])]
TEST2_ICs = [u0(a,b) for (a,b) in zip(TEST_IC["ic_a_rand_out"], TEST_IC["ic_b_rand_out"])]

num_IC = length(TEST_IC["ic_a_rand_in"])

TEST_RES["test1_LE"] = Dict(
    :int   => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :LS    => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :ephec => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :fom   => Array{Array{Float64}}(undef, KSE.Pdim)
)
TEST_RES["test2_LE"] = Dict(
    :int   => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :LS    => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :ephec => Array{Array{Float64}}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :fom   => Array{Array{Float64}}(undef, KSE.Pdim)
)
TEST_RES["test1_dky"] = Dict(
    :int   => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :LS    => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :ephec => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :fom   => Array{Float64}(undef, KSE.Pdim)
)
TEST_RES["test2_dky"] = Dict(
    :int   => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :LS    => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :ephec => Array{Float64}(undef, length(DATA["ro"]), KSE.Pdim, num_IC),
    :fom   => Array{Float64}(undef, KSE.Pdim)
)


## Test 1
options = CG.LE_options(N=1e4, τ=1e3, Δt=10*KSE.Δt, m=9, T=10*KSE.Δt, verbose=true, history=true)
##
compute_LE_oneIC!(TEST_RES, :fom, ["test1_LE", "test1_dky"], KSE, 
    DATA["op_fom_tr"], TEST1_ICs[1], DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options, 1)
compute_LE_allIC!(TEST_RES, :int, ["test1_LE", "test1_dky"], KSE, 
    OPS["op_int"], TEST1_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options)
compute_LE_allIC!(TEST_RES, :LS, ["test1_LE", "test1_dky"], KSE, 
    OPS["op_LS"], TEST1_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options)
compute_LE_allIC!(TEST_RES, :ephec, ["test1_LE", "test1_dky"], KSE, 
    OPS["op_ephec"], TEST1_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options)

## Test 2
compute_LE_oneIC!(TEST_RES, :fom, ["test2_LE", "test2_dky"], model, 
    DATA["op_fom_tr"], TEST2_ICs[1], DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options, 1)
compute_LE_allIC!(TEST_RES, :int, ["test2_LE", "test2_dky"], model, 
    OPS["op_int"], TEST2_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options)
compute_LE_allIC!(TEST_RES, :LS, ["test2_LE", "test2_dky"], model, 
    OPS["op_LS"], TEST2_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options)
compute_LE_allIC!(TEST_RES, :ephec, ["test2_LE", "test2_dky"], model, 
    OPS["op_ephec"], TEST2_ICs, DATA["Vr"], ro, KSE.integrate_FD, KSE.jacob, options)

## Save data
save(testresultfile, "TEST_RES", TEST_RES)

## Plotting Results 
