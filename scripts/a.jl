using DataFrames
using FileIO
using JLD2
using LaTeXStrings
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using ProgressMeter
using StatsBase

include("../src/model/KS.jl")
include("../src/LiftAndLearn.jl")
const LnL = LiftAndLearn

# Settings for the KS equation
KSE = KS(
    [0.0, 22.0], [0.0, 300.0], [1.0, 1.0],
    512, 0.001, 1, "ep"
)

# Create file name to save data
datafile = "examples/data/kse_data_L22.jld2"
opfile = "examples/data/kse_operators_L22.jld2"
resultfile = "examples/data/kse_results_L22.jld2"
testresultfile = "examples/data/kse_test_results_L22.jld2"

# Downsampling rate
DS = 100

# Down-sampled dimension of the time data
Tdim_ds = size(1:DS:KSE.Tdim, 1)  # downsampled time dimension

# Number of random test inputs
num_test_ic = 50

# Training initial conditions
ic_a = [0.8, 1.0, 1.2]
ic_b = [0.2, 0.4, 0.6]

# Parameterized function for the initial condition
L = KSE.Omega[2] - KSE.Omega[1]  # length of the domain
u0 = (a,b) -> a * cos.((2*π*KSE.x)/L) .+ b * cos.((4*π*KSE.x)/L)  # initial condition

DATA = load(datafile)
OPS = load(opfile)
TEST_RES = load(testresultfile)


function analyze_autocorr(op, model, Vr_all, IC, ro, integrator, lags)

    # auto_correletion
    auto_correlation = Array{Array{Float64}}(undef, length(ro), model.Pdim)

    for i in eachindex(model.μs)
        for (j,r) in collect(enumerate(ro))
            Vr = Vr_all[i][:, 1:r]

            Fextract = LnL.extractF(op[i].F, r)
            X = integrator(op[i].A[1:r, 1:r], Fextract, model.t, Vr' * IC)
            Xrecon = Vr * X
            auto_correlation[j, i] = tmean_autocorr(Xrecon, lags)
        end
    end
    return auto_correlation
end


function tmean_autocorr(X::AbstractArray, lags::AbstractVector)
    N, K = size(X)
    M = length(lags)
    Cx = zeros((N, M))
    
    for i in 1:N  # normalzied autocorrelation
        Cx[i,:] = autocor(X[i,:], lags)
    end
    return vec(mean(Cx, dims=1))
end


# Compute the relative error of the autocorrelation for each reduced dimensions
function autocorr_rel_err(AC, AC_fom)
    rdim, pdim = size(AC)
    AC_rel_err = Array{Float64}(undef, rdim)
    for i in 1:rdim
        err = 0
        for j in 1:pdim
            err += norm(AC[i,j] - AC_fom[j], 2) / norm(AC_fom[j], 2)
        end
        AC_rel_err[i] = err / pdim
    end
    return AC_rel_err
end


# Generate random initial condition parameters
ic_a_rand_in = (maximum(ic_a) - minimum(ic_a)) .* rand(num_test_ic) .+ minimum(ic_a)
ic_b_rand_in = (maximum(ic_b) - minimum(ic_b)) .* rand(num_test_ic) .+ minimum(ic_b)

i = 1
μ = KSE.μs[i]

# Generate the FOM system matrices (ONLY DEPENDS ON μ)
A = DATA["op_fom_tr"][i].A
F = DATA["op_fom_tr"][i].F


# lag for autocorrelation
lags = 0:DS:(KSE.Tdim)

# Store some arrays
fom_ac = zeros(length(lags))
int_ac = zeros(length(lags), length(DATA["ro"]))
LS_ac = zeros(length(lags), length(DATA["ro"]))
ephec_ac = zeros(length(lags), length(DATA["ro"]))

# Generate the data for all combinations of the initial condition parameters
prog = Progress(num_test_ic)
Threads.@threads for j in 1:num_test_ic  
    # Step 1: generate test 1 data
    a = ic_a_rand_in[j]
    b = ic_b_rand_in[j]
    Xtest_in = KSE.integrate_FD(A, F, KSE.t, u0(a,b))

    # Step 2: compute autocorrelation of FOM
    fom_ac .+= tmean_autocorr(Xtest_in, lags)

    # Step 3: compute autocorrelation of intrusive ROM 
    tmp = analyze_autocorr(OPS["op_int"], KSE, DATA["Vr"], u0(a,b), DATA["ro"], KSE.integrate_FD, lags)
    for j in eachindex(DATA["ro"])
        int_ac[:,j] .+= tmp[j,1]
    end

    # Step 4: compute autocorrelation of opinf ROM
    tmp = analyze_autocorr(OPS["op_LS"], KSE, DATA["Vr"], u0(a,b), DATA["ro"], KSE.integrate_FD, lags)
    for j in eachindex(DATA["ro"])
        LS_ac[:,j] .+= tmp[j,1]
    end

    # Step 5: compute autocorrelation of ep-opinf ROM
    tmp = analyze_autocorr(OPS["op_ephec"], KSE, DATA["Vr"], u0(a,b), DATA["ro"], KSE.integrate_FD, lags)
    for j in eachindex(DATA["ro"])
        ephec_ac[:,j] .+= tmp[j,1]
    end

    next!(prog)
end

# Compute the average of all initial conditions as the final result
TEST_RES["test1_AC"][:fom][1] = fom_ac / num_test_ic
for j in eachindex(DATA["ro"])
    TEST_RES["test1_AC"][:int][j,1] = int_ac[:,j] / num_test_ic
    TEST_RES["test1_AC"][:LS][j,1] = LS_ac[:,j] / num_test_ic
    TEST_RES["test1_AC"][:ephec][j,1] = ephec_ac[:,j] / num_test_ic
end


# Generate random initial condition parameters
ic_a_out = [0.0, 2.0]
ic_b_out = [0.0, 0.8]
ϵ=1e-2
half_num_test_ic = Int(num_test_ic/2)
ic_a_rand_out = ((minimum(ic_a) - ϵ) - ic_a_out[1]) .* rand(half_num_test_ic) .+ ic_a_out[1]
append!(ic_a_rand_out, (ic_a_out[2] - (maximum(ic_a) + ϵ)) .* rand(half_num_test_ic) .+ (maximum(ic_a) + ϵ))
ic_b_rand_out = ((minimum(ic_b) - ϵ) - ic_b_out[1]) .* rand(half_num_test_ic) .+ ic_b_out[1]
append!(ic_b_rand_out, (ic_b_out[2] - (maximum(ic_b) + ϵ)) .* rand(half_num_test_ic) .+ (maximum(ic_b) + ϵ))

i = 1
μ = KSE.μs[i]

# Generate the FOM system matrices (ONLY DEPENDS ON μ)
A = DATA["op_fom_tr"][i].A
F = DATA["op_fom_tr"][i].F

# lag for autocorrelation
lags = 0:DS:(KSE.Tdim)

# Store some arrays
fom_ac = zeros(length(lags))
int_ac = zeros(length(lags), length(DATA["ro"]))
LS_ac = zeros(length(lags), length(DATA["ro"]))
ephec_ac = zeros(length(lags), length(DATA["ro"]))

# Generate the data for all combinations of the initial condition parameters
prog = Progress(num_test_ic)
Threads.@threads for j in 1:num_test_ic  
    # Step 1: generate test 1 data
    a = ic_a_rand_out[j]
    b = ic_b_rand_out[j]
    Xtest_in = KSE.integrate_FD(A, F, KSE.t, u0(a,b))

    # Step 2: compute autocorrelation of FOM
    fom_ac .+= tmean_autocorr(Xtest_in, lags)

    # Step 3: compute autocorrelation of intrusive ROM 
    tmp = analyze_autocorr(OPS["op_int"], KSE, DATA["Vr"], u0(a,b), DATA["ro"], KSE.integrate_FD, lags)
    for j in eachindex(DATA["ro"])
        int_ac[:,j] .+= tmp[j,1]
    end

    # Step 4: compute autocorrelation of opinf ROM
    tmp = analyze_autocorr(OPS["op_LS"], KSE, DATA["Vr"], u0(a,b), DATA["ro"], KSE.integrate_FD, lags)
    for j in eachindex(DATA["ro"])
        LS_ac[:,j] .+= tmp[j,1]
    end

    # Step 5: compute autocorrelation of ep-opinf ROM
    tmp = analyze_autocorr(OPS["op_ephec"], KSE, DATA["Vr"], u0(a,b), DATA["ro"], KSE.integrate_FD, lags)
    for j in eachindex(DATA["ro"])
        ephec_ac[:,j] .+= tmp[j,1]
    end

    next!(prog)
end

# Compute the average of all initial conditions as the final result
TEST_RES["test2_AC"][:fom][1] = fom_ac / num_test_ic
for j in eachindex(DATA["ro"])
    TEST_RES["test2_AC"][:int][1,j] = int_ac[:,j] / num_test_ic
    TEST_RES["test2_AC"][:LS][1,j] = LS_ac[:,j] / num_test_ic
    TEST_RES["test2_AC"][:ephec][1,j] = ephec_ac[:,j] / num_test_ic
end

save(testresultfile, TEST_RES)