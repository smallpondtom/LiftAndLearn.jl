"""
    LiftAndLearn

Julia package for nonintrusive model reduction of polynomial (and lifted nonlinear)
dynamical systems.

Main features:
- intrusive model reduction by Proper Orthogonal Decomposition (POD) projection (`pod`),
- standard and Tikhonov-regularized Operator Inference (OpInf) (`opinf`),
- Lift & Learn (LnL) with user-defined lifting maps (`lifting`, `lifted_basis`),
- Energy-Preserving OpInf (EP-OpInf) via constrained optimization (`epopinf`),
- Streaming OpInf (one-pass and two-pass variants) for large or streaming data sets.

Data-layout convention: state snapshot matrices have rows = state dimension and
columns = time (each column is a snapshot).
"""
module LiftAndLearn

using Combinatorics: binomial
using LinearAlgebra
using LinearSolve
# Explicit imports: AppleAccelerate (loaded on Apple CPUs below) also exports `solve` and
# `solve!`, which would make the bare names ambiguous and therefore undefined on macOS.
using LinearSolve: LinearProblem, init, solve, solve!, KrylovJL_GMRES
using BlockDiagonals
using Kronecker
using Parameters
using ProgressMeter: Progress, next!
using PROPACK: tsvd
using SciMLOperators: FunctionOperator
using SparseArrays
using StatsBase: sample
using JuMP
using Ipopt, SCS
using DocStringExtensions
using UniqueKronecker
using LoopVectorization: @turbo

# BLAS, LAPACK, and other linear algebra library upgrades based on CPU 
cpu_model = Sys.cpu_info()[1].model
if occursin("Intel", cpu_model)
    using MKL
elseif occursin("Apple", cpu_model)
    using AppleAccelerate
elseif occursin("AMD", cpu_model)
    @info "Using OpenBLAS for AMD CPUs" 
else
    @info "CPU vendor not recognized, using default BLAS"
end

# GPU support
# CUDA is always loaded because the streaming solvers reference `CUDA.CuArray` in their
# type definitions; it is only *used* when `use_gpu=true` on a CUDA-capable machine.
using CUDA
if Sys.isapple()  # macOS
    using Metal
end
# INFO: no support for AMD GPUs yet

"""
$(TYPEDEF)

Abstract supertype of all Operator Inference option structures (e.g. `LSOpInfOption`,
`EPHECOpInfOption`, `EPSICOpInfOption`, `EPPOpInfOption`).

Functions such as `opinf`, `epopinf`, `get_data_matrix` and `time_derivative_approx`
dispatch on this type and read the `system`, `vars`, `data` and `optim` fields of the
concrete option structure.
"""
abstract type AbstractOption end

# Utilities
include("utilities/utilities.jl")

# Options
include("OpInf/OpInf_options.jl")

# Operators 
include("operators/operators.jl")

# Intrusive POD
include("POD/pod.jl")

# Operator Inference
include("OpInf/opinf.jl")

# Analysis
include("utilities/analyze.jl")

# Lift & Learn
include("LnL/lift.jl")
include("LnL/learn.jl")

# Include the optimization methods
include("EP-OpInf/epopinf.jl")

# Streaming-OpInf
include("Streaming/streamify.jl")
include("Streaming/twopass.jl")
include("Streaming/onepass.jl")

end # module LiftAndLearn