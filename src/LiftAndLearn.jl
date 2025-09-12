"""
    LiftAndLearn package main module
"""
module LiftAndLearn

using Combinatorics: binomial
using LinearAlgebra
using LinearSolve
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
if Sys.isapple()  # macOS
    using Metal
else              # Windows/Linux
    using CUDA
    # INFO: no support for AMD GPUs yet
end

"""
    AbstractOption

Abstract type for the options.
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