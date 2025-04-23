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
using SciMLOperators: FunctionOperator
using SparseArrays
using JuMP
using Ipopt, SCS
using DocStringExtensions
using UniqueKronecker

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
include("Streaming/isvd/isvd.jl")
include("Streaming/onepass.jl")

end # module LiftAndLearn