"""
    LiftAndLearn package main module
"""
module LiftAndLearn

using LinearAlgebra
using BlockDiagonals
using Kronecker
using Parameters
using SparseArrays
using JuMP
using Ipopt, SCS
using DocStringExtensions
using UniqueKronecker

"""
    AbstractOption

Abstract type for the options.
"""
abstract type AbstractOption end

# Utilities
include("utilities/utilities.jl")

# Options
include("OpInf/opinf_options.jl")

# Operators and tools
include("operators/operators.jl")

# Intrusive POD
include("POD/pod.jl")

# Operator Inference
include("OpInf/learn.jl")

# Utilities
include("utilities/analyze.jl")

# Lift & Learn
include("LnL/lift.jl")
include("LnL/learn.jl")

# Include the optimization methods
# include("optimizer/NC_Optimize.jl")
# include("optimizer/EP_Optimize.jl")

# Streaming OpInf
# include("Streaming/streaming.jl")
# include("Streaming/streamify.jl")

# [Submodule] Chaos analysis tools
# include("ChaosGizmo/ChaosGizmo.jl")

end # module LiftAndLearn