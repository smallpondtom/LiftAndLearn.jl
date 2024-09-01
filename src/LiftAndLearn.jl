"""
    LiftAndLearn package main module
"""
module LiftAndLearn

using LinearAlgebra
using BlockDiagonals
using Distributions: Uniform
using Kronecker
using Parameters
using SparseArrays
using StatsBase: countmap
using MatrixEquations: lyapc
using Random: rand, rand!, shuffle
using Combinatorics: permutations, factorial, binomial, with_replacement_combinations
using JuMP
using Ipopt, SCS
using FFTW
using DocStringExtensions

"""
    AbstractOption

Abstract type for the options.
"""
abstract type AbstractOption end
"""
    AbstractModel

Abstract type for the model.
"""
abstract type AbstractModel end

# Utilities
include("utilities/utilities.jl")

# Operators and tools
include("operators/operators.jl")

# Intrusive POD
include("POD/pod.jl")

# Operator Inference
include("OpInf/learn.jl")
include("OpInf/OpInf_options.jl")

# Other utilities (for the sake of ordering)
include("utilities/analyze.jl")
include("utilities/integrator.jl")

# Lift & Learn
include("LnL/lift.jl")
include("LnL/learn.jl")

# Include the optimization methods
include("optimizer/NC_Optimize.jl")
include("optimizer/EP_Optimize.jl")

# Streaming OpInf
include("Streaming/streaming.jl")
include("Streaming/streamify.jl")

# [Submodule] Chaos analysis tools
include("ChaosGizmo/ChaosGizmo.jl")

# Include the system models
include("model/Models.jl")

end # module LiftAndLearn