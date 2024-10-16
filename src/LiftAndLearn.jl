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

end # module LiftAndLearn