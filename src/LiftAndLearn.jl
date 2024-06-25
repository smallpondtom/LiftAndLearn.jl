"""
    LiftAndLearn package main module
"""
module LiftAndLearn

using LinearAlgebra
using BlockDiagonals
using Distributions: Uniform
using Kronecker
using Parameters
using QuasiMonteCarlo
using SparseArrays
using StatsBase: countmap
using MatrixEquations: lyapc
using Random: rand, rand!, shuffle
using JuMP
using Ipopt, SCS, Alpine
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
include("utilities/unique_kronecker.jl")
include("utilities/vech.jl")
include("utilities/invec.jl")
export ⊘, vech, invec
include("utilities/streamify.jl")
include("utilities/fat2tall.jl")
include("utilities/tall2fat.jl")

# Operators and tools
include("operators/operators.jl")
include("operators/bilinear.jl")
include("operators/quadratic.jl")
include("operators/cubic.jl")

# Other utilities (for the sake of ordering)
include("utilities/OpInf_options.jl")
include("utilities/analyze.jl")
include("utilities/integrator.jl")

# Intrusive POD
include("POD/pod.jl")

# Operator Inference
include("OpInf/learn.jl")

# Lift & Learn
include("LnL/lift.jl")
include("LnL/learn.jl")

# Include the optimization methods
include("optimizer/NC_Optimize.jl")
include("optimizer/EP_Optimize.jl")

# Streaming OpInf
include("Streaming/streaming.jl")

# [Submodule] Analysis of chaos analysis tools
include("ChaosGizmo/ChaosGizmo.jl")

# Include the system models
include("model/Models.jl")

end # module LiftAndLearn