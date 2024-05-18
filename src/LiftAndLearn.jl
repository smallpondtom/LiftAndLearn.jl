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
include("utilities/batchify.jl")

# Operators and tools
include("operators/operators.jl")
include("operators/bilinear.jl")
include("operators/quadratic.jl")
include("operators/cubic.jl")

# OpInf & LnL
include("OpInf_options.jl")
include("analyze.jl")
include("integrator.jl")
include("lift.jl")
include("learn.jl")
include("intrusiveROM.jl")

# Include the optimizers
include("optimizer/NC_Optimize.jl")
include("optimizer/EP_Optimize.jl")

# Streaming OpInf
include("streaming.jl")

# [Submodule] Analysis of chaos analysis tools
include("ChaosGizmo/ChaosGizmo.jl")

# Include the models
include("model/Heat1D.jl")
include("model/Burgers.jl")
include("model/FHN.jl")
include("model/KS.jl")
using .Heat1D: heat1d
using .Burgers: burgers
using .FHN: fhn
using .KS: ks

end # module LiftAndLearn


