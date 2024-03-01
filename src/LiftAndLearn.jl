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
    Abstract_Option

Abstract type for the options.
"""
abstract type Abstract_Option end
"""
    Abstract_Model

Abstract type for the model.
"""
abstract type Abstract_Model end

include("utils.jl")
export ⊘, vech

include("OpInf_options.jl")
include("analyze.jl")
include("integrator.jl")
include("lift.jl")

# Include the optimizers
include("optimizer/NC_Optimize.jl")
include("optimizer/EP_Optimize.jl")

include("learn.jl")
include("intrusiveROM.jl")
include("streaming.jl")

# [Submodule] Inferring the Lyapunov function
include("LyapInf/LyapInf.jl")
export LyapInf

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


