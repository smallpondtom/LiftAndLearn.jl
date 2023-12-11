"""
LiftAndLearn package main module
"""

module LiftAndLearn

using LinearAlgebra
using BlockDiagonals
using Parameters
using SparseArrays
using MatrixEquations
using Random
using JuMP
using Ipopt, SCS
using FFTW

abstract type Abstract_Options end
abstract type Abstract_Model end

include("utils.jl")
include("OpInf_options.jl")
include("analyze.jl")
include("integrator.jl")
include("lift.jl")

# Include the optimizers
include("optimizer/NC_Optimize.jl")
include("optimizer/EP_Optimize.jl")
include("optimizer/PP-ZQLFI.jl")

include("learn.jl")
include("intrusiveROM.jl")

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


