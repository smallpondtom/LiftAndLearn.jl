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
using Statistics
using JuMP
using Ipopt, SCS
using FFTW
import HSL_jll

include("utils.jl")
include("OpInf_options.jl")
include("analyze.jl")
include("integrator.jl")
include("lift.jl")

# Include the optimizers
include("optimizer/NC_Optimize.jl")
include("optimizer/EP_Optimize.jl")

include("learn.jl")
include("intrusiveROM.jl")

# Inferring the Lyapunov function
include("LyapInf.jl")

end # module LiftAndLearn


