"""
    Lyapunov Function Inference (LyapInf) Submodule
"""

module LyapInf

using DocStringExtensions
using JuMP
using Ipopt, SCS
using LinearAlgebra
using Parameters

import Distributions: Uniform
import MatrixEquations: lyapc
import Random: rand, rand!
import Sobol
import ..LiftAndLearn: operators, lifting, squareMatStates

include("doa.jl")
include("intrusive.jl")
include("nonintrusive.jl")

end