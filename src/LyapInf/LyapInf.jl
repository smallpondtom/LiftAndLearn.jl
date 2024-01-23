"""
    Lyapunov Function Inference (LyapInf) Submodule
"""

module LyapInf

using DocStringExtensions
using JuMP
using Ipopt, SCS
using LinearAlgebra
using MatrixEquations
using Parameters

import ..LiftAndLearn: operators, squareMatStates

include("intrusive.jl")

end