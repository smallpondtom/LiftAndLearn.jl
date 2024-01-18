"""
    Lyapunov Function Inference (LyapInf) Submodule
"""

module LyapInf

using DocStringExtensions
using LinearAlgebra
using Parameters

import ..LiftAndLearn: operators

include("nonintrusive.jl")

end