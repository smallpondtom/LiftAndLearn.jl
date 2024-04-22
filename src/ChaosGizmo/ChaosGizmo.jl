"""
    Gizmos to conduct chaos analysis.
"""
module ChaosGizmo

using DocStringExtensions
using LinearAlgebra
using Parameters
using SparseArrays

import ..LiftAndLearn: Abstract_Model, operators, extractF, extractH

include("integrators.jl")
include("LyapunovExponent.jl")

end