"""
LiftAndLearn package main module
"""

module LiftAndLearn

using LinearAlgebra
using BlockDiagonals
using Parameters
using Plots
using SparseArrays
using MatrixEquations
using Statistics
using JuMP
using Ipopt, NLopt, MadNLP


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

export errBnds, compError, compProjError, compStateError, compOutputError, constraintResidual, symmetryResidual
export forwardEuler, backwardEuler, semiImplicitEuler
export intrusiveMR

export operators, extractF
export lifting
export inferOp, getDataMat

export NC_Optimize, NC_Optimize_output
export EPHEC_Optimize, EPSIC_Optimize

end # module LiftAndLearn


