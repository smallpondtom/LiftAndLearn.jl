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
include("utilities/opinf_options.jl")
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
include("model/Heat1D.jl")
include("model/Burgers.jl")
include("model/FitzHughNagumo.jl")
include("model/KS.jl")
include("model/FisherKPP.jl")
include("model/Heat2D.jl")
include("model/ChafeeInfante.jl")
using .Heat1D: Heat1DModel
using .Burgers: BurgersModel
using .FitzHughNagumo: FitzHughNagumoModel
using .KS: ks
using .FisherKPP: FisherKPPModel
using .Heat2D: Heat2DModel
using .ChafeeInfante: ChafeeInfanteModel

end # module LiftAndLearn