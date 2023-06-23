using DataFrames
using FileIO
using LinearAlgebra
using Plots
using ProgressMeter
using Random
using SparseArrays
using Statistics
using NaNStatistics
using JLD2


include("../../src/model/Burgers.jl")
include("../../src/LiftAndLearn.jl")
const LnL = LiftAndLearn

burger = Burgers(
    [0.0, 1.0], [0.0, 1.0], [0.1, 1.0],
    2^(-7), 1e-4, 10
);
burger.IC = sin.(2 * pi * burger.x)
ic_a = 1.0 # coefficient that changes the initial condition for training data

num_ICs = length(ic_a)
rmin = 1
rmax = 15

options = LnL.OpInf_options(
    reproject=false,
    is_quad=true,
    has_control=false,
    has_output=false,  # suppress output
    optimization="none",  #!!! This options changes the problem into an optimization problem
    opt_verbose=false,
    initial_guess_for_opt=false,
    which_quad_term="F",
    N=1,
    Δt=1e-4,
    deriv_type="SI"
)

# Downsampling rate
DS = 10

# Kinetic viscosity 
μ = 0.1
