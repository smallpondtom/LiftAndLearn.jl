"""
Kuramoto–Sivashinsky equation EP-OpInf main file
"""

#================#
## Generate data
#================#
include("01-datagen.jl")

#============================#
## Compute the POD bases
#============================#
include("02-basis.jl")

#==========================#
## Train the reduced models
#==========================#
include("03-train.jl")

#========================#
## Analysis for training
#========================#
include("04-train-analysis.jl")

#=======================#
## Analysis for testing
#=======================#
include("05-test-analysis.jl")

#===============#
## Plot results
#===============#
include("06-plot.jl")