"""
2D heat equation: main execution file
"""

## Generate data
include("01-datagen.jl")
       
## Generate the POD basis
include("02-basis.jl")

## Train the models
include("03-train.jl")

## Test the models
include("04-test.jl")

## Plot the results
include("05-plot.jl")