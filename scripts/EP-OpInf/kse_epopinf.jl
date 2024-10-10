"""
Kuramoto–Sivashinsky equation EP-OpInf main file
"""

#================#
## Generate data
#================#
include("kse_epopinf_datagen.jl")

#============================#
## Compute reduced operators
#============================#
include("kse_epopinf_reduction.jl")

#========================#
## Analysis for training
#========================#
include("kse_epopinf_training.jl")