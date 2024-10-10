"""
Kuramoto–Sivashinsky equation EP-OpInf main file
"""

#================#
## Generate data
#================#
include("kse/kse_epopinf_datagen.jl")

#============================#
## Compute reduced operators
#============================#
include("kse/kse_epopinf_reduction.jl")

#========================#
## Analysis for training
#========================#
include("kse/kse_epopinf_training.jl")

#======================#
## Analysis for test 1
#======================#
include("kse/kse_epopinf_test1.jl")

#======================#
## Analysis for test 2
#======================#
include("kse/kse_epopinf_test2.jl")