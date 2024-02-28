#############
## Packages
#############
using CairoMakie
using LinearAlgebra

###############
## My modules
###############
using LiftAndLearn
const LnL = LiftAndLearn

##########################
## 1D Heat equation setup
##########################
heat1d = LnL.heat1d(
    [0.0, 1.0], [0.0, 1.0], [0.1, 10],
    2^(-7), 1e-3, 10
)
heat1d.x = heat1d.x[2:end-1]

options = LnL.LS_options(
    system=LnL.sys_struct(
        is_lin=true,
        has_control=true,
        has_output=true,
    ),
    vars=LnL.vars(
        N=1,
    ),
    data=LnL.data(
        Δt=1e-3,
        deriv_type="BE"
    ),
    optim=LnL.opt_settings(
        verbose=true,
    ),
)


