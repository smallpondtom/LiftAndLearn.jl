# Settings for the KS equation 
# NOTE: Only for a single parameter μ
KSE = LnL.ks(
    [0.0, 22.0], [0.0, 300.0], [1.0, 1.0],
    512, 0.001, 1, "ep"
)

# System structure of KSE
KSE_system = LnL.sys_struct(
    is_lin=true,
    is_quad=true,
)

# Variable information of KSE
KSE_vars = LnL.vars(
    N=1,
)

# Data structure of KSE
KSE_data = LnL.data(
    Δt=KSE.Δt,
    DS=100,
)

# Optimization settings for KSE
KSE_optim = LnL.opt_settings(
    verbose=true,
    initial_guess=false,
    max_iter=1000,
    reproject=false,
    SIGE=false,
    with_bnds=true,
    linear_solver="ma86",
)

# Standard OpInf options for KSE
options = LnL.LS_options(
    system=KSE_system,
    vars=KSE_vars,
    data=KSE_data,
    optim=KSE_optim,
)

# Downsampling rate
DS = KSE_data.DS

# Down-sampled dimension of the time data
Tdim_ds = size(1:DS:KSE.Tdim, 1) 

# Number of random test inputs
num_test_ic = 50

# Data pruning settings
prune_idx = KSE.Tdim ÷ 2
t_prune = KSE.t[prune_idx-1:end]

# Parameters of the initial condition
ic_a = [0.8, 1.0, 1.2]
ic_b = [0.2, 0.4, 0.6]
num_ic_params = Int(length(ic_a) * length(ic_b))
L = KSE.Omega[2] - KSE.Omega[1]  # length of the domain

# Parameterized function for the initial condition
u0 = (a,b) -> a * cos.((2*π*KSE.x)/L) .+ b * cos.((4*π*KSE.x)/L)  # initial condition
