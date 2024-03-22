"""
Low-dimensional examples for Lyapunov function Inference
"""

##############
## Packages
##############
using CairoMakie
using Kronecker
using LinearAlgebra
import Random: rand, rand!
import DifferentialEquations: solve, ODEProblem, RK4
import HSL_jll


################
## My modules
################
using LiftAndLearn
const LnL = LiftAndLearn
const LFI = LyapInf


################
## My functions
################ 
include("helper_functions/models.jl")
include("helper_functions/integrate_models.jl")  # load this before visualization.jl
include("helper_functions/visualization.jl")


##########
## CONSTS
##########
SAMPLE = true


#################################################
## Example 1: Lotka-Volterra Predator-Prey 
#################################################
ops = LotkaVolterra()  # Get the operators
datasetting = DataSetting(
    N=2,                    # dimension
    num_ic=20,              # number of initial conditions
    ti=0.0, tf=10.0,        # initial and final time
    dt=0.001,               # time step
    DS=100,                 # down-sampling
    x0_bnds=(-4.0, 4.0),    # initial condition bounds
    model_type="Q"          # model type
)
X, Xdot = generate_data(datasetting, ops)  # Generate the data

## Compute the Lyapunov Function using the intrusive LyapInf
lyapinf_options = LFI.Int_LyapInf_options(
    extra_iter=3,
    optimizer="ipopt",
    ipopt_linear_solver="ma86",
    verbose=true,
    optimize_PandQ="P",
    HSL_lib_path=HSL_jll.libhsl_path,
)
P1, Q, cost, ∇cost = LFI.Int_LyapInf(ops, X, lyapinf_options; Pi=1.0I(2))

## Compute the DoAs
ρ_min, ρ_max = LFI.DoA(P1)
ρ_est = LFI.skp_stability_rad(P1, ops.A, ops.H; dims=(1,2))

## Sample the maximum level c
c_star1, c_all, x_sample = nothing, nothing, nothing
if SAMPLE
    V1(x) = x' * P1 * x
    Vdot1(x) = 2*x' * P1 * ops.A * x + 2*x' * P1 * ops.F * (x ⊘ x)
    c_star1, c_all, x_sample = LFI.doa_sampling(
        V1,
        Vdot1,
        1000, 2, (-5,5);
        method="memory", history=true, n_strata=128, uniform_state_space=true
    )
else
    V1(x) = x' * P1 * x
    Vdot1(x) = 2*x' * P1 * A * x + 2*x' * P1 * F * (x ⊘ x)
    c_star1, _ = LFI.LEDOA(
        V1, Vdot1, 2; linear_solver="ma86", verbose=true, 
        HSL_lib_path=HSL_jll.libhsl_path, ci=1e2, xi=[10,10], δ=1
    )
end

## Print the results
ρ_star1 = sqrt(c_star1) * ρ_min
println("c_star1 = ", c_star1)
println("ρ_est = ", ρ_est)
println("ρ_min = ", ρ_min)
println("ρ_star = ", ρ_star1)

## Plot c convergence and DoAs for intrusive method
fig11 = plot_cstar_convergence(c_all)
fig12 = plot_doa_results(
    ops, c_star1, x_sample, P1[1:2,1:2], Vdot1, (-5,5), (-5,5);
    heatmap_lb=-1, meshsize=1e-2, ax2title="Intrusive LyapInf: DoA"
)
display(fig11)
display(fig12)

## Compute the Lyapunov function using Non-Intrusive LyapInf
lyapinf_options = LFI.NonInt_LyapInf_options(
    extra_iter=3,
    optimizer="ipopt",
    ipopt_linear_solver="ma86",
    verbose=true,
    optimize_PandQ="P",
    HSL_lib_path=HSL_jll.libhsl_path,
)
P2, Q, cost, ∇cost = LFI.NonInt_LyapInf(X, Xdot, lyapinf_options; Pi=1.0I(2))


## Compute the DoAs
ρ_min, ρ_max = LFI.DoA(P2)
ρ_est = LFI.skp_stability_rad(P2, ops.A, ops.H; dims=(1,2))

## Sample the maximum level c
V2 = (x) -> x' * P2 * x
Vdot2 = (x) -> 2*x' * P2 * ops.A * x + 2*x' * P2 * ops.F * (x ⊘ x)
c_star2, c_all, x_sample = LFI.doa_sampling(
    V2,
    Vdot2,
    1000, 2, (-5,5);
    method="memory", history=true, n_strata=8, uniform_state_space=true
)

## Print the results
ρ_star2 = sqrt(c_star2) * ρ_min
println("c_star1 = ", c_star2)
println("ρ_est = ", ρ_est)
println("ρ_min = ", ρ_min)
println("ρ_star = ", ρ_star2)

## Plot c convergence and DoAs for non-intrusive method
fig13 = plot_cstar_convergence(c_all)
fig14 = plot_doa_results(
    ops, c_star2, x_sample, P2[1:2,1:2], Vdot2, (-5,5), (-5,5);
    heatmap_lb=-5, meshsize=1e-2, ax2title="Non-Intrusive LyapInf: DoA"
)
display(fig13)
display(fig14)

## Plot the comparison of intrusive and non-intrusive methods
fig15 = plot_doa_comparison_results(
    ops.A, ops.F, c_star1, c_star2, P1, P2, Vdot1, Vdot2, (-5,5), (-5,5), ρ_est; 
    heatmap_lb=-5, meshsize=1e-2
)
display(fig15)

## Verify the DoA using Monte Carlo
ρ_mc, fig16 = verify_doa(
    ρ_star1, ρ_star2, 
    ([ops.A, ops.F], datasetting.ti, datasetting.tf, datasetting.dt, datasetting.model_type), 
    (-5,5), 5000; M=1.0, dim=datasetting.N
)
display(fig16)





#################################################
## Example 2: Van der Pol Oscillator (Cubic) 
#################################################
ops = VanDerPol_cubic(4.0)  # Get the operators
datasetting = DataSetting(
    N=2,                    # dimension
    num_ic=20,              # number of initial conditions
    ti=0.0, tf=5.0,         # initial and final time
    dt=0.001,               # time step
    DS=100,                 # down-sampling
    x0_bnds=(-4.0, 4.0),    # initial condition bounds
    model_type="C"          # model type
)
X, Xdot = generate_data(datasetting, ops)  # Generate the data

## Compute the Lyapunov Function using the intrusive LyapInf
lyapinf_options = LFI.Int_LyapInf_options(
    extra_iter=3,
    optimizer="SCS",
    ipopt_linear_solver="ma86",
    verbose=true,
    optimize_PandQ="P",
    HSL_lib_path=HSL_jll.libhsl_path,
    is_quad=false,
    is_cubic=true,
)
P1, Q, cost, ∇cost = LFI.Int_LyapInf(ops, X, lyapinf_options; Qi=1.0I(2))

## Compute the DoAs
ρ_min, ρ_max = LFI.DoA(P1)
ρ_est = LFI.skp_stability_rad(P1, ops.A, nothing, ops.G; dims=(1,3))

## Sample the maximum level c
V1(x) = x' * P1 * x
Vdot1(x) = 2*x' * P1 * ops.A * x + 2*x' * P1 * ops.E * ⊘(x,x,x)
c_star1, c_all, x_sample = LFI.doa_sampling(
    V1,
    Vdot1,
    1000, 2, (-3.0,3.0);
    method="memory", history=true
)

## Print the results
ρ_star1 = sqrt(c_star1) * ρ_min
println("c_star1 = ", c_star1)
println("ρ_est = ", ρ_est)
println("ρ_min = ", ρ_min)
println("ρ_star = ", ρ_star1)

## Plot c convergence and DoAs for intrusive method
fig21 = plot_cstar_convergence(c_all)
fig22 = plot_doa_results(
    ops, c_star1, x_sample, P1, Vdot1, (-3.0,3.0), (-3.0,3.0);
    heatmap_lb=-5e-2, meshsize=1e-2, ax2title="Intrusive LyapInf: DoA", dims="C"
)
display(fig21)
display(fig22)

## Compute the Lyapunov function using Non-Intrusive LyapInf
lyapinf_options = LFI.NonInt_LyapInf_options(
    extra_iter=3,
    optimizer="SCS",
    ipopt_linear_solver="ma86",
    verbose=true,
    optimize_PandQ="P",
    HSL_lib_path=HSL_jll.libhsl_path,
)
P2, Q, cost, ∇cost = LFI.NonInt_LyapInf(X, Xdot, lyapinf_options)

## Compute the DoAs
ρ_min, ρ_max = LFI.DoA(P2)
ρ_est = LFI.skp_stability_rad(P2, ops.A, nothing, ops.G; dims=(1,3))

## Sample the maximum level c
V2 = (x) -> x' * P2 * x
Vdot2 = (x) -> 2*x' * P2 * ops.A * x + 2*x' * P2 * ops.E * ⊘(x,x,x)
c_star2, c_all, x_sample = LFI.doa_sampling(
    V2,
    Vdot2,
    1000, 2, (-3.0,3.0);
    method="memory", history=true
)

## Print the results
ρ_star2 = sqrt(c_star2) * ρ_min
println("c_star1 = ", c_star2)
println("ρ_est = ", ρ_est)
println("ρ_min = ", ρ_min)
println("ρ_star = ", ρ_star2)

## Plot c convergence and DoAs for non-intrusive method
fig23 = plot_cstar_convergence(c_all)
fig24 = plot_doa_results(
    ops, c_star2, x_sample, P2, Vdot2, (-3.0,3.0), (-3.0,3.0);
    heatmap_lb=-5e-2, meshsize=1e-2, ax2title="Non-Intrusive LyapInf: DoA", dims="C"
)
display(fig23)
display(fig24)

## Plot the comparison of intrusive and non-intrusive methods
fig25 = plot_doa_comparison_results(
    ops.A, ops.E, c_star1, c_star2, P1, P2, Vdot1, Vdot2, (-3,3), (-3,3), ρ_est; 
    meshsize=1e-2, dims=3
)
display(fig25)

## Verify the DoA
ρ_mc, fig26 = verify_doa(
    ρ_star1, ρ_star2, 
    ([ops.A, ops.E], datasetting.ti, datasetting.tf, datasetting.dt, datasetting.model_type),
    (-3.0,3.0), 5000; M=1.0, dim=datasetting.N
)
display(fig26)





#################################################
## Example 3: Stable 3D System Example
#################################################
using GLMakie  # switch Makie backend for 3D plots

## 
ops = Stable3D()  # Get the operators
datasetting = DataSetting(
    N=3,                    # dimension
    num_ic=20,              # number of initial conditions
    ti=0.0, tf=10.0,         # initial and final time
    dt=0.001,               # time step
    DS=100,                 # down-sampling
    x0_bnds=(-3.0, 3.0),    # initial condition bounds
    model_type="QC"         # model type
)
X, Xdot = generate_data(datasetting, ops)  # Generate the data

## Compute the Lyapunov Function using the intrusive LyapInf
lyapinf_options = LFI.Int_LyapInf_options(
    extra_iter=3,
    optimizer="SCS",
    ipopt_linear_solver="ma86",
    verbose=true,
    optimize_PandQ="P",
    HSL_lib_path=HSL_jll.libhsl_path,
    is_quad=true,
    is_cubic=true,
)
P1, Q, cost, ∇cost = LFI.Int_LyapInf(ops, X, lyapinf_options; Pi=1.0I(3))

## Compute the DoAs
ρ_min, ρ_max = LFI.DoA(P1)
ρ_est = LFI.skp_stability_rad(P1, ops.A, ops.H, ops.G; dims=(1,2,3))

## Sample the maximum level c
V1(x) = x' * P1 * x
Vdot1(x) = 2*x' * P1 * ops.A * x + 2*x' * P1 * ops.F * (x ⊘ x) + 2*x' * P1 * ops.E * ⊘(x,x,x)
c_star1, c_all, x_sample = LFI.doa_sampling(
    V1,
    Vdot1,
    1000, 3, (-4.0,4.0);
    method="memory", history=true
)

## Print the results
ρ_star1 = sqrt(c_star1) * ρ_min
println("c_star1 = ", c_star1)
println("ρ_est = ", ρ_est)
println("ρ_min = ", ρ_min)
println("ρ_star = ", ρ_star1)

## Plot c convergence and DoAs for intrusive method
fig31 = plot_cstar_convergence(c_all)
fig32 = plot_doa_results_3D(
    ops, c_star1, x_sample, P1, Vdot1, (-7.0,7.0), (-7.0,7.0), (-7.0,7.0);
    meshsize=1e-1, ax2title="Intrusive LyapInf: DoA", dims="QC",
    with_streamplot=false, with_samples=false, animate=false, contour_levels=49
)
display(GLMakie.Screen(), fig31)
display(GLMakie.Screen(), fig32)


## Compute the Lyapunov function using Non-Intrusive LyapInf
lyapinf_options = LFI.NonInt_LyapInf_options(
    extra_iter=3,
    optimizer="SCS",
    ipopt_linear_solver="ma86",
    verbose=true,
    optimize_PandQ="P",
    HSL_lib_path=HSL_jll.libhsl_path,
)
P2, Q, cost, ∇cost = LFI.NonInt_LyapInf(X, Xdot, lyapinf_options)

## Compute the DoAs
ρ_min, ρ_max = LFI.DoA(P2)
ρ_est = LFI.skp_stability_rad(P2, ops.A, ops.H, ops.G; dims=(1,2,3))

## Sample the maximum level c
V2 = (x) -> x' * P2 * x
Vdot2 = (x) -> 2*x' * P2 * ops.A * x + 2*x' * P2 * ops.F * (x ⊘ x) + 2*x' * P2 * ops.E * ⊘(x,x,x)
c_star2, c_all, x_sample = LFI.doa_sampling(
    V2,
    Vdot2,
    1000, 3, (-3.0,3.0);
    method="memory", history=true
)

## Print the results
ρ_star2 = sqrt(c_star2) * ρ_min
println("c_star2 = ", c_star2)
println("ρ_est = ", ρ_est)
println("ρ_min = ", ρ_min)
println("ρ_star = ", ρ_star2)

## Plot c convergence and DoAs for non-intrusive method
fig33 = plot_cstar_convergence(c_all)
fig34 = plot_doa_results_3D(
    ops, c_star2, x_sample, P2, Vdot2, (-7.0,7.0), (-7.0,7.0), (-7.0,7.0);
    meshsize=1e-1, ax2title="Non-Intrusive LyapInf: DoA", dims="QC",
    with_streamplot=true, with_samples=true, animate=false, contour_levels=49
)
display(GLMakie.Screen(), fig33)
display(GLMakie.Screen(), fig34)


## Verify the DoA
ρ_mc, fig35 = verify_doa_3D(
    ρ_star1, ρ_star2, 
    ([ops.A, ops.F, ops.E], datasetting.ti, datasetting.tf, datasetting.dt, datasetting.model_type),
    (-7.0,7.0), 5000; M=1.0, dim=datasetting.N, animate=false
)
display(fig35)