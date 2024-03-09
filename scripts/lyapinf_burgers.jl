#############
## Packages
#############
using CairoMakie
using Distributions: Uniform
using Kronecker
using LinearAlgebra
import HSL_jll

###############
## My modules
###############
using LiftAndLearn
const LnL = LiftAndLearn
const LFI = LyapInf

###############
## Setup 
###############
# First order Burger's equation setup
burgers = LnL.burgers(
    [0.0, 1.0], [0.0, 1.0], [0.10, 0.10],
    2^(-7), 1e-4, 1, "periodic"
)

## Reduced dimensions and OpInf options
rmin = 1  # min dimension
rmax = 10  # max dimension

opinf_options = LnL.LS_options(
    system=LnL.sys_struct(
        is_lin=true,
        is_quad=true,
    ),
    vars=LnL.vars(
        N=1,
    ),
    data=LnL.data(
        Δt=burgers.Δt,
        DS=100,
    ),
    optim=LnL.opt_settings(
        verbose=true,
        initial_guess=true,
        max_iter=1000,
        reproject=false,
        SIGE=false,
        linear_solver="ma86",
    ),
)

# Downsampling rate
DS = opinf_options.data.DS
Tdim_ds = size(1:DS:burgers.Tdim, 1)  # downsampled time dimension

## Initial condition
IC = Dict(
    :a => [3, 5, 7],
    :b => [-0.5, 0, 0.5],
    :c => [-0.25, 0, 0.25],
    :test => ([0.1, 10.0], [-1.0, 1.0], [-0.5, 0.5]),
)
IC[:n] = Int(length(IC[:a])*length(IC[:b])*length(IC[:c]))
IC[:f] = (a,b,c) -> exp.(-a * cos.(π .* burgers.x .+ b).^2) .+ c


################################
## Generate the training data
################################
A, F = burgers.generateEPmatrix(burgers, burgers.μs[1])
op_fom = LnL.operators(A=A, F=F)
    
Xall = Vector{Matrix{Float64}}(undef, IC[:n])
Xdotall = Vector{Matrix{Float64}}(undef, IC[:n])

ct = 1
for a in IC[:a], b in IC[:b], c in IC[:c]
    states = burgers.semiImplicitEuler(A, F, burgers.t, IC[:f](a, b, c))
    tmp = states[:, 2:end]
    Xall[ct] = tmp[:, 1:DS:end]  # downsample data
    tmp = (states[:, 2:end] - states[:, 1:end-1]) / burgers.Δt
    Xdotall[ct] = tmp[:, 1:DS:end]  # downsample data
    @info "(Loop #$ct) Generated training data for a = $a, b = $b, c = $c"
    ct += 1
end
X = reduce(hcat, Xall)
R = reduce(hcat, Xdotall)

# Compute the POD basis from the training data
Vrmax = svd(X).U[:, 1:rmax]

#####################################
## Compute the intrusive POD model
#####################################
op_int = LnL.intrusiveMR(op_fom, Vrmax, opinf_options)

#####################################
## Compute the OpInf model
#####################################
op_inf = LnL.inferOp(X, zeros(Tdim_ds,1), zeros(Tdim_ds,1), Vrmax, Vrmax' * R, opinf_options)

## Create a batch of training data
# batchsize = 300
# X_batch = map(Iterators.partition(axes(X,2), batchsize)) do cols
#     X[:, cols]
# end

#####################################
## Intrusive LyapInf for POD model
#####################################
ds2= 1  # another downsampling for lyapinf
int_lyapinf_options = LFI.Int_LyapInf_options(
    extra_iter=3,
    optimizer="SCS",
    ipopt_linear_solver="ma86",
    verbose=true,
    optimize_PandQ="P",
    opt_max_iter=500,
    δJ=1e-5,
    HSL_lib_path=HSL_jll.libhsl_path,
    α=1e-6,
    β=1e-6
)
P_int, Q_int, cost, ∇cost = LFI.Int_LyapInf(op_int, Vrmax' * X[:,1:ds2:end], int_lyapinf_options)

#####################################
## Intrusive LyapInf for OpInf model
#####################################
P_inf, Q_inf, cost, ∇cost = LFI.Int_LyapInf(op_inf, Vrmax' * X[:,1:ds2:end], int_lyapinf_options)

#####################################
## Non-intrusive LyapInf
#####################################
nonint_lyapinf_options = LFI.NonInt_LyapInf_options(
    extra_iter=3,
    optimizer="SCS",
    ipopt_linear_solver="ma86",
    verbose=true,
    optimize_PandQ="P",
    opt_max_iter=500,
    δJ=1e-5,
    HSL_lib_path=HSL_jll.libhsl_path,
    α=1e-6,
)
P_star, Q_star, cost, ∇cost = LFI.NonInt_LyapInf(Vrmax' * X[:,1:ds2:end], Vrmax' * R[:,1:ds2:end], nonint_lyapinf_options)


################################################
## Sample the max level surface (for r = 10)
################################################
sampling = true
##
# POD
V = (x) -> (x' * P_int * x)[1]
Vdot = (x) -> x' * P_int * op_int.A * x + x' * P_int * op_int.F * (x ⊘ x)

if sampling
    c_star1, c_all1, _ = LFI.doa_sampling(
        V,
        Vdot,
        1e6, rmax, [(-4.4,4.4) for _ in 1:rmax]; n_strata=2^4,
        method="enhanced", history=true, uniform_state_space=true, gp=burgers.Xdim,
        sampler="sobol"
    )
else
    c_star1, _ = LFI.LEDOA(V, Vdot, rmax; linear_solver="ma86", verbose=true, HSL_lib_path=HSL_jll.libhsl_path)
end
ρmin1 = sqrt(1/maximum(eigvals(P_int)))
ρstar1 = sqrt(c_star1/maximum(eigvals(P_int)))
# ρskp1 = LFI.skp_stability_rad(P_int, op_int.A, op_int.H)
ρskp1 = 1.0
println("POD: c* = $c_star1, ρ = $ρmin1, ρstar = $ρstar1, ρskp = $ρskp1")

##
fig1 = Figure(fontsize=20)
ax1 = Axis(fig1[1,1],
    title="Level Convergence",
    ylabel=L"c_*",
    xlabel="Sample Number",
    xticks=0:(length(c_all1)÷4):length(c_all1),
)
lines!(ax1, 1:length(c_all1), c_all1)
display(fig1)

## OpInf
V = (x) -> x' * P_inf * x
Vdot = (x) -> x' * P_inf * op_inf.A * x + x' * P_inf * op_inf.F * (x ⊘ x)
c_star2, c_all2, _ = LFI.doa_sampling(
    V,
    Vdot,
    1e6, rmax, [(-40,40) for _ in 1:rmax]; n_strata=2^4,
    method="enhanced", history=true, uniform_state_space=false, gp=burgers.Xdim
)
ρmin2 = sqrt(1/maximum(eigvals(P_inf)))
ρstar2 = sqrt(c_star2/maximum(eigvals(P_inf)))
# ρskp2 = LFI.skp_stability_rad(P_inf, op_inf.A, op_inf.H)
ρskp2 = 1.0
println("OpInf: c* = $c_star2, ρmin = $ρmin2, ρstar = $ρstar2, ρskp = $ρskp2")

##
fig1 = Figure(fontsize=20)
ax1 = Axis(fig1[1,1],
    title="Level Convergence",
    ylabel=L"c_*",
    xlabel="Sample Number",
    xticks=0:2.5e6:length(c_all2),
)
lines!(ax1, 1:length(c_all2), c_all2)
display(fig1)

## Non-intrusive
V = (x) -> x' * P_star * x
Vdot = (x) -> x' * P_star * op_int.A * x + x' * P_star * op_int.F * (x ⊘ x)
c_star3 = LFI.doa_sampling(
    V,
    Vdot,
    1e6, rmax, [(-4.4,4.4) for _ in 1:rmax]; n_strata=2^4,
    method="enhanced", history=false, uniform_state_space=true, gp=burgers.Xdim
)
ρmin3 = sqrt(1/maximum(eigvals(P_star)))
ρstar3 = sqrt(c_star3/maximum(eigvals(P_star)))
# ρskp3 = LFI.skp_stability_rad(P_star, op_int.A, op_int.H)
ρskp3 = 1.0
println("Non-intrusive: c* = $c_star3, ρmin = $ρmin3, ρstar = $ρstar3, ρskp = $ρskp3")



##############################################
## Plot the DoA for all reduced dimensions
##############################################
function interpolate_extrapolate_column(v)
    # Find valid (non-NaN and non-zero) indices and corresponding values
    valid_indices = findall(x -> !isnan(x) && x != 0, v)
    valid_values = v[valid_indices]

    if isempty(valid_indices)
        return v  # Return original if no valid points
    end

    # Extrapolate at the beginning if needed
    if isnan(v[1]) || iszero(v[1])
        if length(valid_indices) > 1
            slope = (valid_values[2] - valid_values[1]) / (valid_indices[2] - valid_indices[1])
            v[1] = valid_values[1] - slope * (valid_indices[1] - 1)
        else
            v[1] = valid_values[1]  # Use the single valid value if only one exists
        end
    end

    # Interpolate missing values
    for i in 2:length(v) - 1
        if isnan(v[i]) || v[i] == 0
            lower_index = findlast(j -> j < i, valid_indices)
            upper_index = findfirst(j -> j > i, valid_indices)
            if isnothing(lower_index) || isnothing(upper_index)
                continue  # Skip if there's no surrounding valid values
            end
            # Linear interpolation
            v[i] = ((valid_values[lower_index] * (valid_indices[upper_index] - i) +
                     valid_values[upper_index] * (i - valid_indices[lower_index])) /
                    (valid_indices[upper_index] - valid_indices[lower_index]))
        end
    end

    # Extrapolate at the end if needed
    if isnan(v[end]) || iszero(v[end])
        if length(valid_indices) > 1
            slope = (valid_values[end] - valid_values[end - 1]) / (valid_indices[end] - valid_indices[end - 1])
            v[end] = valid_values[end] + slope * (length(v) - valid_indices[end])
        else
            v[end] = valid_values[end]  # Use the single valid value if only one exists
        end
    end

    return v
end
##
ρ_all = zeros(rmax,3)
ρskp_all = zeros(rmax,2)
fig2 = Figure(size=(900,600), fontsize=20)
ax = Axis(fig2[1,1],
    ylabel=L"DoA $$",
    xlabel=L"reduced dimension, $r$",
    xticks=rmin:rmax,
    yscale=log10,
)
for (i,r) in enumerate(rmin:rmax)
    # Intrusive POD
    P = P_int[1:r, 1:r]
    A = op_int.A[1:r, 1:r]
    H = LnL.extractH(op_int.H, r)
    F = LnL.extractF(op_int.F, r)
    V = (x) -> x' * P * x
    Vdot = (x) -> x' * P * A * x + x' * P * F * (x ⊘ x)
    c_star = LFI.doa_sampling(
        V,
        Vdot,
        1e6, r, (-100,100);
        method="memory", history=false, uniform_state_space=true
    )
    ρ_all[i,1] = sqrt(c_star/maximum(eigvals(P)))
    ρskp_all[i,1] = LFI.skp_stability_rad(A, H, Q_int[1:r,1:r])

    # OpInf
    P = P_inf[1:r, 1:r]
    A = op_inf.A[1:r, 1:r]
    H = LnL.extractH(op_inf.H, r)
    F = LnL.extractF(op_inf.F, r)
    V = (x) -> x' * P * x
    Vdot = (x) -> x' * P * A * x + x' * P * F * (x ⊘ x)
    c_star = LFI.doa_sampling(
        V,
        Vdot,
        1e6, r, (-100,100);
        method="memory", history=false, uniform_state_space=true
    )
    ρ_all[i,2] = sqrt(c_star/maximum(eigvals(P)))
    ρskp_all[i,2] = LFI.skp_stability_rad(A, H, Q_inf[1:r,1:r])

    # Non-intrusive
    P = P_star[1:r, 1:r]
    A = op_int.A[1:r, 1:r]
    H = LnL.extractH(op_int.H, r)
    F = LnL.extractF(op_int.F, r)
    V = (x) -> x' * P * x
    Vdot = (x) -> x' * P * A * x + x' * P * F * (x ⊘ x)
    c_star = LFI.doa_sampling(
        V,
        Vdot,
        1e6, r, (-100,100);
        method="memory", history=false, uniform_state_space=true
    )
    ρ_all[i,3] = sqrt(c_star/maximum(eigvals(P)))
end
ρ_all_save = copy(ρ_all)
ρ_all = mapslices(interpolate_extrapolate_column, ρ_all; dims=1)
scatterlines!(ax, rmin:rmax, ρ_all[:,1], label="POD")
scatterlines!(ax, rmin:rmax, ρ_all[:,2], label="OpInf")
scatterlines!(ax, rmin:rmax, ρ_all[:,3], label="Non-intrusive")
scatterlines!(ax, rmin:rmax, ρskp_all[:,1], label="POD (SKP)", linestyle=:dash)
scatterlines!(ax, rmin:rmax, ρskp_all[:,2], label="OpInf (SKP)", linestyle=:dash)
fig2[1,2] = Legend(fig2, ax)
display(fig2)



##################
## Verify DoA
##################
## Function
function verify_DoA(
    op, ic_param_bnd, ic_func; 
    Vr=nothing, max_iter=500, full=true, xticks=false, xlim=false, legend_loc=:top
)
    fig = Figure(size=(900,600), fontsize=20)
    if full
        ax = Axis(fig[1,1],
            ylabel=L"\log\left(\frac{\max\Vert\mathbf{x}(t)\Vert_2}{\Vert\mathbf{x}_0\Vert_2}\right)",
            xlabel=L"\Vert\mathbf{x}_0\Vert_2",
        )
    else
        ax = Axis(fig[1,1],
            ylabel=L"\log\left(\frac{\max\Vert\hat{\mathbf{x}}(t)\Vert_2}{\Vert\hat{\mathbf{x}}_0\Vert_2}\right)",
            xlabel=L"\Vert\hat{\mathbf{x}}_0\Vert_2",
        )
    end        

    A, F = op.A, op.F
    N = length(ic_param_bnd)
    ρ_true = Inf
    for _ in 1:max_iter
        foo = zeros(N)
        for (i, p) in enumerate(ic_param_bnd)
            foo[i] = rand(Uniform(p[1], p[2]), 1, 1)[1]
        end        

        if full
            ic_ = ic_func(foo...)
            ic_norm = norm(ic_)
            states = burgers.semiImplicitEuler(A, F, burgers.t, ic_)
        else
            ic_ = ic_func(foo...)
            ic_ = Vr' * ic_  # Project the initial condition to the reduced space
            ic_norm = norm(ic_)
            states = burgers.semiImplicitEuler(A, F, burgers.t, ic_)
        end

        if any(isnan.(states)) || any(isinf.(states))
            scatter!(ax, ic_norm, 0, color=:black, label="", marker=:x, markersize=15)

            # Update the estimated true DoA
            if ρ_true > ic_norm
                ρ_true = ic_norm
            end
        else
            state_norms = norm.(eachcol(states), 2)
            max_norm = maximum(state_norms)
            
            # Check if the maximum norm throughout the trajectory stays within the domain of attraction
            if max_norm <= ic_norm
                scatter!(ax, ic_norm, log.(max_norm ./ ic_norm), color=:green3, strokewidth=0, label="")
            else
                scatter!(ax, ic_norm, log.(max_norm ./ ic_norm), color=:red, strokewidth=0, label="")

                # Update the estimated true DoA
                if ρ_true > ic_norm
                    ρ_true = ic_norm
                end
            end
        end
    end
    vlines!(ax, ρ_true, color=:blue3, linestyle=:dash, label="")

    # Legend
    elem1 = MarkerElement(color=:green3, markersize=10, marker=:circle, strokewidth=0)
    elem2 = MarkerElement(color=:red, markersize=10, marker=:circle, strokewidth=0)
    elem3 = MarkerElement(color=:black, markersize=15, marker=:x)
    elem4 = LineElement(color=:blue3, linestyle=:dash, linewidth=3)
    Legend(fig[1,2],
        [elem1, elem2, elem3, elem4], 
        [
            L"$\max\Vert\textbf{x}(t)\Vert_2 ~\leq~\Vert\textbf{x}_0\Vert_2$", 
            L"$\max\Vert\textbf{x}(t)\Vert_2 ~>~\Vert\textbf{x}_0\Vert_2$",
            L"Unstable: NaN or Inf detected $$",
            L"True DoA: %$(round(ρ_true,digits=4)) $$",
        ], rowgap = 8
    )
    return ρ_true, fig
end

# Initial Condition: exp.(-a * cos.(π .* x .+ b).^2) .+ c
## full
ρ_true1, fig3 = verify_DoA(
    op_fom, ([0.001, 300], [-100, 100], [-100, 100]), IC[:f];
    max_iter=1e3,
)
display(fig3)

## POD-Intrusive model
ρ_true2, fig4 = verify_DoA(
    op_int, ([0.001, 200], [-80, 80], [-80, 80]), IC[:f];
    max_iter=1e3, full=false, Vr=Vrmax
)
display(fig4)

# Initial Condition: a*exp.(-b * cos.(π .* x .+ c).^2) .- d
## POD-Intrusive model
ic_params = (
    [-20, 20],
    [0, 200],
    [-80, 80],
    [-80, 80]
)
ic_func = (a,b,c,d) -> a*exp.(-b * cos.(π .* burgers.x .+ c).^2) .- d
ρ_true3, fig5 = verify_DoA(
    op_int, ic_params, ic_func;
    max_iter=1e3, full=false, Vr=Vrmax
)
display(fig5)

# Initial Condition: a * exp.(-b * cos.(π .* x).^2 .+ c .* sin.(4*π .* x .+ π/3).^2 .- d .* cos.(3*π .* x).^2)
## POD-Intrusive model
ic_params = (
    [-5, 5],
    [-2, 1],
    [-1, 2],
    [-1, 2],
)
ic_func = (a,b,c,d) -> a * exp.(-b * cos.(π .* burgers.x).^2 .+ c .* sin.(4*π .* burgers.x .+ π/3).^2 .- d .* cos.(3*π .* burgers.x).^2) 
ρ_true4, fig6 = verify_DoA(
    op_int, ic_params, ic_func;
    max_iter=5e3, full=false, Vr=Vrmax
)
display(fig6)


# Initial Condition: A * sin(2 * π * ceil(a) * x .+ b)
## POD-Intrusive model
ic_params = (
    [-200, 200],
    [0.01, 100],
    [-100, 100],
)
ic_func = init_wave = (A,a,b) -> A .* sin.(2 * pi * ceil(a) * burgers.x .+ b) 
ρ_true5, fig7 = verify_DoA(
    op_int, ic_params, ic_func;
    max_iter=5e3, full=false, Vr=Vrmax
)
display(fig7)