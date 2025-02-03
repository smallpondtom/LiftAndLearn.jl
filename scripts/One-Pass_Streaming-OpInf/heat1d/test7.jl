"""
One-Pass Streaming-OpInf prototype for 1D heat equation
"""

#=================#
## Load packages
#=================#
using LinearAlgebra
using BlockDiagonals
using CairoMakie
using ProgressMeter
using Random
import PolynomialModelReductionDataset: Heat1DModel
import LiftAndLearn as LnL

#=================#
## Generate data
#=================#
Ω = (0.0, 1.0)
Nx = 2^7; dt = 1e-3
heat1d = Heat1DModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), 
    Δx=((Ω[2]-Ω[1]) + 1/Nx)/Nx, Δt=dt, 
    diffusion_coeffs=0.2, BC=:periodic,
)
heat1d.IC = cos.(2π * heat1d.xspan)

# Some options for operator inference
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=1,
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=dt,
        deriv_type="BE"
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)

μ = heat1d.diffusion_coeffs[1]
A = heat1d.finite_diff_model(heat1d, μ)
op_heat = LnL.Operators(A=A)

# Compute the states with backward Euler
state = heat1d.integrate_model(heat1d.tspan, heat1d.IC; linear_matrix=A,
                            system_input=false, integrator_type=:BackwardEuler)
Xref = copy(state)
Xdot = (state[:, 2:end] - state[:, 1:end-1]) / dt
X = state[:, 2:end]

rmax = 10
tmp = svd(X)
Vrmax = tmp.U[:, 1:rmax]
Σrmax = tmp.S[1:rmax]

#====================================#
## Kaczmarz
#====================================#
function kaczmarz_matrix_vanilla(D, R; maxiter=10_000, tol=1e-6, α=1.0)
    @assert size(D,1) == size(R,1) "D and R must have the same number of rows."
    K, d = size(D)
    _, r  = size(R)

    # Initialize O
    O = zeros(eltype(D), d, r)

    # Count how many row updates we have done
    updates = 0

    while updates < maxiter
        # One sweep over all rows
        for i in 1:K
            updates += 1
            if updates > maxiter
                break
            end
            # Row i of D: shape (1 x d)
            Di = transpose(@view D[i, :])
            # Row i of R: shape (1 x r)
            Ri = transpose(@view R[i, :])

            # residual row: (1 x r)
            res_i = Ri - Di * O

            # row norm-squared (scalar)
            denom = dot(Di, Di)
            # Update O: rank-1 update
            # (d x 1) * (1 x r) = (d x r)
            O .+= (1/denom) * (Di' * res_i) * α
        end

        # Check residual on the full matrix, occasionally or every sweep
        # ||D*O - R||_F
        if norm(D*O - R) < tol
            break
        end
    end

    return O
end

using Distributions

"""
    kaczmarz_matrix_randomized(D, R; maxiter=10_000, tol=1e-6)

Solve min ||R - D*O||_F for O via the randomized Kaczmarz method.

# Arguments
- `D::AbstractMatrix{T}`: A (K x d) matrix.
- `R::AbstractMatrix{T}`: A (K x r) matrix.
- `maxiter::Int`: Maximum number of row-updates (default = 10_000).
- `tol::Real`: Frobenius norm tolerance for early stopping (default = 1e-6).

# Returns
- `O::Matrix{Float64}`: (d x r) matrix approximating the solution of D*O = R.

# Notes
- Each iteration picks row i w.p. proportional to ||D[i,:]||^2.
- The update for row i is: O ← O + (1 / ||D[i,:]||^2) * D[i,:]'*(R[i,:] - D[i,:]*O).
"""
function kaczmarz_matrix_randomized(D, R; maxiter=10_000, tol=1e-6, α=1.0)
    @assert size(D,1) == size(R,1) "D and R must have the same number of rows."
    K, d = size(D)
    _, r  = size(R)

    # Initialize O
    O = zeros(eltype(D), d, r)

    # Precompute row norms squared
    row_norms_sq = [(@views dot(D[i,:], D[i,:])) for i in 1:K]
    total_norm   = sum(row_norms_sq)
    p            = row_norms_sq ./ total_norm

    # Build a discrete distribution for row selection
    row_dist = Distributions.Categorical(p)

    for iter in 1:maxiter
        # pick a random row index i
        i = rand(row_dist)
        Di = transpose(@view D[i, :])
        Ri = transpose(@view R[i, :])

        # row residual
        res_i = Ri - Di * O
        # update
        denom = row_norms_sq[i]
        O .+= (1/denom) * (Di' * res_i) * α

        # Optional check of global residual every iteration (can be expensive)
        # In practice, one might check less frequently for speed.
        if norm(D*O - R) < tol
            break
        end
    end

    return O
end

#====================#
## Generate operators
#====================#
# Compute the values for the intrusive model
op_heat = LnL.Operators(A=A)
op_heat_new = LnL.pod(op_heat, Vrmax, options.system)
Aint = op_heat_new.A

## Compute OpInf
op_infer = LnL.opinf(X, Vrmax, options; Xdot=Xdot)
Ainf = op_infer.A

## Compute One-Pass Streaming-OpInf
# rextra = 0
# Vstream, Λ, Ostream, stream_proj_err, compress_idx = OnePassStreamingOpInf(hcat(X,X), hcat(Xdot,Xdot), rmax+rextra, 1e-12, 1e-9)
# Vsream = Vstream[:,1:rmax]
# Λ = Λ[1:rmax]
# Ostream = (Φ + 1e-12I) \ Ψ

Ostream = kaczmarz_matrix_randomized(X' * Vrmax, Xdot' * Vrmax, maxiter=100_000, tol=1e-12, α=0.8)
Vstream = Vrmax
Λ = Σrmax + 1e-12*rand(rmax)
Astream = Ostream'

#=========#
## Analyze
#=========#
@info "Compute errors"

# Error analysis 
intru_state_err = zeros(rmax)
opinf_state_err = zeros(rmax)
stream_state_err = zeros(rmax)
proj_err = zeros(rmax)
proj_err_stream = zeros(rmax)

@showprogress for i = 1:rmax
    Vr = Vrmax[:,1:i]
    Vr_stream = Vstream[:,1:i]

    # Integrate the intrusive model
    Xint = heat1d.integrate_model(
        heat1d.tspan, Vr' * heat1d.IC,
        linear_matrix=Aint[1:i, 1:i], 
        system_input=false, integrator_type=:BackwardEuler
    )

    # Integrate the inferred model
    Xinf = heat1d.integrate_model(
        heat1d.tspan, Vr' * heat1d.IC,
        linear_matrix=Ainf[1:i, 1:i],
        system_input=false, integrator_type=:BackwardEuler
    )

    # Integrate the streaming model
    Xstream = heat1d.integrate_model(
        heat1d.tspan, Vr_stream' * heat1d.IC,
        linear_matrix=Astream[1:i, 1:i],
        system_input=false, integrator_type=:BackwardEuler
    )

    # Compute errors
    PE = LnL.proj_error(Xref, Vr)
    PE_stream = LnL.proj_error(Xref, Vr_stream)

    # Relative state errors
    SE_int = LnL.rel_state_error(Xref, Xint, Vr)
    SE_inf = LnL.rel_state_error(Xref, Xinf, Vr)
    SE_stream = LnL.rel_state_error(Xref, Xstream, Vr_stream)

    # Sum of error values
    proj_err[i] = PE / heat1d.param_dim
    proj_err_stream[i] = PE_stream / heat1d.param_dim
    intru_state_err[i] = SE_int / heat1d.param_dim
    opinf_state_err[i] = SE_inf / heat1d.param_dim
    stream_state_err[i] = SE_stream / heat1d.param_dim
end

#=================#
## Plot the errors
#=================#
with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 600))
    ax = Axis(
        fig[1, 1], xlabel = "Reduced dimension", ylabel = "Singular Values",
        yscale=log10, xticks=1:rmax, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    scatterlines!(ax, 1:rmax, Σrmax, label="batch", linewidth=8, markersize=30)
    scatterlines!(ax, 1:rmax, sqrt.(Λ), label="stream", linewidth=5, linestyle=:dash, markersize=20)
    axislegend(ax, position = :lb, labelsize=30)
    display(fig)
end

with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 600))
    ax = Axis(
        fig[1, 1], xlabel = "Reduced dimension", ylabel = "mean relative projection error",
        yscale=log10, xticks=1:rmax, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    scatterlines!(ax, 1:rmax, proj_err, label="batch", linewidth=8, markersize=30)
    scatterlines!(ax, 1:rmax, proj_err_stream, label="stream", linewidth=5, linestyle=:dash, markersize=20)
    axislegend(ax, position = :lb, labelsize=30)
    display(fig)
end

with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 600))
    ax = Axis(
        fig[1, 1], xlabel = "Reduced dimension", ylabel = "mean relative state error",
        yscale=log10, xticks=1:rmax, titlesize=30,
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    scatterlines!(ax, 1:rmax, intru_state_err, label = "intrusive", linewidth=8, markersize=30)
    scatterlines!(ax, 1:rmax, opinf_state_err, label = "opinf", linewidth=5, markersize=20, linestyle=:dash)
    scatterlines!(ax, 1:rmax, stream_state_err, label = "stream", linewidth=3, markersize=15, linestyle=:dashdot)
    axislegend(ax, position = :lb, labelsize=30)
    display(fig)
end


