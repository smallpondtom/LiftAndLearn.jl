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
    diffusion_coeffs=0.3,
)

# Some options for operator inference
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=1,
        control=1,
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

# Input from the boundary condition
Ubc = ones(heat1d.time_dim)

μ = heat1d.diffusion_coeffs[1]
A, B = heat1d.finite_diff_model(heat1d, μ)
C = ones(1, heat1d.spatial_dim) / heat1d.spatial_dim
op_heat = LnL.Operators(A=A, B=B, C=C)

# Only one initial condition and input
heat1d.IC = cos.(2π * heat1d.xspan)

# Compute the states with backward Euler
state = heat1d.integrate_model(heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
                            system_input=true, integrator_type=:BackwardEuler)
Xref = copy(state)
Uref = Ubc
Xdot = (state[:, 2:end] - state[:, 1:end-1]) / dt

ICref = heat1d.IC
X = state[:, 2:end]
U = Ubc[2:end]'

# Different initial conditions and inputs
# rng = MersenneTwister(1234)
# for i in 1:9
#     heat1d.IC = randn(rng) * cos.(2π * heat1d.xspan) + randn(rng) * sin.(2π * heat1d.xspan) * 0.01
#     # heat1d.IC[2:end-1] += randn(heat1d.spatial_dim-2) * 0.1
#     Ubc = ones(heat1d.time_dim) * (rand(rng) * 2 - 1)

#     state = heat1d.integrate_model(heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
#                             system_input=true, integrator_type=:BackwardEuler)
#     X = hcat(X, state[:, 2:end])
#     Xdot = hcat(Xdot, (state[:, 2:end] - state[:, 1:end-1]) / dt)
#     U = hcat(U, Ubc[2:end]')
# end

rmax = 10
tmp = svd(X)
Vrmax = tmp.U[:, 1:rmax]
Σrmax = tmp.S[1:rmax]

#====================================#
## One-Pass Streaming-OpInf function
#====================================#
"""
Faster QR factorization that returns Q without processing the Householder vectors.

Reference:
https://github.com/JuliaLinearAlgebra/IncrementalSVD.jl/blob/da75cd435ed3f57bc56afab3d2faec7155a9b913/src/IncrementalSVD.jl#L207C1-L217C4
"""
function qrf!(P::AbstractArray{T}, R::AbstractArray{T}) where {T<:Number}
    m, b = checksize(P)
    m >= b || throw(DimensionMismatch("Works only for m > b"))
    P, tau = LAPACK.geqrf!(P)
    fill!(R, zero(T))
    @inbounds for j = 1:b, i = 1:j
        R[i,j] = P[i,j]
    end
    LAPACK.orgqr!(P, tau)
    return R
end

"""
Dispatch
"""
function qrf!(P::AbstractArray{<:Number})
    m, b = checksize(P)
    m >= b || throw(DimensionMismatch("Works only for m > b"))
    P, tau = LAPACK.geqrf!(P)
    LAPACK.orgqr!(P, tau)
end

"""
    checksize(A::AbstractArray)

Check the size of the input matrix and return the number of rows and columns.

# Arguments
- `A::AbstractArray`: input matrix

# Returns
- `m::Int`: number of rows
- `n::Int`: number of columns
"""
function checksize(A::AbstractArray)
    m, n = nothing, nothing
    try
        m, n = size(A)
    catch e
        if isa(e, BoundsError)
            m, n = length(A), 1
        else
            rethrow(e)
        end
    end
    return m, n
end

function reorthogonalize!(V::AbstractMatrix{T}, tol::Real) where {T<:Number}
    # Dimension
    r = size(V, 2)
    R = zeros(T, r, r)
    if abs(dot(V[:, end], V[:, 1])) > tol
        @views for k in 1:r
            for _ = 1:2  # do this twice (from p307 algo 6.11 in [GanderGK2014])
                for i = 1:k-1
                    E = dot(V[:, i], V[:, k])
                    V[:, k] .-= E * V[:, i]
                    R[i, k] += E
                end
            end
            R[k, k] = sqrt(dot(V[:, k], V[:, k]))
            V[:, k] ./= R[k, k]
        end
    end
end

# function OnePassStreamingOpInf(X, Xdot, U, rmax, α, γ)
#     n, K = size(X)
#     m = size(U,1)

#     # Initial data
#     x1 = X[:,1]  # n x 1
#     xdot1 = Xdot[:,1]  # n x 1
#     u1 = U[:,1]  # m x 1
   
#     # POD basis
#     V = x1 / norm(x1)

#     # Singular value
#     Σ = norm(x1)

#     # Initialize the reduced dimensions
#     r = 1      # state

#     # Covariance matrix EVD components
#     rd = 1
#     d1 = vcat(x1, u1)
#     Vϕ = d1 / norm(d1)
#     Λϕ = dot(d1, d1)

#     # Cross-covariance matrix SVD components
#     rr = 1
#     Vψ = copy(Vϕ)
#     Σψ = norm(d1) * norm(xdot1)
#     Wψ = xdot1 / norm(xdot1)

#     # Streaming process
#     for i in 2:K 
#         xi = X[:,i] # n x 1
#         xdoti = Xdot[:,i] # n x 1
#         ui = U[:,i] # m x 1

#         # POD basis
#         q1 = V' * xi
#         xperp = xi - V * q1
#         q2 = V' * xperp
#         xperp = xperp - V * q2
#         q = q1 + q2
#         p = norm(xperp)

#         # if p < ϵ
#         #     p = 0.0
#         # else
#         #     xperp /= p
#         # end

#         p = [p]
#         xperp = reshape(xperp, :, 1)
#         qrf!(xperp, p)
#         p = p[1]

#         C = zeros(r+1, r+1)
#         for j in 1:r
#             C[j,j] = Σ[j]
#             C[j,end] = q[j]
#         end
#         C[end,end] = p

#         Vc, Σc, _ = svd(C)

#         # if p < ϵ  # No increment
#         #     V = V * Vc[1:r,1:r]
#         #     Σ = Σc[1:r]
#         # else  # Increment
#         #     V = hcat(V, xperp) * Vc
#         #     Σ = Σc
#         #     r += 1
#         # end

#         V = hcat(V, xperp) * Vc
#         Σ = Σc
#         r += 1

#         if r > rmax
#             V = V[:,1:rmax]
#             Σ = Σ[1:rmax]
#             r = rmax
#         end

#         # Covariance matrix 
#         di = vcat(xi, ui)

#         qd1 = Vϕ' * di
#         dperp = di - Vϕ * qd1
#         qd2 = Vϕ' * dperp
#         dperp = dperp - Vϕ * qd2
#         qd = qd1 + qd2
#         pd = norm(dperp)

#         # if pd < ϵ
#         #     pd = 0.0
#         # else
#         #     dperp /= pd
#         # end

#         pd = [pd]
#         dperp = reshape(dperp, :, 1)
#         qrf!(dperp, pd)
#         pd = pd[1]

#         Cϕ = zeros(rd+1, rd+1)
#         for j in 1:rd
#             for k in 1:rd
#                 if j == k
#                     Cϕ[j,k] = Λϕ[j] + qd[j] * qd[k]
#                 else
#                     Cϕ[j,k] = qd[j] * qd[k]
#                 end
#             end
#             Cϕ[j,end] = qd[j] * pd
#             Cϕ[end,j] = qd[j] * pd
#         end
#         Cϕ[end,end] = pd^2

#         Vcϕ, Λcϕ, _ = svd(Cϕ)

#         # if pd < ϵ  # No increment
#         #     Vϕ = Vϕ * Vcϕ[1:rd,1:rd]
#         #     Λϕ = Λcϕ[1:rd]
#         # else  # Increment
#         #     Vϕ = hcat(Vϕ, dperp) * Vcϕ
#         #     Λϕ = Λcϕ
#         #     rd += 1
#         # end

#         Vϕ = hcat(Vϕ, dperp) * Vcϕ
#         Λϕ = Λcϕ
#         rd += 1

#         if rd > rmax + α
#             Vϕ = Vϕ[:,1:rmax+α]
#             Λϕ = Λϕ[1:rmax+α]
#             rd = rmax + α
#         end

#         # Cross-covariance matrix
#         qd1 = Vψ' * di 
#         dperp = di - Vψ * qd1
#         qd2 = Vψ' * dperp
#         dperp = dperp - Vψ * qd2
#         qd = qd1 + qd2
#         pd = norm(dperp)

#         qr1 = Wψ' * xdoti
#         rperp = xdoti - Wψ * qr1
#         qr2 = Wψ' * rperp
#         rperp = rperp - Wψ * qr2
#         qr = qr1 + qr2
#         pr = norm(rperp)

#         # if pr < ϵ
#         #     pr = 0.0
#         # else
#         #     rperp /= pr
#         # end

#         pd = [pd]
#         dperp = reshape(dperp, :, 1)
#         qrf!(dperp, pd)
#         pd = pd[1]

#         pr = [pr]
#         rperp = reshape(rperp, :, 1)
#         qrf!(rperp, pr)
#         pr = pr[1]

#         Cψ = zeros(rr+1, rr+1)
#         for j in 1:rr
#             for k in 1:rr
#                 if j == k
#                     Cψ[j,k] = Σψ[j] + qd[j] * qr[k]
#                 else
#                     Cψ[j,k] = qd[j] * qr[k]
#                 end
#             end
#             Cψ[j,end] = qd[j] * pr
#             Cψ[end,j] = qr[j] * pd
#         end
#         Cψ[end,end] = pr * pd

#         Vcψ, Σcψ, Wcψ = svd(Cψ)

#         # if pr < ϵ  # No increment
#         #     Σψ = Σcψ[1:rr]
#         #     Wψ = Wψ * Wcψ[:,1:rr]
#         # else  # Increment
#         #     Σψ = Σcψ
#         #     Wψ = hcat(Wψ, rperp) * Wcψ
#         #     rr += 1
#         # end

#         Vψ = hcat(Vψ, dperp) * Vcψ
#         Σψ = Σcψ
#         Wψ = hcat(Wψ, rperp) * Wcψ
#         rr += 1

#         if rr > rmax + α
#             Vψ = Vψ[:,1:rmax+α]
#             Σψ = Σψ[1:rmax+α]
#             Wψ = Wψ[:,1:rmax+α]
#             rr = rmax + α
#         end

#         # @views reorthogonalize!(V, ϵ)
#         # @views reorthogonalize!(Vϕ, ϵ)
#         # @views reorthogonalize!(Wψ, ϵ)
#     end

#     Φ = Vϕ * Diagonal(Λϕ) * Vϕ'
#     Ψ = Vψ * Diagonal(Σψ) * Wψ'

#     # Φinv = Vϕ * Diagonal(1 ./ (sqrt.(Λϕ) .+ γ)) * Vϕ'
#     VV = BlockDiagonal([V, 1.0I(m)])
#     # Ostream = VV' * (Φinv * Ψ) * V
#     Ostream = VV' * ((Φ + γ*I) \ Ψ) * V

#     return Ostream, V, Σ, Φ, Ψ, Vϕ, Λϕ, Vψ, Σψ, Wψ
# end

function OnePassStreamingOpInf(X, Xdot, U, rmax, λ)
    n, K = size(X)
    m = size(U,1)

    x1 = X[:,1] 
    xdot1 = Xdot[:,1] 
   
    V = x1 / norm(x1)
    Σ = norm(x1)
    W = 1.0
    r1 = 1  

    r2 = 1
    Vd = xdot1 / norm(xdot1)
    Σd = norm(xdot1)
    Wd = 1.0

    for i in 2:K 
        xi = X[:,i]
        xdoti = Xdot[:,i]

        q1 = V' * xi
        xperp = xi - V * q1
        q2 = V' * xperp
        xperp = xperp - V * q2
        q = q1 + q2
        p = norm(xperp)

        p = [p]
        xperp = reshape(xperp, :, 1)
        qrf!(xperp, p)
        p = p[1]

        C = zeros(r1+1, r1+1)
        for j in 1:r1
            C[j,j] = Σ[j]
            C[j,end] = q[j]
        end
        C[end,end] = p

        Vc, Σc, Wc = svd(C)
        V = hcat(V, xperp) * Vc
        Σ = Σc
        W = [W zeros(size(W,1), 1); zeros(1, r1) 1.0] * Wc
        r1 += 1

        q1 = Vd' * xdoti
        xdotperp = xdoti - Vd * q1
        q2 = Vd' * xdotperp
        xdotperp = xdotperp - Vd * q2
        q = q1 + q2
        p = norm(xdotperp)

        p = [p]
        xdotperp = reshape(xdotperp, :, 1)
        qrf!(xdotperp, p)
        p = p[1]

        C = zeros(r2+1, r2+1)
        for j in 1:r2
            C[j,j] = Σd[j]
            C[j,end] = q[j]
        end
        C[end,end] = p

        Vcd, Σcd, Wcd = svd(C)
        Vd = hcat(Vd, xdotperp) * Vcd
        Σd = Σcd
        Wd = [Wd zeros(size(Wd,1), 1); zeros(1, r2) 1.0] * Wcd
        r2 += 1

        if r1 > rmax
            V = V[:,1:rmax]
            Σ = Σ[1:rmax]
            W = W[:,1:rmax]
            r1 = rmax
        end
        if r2 > rmax
            Vd = Vd[:,1:rmax]
            Σd = Σd[1:rmax]
            Wd = Wd[:,1:rmax]
            r2 = rmax
        end
    end

    Σ = Diagonal(Σ)
    Σd = Diagonal(Σd)
    Xd = Vd * Σd * Wd'

    D = [W * Σ    U']
    R = Xd' * V
    if !iszero(λ)
        D = vcat(D, λ * I(size(D,2)))
        R = vcat(R, zeros(size(D, 2), size(R, 2)))
    end

    O = D \ R

    # Φ = zeros(rmax+m, rmax+m)
    # Φ[1:rmax, 1:rmax] .= Σ.^2
    # Φ[rmax+1:rmax+m, 1:rmax] .= U * W * Σ
    # Φ[1:rmax, rmax+1:rmax+m] .= Σ * W' * U'
    # Φ[rmax+1:rmax+m, rmax+1:rmax+m] .= U * U'

    # Ψ = zeros(rmax+m, rmax)
    # Ψ[1:rmax, :] .= Σ * W' * Xd' * V 
    # Ψ[rmax+1:rmax+m, :] .= U * Xd' * V

    # O = (Φ + λ*I) \ Ψ

    # return O, V, diag(Σ), W, Vd, diag(Σd), Wd, Φ, Ψ
    return O, V, diag(Σ), W, Vd, diag(Σd), Wd
end

# function SparseMat(k::Int, n::Int; zeta::Int=min(k,8))
#     # if k > n
#     #     error("k should be less than or equal to n.")
#     # end
#     if zeta < 1 || zeta > k
#         error("zeta should be between 1 and k.")
#     end

#     # Create indCol: repeat each column index zeta times
#     indCol = repeat(1:n, inner=zeta)

#     # Initialize indRow
#     indRow = Vector{Int}(undef, n * zeta)
#     idx = 1
#     for _ in 1:n
#         rows = sort(sample(1:k, zeta; replace=false))
#         indRow[idx:idx+zeta-1] = rows
#         idx += zeta
#     end

#     # Generate values
#     vals = sign.(randn(n * zeta))
#     Xi = sparse(indRow, indCol, vals, k, n)
#     return Xi
# end

#====================#
## Generate operators
#====================#
# Compute the values for the intrusive model
op_heat = LnL.Operators(A=A, B=B)
op_heat_new = LnL.pod(op_heat, Vrmax, options.system)
Aint = op_heat_new.A
Bint = op_heat_new.B

## Compute OpInf
op_infer = LnL.opinf(X, Vrmax, options; U=U, Xdot=Xdot)
Ainf = op_infer.A
Binf = op_infer.B

## Compute One-Pass Streaming-OpInf
rextra = 0
Ostream, Vstream, Λ, W, Vd, Λd, Wd = OnePassStreamingOpInf(X, Xdot, U, rmax+rextra,  0.0)
Astream = Ostream[1:rmax,1:rmax]'
Bstream = Ostream[rmax+rextra+1:rmax+rextra+1,1:rmax]'

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
        heat1d.tspan, Vr' * ICref, Uref,
        linear_matrix=Aint[1:i, 1:i], control_matrix=Bint[1:i,:],
        system_input=true, integrator_type=:BackwardEuler
    )

    # Integrate the inferred model
    Xinf = heat1d.integrate_model(
        heat1d.tspan, Vr' * ICref, Uref,
        linear_matrix=Ainf[1:i, 1:i], control_matrix=Binf[1:i,:],
        system_input=true, integrator_type=:BackwardEuler
    )

    # Integrate the streaming model
    Xstream = heat1d.integrate_model(
        heat1d.tspan, Vr_stream' * ICref, Uref,
        linear_matrix=Astream[1:i, 1:i], control_matrix=Bstream[1:i,:],
        system_input=true, integrator_type=:BackwardEuler
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
    # scatterlines!(ax, 1:rmax, sqrt.(Λ[1:rmax]), label="stream", linewidth=5, linestyle=:dash, markersize=20)
    scatterlines!(ax, 1:rmax, Λ[1:rmax], label="stream", linewidth=5, linestyle=:dash, markersize=20)
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
        # limits=(nothing, nothing, 1e-6, 1e+6),
    )
    scatterlines!(ax, 1:rmax, intru_state_err, label = "intrusive", linewidth=8, markersize=30)
    scatterlines!(ax, 1:rmax, opinf_state_err, label = "opinf", linewidth=5, markersize=20, linestyle=:dash)
    scatterlines!(ax, 1:rmax, stream_state_err, label = "stream", linewidth=3, markersize=15, linestyle=:dashdot)
    axislegend(ax, position = :lb, labelsize=30)
    display(fig)
end

# with_theme(theme_latexfonts()) do
#     fig = Figure(size = (800, 600))
#     ax = Axis(
#         fig[1, 1], xlabel = "streams", ylabel = "absolute projection error",
#         yscale=log10, titlesize=30,
#         xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
#     )
#     L = length(pe)
#     scatterlines!(ax, 1:L, pe, linewidth=8, markersize=30)
#     display(fig)
# end

# with_theme(theme_latexfonts()) do
#     fig = Figure(size = (800, 600))
#     ax = Axis(
#         fig[1, 1], xlabel = "streams", ylabel = "subspace angle errors",
#         yscale=log10, titlesize=30,
#         xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
#     )
#     popfirst!(sae)
#     L = length(sae)
#     scatterlines!(ax, 3:L+2, sae, linewidth=8, markersize=30)
#     display(fig)
# end

