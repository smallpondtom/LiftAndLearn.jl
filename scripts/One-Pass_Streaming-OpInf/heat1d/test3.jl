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
using SparseArrays
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

#=================#
## Plot the data
#=================#
with_theme(theme_latexfonts()) do
    fig = Figure(size=(800, 600))
    ax = Axis3(
        fig[1, 1], xlabel=L"t", ylabel=L"\omega", zlabel=L"x(\omega,t)",
        titlesize=30, xlabelsize=30, ylabelsize=30, zlabelsize=30,
        xticklabelsize=25, yticklabelsize=25, zticklabelsize=25,
    )
    surface!(ax, heat1d.xspan, heat1d.tspan, Xref, colormap=:plasma)
    display(fig)
end

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

# function OnePassStreamingOpInf(X, Xdot, rmax, basis_tol, ϵ, λ)
#     n, K = size(X)
#     m = 0

#     # Initial data
#     x1 = X[:,1]  # n x 1
#     xdot1 = Xdot[:,1]  # n x 1
   
#     # POD basis
#     V = x1 / norm(x1)

#     # Eigenvalue 
#     Λ = dot(x1, x1)

#     # Initialize the reduced dimensions
#     r = 1      # state
#     d = r + m  # data (state + input)
#     dmax = rmax + m

#     # Input-state correlation matrix
#     Φ = x1 * x1'

#     # State-derivative correlation matrix
#     Ψ = x1 * xdot1'

#     compression = false
#     not_initial_compression = false

#     proj_err = zeros(K)
#     proj_err[1] = norm(X - V * (V' * X)) / norm(X)
#     compressed = []

#     # Streaming process
#     for i in 2:K 
#         xi = X[:,i] # n x 1
#         xdoti = Xdot[:,i] # n x 1
#         # ui = U[:,i] # m x 1

#         w1 = V' * xi
#         xperp = xi - V * w1
#         w2 = V' * xperp
#         xperp = xperp - V * w2
#         w = w1 + w2
#         xperp_mag = norm(xperp)

#         if xperp_mag < ϵ
#             xperp_mag = 0.0
#         else
#             xperp /= xperp_mag
#         end

#         C = zeros(r+1, r+1)
#         @simd for j in 1:r
#             for k in 1:r
#                 if j == k
#                     C[j,k] = Λ[j] + w[j] * w[k]
#                 else
#                     C[j,k] = w[j] * w[k]
#                 end
#             end
#             C[j,end] = w[j] * xperp_mag
#             C[end,j] = w[j] * xperp_mag
#         end
#         C[end,end] = xperp_mag^2

#         Vc, Λc, _ = svd(C)

#         if xperp_mag < ϵ  # No increment
#             V = V * Vc[1:r,1:r]
#             Λ = Λc[1:r]
#         else  # Increment
#             V = hcat(V, xperp) * Vc
#             Λ = Λc

#             if compression
#                 # Zero-pad the correlation matrices
#                 Φ = [Φ           zeros(d,1);
#                     zeros(1,d)         0.0]
#                 Ψ = [Ψ           zeros(d,1);
#                     zeros(1,r)         0.0]
#             end

#             # Update the reduced dimensions
#             r += 1
#             d += 1
#         end

#         if r > rmax 
#             V = V[:,1:rmax]
#             Λ = Λ[1:rmax]

#             Vc = Vc[:,1:rmax]
#             VVc = Vc
            
#             r = rmax
#             d = r + m
#         end

#         @views reorthogonalize!(V, ϵ)
#         PE = norm(X - V * (V' * X)) / norm(X)
#         proj_err[i] = PE

#         if PE < basis_tol
#             compression = true
#         end

#         if compression && not_initial_compression
#             xhat = V' * xi
#             rvec = V' * xdoti
#             dvec = xhat

#             Φ *= λ
#             Ψ *= λ
#             @inbounds @fastmath for j in 1:d
#                 for k in 1:d
#                     Φ[j, k] += dvec[j] * dvec[k]
#                 end
#                 for k in 1:r
#                     Ψ[j, k] += dvec[j] * rvec[k]
#                 end
#             end

#             Φ = VVc' * Φ * VVc
#             Ψ = VVc' * Ψ * Vc
#             push!(compressed, i)
#         elseif compression
#             Φ += xi * xi'
#             Ψ += xi * xdoti'
#             Φ = V' * Φ * V
#             Ψ = V' * Ψ * V
#             not_initial_compression = true
#         else
#             Φ += xi * xi'
#             Ψ += xi * xdoti'
#         end
#     end

#     return V, Λ, Φ, Ψ, proj_err, compressed
# end

# function OnePassStreamingOpInf(X, Xdot, rmax, ϵ, λ)
#     # (0) setup
#     n, K = size(X)
#     m = 0

#     # (1) Initialization 
#     # Initial data
#     x1 = X[:,1]  # n x 1
#     xdot1 = Xdot[:,1]  # n x 1
   
#     # POD basis
#     V = x1 / norm(x1)

#     # Eigenvalue 
#     Λ = dot(x1, x1)

#     # Initialize the reduced dimensions
#     r = 1      # state
#     d = r + m  # data (state + input)
#     dmax = rmax + m

#     # Input-state correlation matrix
#     Φ = x1 * x1'

#     # State-derivative correlation matrix
#     Ψ = x1 * xdot1'

#     compression = false

#     proj_err = zeros(K)
#     proj_err[1] = norm(X - V * (V' * X)) / norm(X)
#     compressed = []

#     # Streaming process
#     for i in 2:K 
#         # (2) Receive new data
#         xi = X[:,i] # n x 1
#         xdoti = Xdot[:,i] # n x 1
#         # ui = U[:,i] # m x 1

#         # (3) Compute the orthogonal component
#         w1 = V' * xi
#         xperp = xi - V * w1
#         w2 = V' * xperp
#         xperp = xperp - V * w2
#         w = w1 + w2
#         xperp_mag = norm(xperp)

#         if xperp_mag < ϵ
#             xperp_mag = 0.0
#         else
#             xperp /= xperp_mag
#         end

#         # (5) Construct the core matrix
#         C = zeros(r+1, r+1)
#         @simd for j in 1:r
#             for k in 1:r
#                 if j == k
#                     C[j,k] = Λ[j] + w[j] * w[k]
#                 else
#                     C[j,k] = w[j] * w[k]
#                 end
#             end
#             C[j,end] = w[j] * xperp_mag
#             C[end,j] = w[j] * xperp_mag
#         end
#         C[end,end] = xperp_mag^2

#         # (6) Take the SVD of the core matrix
#         Vc, Λc, _ = svd(C)

#         # (7) Update the POD basis and Eigenvalue matrix
#         if xperp_mag < ϵ  # No increment
#             V = V * Vc[1:r,1:r]
#             Λ = Λc[1:r]
#         else  # Increment
#             V = hcat(V, xperp) * Vc
#             Λ = Λc

#             if compression
#                 # Zero-pad the correlation matrices
#                 Φ = [Φ           zeros(d,1);
#                     zeros(1,d)         0.0]
#                 Ψ = [Ψ           zeros(d,1);
#                     zeros(1,r)         0.0]
#             end

#             # Update the reduced dimensions
#             r += 1
#             d += 1
#         end

#         # (9) Compress matrices
#         if r > rmax 
#             V = V[:,1:rmax]
#             Λ = Λ[1:rmax]

#             Vc = Vc[:,1:rmax]
#             VVc = Vc
            
#             Φ = spdiagm(Λ)

#             if compression
#                 Ψ = VVc' * Ψ * Vc
#                 push!(compressed, i)
#             else
#                 Ψ = V' * Ψ * V
#             end

#             compression = true

#             r = rmax
#             d = r + m
#         end

#         if compression
#             # (10) Project onto basis
#             xhat = V' * xi
#             rvec = V' * xdoti
            
#             # (11) Form the data vector, d 
#             # dvec = vcat(xhat, ui)
#             dvec = xhat

#             # (12) Update the covariance and correlation matrices
#             Φ *= λ
#             Ψ *= λ
#             @inbounds @fastmath for j in 1:d
#                 for k in 1:d
#                     Φ[j, k] += dvec[j] * dvec[k]
#                 end
#                 for k in 1:r
#                     Ψ[j, k] += dvec[j] * rvec[k]
#                 end
#             end

#         else
#             Φ += xi * xi'
#             Ψ += xi * xdoti'
#         end

#         # (13) Reorthogonalize the basis
#         @views reorthogonalize!(V, ϵ)

#         proj_err[i] = norm(X - V * (V' * X)) / norm(X)
#     end

#     return V, Λ, Φ, Ψ, proj_err, compressed
# end

# function two_step_ortho_component(V::AbstractArray{T}, x::AbstractVector{T}, ϵ::Real) where {T<:Number}
#     w1 = V' * x
#     xperp = x - V * w1
#     w2 = V' * xperp
#     xperp = xperp - V * w2
#     w = w1 + w2
#     xperp_mag = norm(xperp)

#     if xperp_mag < ϵ
#         xperp_mag = 0.0
#     else
#         xperp /= xperp_mag
#     end

#     return w, xperp, xperp_mag
# end

# function construct_core_matrix!(C::AbstractMatrix{T}, Λ::Union{AbstractVector{T},T}, 
#                                 w::Union{AbstractVector{T},T}, xperp_mag::T) where {T<:Number}
#     for j in eachindex(w)
#         for k in eachindex(w)
#             if j == k
#                 C[j,k] = Λ[j] + w[j] * w[k]
#             else
#                 C[j,k] = w[j] * w[k]
#             end
#         end
#         C[j,end] = w[j] * xperp_mag
#         C[end,j] = w[j] * xperp_mag
#     end
#     C[end,end] = xperp_mag^2
# end

# function OnePassStreamingOpInf(X, Xdot, rmax, ϵ, λ)
#     # (0) setup
#     n, K = size(X)
#     m = 0

#     # Initialization 
#     x1 = X[:,1]  # n x 1
#     xdot1 = Xdot[:,1]  # n x 1
   
#     # POD basis (state)
#     Vx = x1 / norm(x1)

#     # POD basis (derivative)
#     Vxdot = xdot1 / norm(xdot1)

#     # Eigenvalue (state)
#     Λx = dot(x1, x1)

#     # Eigenvalue (derivative)
#     Λxdot = dot(xdot1, xdot1)

#     # Initialize the reduced dimensions
#     rx = 1     
#     rxdot = 1
#     d = rx + m  # data (state + input)
#     dmax = rmax + m

#     # Input-state correlation matrix
#     Φ = x1 * x1'

#     # State-derivative correlation matrix
#     Ψ = x1 * xdot1'

#     # Flags for reaching the maximum reduced dimensions
#     rx_reached_rmax    = false
#     rxdot_reached_rmax = false

#     # Streaming process
#     for i in 2:K 
#         # Receive new data
#         xi = X[:,i] # n x 1
#         xdoti = Xdot[:,i] # n x 1

#         # Compute the orthogonal component (state)
#         wx, xperp, xperp_mag = two_step_ortho_component(Vx, xi, ϵ)

#         # Compute the orthogonal component (derivative)
#         wxdot, xperpdot, xperpdot_mag = two_step_ortho_component(Vxdot, xdoti, ϵ)

#         C = zeros(rx+1, rx+1)
#         construct_core_matrix!(C, Λx, wx, xperp_mag)

#         Ctilde = zeros(rxdot+1, rxdot+1)
#         construct_core_matrix!(Ctilde, Λxdot, wxdot, xperpdot_mag)

#         # Take the SVD of the core matrix (state)
#         Vc, Λc, _ = svd(C)

#         # Take the SVD of the core matrix (derivative)
#         Vctilde, Λctilde, _ = svd(Ctilde)

#         # Update the POD basis and Eigenvalue matrix (state)
#         if xperp_mag < ϵ  # No increment
#             Vx = Vx * Vc[1:rx,1:rx]
#             Λx = Λc[1:rx]
#         else  # Increment
#             Vx = hcat(Vx, xperp) * Vc
#             Λx = Λc

#             if rx_reached_rmax
#                 # Zero-pad the correlation matrices
#                 Φ = [Φ           zeros(d,1);
#                     zeros(1,d)         0.0]
#                 Ψ = vcat(Ψ, zeros(1,rxdot))
#             end

#             # Update the reduced dimensions
#             rx += 1
#             d += 1
#         end

#         # Update the POD basis and Eigenvalue matrix (derivative)
#         if xperpdot_mag < ϵ  # No increment
#             Vxdot = Vxdot * Vctilde[1:rxdot,1:rxdot]
#             Λxdot = Λctilde[1:rxdot]
#         else  # Increment
#             Vxdot = hcat(Vxdot, xperpdot) * Vctilde
#             Λxdot = Λctilde

#             if rxdot_reached_rmax
#                 # Zero-pad the correlation matrices
#                 Ψ = hcat(Ψ, zeros(d,1))
#             end

#             # Update the reduced dimensions
#             rxdot += 1
#         end

#         # Compress matrices
#         if rx > rmax 
#             Vx = Vx[:,1:rmax]
#             Λx = Λx[1:rmax]
#             Vc = Vc[:,1:rmax]
#             Φ = (Matrix ∘ Diagonal)(Λx)

#             if rx_reached_rmax
#                 Ψ = Vc' * Ψ
#             else
#                 Ψ = Vx' * Ψ
#             end

#             rx_reached_rmax = true
#             rx = rmax
#             d = rx + m
#         end

#         if rxdot > rmax 
#             Vxdot = Vxdot[:,1:rmax]
#             Λxdot = Λxdot[1:rmax]
#             Vctilde = Vctilde[:,1:rmax]

#             if rxdot_reached_rmax
#                 Ψ = Ψ * Vctilde
#             else
#                 Ψ = Ψ * Vxdot
#             end

#             rxdot_reached_rmax = true
#             rxdot = rmax
#         end

#         if rx_reached_rmax
#             xhat = Vx' * xi
#             dvec = xhat
#         else
#             dvec = xi
#         end

#         if rxdot_reached_rmax
#             rvec = Vxdot' * xdoti
#         else
#             rvec = xdoti
#         end

#         # Update the correlation and cross-correlation matrices
#         Φ *= λ
#         Ψ *= λ
#         @inbounds @simd for j in eachindex(dvec)
#             for k in eachindex(dvec)
#                 Φ[j, k] += dvec[j] * dvec[k]
#             end
#             for k in eachindex(rvec)
#                 Ψ[j, k] += dvec[j] * rvec[k]
#             end
#         end

#         # Reorthogonalize the basis
#         @views reorthogonalize!(Vx, ϵ)
#         @views reorthogonalize!(Vxdot, ϵ)
#     end

#     return Vx, Λx, Φ, Ψ, Vxdot, Λxdot
# end

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
rextra = 0
Vstream, Λ, Φ, Ψ, stream_proj_err, compress_idx = OnePassStreamingOpInf(X, Xdot, rmax+rextra, 5e-10, 1e-12, 1.0)
# Vstream, Λ, Φ, Ψ, Vxdot, Λxdot = OnePassStreamingOpInf(X, Xdot, rmax+rextra, 1e-12, 1.0)
Vsream = Vstream[:,1:rmax]
Λ = Λ[1:rmax]
Ostream = (Φ + 1e-12I) \ Ψ
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

with_theme(theme_latexfonts()) do 
    fig = Figure(size = (800, 600))
    ax = Axis(
        fig[1, 1], xlabel = "stream", ylabel = "relative rojection error",
        yscale=log10, titlesize=30, xlabelsize=30, ylabelsize=30,
        xticklabelsize=25, yticklabelsize=25,
    )
    lines!(ax, 1:minimum(compress_idx)-1, stream_proj_err[1:minimum(compress_idx)-1], linewidth=5, label="full")
    lines!(ax, minimum(compress_idx):size(X,2), stream_proj_err[minimum(compress_idx):end], linewidth=5, label="compressed")
    axislegend(ax, position = :rt, labelsize=30)
    display(fig) 
end