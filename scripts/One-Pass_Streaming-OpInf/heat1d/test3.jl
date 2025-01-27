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

#=============================#
## Plot the data just in case
#=============================#
fig, ax, sf = CairoMakie.surface(heat1d.xspan, heat1d.tspan, X)
CairoMakie.Colorbar(fig[1, 2], sf)
display(fig)

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

# function OnePassStreamingOpInf(X, Xdot, rmax, ϵ)
#     # (0) setup
#     n, K = size(X)
#     m = 0

#     # Shuffle the data (not possible in the streaming setting)
#     ridx = shuffle(1:K)
#     X = X[:, ridx]
#     Xdot = Xdot[:, ridx]
#     # U = U[:, ridx]

#     # (1) Initialization 
#     # Initial data
#     x1 = X[:,1]  # n x 1
#     xdot1 = Xdot[:,1]  # n x 1
#     # u1 = U[:,1]  # m x 1
   
#     # POD basis
#     V = x1 / norm(x1)

#     # Eigenvalue 
#     Λ = sqrt(dot(x1, x1))

#     # Initialize the reduced dimensions
#     r = 1      # state
#     d = r + m  # data (state + input)
#     dmax = rmax + m

#     # Input-state correlation matrix
#     Φ = zeros(d, d)
#     Φ[1,1] = dot(x1, x1)
#     # Φ[2:end,2:end] = u1 * u1'

#     # State-derivative correlation matrix
#     Ψ = zeros(d, r)
#     # Ψ[1,1] = norm(x1) * norm(xdot1)
#     # Ψ[2:end,1] = u1 * norm(xdot1)

#     # Streaming process
#     for i in 2:K 
#         # (2) Receive new data
#         xi = X[:,i] # n x 1
#         xdoti = Xdot[:,i] # n x 1
#         # ui = U[:,i] # m x 1

#         # (3) Compute the orthogonal component
#         w = V' * xi
#         xperp = xi - V * w
#         xperp_mag = norm(xperp)

#         if xperp_mag < ϵ
#             xperp_mag = 0.0
#         else
#             xperp /= xperp_mag
#         end

#         # # (4) Augment the POD basis
#         # V = hcat(V, xperp)

#         # (5) Construct the core matrix
#         C = zeros(r+1, r+1)
#         for j in 1:r
#             C[j,j] = Λ[j]
#             C[j,end] = w[j]
#         end
#         C[end,end] = xperp_mag

#         # (6) Take the SVD of the core matrix
#         Vc, Λc, _ = svd(C)

#         # (7) Update the POD basis and Eigenvalue matrix
#         if norm(xperp) < ϵ  # No increment
#             V = V * Vc[1:r,1:r]
#             Λ = Λc[1:r]
#         else  # Increment
#             V = hcat(V, xperp) * Vc
#             Λ = Λc

#             # Zero-pad the correlation matrices
#             Φ = [Φ           zeros(d,1);
#                  zeros(1,d)         0.0]
#             Ψ = [Ψ           zeros(d,1);
#                  zeros(1,r)         0.0]

#             # Update the reduced dimensions
#             r += 1
#             d += 1
#         end

#         # (9) Compress matrices
#         if r > rmax
#             V = V[:,1:rmax]
#             Λ = Λ[1:rmax]

#             Vc = Vc[:,1:rmax]
#             # VVc = BlockDiagonal([Vc, 1.0I(m)])
#             VVc = Vc

#             # Φ = VVc' * Φ * VVc
#             # Ψ = VVc' * Ψ * Vc

#             Vϕ, Λϕ, _ = svd(Φ)
#             Φ = (Matrix ∘ Diagonal)(Λϕ[1:dmax])

#             Ψ = VVc' * Ψ * Vc

#             # Vϕ = Vϕ[:,1:dmax]
#             # VVϕ = Vϕ
#             # Ψ = (VVϕ)' * Ψ * (Vϕ)

#             # Λψ = svdvals(Ψ)
#             # Ψ = zeros(dmax, rmax)
#             # for j in 1:rmax
#             #     Ψ[j,j] = Λψ[j]
#             # end

#             r = rmax
#             d = r + m
#         end

#         # (10) Project onto basis
#         xhat = V' * xi
#         rvec = V' * xdoti
        
#         # (11) Form the data vector, d 
#         # dvec = vcat(xhat, ui)
#         dvec = xhat

#         # (12) Update the covariance and correlation matrices
#         @inbounds @fastmath for j in 1:d
#             for k in 1:d
#                 Φ[j, k] += dvec[j] * dvec[k]
#             end
#             for k in 1:r
#                 Ψ[j, k] += dvec[j] * rvec[k]
#             end
#         end

#         # (13) Reorthogonalize the basis
#         @views reorthogonalize!(V, ϵ)
#     end

#     return V, Λ, Φ, Ψ
# end

# function OnePassStreamingOpInf(X, Xdot, rmax, ϵ)
#     # (0) setup
#     n, K = size(X)
#     m = 0

#     # Shuffle the data (not possible in the streaming setting)
#     # ridx = shuffle(1:K)
#     # X = X[:, ridx]
#     # Xdot = Xdot[:, ridx]
#     # U = U[:, ridx]

#     # (1) Initialization 
#     # Initial data
#     x1 = X[:,1]  # n x 1
#     xdot1 = Xdot[:,1]  # n x 1
#     # u1 = U[:,1]  # m x 1
   
#     # POD basis
#     V = x1 / norm(x1)
#     Vdot = xdot1 / norm(xdot1)

#     # Eigenvalue 
#     Λ = sqrt(dot(x1, x1))
#     Λdot = sqrt(dot(xdot1, xdot1))

#     # Initialize the reduced dimensions
#     r = 1      # state
#     rdot = 1
#     d = r + m  # data (state + input)
#     dmax = rmax + m

#     # Input-state correlation matrix
#     # Φ = zeros(d, d)
#     # Φ[1,1] = dot(x1, x1)
#     # Φ[2:end,2:end] = u1 * u1'

#     Φ = x1 * x1'

#     # State-derivative correlation matrix
#     # Ψ = zeros(d, r)
#     # Ψ[1,1] = norm(x1) * norm(xdot1)
#     # Ψ[2:end,1] = u1 * norm(xdot1)

#     Ψ = x1 * xdot1'

#     reached_r = false
#     reached_rdot = false

#     # Streaming process
#     for i in 2:K 
#         # (2) Receive new data
#         xi = X[:,i] # n x 1
#         xdoti = Xdot[:,i] # n x 1
#         # ui = U[:,i] # m x 1

#         # (3) Compute the orthogonal component
#         w = V' * xi
#         xperp = xi - V * w
#         xperp_mag = norm(xperp)

#         w = Vdot' * xdoti
#         xperp_dot = xdoti - Vdot * w
#         xperp_dot_mag = norm(xperp_dot)

#         if xperp_mag < ϵ
#             xperp_mag = 0.0
#         else
#             xperp /= xperp_mag
#         end

#         if xperp_dot_mag < ϵ
#             xperp_dot_mag = 0.0
#         else
#             xperp_dot /= xperp_dot_mag
#         end

#         # # (4) Augment the POD basis
#         # V = hcat(V, xperp)

#         # (5) Construct the core matrix
#         C = zeros(r+1, r+1)
#         for j in 1:r
#             C[j,j] = Λ[j]
#             C[j,end] = w[j]
#         end
#         C[end,end] = xperp_mag

#         Cdot = zeros(rdot+1, rdot+1)
#         for j in 1:rdot
#             Cdot[j,j] = Λdot[j]
#             Cdot[j,end] = w[j]
#         end
#         Cdot[end,end] = xperp_dot_mag

#         # (6) Take the SVD of the core matrix
#         Vc, Λc, _ = svd(C)

#         Vcdot, Λcdot, _ = svd(Cdot)

#         # (7) Update the POD basis and Eigenvalue matrix
#         if xperp_mag < ϵ && xperp_dot_mag < ϵ  # No increment
#             V = V * Vc[1:r,1:r]
#             Λ = Λc[1:r]

#             Vdot = Vdot * Vcdot[1:rdot,1:rdot]
#             Λdot = Λcdot[1:rdot]
#         elseif xperp_mag >= ϵ && xperp_dot_mag < ϵ  # Increment
#             V = hcat(V, xperp) * Vc
#             Λ = Λc

#             Vdot = Vdot * Vcdot[1:rdot,1:rdot]
#             Λdot = Λcdot[1:rdot]

#             if reached_r
#                 # Zero-pad the correlation matrices
#                 Φ = [Φ           zeros(d,1);
#                     zeros(1,d)         0.0]
#                 Ψ = [Ψ; zeros(1,rdot)]
#             end

#             r += 1
#             d += 1

#         elseif xperp_mag < ϵ && xperp_dot_mag >= ϵ  # Increment
#             V = V * Vc[1:r,1:r]
#             Λ = Λc[1:r]

#             Vdot = hcat(Vdot, xperp_dot) * Vcdot
#             Λdot = Λcdot

#             if reached_r
#                 # Zero-pad the correlation matrices
#                 Ψ = [Ψ, zeros(d,1)]
#             end

#             rdot += 1
        
#         else  # Increment
#             V = hcat(V, xperp) * Vc
#             Λ = Λc

#             Vdot = hcat(Vdot, xperp_dot) * Vcdot
#             Λdot = Λcdot

#             if reached_r
#                 # Zero-pad the correlation matrices
#                 Φ = [Φ           zeros(d,1);
#                     zeros(1,d)         0.0]
#                 Ψ = [Ψ           zeros(d,1);
#                     zeros(1,rdot)         0.0]
#             end

#             # Update the reduced dimensions
#             r += 1
#             d += 1
#             rdot += 1
#         end

#         # (9) Compress matrices
#         if r > rmax
#             V = V[:,1:rmax]
#             Λ = Λ[1:rmax]

#             Vc = Vc[:,1:rmax]
#             # VVc = BlockDiagonal([Vc, 1.0I(m)])
#             VVc = Vc

#             # Φ = VVc' * Φ * VVc
#             # Ψ = VVc' * Ψ * Vc

#             Vϕ, Λϕ, _ = svd(Φ)
#             Φ = (Matrix ∘ Diagonal)(Λϕ[1:dmax])

#             if reached_r
#                 Ψ = VVc' * Ψ
#             else
#                 Ψ = V' * Ψ
#             end
#             # Ψ = VVc' * Ψ * Vc

#             # Vϕ = Vϕ[:,1:dmax]
#             # VVϕ = Vϕ
#             # Ψ = (VVϕ)' * Ψ * (Vϕ)

#             # Λψ = svdvals(Ψ)
#             # Ψ = zeros(dmax, rmax)
#             # for j in 1:rmax
#             #     Ψ[j,j] = Λψ[j]
#             # end

#             r = rmax
#             d = r + m

#             reached_r = true
#         end

#         if rdot > rmax
#             Vdot = Vdot[:,1:rmax]
#             Λdot = Λdot[1:rmax]

#             Vcdot = Vcdot[:,1:rmax]

#             if reached_rdot
#                 Ψ = Ψ * Vcdot
#             else
#                 Ψ = Ψ * Vdot
#             end

#             rdot = rmax

#             reached_rdot = true
#         end

#         if reached_r && reached_rdot
#             # (10) Project onto basis
#             xhat = V' * xi
#             rvec = Vdot' * xdoti
            
#             # (11) Form the data vector, d 
#             # dvec = vcat(xhat, ui)
#             dvec = xhat

#             # (12) Update the covariance and correlation matrices
#             @inbounds @fastmath for j in 1:d
#                 for k in 1:d
#                     Φ[j, k] += dvec[j] * dvec[k]
#                 end
#                 for k in 1:rdot
#                     Ψ[j, k] += dvec[j] * rvec[k]
#                 end
#             end

#         else
#             Φ += xi * xi'
#             Ψ += xi * xdoti'
#         end

#         # (13) Reorthogonalize the basis
#         @views reorthogonalize!(V, ϵ)
#     end

#     return V, Λ, Φ, Ψ
# end

function OnePassStreamingOpInf(X, Xdot, rmax, ϵ, α)
    # (0) setup
    n, K = size(X)
    m = 0

    # Shuffle the data (not possible in the streaming setting)
    # ridx = shuffle(1:K)
    # X = X[:, ridx]
    # Xdot = Xdot[:, ridx]
    # U = U[:, ridx]

    # (1) Initialization 
    # Initial data
    x1 = X[:,1]  # n x 1
    xdot1 = Xdot[:,1]  # n x 1
    # u1 = U[:,1]  # m x 1
   
    # POD basis
    V = x1 / norm(x1)

    # Eigenvalue 
    Λ = sqrt(dot(x1, x1))

    # Initialize the reduced dimensions
    r = 1      # state
    d = r + m  # data (state + input)
    dmax = rmax + m

    # Input-state correlation matrix
    # Φ = zeros(d, d)
    # Φ[1,1] = dot(x1, x1)
    # Φ[2:end,2:end] = u1 * u1'

    Φ = x1 * x1' + α * I(n)

    # State-derivative correlation matrix
    # Ψ = zeros(d, r)
    # Ψ[1,1] = norm(x1) * norm(xdot1)
    # Ψ[2:end,1] = u1 * norm(xdot1)

    Ψ = x1 * xdot1'

    reached_r = false

    proj_err = zeros(K)
    proj_err[1] = norm(X - V * (V' * X)) / norm(X)
    compressed = []

    # Streaming process
    for i in 2:K 
        # (2) Receive new data
        xi = X[:,i] # n x 1
        xdoti = Xdot[:,i] # n x 1
        # ui = U[:,i] # m x 1

        # (3) Compute the orthogonal component
        w = V' * xi
        xperp = xi - V * w
        xperp_mag = norm(xperp)


        if xperp_mag < ϵ
            xperp_mag = 0.0
        else
            xperp /= xperp_mag
        end

        # # (4) Augment the POD basis
        # V = hcat(V, xperp)

        # (5) Construct the core matrix
        C = zeros(r+1, r+1)
        for j in 1:r
            C[j,j] = Λ[j]
            C[j,end] = w[j]
        end
        C[end,end] = xperp_mag

        # (6) Take the SVD of the core matrix
        Vc, Λc, _ = svd(C)

        # (7) Update the POD basis and Eigenvalue matrix
        if xperp_mag < ϵ  # No increment
            V = V * Vc[1:r,1:r]
            Λ = Λc[1:r]
        else  # Increment
            V = hcat(V, xperp) * Vc
            Λ = Λc

            if reached_r
                # Zero-pad the correlation matrices
                Φ = [Φ           zeros(d,1);
                    zeros(1,d)         α]
                Ψ = [Ψ           zeros(d,1);
                    zeros(1,r)         0.0]
            end

            # Update the reduced dimensions
            r += 1
            d += 1
        end

        # (9) Compress matrices
        if r > rmax
            V = V[:,1:rmax]
            Λ = Λ[1:rmax]

            Vc = Vc[:,1:rmax]
            VVc = Vc

            # if cnt == 0
            #     cnt += 1
            # else
            #     Λϕ = svdvals(Φ)
            #     Φ = (Matrix ∘ Diagonal)(Λϕ[1:dmax])

            #     if reached_r
            #         Ψ = VVc' * Ψ * Vc
            #     else
            #         Ψ = V' * Ψ * V
            #     end

            #     reached_r = true
            #     push!(compressed, i)
            # end

            Λϕ = svdvals(Φ)
            Φ = (Matrix ∘ Diagonal)(Λϕ[1:dmax])

            if reached_r
                Ψ = VVc' * Ψ * Vc
            else
                Ψ = V' * Ψ * V
            end

            push!(compressed, i)

            reached_r = true

            r = rmax
            d = r + m
        end

        if reached_r
            # (10) Project onto basis
            xhat = V' * xi
            rvec = V' * xdoti
            
            # (11) Form the data vector, d 
            # dvec = vcat(xhat, ui)
            dvec = xhat

            # (12) Update the covariance and correlation matrices
            @inbounds @fastmath for j in 1:d
                for k in 1:d
                    Φ[j, k] += dvec[j] * dvec[k]
                end
                for k in 1:r
                    Ψ[j, k] += dvec[j] * rvec[k]
                end
            end

        else
            Φ += xi * xi'
            Ψ += xi * xdoti'
        end

        # (13) Reorthogonalize the basis
        @views reorthogonalize!(V, ϵ)

        proj_err[i] = norm(X - V * (V' * X)) / norm(X)
    end

    return V, Λ, Φ, Ψ, proj_err, compressed
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
rextra = 0
Vstream, Λ, Φ, Ψ, stream_proj_err, compress_idx = OnePassStreamingOpInf(X, Xdot, rmax+rextra, 1e-12, 1e-4)
Vsream = Vstream[:,1:rmax]
Λ = Λ[1:rmax]
Ostream = (Φ) \ Ψ
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
    scatterlines!(ax, 1:rmax, Λ, label="stream", linewidth=5, linestyle=:dash, markersize=20)
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