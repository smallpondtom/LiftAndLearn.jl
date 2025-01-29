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
    diffusion_coeffs=0.2,
)
heat1d.IC = cos.(2π * heat1d.xspan)

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

# Compute the states with backward Euler
state = heat1d.integrate_model(heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
                            system_input=true, integrator_type=:BackwardEuler)
Xref = copy(state)
Xdot = (state[:, 2:end] - state[:, 1:end-1]) / dt
X = state[:, 2:end]
U = Ubc[2:end]'

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

# function OnePassStreamingOpInf(X, Xdot, U, rmax)
#     # (1) Initialization 
#     n, K = size(X)
#     m = size(U,1)
#     x1 = X[:,1]  # n x 1
#     xdot1 = Xdot[:,1]  # n x 1
#     u1 = U[:,1]  # m x 1
#     dvec1 = vcat(x1, u1)  # (n+m) x 1
   
#     # POD basis
#     V = zeros(n, rmax+1)
#     q1, r1 = qr(x1)
#     V[:,1] .= Matrix(q1)

#     # Singular value matrix
#     Σ = zeros(rmax+1)
#     Σ[1] = r1[1]

#     # # State covariance matrix
#     # Ξ = zeros(rmax+1, rmax+1)
#     # Ξ[1,1] = dot(x1,x1) 

#     # Input-state correlation matrix
#     dmax = rmax + m
#     Φ = zeros(dmax+1, dmax)
#     Φ[1,1] = dot(dvec1, dvec1)

#     # State-derivative correlation matrix
#     Ψ = zeros(dmax+1, rmax+1)
#     Ψ[1,1] = norm(dvec1) * norm(xdot1)

#     # Initialize the reduced dimension
#     r = 1

#     # Streaming process
#     for i in 2:K 
#         # (2) Receive new data
#         xi = X[:,i] # n x 1
#         xdoti = Xdot[:,i] # n x 1
#         ui = U[:,i] # m x 1

#         Vr = @view V[:,1:r] # n x r

#         # (3) Compute the orthogonal component
#         xproj1 = Vr' * xi
#         xperp1 = xi - Vr * xproj1
#         xproj = Vr' * xperp1
#         xperp = xperp1 - Vr * xproj
#         xproj += xproj1

#         # (4) Take the QR decomposition
#         xperp_mag = [0.0]
#         xperp = reshape(xperp, n, 1)
#         qrf!(xperp, xperp_mag)

#         # (5) Augment the POD basis
#         V[:,r+1] .= xperp

#         # # (6) Augment the state covariance matrix
#         # Ξ[1:r,r+1] .= xproj
#         # Ξ[r+1,r+1] = xperp_mag[1]
#         R = [Diagonal(Σ[1:r]) xproj; zeros(1, r) xperp_mag[1]]

#         # (7) Zero-pad the correlation matrices 
#         # Which is unnecessary in this case since we already preallocated the matrix
        
#         # (8) Update the reduced dimension
#         r += 1

#         # (9) Compress matrices
#         if r > rmax
#             W, S, _ = svd(R[1:rmax,1:rmax])
#             # Λ = reverse(Λ) # Sort in descending order
#             # Θ = reverse(Θ, dims=2)  # Sort in descending order

#             # Λ = Λ[1:rmax]  # rmax x 1
#             # Θ = Θ[:,1:rmax]  # n x rmax
#             W = W[:,1:rmax]
#             S = S[1:rmax]

#             V[:,1:rmax] .= V[:,1:rmax] * W  # n x rmax
#             Σ[1:rmax] .= S
#             # @inbounds for j in 1:rmax
#             #     Ξ[j,j] = Λ[j]
#             # end
#             Γ = BlockDiagonal([W, 1.0I(m)])  
#             # Φ[1:dmax,1:dmax] .= BlockDiagonal([Diagonal(Λ), Φ[end-m+1:end,end-m+1:end]])
#             Φ[1:dmax,1:dmax] .= BlockDiagonal([Diagonal(S.^2), Φ[end-m+1:end,end-m+1:end]])
#             # Ψ[1:dmax,1:rmax] .= Γ' * Ψ[1:dmax,1:rmax] * Θ
#             # println(size(Γ), size(Ψ[1:dmax,1:rmax]), size(W))
#             Ψ[1:dmax,1:rmax] .= Γ' * Ψ[1:dmax,1:rmax] * W
#             r = rmax
#         end

#         # (10) Project onto basis
#         Vr = @view V[:,1:r] # n x r
#         xhat = Vr' * xi
#         rvec = Vr' * xdoti
        
#         # (11) Form the data vector, d 
#         dvec = vcat(xhat, ui)
#         di = length(dvec)

#         # (12) Update the covariance and correlation matrices
#         # Ξ[1:r,1:r] .+= xhat * xhat'
#         Φ[1:di,1:di] .+= dvec * dvec'
#         Ψ[1:di,1:r] .+= dvec * rvec'

#         # (13) Reorthogonalize the basis
#         @views reorthogonalize!(V[:,1:r], 1e-12)
#     end

#     # return V[:,1:rmax], Ξ[1:rmax,1:rmax], Φ[1:dmax,1:dmax], Ψ[1:dmax,1:rmax]
#     return V[:,1:rmax], Σ[1:rmax], Φ[1:dmax,1:dmax], Ψ[1:dmax,1:rmax]
# end

# function OnePassStreamingOpInf(X, Xdot, U, rmax, ϵ)
#     # (1) Initialization 
#     n, K = size(X)
#     m = size(U,1)
#     x1 = X[:,1]  # n x 1
#     xdot1 = Xdot[:,1]  # n x 1
#     u1 = U[:,1]  # m x 1
#     dvec1 = vcat(x1, u1)  # (n+m) x 1
   
#     # POD basis
#     V = zeros(n, rmax+1)
#     V[:,1] .= x1 / norm(x1)

#     # State covariance matrix
#     Ξ = zeros(rmax+1, rmax+1)
#     Ξ[1,1] = dot(x1,x1) 

#     # Input-state correlation matrix
#     dmax = rmax + m
#     Φ = zeros(dmax+1, dmax)
#     Φ[1,1] = dot(dvec1, dvec1)

#     # State-derivative correlation matrix
#     Ψ = zeros(dmax+1, rmax+1)
#     Ψ[1,1] = norm(dvec1) * norm(xdot1)

#     # Initialize the reduced dimension
#     r = 1

#     # Streaming process
#     for i in 2:K 
#         # (2) Receive new data
#         xi = X[:,i] # n x 1
#         xdoti = Xdot[:,i] # n x 1
#         ui = U[:,i] # m x 1

#         Vr = @view V[:,1:r] # n x r

#         # (3) Compute the orthogonal component
#         xperp1 = xi - Vr * Vr' * xi
#         xperp = xperp1 - Vr * Vr' * xperp1
#         xperp_mag = norm(xperp)

#         if xperp_mag > ϵ
#             # (4) Augment the POD basis
#             V[:,r+1] .= xperp / xperp_mag

#             # (5) Zero-pad the state covariance matrix (which is unnecessary in this case)

#             # (6) Zero-pad the correlation matrices 
#             # Which is unnecessary in this case since we already preallocated the matrix

#             # (7) Update the reduced dimension
#             r += 1
#         end

#         # (9) Compress matrices
#         if r > rmax
#             Λ, Θ = eig(Ξ[1:rmax,1:rmax])
#             Λ = reverse(Λ) # Sort in descending order
#             Θ = reverse(Θ, dims=2)  # Sort in descending order

#             Λ = Λ[1:rmax]  # rmax x 1
#             Θ = Θ[:,1:rmax]  # n x rmax

#             V[:,1:rmax] .= V[:,1:rmax] * Θ  # n x rmax
#             @inbounds for j in 1:rmax
#                 Ξ[j,j] = Λ[j]
#             end
#             Γ = BlockDiagonal([V, 1.0I(m)])  
#             Φ[1:dmax,1:dmax] .= BlockDiagonal([Diagonal(Λ), Φ[end-m+1:end,end-m+1:end]])
#             Ψ[1:dmax,1:rmax] .= Γ' * Ψ[1:dmax,1:rmax] * Θ
#             r = rmax
#         end

#         # (10) Project onto basis
#         Vr = @view V[:,1:r] # n x r
#         xhat = Vr' * xi
#         rvec = Vr' * xdoti
        
#         # (11) Form the data vector, d 
#         dvec = vcat(xhat, ui)
#         di = length(dvec)

#         # (12) Update the covariance and correlation matrices
#         Ξ[1:r,1:r] .+= xhat * xhat'
#         Φ[1:di,1:di] .+= dvec * dvec'
#         Ψ[1:di,1:r] .+= dvec * rvec'

#         # (13) Reorthogonalize the basis
#         @views reorthogonalize!(V[:,1:r], 1e-12)
#     end
# 
#     return V[:,1:rmax], Ξ[1:rmax,1:rmax], Φ[1:dmax,1:dmax], Ψ[1:dmax,1:rmax]
# end


function OnePassStreamingOpInf(X, Xdot, U, rmax, ϵ)
    # (0) setup
    n, K = size(X)
    m = size(U,1)

    # (1) Initialization 
    # Initial data
    x1 = X[:,1]  # n x 1
    xdot1 = Xdot[:,1]  # n x 1
    u1 = U[:,1]  # m x 1
   
    # POD basis
    V = x1 / norm(x1)

    # Eigenvalue 
    Λ = dot(x1, x1)

    # Initialize the reduced dimensions
    r = 1      # state
    d = r + m  # data (state + input)
    dmax = rmax + m

    # Input-state correlation matrix
    Φ = zeros(d, d)
    Φ[1,1] = dot(x1, x1)
    Φ[2:end,2:end] = u1 * u1'

    # State-derivative correlation matrix
    Ψ = zeros(d, r)
    Ψ[1,1] = norm(x1) * norm(xdot1)
    Ψ[2:end,1] = u1 * norm(xdot1)

    # Streaming process
    for i in 2:K 
        # (2) Receive new data
        xi = X[:,i] # n x 1
        xdoti = Xdot[:,i] # n x 1
        ui = U[:,i] # m x 1

        # (3) Compute the orthogonal component
        w1 = V' * xi
        xperp = xi - V * w1
        w2 = V' * xperp
        xperp = xperp - V * w2
        w = w1 + w2
        xperp_mag = norm(xperp)

        # (4) Augment the POD basis
        # V = hcat(V, xperp)

        # (5) Construct the core matrix
        C = zeros(r+1, r+1)
        for j in 1:r
            for k in 1:r
                if j == k
                    C[j,k] = Λ[j] + w[j] * w[k]
                else
                    C[j,k] = w[j] * w[k]
                end
            end
            C[j,end] = w[j] * xperp_mag
            C[end,j] = w[j] * xperp_mag
        end
        C[end,end] = xperp_mag^2

        # (6) Take the EVD of the core matrix
        Λc, Vc = eigen(C)
        # Sort in descending order
        Λc = reverse(Λc)
        Vc = reverse(Vc, dims=2) 

        # (7) Update the POD basis and Eigenvalue matrix
        if norm(xperp) < ϵ  # No increment
            V = V * Vc[1:r,1:r]
            Λ = Λc[1:r]
        else  # Increment
            V = hcat(V, xperp ./ xperp_mag) * Vc
            Λ = Λc

            # Zero-pad the correlation matrices
            Φ = [Φ           zeros(d,1);
                 zeros(1,d)         0.0]
            Ψ = [Ψ           zeros(d,1);
                 zeros(1,r)         0.0]

            # Update the reduced dimensions
            r += 1
            d += 1
        end

        # (9) Compress matrices
        if r > rmax
            V = V[:,1:rmax]
            Λ = Λ[1:rmax]
            Vc = Vc[:,1:rmax]
            VVc = BlockDiagonal([Vc, 1.0I(m)])
            Φ = VVc' * Φ * VVc
            # Φ = (Matrix ∘ Diagonal)(svdvals(Φ)[1:dmax])
            Ψ = VVc' * Ψ * Vc
            r = rmax
            d = r + m
        end

        # (10) Project onto basis
        xhat = V' * xi
        rvec = V' * xdoti
        
        # (11) Form the data vector, d 
        dvec = vcat(xhat, ui)

        # (12) Update the covariance and correlation matrices
        @inbounds @fastmath for j in 1:d
            for k in 1:d
                Φ[j, k] += dvec[j] * dvec[k]
            end
            for k in 1:r
                Ψ[j, k] += dvec[j] * rvec[k]
            end
        end

        # (13) Reorthogonalize the basis
        @views reorthogonalize!(V, ϵ)
    end

    return V, Λ, Φ, Ψ
end


# function OnePassStreamingOpInf(X, Xdot, U, rmax, ϵ)
#     # (0) setup
#     n, K = size(X)
#     m = size(U,1)

#     # (1) Initialization 
#     # Initial data
#     x1 = X[:,1]  # n x 1
#     xdot1 = Xdot[:,1]  # n x 1
#     u1 = U[:,1]  # m x 1
   
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
#     Φ[2:end,2:end] = u1 * u1'

#     # State-derivative correlation matrix
#     Ψ = zeros(d, r)
#     Ψ[1,1] = norm(x1) * norm(xdot1)
#     Ψ[2:end,1] = u1 * norm(xdot1)

#     # Streaming process
#     for i in 2:K 
#         # (2) Receive new data
#         xi = X[:,i] # n x 1
#         xdoti = Xdot[:,i] # n x 1
#         ui = U[:,i] # m x 1

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
#             VVc = BlockDiagonal([Vc, 1.0I(m)])
#             Φ = VVc' * Φ * VVc
#             Ψ = VVc' * Ψ * Vc

#             # Vϕ, Λϕ, _ = svd(Φ)
#             # Φ = (Matrix ∘ Diagonal)(Λϕ[1:dmax])
#             # Ψ = Vϕ[:,1:dmax]' * Ψ * Vc[:,1:rmax]

#             r = rmax
#             d = r + m
#         end

#         # (10) Project onto basis
#         xhat = V' * xi
#         rvec = V' * xdoti
        
#         # (11) Form the data vector, d 
#         dvec = vcat(xhat, ui)

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
Vstream, Λ, Φ, Ψ = OnePassStreamingOpInf(X, Xdot, U, rmax+rextra, 1e-12)
Vsream = Vstream[:,1:rmax]
Λ = Λ[1:rmax]
Ostream = (Φ + 1e-12I) \ Ψ
Astream = Ostream[1:rmax,1:rmax]'
Bstream = Ostream[rmax+rextra+1:end,1:rmax]'

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
        heat1d.tspan, Vr' * heat1d.IC, Ubc,
        linear_matrix=Aint[1:i, 1:i], control_matrix=Bint[1:i,:],
        system_input=true, integrator_type=:BackwardEuler
    )

    # Integrate the inferred model
    Xinf = heat1d.integrate_model(
        heat1d.tspan, Vr' * heat1d.IC, Ubc,
        linear_matrix=Ainf[1:i, 1:i], control_matrix=Binf[1:i,:],
        system_input=true, integrator_type=:BackwardEuler
    )

    # Integrate the streaming model
    Xstream = heat1d.integrate_model(
        heat1d.tspan, Vr_stream' * heat1d.IC, Ubc,
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
        limits=(nothing, nothing, 1e-6, 1e+1),
    )
    scatterlines!(ax, 1:rmax, intru_state_err, label = "intrusive", linewidth=8, markersize=30)
    scatterlines!(ax, 1:rmax, opinf_state_err, label = "opinf", linewidth=5, markersize=20, linestyle=:dash)
    scatterlines!(ax, 1:rmax, stream_state_err, label = "stream", linewidth=3, markersize=15, linestyle=:dashdot)
    axislegend(ax, position = :lb, labelsize=30)
    display(fig)
end