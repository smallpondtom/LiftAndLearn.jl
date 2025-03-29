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
    diffusion_coeffs=0.7,
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
op_heat = LnL.Operators(A=A, B=B)

# Compute the states with backward Euler
state = heat1d.integrate_model(heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
                            system_input=true, integrator_type=:BackwardEuler)
Xref = copy(state)
Xdot = (state[:, 2:end] - state[:, 1:end-1]) / dt
X = state[:, 2:end]
U = Ubc[2:end]'

# Another set of data with different initial condition
# heat1d.IC = cos.(4π * heat1d.xspan)
# state = heat1d.integrate_model(heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
#                             system_input=true, integrator_type=:BackwardEuler)
# Xdot = hcat(Xdot, (state[:, 2:end] - state[:, 1:end-1]) / dt)
# X = hcat(X, state[:, 2:end])
# U = hcat(U, Ubc[2:end]')

rmax = 12
tmp = svd(X)
Vrmax = tmp.U[:, 1:rmax]
Σrmax = tmp.S[1:rmax]

# Use same initial condition as reference data 
heat1d.IC = cos.(2π * heat1d.xspan)

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

# Prototype 4
function OnePassStreamingOpInf(X, Xdot, U, rmax, basis_tol, ϵ, λ, no_full=false)
    n, K = size(X)
    m = size(U,1)

    # Initial data
    x1 = X[:,1]  # n x 1
    u1 = U[:,1]  # m x 1
    xdot1 = Xdot[:,1]  # n x 1
   
    # POD basis
    V = x1 / norm(x1)

    # Eigenvalue 
    Λ = dot(x1, x1)

    # Initialize the reduced dimensions
    r = 1      # state
    d = r + m  # data (state + input)
    dmax = rmax + m

    # Input-state correlation matrix
    dvec1 = vcat(x1, u1)
    Φ = dvec1 * dvec1'

    # State-derivative correlation matrix
    Ψ = dvec1 * xdot1'

    compression = no_full ? true : false
    not_initial_compression = no_full ? true : false

    proj_err = zeros(K)
    proj_err[1] = norm(X - V * (V' * X)) / norm(X)
    compressed = []

    # Streaming process
    for i in 2:K 
        xi = X[:,i] # n x 1
        xdoti = Xdot[:,i] # n x 1
        ui = U[:,i] # m x 1

        w1 = V' * xi
        xperp = xi - V * w1
        w2 = V' * xperp
        xperp = xperp - V * w2
        w = w1 + w2
        xperp_mag = norm(xperp)

        if xperp_mag < ϵ
            xperp_mag = 0.0
        else
            xperp /= xperp_mag
        end

        C = zeros(r+1, r+1)
        @simd for j in 1:r
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

        Vc, Λc, _ = svd(C)

        if xperp_mag < ϵ  # No increment
            V = V * Vc[1:r,1:r]
            Λ = Λc[1:r]
        else  # Increment
            V = hcat(V, xperp) * Vc
            Λ = Λc

            if compression
                # # Zero-pad the correlation matrices
                # Φ = [Φ           zeros(d,1);
                #     zeros(1,d)         0.0]
                # Ψ = [Ψ           zeros(d,1);
                #     zeros(1,r)         0.0]

                # (perhaps correct zero-padding)
                Φx = zeros(r+1, r+1)
                Φx[1:r, 1:r] .= Φ[1:r, 1:r]
                Φux = zeros(m, r+1)
                Φux[:, 1:r] .= Φ[r+1:r+m, 1:r]
                Φxu = zeros(r+1, m)
                Φxu[1:r, :] .= Φ[1:r, r+1:r+m]
                Φu = Φ[r+1:r+m, r+1:r+m]
                Φ = [Φx Φxu;
                     Φux Φu]

                Ψx = zeros(r+1, r+1)
                Ψx[1:r, 1:r] .= Ψ[1:r, 1:r]
                Ψux = zeros(m, r+1)
                Ψux[:, 1:r] .= Ψ[r+1:r+m, 1:r]
                Ψ = vcat(Ψx, Ψux)
            end

            # Update the reduced dimensions
            r += 1
            d += 1
        end

        if r > rmax 
            V = V[:,1:rmax]
            Λ = Λ[1:rmax]

            Vc = Vc[:,1:rmax]
            VVc = BlockDiagonal([Vc, 1.0I(m)])

            if no_full
                Φ = VVc' * Φ * VVc
                Ψ = VVc' * Ψ * Vc
            end
            
            r = rmax
            d = r + m
        end

        @views reorthogonalize!(V, ϵ)
        PE = norm(X - V * (V' * X)) / norm(X)
        proj_err[i] = PE

        if PE < basis_tol
            compression = true
        end

        if compression && not_initial_compression
            if !no_full
                Φ = VVc' * Φ * VVc
                Ψ = VVc' * Ψ * Vc
            end

            xhat = V' * xi
            rvec = V' * xdoti
            dvec = vcat(xhat, ui)

            Φ *= λ
            Ψ *= λ
            @inbounds @fastmath for j in 1:d
                for k in 1:d
                    Φ[j, k] += dvec[j] * dvec[k]
                end
                for k in 1:r
                    Ψ[j, k] += dvec[j] * rvec[k]
                end
            end

            if no_full
                if r >= rmax
                    push!(compressed, i)
                end
            else
                push!(compressed, i)
            end
        elseif compression
            VV = BlockDiagonal([V, 1.0I(m)])
            Φ = VV' * Φ * VV
            Ψ = VV' * Ψ * V

            xhat = V' * xi
            rvec = V' * xdoti
            dvec = vcat(xhat, ui)

            Φ += dvec * dvec'
            Ψ += dvec * rvec'
            not_initial_compression = true
        else
            dvec = vcat(xi, ui)
            Φ += dvec * dvec'
            Ψ += dvec * xdoti'
        end
    end

    return V, Λ, Φ, Ψ, proj_err, compressed
end

# function OnePassStreamingOpInf(X, Xdot, U, rmax, basis_tol, ϵ, λ)
#     n, K = size(X)
#     m = size(U,1)

#     # # Randomly shuffle data
#     # idx = randperm(K)
#     # X = X[:,idx]
#     # Xdot = Xdot[:,idx]
#     # U = U[:,idx]

#     # Initial data
#     x1 = X[:,1]  # n x 1
#     u1 = U[:,1]  # m x 1
#     xdot1 = Xdot[:,1]  # n x 1
   
#     # POD basis
#     V = x1 / norm(x1)

#     # Eigenvalue 
#     Σ = norm(x1)

#     # Initialize the reduced dimensions
#     r = 1      # state
#     d = r + m  # data (state + input)

#     # Input-state correlation matrix
#     dvec1 = vcat(x1, u1)
#     Φ = dvec1 * dvec1'

#     # State-derivative correlation matrix
#     Ψ = dvec1 * xdot1'

#     compression = false
#     not_initial_compression = false

#     proj_err = zeros(K)
#     proj_err[1] = norm(X - V * (V' * X)) / norm(X)
#     compressed = []

#     # Streaming process
#     for i in 2:K 
#         xi = X[:,i] # n x 1
#         xdoti = Xdot[:,i] # n x 1
#         ui = U[:,i] # m x 1

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
#             C[j,j] = Σ[j]
#             C[j,r+1] = w[j]
#         end
#         C[r+1,r+1] = xperp_mag

#         Vc, Σc, _ = svd(C)

#         if xperp_mag < ϵ  # No increment
#             V = V * Vc[1:r,1:r]
#             Σ = Σc[1:r]
#         else  # Increment
#             V = hcat(V, xperp) * Vc
#             Σ = Σc

#             if compression
#                 # Zero-pad the correlation matrices
#                 Φ = [Φ           zeros(d,1);
#                     zeros(1,d)         1e-12]
#                 Ψ = [Ψ           zeros(d,1);
#                     zeros(1,r)         1e-12]
#             end

#             # Update the reduced dimensions
#             r += 1
#             d += 1
#         end

#         if r > rmax 
#             V = V[:,1:rmax]
#             Σ = Σ[1:rmax]

#             Vc = Vc[:,1:rmax]
#             VVc = BlockDiagonal([Vc, 1.0I(m)])
            
#             r = rmax
#             d = r + m
#         end

#         reorthogonalize!(V, ϵ)
#         PE = norm(X - V * (V' * X)) / norm(X)
#         proj_err[i] = PE

#         if PE < basis_tol
#             compression = true
#         end

#         if compression && not_initial_compression
#             Φ = VVc' * Φ * VVc
#             Ψ = VVc' * Ψ * Vc

#             xhat = V' * xi
#             rvec = V' * xdoti
#             dvec = vcat(xhat, ui)

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

#             push!(compressed, i)
#         elseif compression
#             VV = BlockDiagonal([V, 1.0I(m)])
#             Φ = VV' * Φ * VV
#             Ψ = VV' * Ψ * V

#             xhat = V' * xi
#             rvec = V' * xdoti
#             dvec = vcat(xhat, ui)

#             Φ += dvec * dvec'
#             Ψ += dvec * rvec'
#             not_initial_compression = true
#         else
#             dvec = vcat(xi, ui)
#             Φ += dvec * dvec'
#             Ψ += dvec * xdoti'
#         end
#     end

#     return V, Σ, Φ, Ψ, proj_err, compressed
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
Vstream, Λ, Φ, Ψ, stream_proj_err, compress_idx = OnePassStreamingOpInf(X, Xdot, U, rmax+rextra, 1e-5, 1e-12, 1.0)
Vsream = Vstream[:,1:rmax]
Λ = Λ[1:rmax]
Λ = sqrt.(Λ)
Ostream = (Φ) \ Ψ
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
stream_op_err = zeros(rmax)
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

    # Operator errors
    ind = vcat(1:i, rmax+1)
    Ostar = vcat(Aint', Bint')
    stream_op_err[i] = norm(Ostream[ind,:] - Ostar[ind,:]) / norm(Ostar[ind,:])
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
        fig[1, 1], xlabel = "Reduced dimension", ylabel = "Singular Value Errors",
        yscale=log10, xticks=1:rmax, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    scatterlines!(ax, 1:rmax, abs.(Σrmax - Λ) ./ Σrmax, linewidth=8, markersize=30)
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
        fig[1, 1], xlabel = "Reduced dimension", ylabel = "Streaming Operator Errors",
        xticks=1:rmax, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    scatterlines!(ax, 1:rmax, stream_op_err, linewidth=8, markersize=30)
    display(fig)
end

with_theme(theme_latexfonts()) do
    fig = Figure(size = (800, 600))
    ax = Axis(
        fig[1, 1], xlabel = "Reduced dimension", ylabel = "mean relative state error",
        yscale=log10, xticks=1:rmax, titlesize=30,
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
        limits=(nothing, nothing, 1e-7, 1e+0),
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
    lines!(ax, 1:minimum(compress_idx)-1, stream_proj_err[1:minimum(compress_idx)-1], linewidth=8, label="full")
    lines!(ax, minimum(compress_idx):length(stream_proj_err), stream_proj_err[minimum(compress_idx):end], linewidth=8, label="compressed")
    axislegend(ax, position = :rt, labelsize=30)
    display(fig) 
end