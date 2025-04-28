"""
One-Pass Streaming-OpInf prototype for the Viscous Burgers Equation
"""

#=================#
## Load packages
#=================#
using LinearAlgebra
using BlockDiagonals
using CairoMakie
using ProgressMeter
using Random
import PolynomialModelReductionDataset: BurgersModel
import LiftAndLearn as LnL
using Kronecker
using UniqueKronecker
using SparseArrays

#=================#
## Generate data
#=================#
Ω = (0.0, 1.0)
Nx = 2^7; dt = 1e-4
burgers = BurgersModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    diffusion_coeffs=0.1, BC=:dirichlet,
)
burgers.IC = 0.1*cos.(π*burgers.xspan)
num_inputs = 10  # number of random inputs for training data
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=[1,2],
        control=1,
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=dt,
        deriv_type="SI",
        DS=20,  # downsampling factor
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)

seed = 1234
rgen = Random.MersenneTwister(seed)
Amp = 0.1 * randn(rgen, num_inputs)'
Freq = randn(rgen, num_inputs)' 
Urand = Amp .* sin.(2π*Freq .* burgers.tspan) .+ 1.0 # Random input/boundary condition for training data

# Urand = randn(rgen, burgers.time_dim, num_inputs) # Random input/boundary condition for training data

μ = burgers.diffusion_coeffs[1]
A, F, B = burgers.finite_diff_model(burgers, μ)
op_burgers = LnL.Operators(A=A, B=B, A2u=F)

# Compute the reference data with the reference input
Uref = ones(burgers.time_dim, 1);  # Reference input/boundary condition for OpInf testing 
Xref = burgers.integrate_model(
    burgers.tspan, burgers.IC, Uref; linear_matrix=A,
    control_matrix=B, quadratic_matrix=F, system_input=true
)

# Compute the training with random input 
Xall = Vector{Matrix{Float64}}(undef, num_inputs)
Xdotall = Vector{Matrix{Float64}}(undef, num_inputs)
for j in 1:num_inputs
    states = burgers.integrate_model(
        burgers.tspan, burgers.IC, Urand[:, j], linear_matrix=A,
        control_matrix=B, quadratic_matrix=F, system_input=true
    ) 
    Xall[j] = states[:, 2:end]
    Xdotall[j] = (states[:, 2:end] - states[:, 1:end-1]) / burgers.Δt
end
X = reduce(hcat, Xall)
Xdot = reduce(hcat, Xdotall)
U = reshape(Urand[2:end,:], (burgers.time_dim - 1) * num_inputs, 1)

# Down sample the training data
X = X[:, 1:options.data.DS:end]
Xdot = Xdot[:, 1:options.data.DS:end]
U = U[1:options.data.DS:end]

# Compute the SVD
rmax = 15
tmp = svd(X)
Vrmax = tmp.U[:, 1:rmax]
Σrmax = tmp.S[1:rmax]

#=================#
## Plot the data
#=================#
with_theme(theme_latexfonts()) do
    fig = Figure(size=(800, 600))
    ax = Axis3(
        fig[1, 1], xlabel=L"\omega", ylabel=L"t", zlabel=L"x(\omega,t)",
        titlesize=30, xlabelsize=30, ylabelsize=30, zlabelsize=30,
        xticklabelsize=25, yticklabelsize=25, zticklabelsize=25,
    )
    surface!(ax, burgers.xspan, burgers.tspan, Xref, colormap=:plasma)
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

function OnePassStreamingOpInf(X, Xdot, U, rmax, ϵ, λ)
    # (0) setup
    n, K = size(X)
    m = size(U, 1)

    # (1) Initialization 
    # Initial data
    x1 = X[:,1]  # n x 1
    # xx1 = kron(x1, x1) # n^2 x 1
    xx1 = x1 ⊘ x1 # n(n+1)/2 x 1
    xdot1 = Xdot[:,1]  # n x 1
    u1 = U[:,1]  # m x 1
    d1 = vcat(x1, u1)
    d1 = vcat(d1, xx1)
   
    # POD basis
    V = x1 / norm(x1)

    # Eigenvalue 
    Λ = dot(x1, x1)

    # Initialize the reduced dimensions
    r = 1                       # state
    d = r + m + Int(r*(r+1)/2)  # data (state + input + state-square)
    dmax = Int(rmax + m + rmax*(rmax+1)/2)

    # Input-state correlation matrix
    Φ = d1 * d1'

    # State-derivative correlation matrix
    Ψ = d1 * xdot1'

    reached_r = false

    proj_err = zeros(K)
    proj_err[1] = norm(X - V * (V' * X)) / norm(X)
    compressed = []

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

        if xperp_mag < ϵ
            xperp_mag = 0.0
        else
            xperp /= xperp_mag
        end

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

        # (6) Take the SVD of the core matrix
        Vc, Λc, _ = svd(C)

        # (7) Update the POD basis and Eigenvalue matrix
        if xperp_mag < ϵ  # No increment
            V = V * Vc[1:r,1:r]
            Λ = Λc[1:r]
        else  # Increment
            V = hcat(V, xperp) * Vc
            Λ = Λc

            # Update the reduced dimensions
            dold = copy(d)
            r += 1
            d = r + m + Int(r*(r+1)/2)
            ddiff = d - dold

            if reached_r
                # Zero-pad the correlation matrices
                Φ = [Φ                 zeros(dold,ddiff);
                    zeros(ddiff,dold)  zeros(ddiff,ddiff)]
                Ψ = [Ψ                 zeros(dold,1);
                    zeros(ddiff,r-1)   zeros(ddiff,1)]
            end
        end

        # (9) Compress matrices
        if r > rmax 
            V = V[:,1:rmax]
            Λ = Λ[1:rmax]
            Vc = Vc[:,1:rmax]

            # Φ = spzeros(dmax, dmax)
            # Λ_quad_red = Λ ⊘ Λ
            # for j in 1:dmax
            #     for k in 1:dmax
            #         if j == k
            #             if j <= rmax
            #                 Φ[j, k] = Λ[j]
            #             elseif rmax+1 <= j <= rmax+m
            #                 Φ[j, k] = 1.0
            #             else
            #                 Φ[j, k] = Λ_quad_red[j-rmax-m]
            #             end
            #         end
            #     end
            # end

            # Φold = copy(Φ)
            # Φ = zeros(dmax, dmax)
            # # Fill the diagonal of Φ
            # Λ_quad_red = Λ ⊘ Λ
            # for j in 1:dmax
            #     if j <= rmax
            #         Φ[j, j] = Λ[j]
            #     elseif rmax+1 <= j <= rmax+m
            #         continue
            #     else
            #         Φ[j, j] = Λ_quad_red[j-rmax-m]
            #     end
            # end

            if reached_r 
                Lr = UniqueKronecker.elimat(r,2)
                Drmax = UniqueKronecker.dupmat(rmax,2)
                Γ = Lr * (Vc ⊗ Vc) * Drmax

                VVc = (sparse ∘ BlockDiagonal)([Vc, 1.0I(m), Γ])
                Φ = VVc' * Φ * VVc

                # # Fill other blocks of Φ
                # Φ[1:rmax, rmax+1:rmax+m] = Vc' * Φold[1:r, r+1:r+m] 
                # Φ[1:rmax, rmax+m+1:end] = Vc' * Φold[1:r, r+m+1:end] * Γ
                # Φ[rmax+1:rmax+m, rmax+1:rmax+m] = Φold[r+1:r+m, r+1:r+m]
                # Φ[rmax+1:rmax+m, rmax+m+1:end] = Φold[r+1:r+m, r+m+1:end] * Γ
                # Φ[rmax+m+1:end, rmax+m+1:end] = Γ' * Φold[r+m+1:end, r+m+1:end] * Γ
                # # Fill the lower triangle of Φ
                # for j in 2:dmax
                #     for k in 1:j-1
                #         Φ[k, j] = Φ[j, k]
                #     end
                # end

                # VVc = (sparse ∘ BlockDiagonal)([Vc, 1.0I(m), Lr*(Vc ⊗ Vc)*Drmax])
                # Ψ = VVc' * Ψ * Vc

                Ψ1 = @view Ψ[1:r, :]
                Ψ2 = @view Ψ[r+1:r+m, :]
                Ψ3 = @view Ψ[r+m+1:end, :]
                Ψ = vcat(Vc' * Ψ1, Ψ2, Γ' * Ψ3) * Vc
                push!(compressed, i)
            else
                Ln = UniqueKronecker.elimat(n,2)
                Drmax = UniqueKronecker.dupmat(rmax,2)
                Γ = Ln * (V ⊗ V) * Drmax

                VVc = (sparse ∘ BlockDiagonal)([V, 1.0I(m), Γ])
                Φ = VVc' * Φ * VVc

                # # Fill other blocks of Φ
                # Φ[1:rmax, rmax+1:rmax+m] = V' * Φold[1:n, n+1:n+m] 
                # Φ[1:rmax, rmax+m+1:end] = V' * Φold[1:n, n+m+1:end] * Γ
                # Φ[rmax+1:rmax+m, rmax+1:rmax+m] = Φold[n+1:n+m, n+1:n+m]
                # Φ[rmax+1:rmax+m, rmax+m+1:end] = Φold[n+1:n+m, n+m+1:end] * Γ
                # Φ[rmax+m+1:end, rmax+m+1:end] = Γ' * Φold[n+m+1:end, n+m+1:end] * Γ
                # # Fill the lower triangle of Φ
                # for j in 2:dmax
                #     for k in 1:j-1
                #         Φ[k, j] = Φ[j, k]
                #     end
                # end

                # Vtilde = (sparse ∘ BlockDiagonal)([V, 1.0I(m), Ln*(V ⊗ V)*Drmax])
                # Ψ = Vtilde' * Ψ * V

                Ψ1 = @view Ψ[1:n, :]
                Ψ2 = @view Ψ[n+1:n+m, :]
                Ψ3 = @view Ψ[n+m+1:end, :]
                Ψ = vcat(V' * Ψ1, Ψ2, Γ' * Ψ3) * V
            end

            reached_r = true

            r = rmax
            d = r + m + Int(r*(r+1)/2)
        end

        if reached_r
            # (10) Project onto basis
            xhat = V' * xi
            rvec = V' * xdoti
            
            # (11) Form the data vector, d 
            dvec = vcat(xhat, ui)
            # dvec = vcat(dvec, kron(xhat, xhat))
            dvec = vcat(dvec, xhat ⊘ xhat)

            # (12) Update the covariance and correlation matrices
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

        else
            # xxi = kron(xi, xi)
            xxi = xi ⊘ xi
            di = vcat(xi, ui)
            di = vcat(di, xxi)
            Φ += di * di'
            Ψ += di * xdoti'
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
op_heat = LnL.Operators(A=A, B=B, A2u=F)
op_heat_new = LnL.pod(op_heat, Vrmax, options.system)
Aint = op_heat_new.A
Bint = op_heat_new.B 
Fint = op_heat_new.A2u

## Compute OpInf
op_infer = LnL.opinf(X, Vrmax, options; U=U, Xdot=Xdot)
Ainf = op_infer.A
Binf = op_infer.B 
Finf = op_infer.A2u

## Compute One-Pass Streaming-OpInf
rextra = 0
Vstream, Λ, Φ, Ψ, stream_proj_err, compress_idx = OnePassStreamingOpInf(X, Xdot, reshape(U, 1, :), rmax+rextra, 1e-12, 1.0)
Vstream = Vstream[:,1:rmax]
Λ = Λ[1:rmax]
##
Ostream = (Φ + 1e-12I) \ Ψ
Astream = Ostream[1:rmax,:]'
Bstream = Ostream[rmax+1,:]
Fstream = Ostream[rmax+2:end,:]'
# Hstream = Ostream[rmax+2:end,:]'
# Fstream = UniqueKronecker.eliminate(Hstream, 2)

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
    Xint = burgers.integrate_model(
        burgers.tspan, Vr' * burgers.IC, Uref,
        linear_matrix=Aint[1:i, 1:i], control_matrix=Bint[1:i,:], 
        quadratic_matrix=UniqueKronecker.extractF(Fint, i), 
        system_input=true,
    )

    # Integrate the inferred model
    Xinf = burgers.integrate_model(
        burgers.tspan, Vr' * burgers.IC, Uref,
        linear_matrix=Ainf[1:i, 1:i], control_matrix=Binf[1:i,:],
        quadratic_matrix=UniqueKronecker.extractF(Finf, i), 
        system_input=true, 
    )

    # Integrate the streaming model
    Xstream = burgers.integrate_model(
        burgers.tspan, Vr_stream' * burgers.IC, Uref,
        linear_matrix=Astream[1:i, 1:i], control_matrix=Bstream[1:i,:],
        quadratic_matrix=UniqueKronecker.extractF(Fstream, i),
        system_input=true, 
    )

    # Compute errors
    PE = LnL.proj_error(Xref, Vr)
    PE_stream = LnL.proj_error(Xref, Vr_stream)

    # Relative state errors
    SE_int = LnL.rel_state_error(Xref, Xint, Vr)
    SE_inf = LnL.rel_state_error(Xref, Xinf, Vr)
    SE_stream = LnL.rel_state_error(Xref, Xstream, Vr_stream)

    # Sum of error values
    proj_err[i] = PE / burgers.param_dim
    proj_err_stream[i] = PE_stream / burgers.param_dim
    intru_state_err[i] = SE_int / burgers.param_dim
    opinf_state_err[i] = SE_inf / burgers.param_dim
    stream_state_err[i] = SE_stream / burgers.param_dim
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
    lines!(ax, 1:minimum(compress_idx)-1, stream_proj_err[1:minimum(compress_idx)-1], linewidth=5, label="full data")
    lines!(ax, minimum(compress_idx):size(X,2), stream_proj_err[minimum(compress_idx):end], linewidth=5, label="compressed")
    axislegend(ax, position = :rt, labelsize=30)
    display(fig) 
end
