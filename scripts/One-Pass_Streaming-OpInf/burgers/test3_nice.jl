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
    diffusion_coeffs=0.5, BC=:dirichlet,
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
        DS=10,  # downsampling factor
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)

seed = 1234
rgen = Random.MersenneTwister(seed)
# Amp = 0.1 * randn(rgen, num_inputs)'
# Freq = randn(rgen, num_inputs)' 
# Urand = Amp .* sin.(2π*Freq .* burgers.tspan) .+ 1.0 # Random input/boundary condition for training data

Urand = randn(rgen, burgers.time_dim, num_inputs) # Random input/boundary condition for training data

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


function OnePassStreamingOpInf(X, Xdot, U, rmax, rdmax, λ)
    n, K = size(X)
    m = size(U,1)

    x1 = X[:,1] 
    xdot1 = Xdot[:,1] 
   
    V = x1 / norm(x1)
    Σ = norm(x1)
    W = 1.0
    r1 = 1  

    Vd = xdot1 / norm(xdot1)
    Σd = norm(xdot1)
    Wd = 1.0
    r2 = 1

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
        if r2 > rdmax
            Vd = Vd[:,1:rdmax]
            Σd = Σd[1:rdmax]
            Wd = Wd[:,1:rdmax]
            r2 = rdmax
        end
    end

    Σ_diag = Diagonal(Σ)
    Σd_diag = Diagonal(Σd)
    Xd = Vd * Σd_diag * Wd'

    # D = [W * Σ   (W ⊖ W) * (Σ ⊗ Σ)   U']
    D = [W * Σ_diag   (W ⧁ W) * Diagonal(Σ ⊘ Σ)   U']
    R = Xd' * V
    if !iszero(λ)
        D = vcat(D, λ * I(size(D,2)))
        R = vcat(R, zeros(size(D, 2), size(R, 2)))
    end

    # Φ = Z * Z'
    # Ψ = Z * Xd' * V

    O = D \ R

    # O = Z \ Y        # define Φ and Ψ for least‑squares (and ridge if λ > 0)
    # O = (Φ + λ*I) \ Ψ
    # return O, V, Σ, W, Vd, Σd, Wd, Φ, Ψ
    return O, V, Σ, W, Vd, Σd, Wd
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
Ostream, Vstream, Λ, W, Vd, Λd, Wd = OnePassStreamingOpInf(X, Xdot, reshape(U, 1, :), rmax+rextra, rmax, 0.0)
Vstream = Vstream[:,1:rmax]
Λ = Λ[1:rmax]

# Astream = Ostream[1:rmax,:]'
# Hstream = Ostream[rmax+1:rmax+Int(rmax^2),:]'
# Bstream = Ostream[rmax+Int(rmax^2)+1,:]
# Fstream = UniqueKronecker.eliminate(Hstream, 2)

rmax2 = Int(rmax * (rmax + 1) / 2)
Astream = Ostream[1:rmax,:]'
Fstream = Ostream[rmax+1:rmax+rmax2,:]'
Bstream = Ostream[rmax+rmax2+1,:]

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

# with_theme(theme_latexfonts()) do 
#     fig = Figure(size = (800, 600))
#     ax = Axis(
#         fig[1, 1], xlabel = "stream", ylabel = "relative rojection error",
#         yscale=log10, titlesize=30, xlabelsize=30, ylabelsize=30,
#         xticklabelsize=25, yticklabelsize=25,
#     )
#     lines!(ax, 1:minimum(compress_idx)-1, stream_proj_err[1:minimum(compress_idx)-1], linewidth=5, label="full data")
#     lines!(ax, minimum(compress_idx):size(X,2), stream_proj_err[minimum(compress_idx):end], linewidth=5, label="compressed")
#     axislegend(ax, position = :rt, labelsize=30)
#     display(fig) 
# end

