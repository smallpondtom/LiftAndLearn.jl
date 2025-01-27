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
    diffusion_coeffs=range(0.1, 10, 10),
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

# Store the training data and reference data
Xfull = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
Xdotfull = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
Ufull = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
Xref = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
Yref = Vector{Matrix{Float64}}(undef, heat1d.param_dim)

# Store the operators 
A_full = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
B_full = Vector{Matrix{Float64}}(undef, heat1d.param_dim)

A_intru = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
B_intru = Vector{Matrix{Float64}}(undef, heat1d.param_dim)

A_opinf = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
B_opinf = Vector{Matrix{Float64}}(undef, heat1d.param_dim)

A_stream = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
B_stream = Vector{Matrix{Float64}}(undef, heat1d.param_dim)
V_stream = Vector{Matrix{Float64}}(undef, heat1d.param_dim)

# Input from the boundary condition
Ubc = ones(heat1d.time_dim)

@info "Generate the data"
@showprogress for (idx, μ) in enumerate(heat1d.diffusion_coeffs)
    A, B = heat1d.finite_diff_model(heat1d, μ)
    C = ones(1, heat1d.spatial_dim) / heat1d.spatial_dim
    op_heat = LnL.Operators(A=A, B=B, C=C)
    A_full[idx] = A
    B_full[idx] = B

    # Compute the states with backward Euler
    state = heat1d.integrate_model(heat1d.tspan, heat1d.IC, Ubc; linear_matrix=A, control_matrix=B,
                               system_input=true, integrator_type=:BackwardEuler)
    Xref[idx] = state
    Xdotfull[idx] = (state[:, 2:end] - state[:, 1:end-1]) / dt
    Xfull[idx] = state[:, 2:end]
    Ufull[idx] = Ubc[2:end]'
end

X = reduce(hcat, Xfull)
Xdot = reduce(hcat, Xdotfull)
U = reduce(hcat, Ufull)

rmax = 10
tmp = svd(X)
Vrmax = tmp.U[:, 1:rmax]

#====================================#
## One-Pass Streaming-OpInf function
#====================================#
function OnePassStreamingOpInf(X, Xdot, U, rmax)
    # (1) Initialization 
    n, K = size(X)
    m = size(U,2)
    x1 = X[:,1]  # n x 1
    xdot1 = Xdot[:,1]  # n x 1
    u1 = U[:,1]  # m x 1
    d1 = vcat(x1, u1)  # (n+m) x 1
   
    # POD basis
    V = zeros(n, rmax)
    V[:,1] .= x1 / norm(x1)

    # State covariance matrix
    Ξ = zeros(rmax, rmax)
    Ξ[1,1] = dot(x1,x1) 

    # Input-state correlation matrix
    dmax = rmax + m
    Φ = zeros(dmax, dmax)
    Φ[1,1] = dot(d1, d1)

    # State-derivative correlation matrix
    Ψ = zeros(dmax, rmax)
    Ψ[1,1] = norm(d1) * norm(xdot1)

    # Initialize the reduced dimension
    r = 1

    # Streaming process
    for i in 2:K 
        # (2) Receive new data
        xi = X[:,i] # n x 1
        xdoti = Xdot[:,i] # n x 1
        ui = U[:,i] # m x 1

        Vr = @view V[:,1:r] # n x r

        # (3) Compute the orthogonal component
        xperp1 = xi - Vr * Vr' * xi
        xperp = xperp1 - Vr * Vr' * xperp1

        # (4) Take the QR decomposition
        q, xperp_mag = qr(xperp)

        # (5) Augment the POD basis
        V[:,r+1] = q

        # (6) Augment the state covariance matrix
        Ξ[1:r,r+1] = Vr' * q
        Ξ[r+1,r+1] = xperp_mag

        # (7) Zero-pad the correlation matrices 
        # Which is unnecessary in this case since we already preallocated the matrix
        
        # (8) Update the reduced dimension
        r += 1

        # (9) Compress matrices
        if r > rmax
            Λ, Θ = eigen(Ξ)
            Λ = reverse(Λ) # Sort in descending order
            Θ = reverse(Θ, dims=2)  # Sort in descending order

            Λ = Λ[1:rmax]  # rmax x 1
            Θ = Θ[:,1:rmax]  # n x rmax

            V = V * Θ  # n x rmax
            @inbounds for j in 1:rmax
                Ξ[j,j] = S[j]
            end
            Γ = BlockDiagonal([V, kron(V,V), 1.0I(m)])  
            Φ[1:rmax,1:rmax] .= BlockDiagonal([Λ, kron(Λ,Λ), Φ[end-m+1:end,end-m+1:end]])
            Ψ[1:dmax,1:rmax] .= Γ' * Ψ * Θ
            r = rmax
        end

        # (10) Project onto basis
        xhat = Vr' * xi
        rvec = Vr' * xdoti
        
        # (11) Form the data vector, d 
        dvec = vcat(xhat, ui)

        # (12) Update the covariance and correlation matrices
        Ξ[1:r,1:r] += xhat * xhat'
        Φ[1:dmax,1:dmax] += dvec * dvec'
        Ψ[1:dmax,1:r] += dvec * rvec'
    end

    return V, Ξ, Φ, Ψ
end

#====================#
## Generate operators
#====================#
rmax = 10
@showprogress for (idx, μ) in enumerate(heat1d.diffusion_coeffs)
    A = A_full[idx]
    B = B_full[idx]
    X = Xfull[idx]
    Xdot = Xdotfull[idx]
    U = Ufull[idx]

    # Compute the values for the intrusive model
    op_heat = LnL.Operators(A=A, B=B)
    op_heat_new = LnL.pod(op_heat, Vrmax, options.system)
    A_intru[idx] = op_heat_new.A
    B_intru[idx] = op_heat_new.B

    # Compute OpInf
    op_infer = LnL.opinf(X, Vrmax, options; U=U, Xdot=Xdot)
    A_opinf[idx] = op_infer.A
    B_opinf[idx] = op_infer.B
    
    # Compute One-Pass Streaming-OpInf
    V, Ξ, Φ, Ψ = OnePassStreamingOpInf(X, Xdot, U, rmax)
    Ostream = Φ \ Ψ
    A_stream[idx] = Ostream[1:rmax,1:rmax]'
    B_stream[idx] = Ostream[rmax+1:end,1:rmax]'
    V_stream[idx] = V
end

#=========#
## Analyze
#=========#
@info "Compute errors"

# Error analysis 
intru_state_err = zeros(r, 1)
opinf_state_err = zeros(r, 1)
stream_state_err = zeros(r, 1)
proj_err = zeros(r, 1)

@showprogress for i = 1:r, j = 1:heat1d.param_dim
    X = Xref[j]  # full order model states
    U = Uref[j]
    Vr = Vrmax[:, 1:i]

    # Unpack intrusive operators
    Aint = A_intru[j]
    Bint = B_intru[j]

    # Unpack inferred operators
    Ainf = A_opinf[j]
    Binf = B_opinf[j]

    # Unpack the streaming operators
    Astream = A_stream[j]
    Bstream = B_stream[j]
    Vr_stream = V_stream[j][:,1:i]

    # Integrate the intrusive model
    Xint = heat1d.integrate_model(
        heat1d.tspan, Vr' * heat1d.IC, U,
        linear_matrix=Aint[1:i, 1:i], control_matrix=Bint[1:i,:],
        system_input=true, integrator_type=:BackwardEuler
    )

    # Integrate the inferred model
    Xinf = heat1d.integrate_model(
        heat1d.tspan, Vr' * heat1d.IC, U,
        linear_matrix=Ainf[1:i, 1:i], control_matrix=Binf[1:i,:],
        system_input=true, integrator_type=:BackwardEuler
    )

    # Integrate the streaming model
    Xstream = heat1d.integrate_model(
        heat1d.tspan, Vr_stream' * heat1d.IC, U,
        linear_matrix=Astream[1:i, 1:i], control_matrix=Bstream[1:i,:],
        system_input=true, integrator_type=:BackwardEuler
    )

    # Compute errors
    PE = LnL.proj_error(X, Vr)
    PE_stream = LnL.proj_error(X, Vr_stream)

    # Relative state errors
    SE_int = LnL.rel_state_error(X, Xint, Vr)
    SE_inf = LnL.rel_state_error(X, Xinf, Vr)
    SE_stream = LnL.rel_state_error(X, Xstream, Vr_stream)

    # Sum of error values
    proj_err[i] += PE / heat1d.param_dim
    proj_err_stream[i] += PE_stream / heat1d.param_dim
    intru_state_err[i] += SE_int / heat1d.param_dim
    opinf_state_err[i] += SE_inf / heat1d.param_dim
    stream_state_err[i] += SE_stream / heat1d.param_dim
end

#=================#
## Plot the errors
#=================#
with_theme(theme_latexfonts()) do
    fig = Figure(resolution = (800, 600))
    ax = Axis(fig[1, 1], xlabel = "Reduced dimension", ylabel = "mean relative projection error")
    scatterlines!(ax, 1:r, proj_err, label = "batch")
    scatterlines!(ax, 1:r, proj_err_stream, label = "stream")
    axislegend(ax, position = :rt)
    display(fig)
end

with_theme(theme_latexfonts()) do
    fig = Figure(resolution = (800, 600))
    ax = Axis(fig[1, 1], xlabel = "Reduced dimension", ylabel = "mean relative state error")
    scatterlines!(ax, 1:r, intru_state_err, label = "intrusive")
    scatterlines!(ax, 1:r, opinf_state_err, label = "opinf")
    scatterlines!(ax, 1:r, stream_state_err, label = "stream")
    axislegend(ax, position = :rt)
    display(fig)
end