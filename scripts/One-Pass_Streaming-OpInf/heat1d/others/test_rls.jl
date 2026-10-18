#================#
## Load packages
#================#
using CairoMakie
using LinearAlgebra
using IncrementalSVD
import LiftAndLearn as LnL
using PolynomialModelReductionDataset: Heat2DModel
using Printf

#================#
## Generate data
#================#
Ω = ((0.0, 1.0), (0.0, 1.0))  # or ω1 ∈ [0,1], ω2 ∈ [0,1.25]
Nx = 32
Ny = 32 # or 40
M = 10
μs = range(0.1, 1.0, length=M)
heat2d = Heat2DModel(
    spatial_domain=Ω, time_domain=(0,1.0), 
    Δx=(Ω[1][2] + 1/Nx)/Nx, Δy=(Ω[2][2] + 1/Ny)/Ny, Δt=1e-3,
    diffusion_coeffs=μs, BC=(:dirichlet, :dirichlet)
)
xgrid0 = heat2d.yspan' .* ones(heat2d.spatial_dim[1])
ygrid0 = ones(heat2d.spatial_dim[2])' .* heat2d.xspan
ux0 = sin.(2π * xgrid0) .* cos.(2π * ygrid0)
heat2d.IC = vec(ux0)  # initial condition

# Some options for operator inference
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        state=1,
        control=1,
        # output=1,
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=1e-3,
        deriv_type="BE"
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)

# Training
Uref = [1.0, 1.0, -1.0, -1.0]
Uref = repeat(Uref, 1, heat2d.time_dim)
X_all = Vector{Matrix{Float64}}(undef, length(heat2d.diffusion_coeffs))
Xdot_all = Vector{Matrix{Float64}}(undef, length(heat2d.diffusion_coeffs))
U_all = Vector{Matrix{Float64}}(undef, length(heat2d.diffusion_coeffs))
Threads.@threads for (i, μ) in collect(enumerate(heat2d.diffusion_coeffs))
    A, B = heat2d.finite_diff_model(heat2d, μ)
    # Compute the (reference) state snapshot data with backward Euler
    Xref = heat2d.integrate_model(
        heat2d.tspan, heat2d.IC, Uref; linear_matrix=A, control_matrix=B, 
        system_input=true, integrator_type=:BackwardEuler
    )
    Xdot_all[i] = (Xref[:, 2:end] - Xref[:, 1:end-1]) / heat2d.Δt
    X_all[i] = Xref[:, 2:end]
    U_all[i] = Uref[:, 2:end]
end
X = reduce(hcat, X_all)

#================#
## POD basis
#================#
rmax = 10

# Vrmax = svd(X).U[:,1:rmax]

baker = iSVD(x1=X[:,1], algo=:baker, max_rank=rmax)
full_increment!(baker, X[:,2:end], verbose=true)
Vrmax = baker.Q[:,1:rmax]

#================#
## RLS function
#================#
# Recursive Least-Squares solution of
# O = argmin_{O} ||DO - R||_F
function rls(D::Matrix, R::Matrix; γ=1e-3)
    # Initialize
    N = size(D,2)
    M = 1
    n = size(R, 2)
    O = zeros(N, n)
    P = Matrix(1.0I(N) / γ)
    K = zeros(N,M)
    u = zeros(N)
    c = zeros(M,M)
    ξpre = zeros(M,n)

    for (d,r) in zip(eachrow(D), eachrow(R))
        d = reshape(d, 1, :)
        r = reshape(r, 1, :)

        # (1)
        # K = P * d' * c[1,1]
        # P -= K * K' / c[1,1]
        # c[1,1] = 1 / (1 + dot(d, P*d'))
        # mul!(K, P, d[:], c[1,1], 0)
        # P -= K * K' / c[1,1]


        # (2)
        # c .= 1 / (1 + dot(d, P*d'))
        # P -= (P * d') * (d * P) * c[1,1]
        # mul!(K, P, d[:], 1.0, 0.0)

        # (3)
        # g = P * d' * c
        # P -= g * d * P
        # c[1,1] = 1 / (1 + dot(d, P*d'))
        # mul!(K, P, d[:], c[1,1], 0.0)
        # P -= K * d * P

        # ξpre .= r
        # mul!(ξpre, d, O, -1.0, 1.0)
        # O += K * ξpre

        mul!(u, P, d[:], 1.0, 0.0)
        # u .= P * d'
        denom = 1 + dot(d, u)
        c .= 1 / denom 
        mul!(K, P, d[:], c[1,1], 0.0)
        BLAS.syr!('U', -1.0 / denom, u, P)
        @inbounds for i in 1:N, j in i+1:N
            P[j, i] = P[i, j]
        end
        # mul!(K, P, d[:], 1.0, 0.0)

        # mul!(K, P, d[:], c[1,1], 0.0)
        # P -= K * d * P

        ξpre .= r
        mul!(ξpre, d, O, -1.0, 1.0)
        mul!(O, K, ξpre, 1.0, 1.0)

        # ξpre .= r - d * O
        # O .+= K * ξpre
    end
    return O
end



##
num_of_streams = size(X_all[1],2)
true_stream_error = zeros(M+1, num_of_streams)
stream_error = zeros(M, num_of_streams)
err = zeros(M+1, num_of_streams)
for idx in 1:length(heat2d.diffusion_coeffs)
    Xhat = Vrmax' * X_all[idx]
    U = U_all[idx]
    D = hcat(Xhat', U')
    R = Xdot_all[idx]' * Vrmax

    γ = 1e-9

    options.λ = LnL.TikhonovParameter(
        A = γ,
        B = γ,
    )
    options.with_reg = true
    op_trinf = LnL.opinf(X_all[idx], Vrmax, options; U=U_all[idx], Xdot=Xdot_all[idx])
    Ostar = op_trinf.O'

    Ohat = rls(D, R; γ=1e-9)
    @printf("||O - Ostar||_F / ||Ostar||_F = %.5e\n", norm(Ohat - Ostar, 2)/norm(Ostar, 2))

    streamsize = 1
    X_stream = LnL.streamify(Xhat, streamsize)
    U_stream = LnL.streamify(U, streamsize)
    Xdot_stream = LnL.streamify(R', streamsize)
    num_of_streams = length(X_stream)
    rls_stream  = LnL.StreamingOpInf(options=options, n=rmax, m=4, algorithm=:iQRRLS, Γs=γ)

    Eps = nothing

    for i in 1:num_of_streams
        d = LnL.stream!(rls_stream, X_stream[i], Xdot_stream[i]; U=U_stream[i])
        foo = Ostar - rls_stream.cache.O
        err_factor = 1.0I - rls_stream.cache.K * d
        if i == 1
            Eps = foo
        else 
            Eps = err_factor * Eps
        end
        tmp = rls_stream.cache.K * (Xdot_stream[i]' - d * Ostar)
        err[idx,i] = norm(tmp, 2) / norm(Ostar, 2)
        true_stream_error[idx,i] = norm(foo, 2)/norm(Ostar, 2)
        stream_error[idx,i] = norm(Eps, 2)/norm(Ostar, 2)
    end

    op_rls = LnL.terminate_stream(rls_stream)
    Ohat = op_rls.O

    @printf("||O - Ostar||_F / ||Ostar||_F = %.5e\n", norm(Ohat - Ostar, 2)/norm(Ostar, 2))
end

##
true_stream_error[end,:] = sum(true_stream_error[1:end-1,:], dims=1) / M
stream_error[end,:] = sum(stream_error[1:end-1,:], dims=1) / M
err[end,:] = sum(err[1:end-1,:], dims=1) / M
with_theme(theme_latexfonts()) do
    fig = Figure(size=(2100, 700))
    ax1 = Axis(
        fig[1, 1], xlabel="Stream", 
        ylabel=L"\Vert \mathbf{O}_* - \mathbf{O} \Vert_F / \Vert \mathbf{O}_* \Vert_F",
        yscale=log10, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    ax2 = Axis(
        fig[1, 2], xlabel="Stream", 
        ylabel=L"\Vert (\mathbf{I} - \mathbf{g}_k\mathbf{d}_k)\mathcal{E}_{k-1} \Vert_F / \Vert \mathbf{O}_* \Vert_F",
        yscale=log10, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    ax3 = Axis(
        fig[1, 3], xlabel="Stream", 
        ylabel=L"\Vert \mathbf{g}_k\mathbf{\epsilon}_k \Vert_F / \Vert \mathbf{O}_* \Vert_F",
        yscale=log10, titlesize=30, 
        xlabelsize=30, ylabelsize=30, xticklabelsize=25, yticklabelsize=25,
    )
    for i in 1:M
        lines!(ax1, 1:num_of_streams, true_stream_error[i,:])
        lines!(ax2, 1:num_of_streams, stream_error[i,:])
        lines!(ax3, 1:num_of_streams, err[i,:])
    end
    lines!(ax1, 1:num_of_streams, true_stream_error[end,:], color=:red, linewidth=5, linestyle=:dash)
    lines!(ax2, 1:num_of_streams, stream_error[end,:], color=:red, linewidth=5, linestyle=:dash)
    lines!(ax3, 1:num_of_streams, err[end,:], color=:red, linewidth=5, linestyle=:dash)
    display(fig)
end

#================#
## Data for RLS
#================#
idx = rand(1:length(heat2d.diffusion_coeffs), 1)[1]
idx = 1
Xhat = Vrmax' * X_all[idx]
U = U_all[idx]
D = hcat(Xhat', U')
R = Xdot_all[idx]' * Vrmax


#================#
## Solve for O
#================#
γ = 1e-9

##
# Ostar = (D'*D + γ*I) \ D' * R
options.λ = LnL.TikhonovParameter(
    A = γ,
    B = γ,
)
options.with_reg = true
op_trinf = LnL.opinf(X_all[idx], Vrmax, options; U=U_all[idx], Xdot=Xdot_all[idx])
Ostar = op_trinf.O'

##
Ohat = rls(D, R; γ=1e-9)
##
streamsize = 1
X_stream = LnL.streamify(Xhat, streamsize)
U_stream = LnL.streamify(U, streamsize)
Xdot_stream = LnL.streamify(R', streamsize)
num_of_streams = length(X_stream)
rls_stream  = LnL.StreamingOpInf(options=options, n=rmax, m=4, algorithm=:RLS, Γs=γ)

stream_error = zeros(num_of_streams)
for i in 1:num_of_streams
    LnL.stream!(rls_stream, X_stream[i], Xdot_stream[i]; U=U_stream[i])
    stream_error[i] = norm(rls_stream.cache.O - Ostar, 2)/norm(Ostar, 2)
end

# LnL.stream_all!(rls_stream, X_stream, Xdot_stream; U=U_stream)
op_rls = LnL.terminate_stream(rls_stream)
Ohat = op_rls.O


##
@printf("||O - Ostar||_F / ||Ostar||_F = %.5e\n", norm(Ohat - Ostar, 2)/norm(Ostar, 2))

##
plot(log10.(stream_error))