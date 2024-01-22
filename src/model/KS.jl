"""
    Kuramoto-Sivashinsky equation PDE model
"""
module KS

using DocStringExtensions
using FFTW
using LinearAlgebra
using SparseArrays

export ks

"""
    Abstract_Models

Abstract type for the models.
"""
abstract type Abstract_Models end


"""
$(TYPEDEF)

Kuramoto-Sivashinsky equation PDE model
    
```math
\\frac{\\partial u}{\\partial t} = -\\mu\\frac{\\partial^4 u}{\\partial x^4} - \\frac{\\partial^2 u}{\\partial x^2} - u\\frac{\\partial u}{\\partial x}
```

where ``u`` is the state variable and ``\\mu`` is the viscosity coefficient.

## Fields
- `Omega::Vector{Float64}`: spatial domain
- `T::Vector{Float64}`: temporal domain
- `D::Vector{Float64}`: parameter domain
- `nx::Float64`: number of spatial grid points
- `Δx::Float64`: spatial grid size
- `Δt::Float64`: temporal step size
- `IC::VecOrMat{Float64}`: initial condition
- `x::Vector{Float64}`: spatial grid points
- `t::Vector{Float64}`: temporal points
- `k::Vector{Float64}`: Fourier modes
- `μs::Union{Vector{Float64},Float64}`: parameter vector
- `Xdim::Int64`: spatial dimension
- `Tdim::Int64`: temporal dimension
- `Pdim::Int64`: parameter dimension
- `type::String`: model type
- `model_PS::Function`: model using Pseudo-Spectral Method/Fast Fourier Transform
- `model_PS_ew::Function`: model using Pseudo-Spectral Method/Fast Fourier Transform (element-wise)
- `model_SG::Function`: model using Spectral-Galerkin Method
- `model_FD::Function`: model using Finite Difference
- `integrate_FD::Function`: integrator using Crank-Nicholson Adams-Bashforth method
- `integrate_PS::Function`: integrator using Crank-Nicholson Adams-Bashforth method in the Fourier space
- `integrate_PS_ew::Function`: integrator using Crank-Nicholson Adams-Bashforth method in the Fourier space (element-wise)
- `integrate_SG::Function`: integrator for second method of Fourier Transform without FFT
"""
mutable struct ks <: Abstract_Models
    Omega::Vector{Float64}  # spatial domain
    T::Vector{Float64}  # temporal domain
    D::Vector{Float64}  # parameter domain
    nx::Float64  # number of spatial grid points
    Δx::Float64  # spatial grid size
    Δt::Float64  # temporal step size
    IC::VecOrMat{Float64}  # initial condition
    x::Vector{Float64}  # spatial grid points
    t::Vector{Float64}  # temporal points
    k::Vector{Float64}  # Fourier modes
    μs::Union{Vector{Float64},Float64}  # parameter vector
    Xdim::Int64  # spatial dimension
    Tdim::Int64  # temporal dimension
    Pdim::Int64  # parameter dimension

    type::String  # model type

    model_PS::Function  # model using Pseudo-Spectral Method/Fast Fourier Transform
    model_PS_ew::Function  # model using Pseudo-Spectral Method/Fast Fourier Transform (element-wise)
    model_SG::Function  # model using Spectral-Galerkin Method
    model_FD::Function  # model using Finite Difference
    integrate_FD::Function  # integrator using Crank-Nicholson Adams-Bashforth method
    integrate_PS::Function  # integrator using Crank-Nicholson Adams-Bashforth method in the Fourier space
    integrate_PS_ew::Function  # integrator using Crank-Nicholson Adams-Bashforth method in the Fourier space (element-wise)
    integrate_SG::Function  # integrator for second method of Fourier Transform without FFT
end


"""
    ks(Omega, T, D, nx, Δt, Pdim, type) → ks

Kuramoto-Sivashinsky equation PDE model constructor

## Arguments
- `Omega::Vector{Float64}`: spatial domain
- `T::Vector{Float64}`: temporal domain
- `D::Vector{Float64}`: parameter domain
- `nx::Float64`: number of spatial grid points
- `Δt::Float64`: temporal step size
- `Pdim::Int64`: parameter dimension

## Returns
- `ks`: Kuramoto-Sivashinsky equation PDE model
"""
function ks(Omega, T, D, nx, Δt, Pdim, type)
    Δx = (Omega[2] - Omega[1]) / nx
    x = collect(Omega[1]:Δx:Omega[2]-Δx)  # assuming a periodic boundary condition
    t = collect(T[1]:Δt:T[2])
    k = collect(-nx/2:1.0:nx/2-1)  

    μs = Pdim == 1 ? D[1] : collect(range(D[1], D[2], Pdim))
    Xdim = length(x)
    Tdim = length(t)
    IC = zeros(Xdim, 1)

    @assert nx == Xdim "nx must be equal to Xdim"
    @assert (type == "c" || type == "nc" || type == "ep") "type must be either c, nc, or ep"

    ks(
        Omega, T, D, nx, Δx, Δt, IC, x, t, k, μs, Xdim, Tdim, Pdim, type,
        model_PS, model_PS_ew, model_SG, model_FD, integrate_FD, 
        integrate_PS, integrate_PS_ew, integrate_SG
    )
end


"""
    model_FD(model, μ) → A, F

Generate A, F matrices for the Kuramoto-Sivashinsky equation using the Finite Difference method.

## Arguments
- `model`: Kuramoto-Sivashinsky equation model
- `μ`: parameter value

## Returns
- `A`: A matrix
- `F`: F matrix
"""
function model_FD(model::ks, μ::Float64)
    N = model.Xdim
    Δx = model.Δx

    # Create A matrix
    ζ = 2/Δx^2 - 6*μ/Δx^4
    η = 4*μ/Δx^4 - 1/Δx^2
    ϵ = -μ/Δx^4

    A = spdiagm(
        0 => ζ * ones(N), 
        1 => η * ones(N - 1), -1 => η * ones(N - 1),
        2 => ϵ * ones(N - 2), -2 => ϵ * ones(N - 2)
    )
    # For the periodicity for the first and final few indices
    A[1, end-1:end] = [ϵ, η]
    A[2, end] = ϵ
    A[end-1, 1] = ϵ
    A[end, 1:2] = [η, ϵ]
    
    # Create F matrix
    S = Int(N * (N + 1) / 2)
    if model.type == "nc"
        if N >= 3
            Fval = repeat([1.0, -1.0], outer=N - 2)
            row_i = repeat(2:(N-1), inner=2)
            seq = Int.([2 + (N + 1) * (x - 1) - x * (x - 1) / 2 for x in 1:(N-1)])
            col_i = vcat(seq[1], repeat(seq[2:end-1], inner=2), seq[end])
            F = sparse(row_i, col_i, Fval, N, S) / 2 / Δx

            # For the periodicity for the first and final indices
            F[1, 2] = - 1 / 2 / Δx
            F[1, N] = 1 / 2 / Δx
            F[N, N] = - 1 / 2 / Δx
            F[N, end-1] = 1 / 2 / Δx 
        else
            F = zeros(N, S)
        end
    elseif model.type == "c"
        if N >= 3
            ii = repeat(2:(N-1), inner=2)
            m = 2:N-1
            mm = Int.([N*(N+1)/2 - (N-m).*(N-m+1)/2 - (N-m) - (N-(m-2)) for m in 2:N-1])  # this is where the x_{i-1}^2 term is
            mp = Int.([N*(N+1)/2 - (N-m).*(N-m+1)/2 - (N-m) + (N-(m-1)) for m in 2:N-1])  # this is where the x_{i+1}^2 term is
            jj = reshape([mp'; mm'],2*N-4);
            vv = reshape([-ones(1,N-2); ones(1,N-2)],2*N-4)/(4*Δx);
            F = sparse(ii,jj,vv,N,S)

            # Boundary conditions (Periodic)
            F[1,N+1] = -1/4/Δx
            F[1,end] = 1/4/Δx
            F[N,end-2] = 1/4/Δx
            F[N,1] = -1/4/Δx
        else
            F = zeros(N, S)
        end
    elseif model.type == "ep"
        if N >= 3
            ii = repeat(2:(N-1), inner=4)
            m = 2:N-1
            mi = Int.([N*(N+1)/2 - (N-m)*(N-m+1)/2 - (N-m) for m in 2:N-1])               # this is where the xi^2 term is
            mm = Int.([N*(N+1)/2 - (N-m).*(N-m+1)/2 - (N-m) - (N-(m-2)) for m in 2:N-1])  # this is where the x_{i-1}^2 term is
            mp = Int.([N*(N+1)/2 - (N-m).*(N-m+1)/2 - (N-m) + (N-(m-1)) for m in 2:N-1])  # this is where the x_{i+1}^2 term is
            jp = mi .+ 1  # this is the index of the x_{i+1}*x_i term
            jm = mm .+ 1  # this is the index of the x_{i-1}*x_i term
            jj = reshape([mp'; mm'; jp'; jm'],4*N-8);
            vv = reshape([-ones(1,N-2); ones(1,N-2); -ones(1,N-2); ones(1,N-2)],4*N-8)/(6*Δx);
            F = sparse(ii,jj,vv,N,S)

            # Boundary conditions (Periodic)
            F[1,2] = -1/6/Δx
            F[1,N+1] = -1/6/Δx
            F[1,N] = 1/6/Δx
            F[1,end] = 1/6/Δx
            F[N,end-1] = 1/6/Δx
            F[N,end-2] = 1/6/Δx
            F[N,1] = -1/6/Δx
            F[N,N] = -1/6/Δx
        else
            F = zeros(N, S)
        end
    else
        error("type must be either c, nc, or ep")
    end

    return A, sparse(F)
end


"""
    model_PS(model, μ) → A, F

Generate A, F matrices for the Kuramoto-Sivashinsky equation using the Pseudo-Spectral/Fast Fourier Transform method.

## Arguments
- `model`: Kuramoto-Sivashinsky equation model
- `μ`: parameter value

## Returns
- `A`: A matrix
- `F`: F matrix  (take out 1.0im)
"""
function model_PS(model::ks, μ::Float64)
    L = model.Omega[2]

    # Create A matrix
    A = spdiagm(
        0 => [(2 * π * k / L)^2 - μ*(2 * π * k / L)^4 for k in model.k]
    )
    
    # Create F matix
    F = spdiagm(
        0 => [-π * k / L for k in model.k]
    )
    return A, F
end


"""
    model_PS_ew(model, μ) → A, F

Generate A, F matrices for the Kuramoto-Sivashinsky equation using the Fast Fourier Transform method (element-wise).

## Arguments
- `model`: Kuramoto-Sivashinsky equation model
- `μ`: parameter value

## Returns
- `A`: A matrix
- `F`: F matrix
"""
function model_PS_ew(model::ks, μ::Float64)
    L = model.Omega[2]

    # Create A matrix
    A = [(2 * π * k / L)^2 - μ*(2 * π * k / L)^4 for k in model.k]
    # Create F matix
    F = [-π * 1.0im * k / L for k in model.k]
    return A, F
end


"""
    model_SG(model, μ) → A, F

Generate A, F matrices for the Kuramoto-Sivashinsky equation using the Spectral-Galerkin method.

## Arguments
- `model`: Kuramoto-Sivashinsky equation model
- `μ`: parameter value

## Returns
- `A`: A matrix
- `F`: F matrix
"""
function model_SG(model::ks, μ::Float64)
    N = model.Xdim
    L = model.Omega[2]

    # Create A matrix
    A = spdiagm(
        0 => [(2 * π * k / L)^2 - μ*(2 * π * k / L)^4 for k in model.k]
    )

    # Create F matrix
    # WARNING: The 1.0im is taken out from F
    F = spzeros(N, Int(N * (N + 1) / 2))
    for k in model.k
        foo = zeros(N, N)
        for p in model.k, q in model.k
            if  p + q == k
                pshift = Int(p + N/2 + 1)
                qshift = Int(q + N/2 + 1)
                if pshift > qshift
                    foo[pshift, qshift] += -π * k / L
                else 
                    foo[qshift, pshift] += -π * k / L
                end
            end
        end
        F[Int(k+N/2+1), :] =  vech(foo)
    end

    # # INFO: Another way to creat F matrix inspired by the convolution operator
    # F = spzeros(N, Int(N * (N + 1) / 2))
    # D = Dict{Integer, AbstractVector}()
    # for i in 0:-1:-N+1
    #     D[i] = abs(i-1)*ones(N) .+ 1im*collect(1:N)
    # end
    # ConvIdxMat = spdiagm(2*N-1, N, D...)
    # ConvIdxMat = ConvIdxMat[Int(N/2)+1:Int(N+N/2), :]
    # F = spzeros(N, Int(N*(N+1)/2))

    # for (i,k) in enumerate(model.k)
    #     foo = spzeros(N, N)
    #     for j in 1:N
    #         idx = ConvIdxMat[i, j]
    #         if idx != 0.0
    #             p = idx |> real |> Int
    #             q = idx |> imag |> Int
    #             if p > q
    #                 foo[p, q] += -π * k / L
    #             else
    #                 foo[q, p] += -π * k / L
    #             end
    #         end
    #     end
    #     F[i, :] = vech(foo)
    # end

    return A, F
end


"""
    vech(A) → v

Half-vectorization operation

## Arguments
- `A`: matrix to half-vectorize

## Returns
- `v`: half-vectorized form
"""
function vech(A::AbstractMatrix{T}) where {T}
    m = LinearAlgebra.checksquare(A)
    v = Vector{T}(undef, (m * (m + 1)) >> 1)
    k = 0
    for j = 1:m, i = j:m
        @inbounds v[k+=1] = A[i, j]
    end
    return v
end


"""
    integrate_FD(A, F, tdata, IC; const_stepsize=true, u2_lm1=nothing) → u

Integrator using Crank-Nicholson Adams-Bashforth method for (FD)

## Arguments
- `A`: A matrix
- `F`: F matrix
- `tdata`: temporal points
- `IC`: initial condition
- `const_stepsize`: whether to use a constant time step size
- `u2_lm1`: u2 at j-2

## Returns
- `u`: state matrix
"""
function integrate_FD(A, F, tdata, IC; const_stepsize=true, u2_lm1=nothing)
    Xdim = length(IC)
    Tdim = length(tdata)
    u = zeros(Xdim, Tdim)
    u[:, 1] = IC
    # u2_lm1 = Vector{Float64}()  # u2 at j-2 placeholder

    if const_stepsize
        Δt = tdata[2] - tdata[1]  # assuming a constant time step size
        ImdtA_inv = Matrix(1.0I(Xdim) - Δt/2 * A) \ 1.0I(Xdim) # |> sparse
        IpdtA = (1.0I(Xdim) + Δt/2 * A)

        for j in 2:Tdim
            u2 = vech(u[:, j-1] * u[:, j-1]')
            if j == 2 && isnothing(u2_lm1)
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * Δt)
            else
                u[:, j] = ImdtA_inv * (IpdtA * u[:, j-1] + F * u2 * 3*Δt/2 - F * u2_lm1 * Δt/2)
            end
            u2_lm1 = u2
        end
    else
        for j in 2:Tdim
            Δt = tdata[j] - tdata[j-1]
            u2 = vech(u[:, j-1] * u[:, j-1]')
            if j == 2 && isnothing(u2_lm1)
                u[:, j] = (1.0I(Xdim) - Δt/2 * A) \ ((1.0I(Xdim) + Δt/2 * A) * u[:, j-1] + F * u2 * Δt)
            else
                u[:, j] = (1.0I(Xdim) - Δt/2 * A) \ ((1.0I(Xdim) + Δt/2 * A) * u[:, j-1] + F * u2 * 3*Δt/2 - F * u2_lm1 * Δt/2)
            end
            u2_lm1 = u2
        end
    end
    return u
end


"""
    integrate_PS(A, F, tdata, IC) → u, uhat

Integrator using Crank-Nicholson Adams-Bashforth method for (FFT)

## Arguments
- `A`: A matrix
- `F`: F matrix
- `tdata`: temporal points
- `IC`: initial condition

## Returns
- `u`: state matrix
- `uhat`: state matrix in the Fourier space
"""
function integrate_PS(A, F, tdata, IC)
    Xdim = length(IC)
    Tdim = length(tdata)
    foo = zeros(ComplexF64, Xdim)

    # Plan the fourier transform and inverse fourier transform
    pfft = plan_fft(IC)
    pifft = plan_ifft(foo)

    u = zeros(Xdim, Tdim)  # state in the physical space
    u[:, 1] = IC
    uhat = zeros(ComplexF64, Xdim, Tdim)  # state in the Fourier space
    uhat[:, 1] = fftshift(pfft * u[:, 1]) / Xdim
    uhat2_lm1 = Vector{ComplexF64}()  # uhat2 at j-2 placeholder

    for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        uhat2 = fftshift(pfft * (u[:, j-1].^2)) / Xdim

        # WARNING: The 1.0im is taken out from F
        if j == 2
            uhat[:, j] = (1.0I(Xdim) - Δt/2 * A) \ ((1.0I(Xdim) + Δt/2 * A) * uhat[:, j-1] + 1.0im * F * uhat2 * Δt)
        else
            uhat[:, j] = (1.0I(Xdim) - Δt/2 * A) \ ((1.0I(Xdim) + Δt/2 * A) * uhat[:, j-1] + 1.0im * F * uhat2 * 3*Δt/2 - 1.0im * F * uhat2_lm1 * Δt/2)
        end

        # Get the state in the physical space
        u[:, j] = real.(Xdim * pifft * (ifftshift(uhat[:, j])))

        # Clean
        uhat[:, j] = fftshift(pfft * u[:, j]) / Xdim
        uhat2_lm1 = uhat2
    end
    return u, uhat
end


"""
    integrate_PS_ew(A, F, tdata, IC) → u, uhat

Integrator using Crank-Nicholson Adams-Bashforth method for (FFT) (element-wise)

## Arguments
- `A`: A matrix
- `F`: F matrix
- `tdata`: temporal points
- `IC`: initial condition

## Returns
- `u`: state matrix
- `uhat`: state matrix in the Fourier space
"""
function integrate_PS_ew(A, F, tdata, IC)
    Xdim = length(IC)
    Tdim = length(tdata)
    foo = zeros(ComplexF64, Xdim)

    # Plan the fourier transform and inverse fourier transform
    pfft = plan_fft(IC)
    pifft = plan_ifft(foo)

    u = zeros(Xdim, Tdim)  # state in the physical space
    u[:, 1] = IC
    uhat = zeros(ComplexF64, Xdim, Tdim)  # state in the Fourier space
    uhat[:, 1] = fftshift(pfft * u[:, 1]) / Xdim
    uhat2_lm1 = Vector{ComplexF64}()  # uhat2 at j-2 placeholder

    for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        uhat2 = fftshift(pfft * (u[:, j-1].^2)) / Xdim

        if j == 2
            uhat[:, j] = (1.0 ./ (1.0 .- Δt/2 * A)) .* ((1.0 .+ Δt/2 * A) .* uhat[:, j-1] + F .* uhat2 * Δt)
        else
            uhat[:, j] = (1.0 ./ (1.0 .- Δt/2 * A)) .* ((1.0 .+ Δt/2 * A) .* uhat[:, j-1] + F .* uhat2 * 3*Δt/2 - F .* uhat2_lm1 * Δt/2)
        end

        # Get the state in the physical space
        u[:, j] = real.(Xdim * pifft * (ifftshift(uhat[:, j])))

        # Clean
        uhat[:, j] = fftshift(pfft * u[:, j]) / Xdim
        uhat2_lm1 = uhat2
    end
    return u, uhat
end


"""
    integrate_SG(A, F, tdata, IC) → u, uhat

Integrator for model produced with Spectral-Galerkin method.

## Arguments
- `A`: A matrix
- `F`: F matrix
- `tdata`: temporal points
- `IC`: initial condition

## Returns
- `u`: state matrix
- `uhat`: state matrix in the Fourier space
"""
function integrate_SG(A, F, tdata, IC)
    Xdim = length(IC)
    Tdim = length(tdata)
    foo = zeros(ComplexF64, Xdim)

    # Plan the fourier transform and inverse fourier transform
    pfft = plan_fft(IC)
    pifft = plan_ifft(foo)

    u = zeros(Xdim, Tdim)  # state in the physical space
    u[:, 1] = IC
    uhat = zeros(ComplexF64, Xdim, Tdim)  # state in the Fourier space
    uhat[:, 1] = fftshift(pfft * u[:, 1]) / Xdim
    uhat2_lm1 = Vector{ComplexF64}()  # uhat2 at j-2 placeholder

    for j in 2:Tdim
        Δt = tdata[j] - tdata[j-1]
        uhat2 = vech(uhat[:, j-1] * transpose(uhat[:, j-1]))

        # WARNING: The 1.0im is taken out from F
        if j == 2
            uhat[:, j] = (1.0I(Xdim) - Δt/2 * A) \ ((1.0I(Xdim) + Δt/2 * A) * uhat[:, j-1] + 1.0im * F * uhat2 * Δt)
        else
            uhat[:, j] = (1.0I(Xdim) - Δt/2 * A) \ ((1.0I(Xdim) + Δt/2 * A) * uhat[:, j-1] + 1.0im * F * uhat2 * 3*Δt/2 - 1.0im * F * uhat2_lm1 * Δt/2)
        end

        # Get the state in the physical space
        u[:, j] = real.(Xdim * pifft * (ifftshift(uhat[:, j])))

        # Clean
        uhat[:, j] = fftshift(pfft * u[:, j]) / Xdim
        uhat2_lm1 = uhat2
    end
    return u, uhat
end

end