module FHN

using LinearAlgebra
using SparseArrays

export fhn

abstract type Abstract_Models end

"""
Struct for Fitzhugh-Nagumo PDE settings
"""
mutable struct fhn <: Abstract_Models
    Ω::Vector{Float64}  # spatial domain 
    T::Vector{Float64}  # temporal domain
    αD::Vector{Float64}  # alpha parameter domain
    βD::Vector{Float64}  # beta parameter domain
    Δx::Float64  # spatial grid size
    Δt::Float64  # temporal step size
    Ubc::Matrix{Float64}  # boundary condition (input)
    ICx::Matrix{Float64}  # initial condition for the original states
    ICw::Matrix{Float64}  # initial condition for the lifted states
    x::Vector{Float64}  # spatial grid points
    t::Vector{Float64}  # temporal points
    Xdim::Int64  # spatial dimension
    Tdim::Int64  # temporal dimension
    
    FOM::Function
    generateFHNmatrices::Function
end


function fhn(Ω, T, αD, βD, Δx, Δt)
    x = (Ω[1]:Δx:Ω[2]-Δx)  # do not include final boundary conditions
    t = T[1]:Δt:T[2]
    Xdim = length(x)
    Tdim = length(t)
    Ubc = ones(Tdim, 1)
    ICx = zeros(Xdim*2, 1)
    ICw = zeros(Xdim*3, 1)

    fhn(Ω, T, αD, βD, Δx, Δt, Ubc, ICx, ICw, x, t, Xdim, Tdim, FOM, generateFHNmatrices)
end


"""
Create the full order operators with the nonlinear operator expressed as f(x). 
Referenced Elizabeth's matlab function.
"""
function FOM(k, l)
    h = l / (k - 1)
    Alift = spzeros(3 * k, 3 * k)
    Blift = spzeros(3 * k, 2)
    gamma = 0.015
    R = 0.5
    c = 2
    g = 0.05

    E = sparse(1.0I, 3 * k, 3 * k)
    for i in 1:k
        E[i, i] = gamma
        E[i+2*k, i+2*k] = gamma
    end

    # Left boundary
    Alift[1, 1] = -gamma^2 / h^2 - 0.1
    Alift[1, 2] = gamma^2 / h^2
    Alift[1, k+1] = -1
    # Alift[1,2*k+1] = 1.1

    Alift[k+1, 1] = R
    Alift[k+1, k+1] = -c

    Alift[2*k+1, 1] = 2 * g
    Alift[2*k+1, 2*k+1] = -2 * gamma^2 / h^2 - 0.2

    # Right boundary 
    Alift[k, k-1] = gamma^2 / h^2
    Alift[k, k] = -gamma^2 / h^2 - 0.1
    Alift[k, 2*k] = -1
    # Alift[k,3*k] = 1.1

    Alift[2*k, k] = R
    Alift[2*k, 2*k] = -c

    Alift[3*k, 3*k] = -2 * gamma^2 / h^2 - 0.2
    Alift[3*k, k] = 2 * g

    # Inner points
    for i in 2:k-1
        Alift[i, i-1] = gamma^2 / h^2
        Alift[i, i] = -2 * gamma^2 / h^2 - 0.1
        Alift[i, i+1] = gamma^2 / h^2
        Alift[i, i+k] = -1

        Alift[i+k, i] = R
        Alift[i+k, i+k] = -c

        Alift[i+2*k, i] = 2 * g
        Alift[i+2*k, 1+2*k] = -4 * gamma^2 / h^2 - 0.2
    end

    # B matrix
    Blift[1, 1] = gamma^2 / h
    # Blift[:,2] = g
    # NOTE: The second column of the input matrix B corresponds to the constant
    # terms of the FHN PDE.
    Blift[1:2*k, 2] .= g

    Atmp = E \ Alift
    Btmp = E \ Blift

    # A and B matrix
    A = Atmp[1:2*k, 1:2*k]
    B = Btmp[1:2*k, 1]
    K = Btmp[1:2*k, 2]

    # C matrix
    Clift = spzeros(2, 3 * k)
    Clift[1, 1] = 1
    Clift[2, 1+k] = 1
    C = Clift[:, 1:2*k]

    f = x -> [-x[1:k, :] .^ 3 + 1.1 * x[1:k, :] .^ 2; spzeros(k, size(x, 2))] / gamma

    return A, B, C, K, f
end


"""
Generate the full order operators used for the intrusive model operators. 
Referenced Elizabeth's matlab function.
"""
function generateFHNmatrices(k, l)
    h = l / (k - 1)
    E = sparse(1.0I, 3 * k, 3 * k)
    A = spzeros(3 * k, 3 * k)
    H = spzeros(3 * k, 9 * k^2)
    N = spzeros(3 * k, 3 * k)
    B = spzeros(3 * k, 2)

    gamma = 0.015
    R = 0.5
    c = 2
    g = 0.05

    for i in 1:k
        E[i, i] = gamma
        E[i+2*k, i+2*k] = gamma
    end

    # Left boundary
    A[1, 1] = -gamma^2 / h^2 - 0.1
    A[1, 2] = gamma^2 / h^2
    A[1, k+1] = -1

    A[k+1, 1] = R
    A[k+1, k+1] = -c

    A[2*k+1, 1] = 2 * g
    A[2*k+1, 2*k+1] = -2 * gamma^2 / h^2 - 0.2

    # Right Boundary
    A[k, k-1] = gamma^2 / h^2
    A[k, k] = -gamma^2 / h^2 - 0.1
    A[k, 2*k] = -1

    A[2*k, k] = R
    A[2*k, 2*k] = -c

    A[3*k, 3*k] = -2 * gamma^2 / h^2 - 0.2
    A[3*k, k] = 2 * g

    # inner points
    for i = 2:k-1
        A[i, i-1] = gamma^2 / h^2
        A[i, i] = -2 * gamma^2 / h^2 - 0.1
        A[i, i+1] = gamma^2 / h^2
        A[i, i+k] = -1

        A[i+k, i] = R
        A[i+k, i+k] = -c

        A[i+2*k, i] = 2 * g
        A[i+2*k, i+2*k] = -4 * gamma^2 / h^2 - 0.2
    end

    # left boundary
    H[1, 1] = 1.1
    H[1, 2*k+1] = -0.5
    H[1, 2*k*3*k+1] = -0.5

    H[2*k+1, 2] = gamma^2 / h^2
    H[2*k+1, 3*k+1] = gamma^2 / h^2
    H[2*k+1, 2*k+1] = 1.1
    H[2*k+1, 2*k*3*k+1] = 1.1
    H[2*k+1, 2*k*3*k+2*k+1] = -2
    H[2*k+1, k+1] = -1
    H[2*k+1, k*3*k+1] = -1

    # right boundary
    H[k, 3*k*(k-1)+k] = 1.1

    H[k, (k-1)*3*k+3*k] = -0.5
    H[k, (3*k-1)*3*k+k] = -0.5

    H[3*k, (k-2)*3*k+k] = gamma^2 / h^2
    H[3*k, (k-1)*3*k+k-1] = gamma^2 / h^2
    H[3*k, (k-1)*3*k+3*k] = 1.1
    H[3*k, (3*k-1)*3*k+k] = 1.1
    H[3*k, 9*k^2] = -2
    H[3*k, (k-1)*3*k+2*k] = -1
    H[3*k, (2*k-1)*3*k+k] = -1

    # inner points
    for i = 2:k-1
        H[i, (i-1)*3*k+i] = 1.1
        H[i, (i-1)*3*k+2*k+i] = -0.5
        H[i, (i+2*k-1)*3*k+i] = -0.5

        H[i+2*k, (i-1)*3*k+i+1] = gamma^2 / h^2
        H[i+2*k, i*3*k+i] = gamma^2 / h^2
        H[i+2*k, (i-1)*3*k+i-1] = gamma^2 / h^2
        H[i+2*k, (i-2)*3*k+i] = gamma^2 / h^2
        H[i+2*k, (i-1)*3*k+i+2*k] = 1.1
        H[i+2*k, (i-1+2*k)*3*k+i] = 1.1
        H[i+2*k, (i+2*k-1)*3*k+i+2*k] = -2
        H[i+2*k, (i-1)*3*k+k+i] = -1
        H[i+2*k, (i+k-1)*3*k+i] = -1
    end

    N[2*k+1, 1] = 2 * gamma^2 / h

    B[1, 1] = gamma^2 / h
    # B[:,2] = g
    B[1:2*k, 2] .= g

    C = spzeros(2, 3 * k)
    C[1, 1] = 1
    C[2, 1+k] = 1

    A = E \ A
    tmp = E \ B
    B = tmp[:,1]
    K = tmp[:,2]
    H = E \ H
    N = E \ N

    # NOTE: (BELOW) making the sparse H matrix symmetric
    # n = 3 * k
    # Htensor = ndSparse(H, n)
    # H2 = reshape(permutedims(Htensor, [2, 1, 3]), (n, n^2))
    # H3 = reshape(permutedims(Htensor, [3, 1, 2]), (n, n^2))
    # symH2 = 0.5 * (H2 + H3)
    # symHtensor2 = reshape(symH2, (n, n, n))
    # symHtensor = permutedims(symHtensor2, [2 1 3])
    # symH = reshape(symHtensor, (n, n^2))
    # H = sparse(symH)

    return A, B, C, H, N, K
end

end