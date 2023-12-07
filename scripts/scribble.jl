## Import necessary libraries
using DifferentialEquations, Plots, SparseArrays
using FiniteDiff, LinearAlgebra, FFTW

function burgers_equation(dx, u, params)
    ν = params
    du = similar(u)
    # Using central differences for spatial derivatives
    for i in 2:length(u)-1
        du[i] = ν * (u[i+1] - 2*u[i] + u[i-1]) / dx^2 - u[i] * (u[i+1] - u[i-1]) / (2*dx)
    end
    # Boundary conditions
    du[1] = du[end] = 0
    return du
end

## Define the spatial domain and initial condition
x_min = 0.0
x_max = 2.0
N = 128
dx = (x_max - x_min) / (N - 1)
x = range(x_min, stop=x_max, length=N)
u0 = 0.2*sin.(π * x) + 0.8*sin.(2π * x)  # example initial condition
tf = 3.0
tspan = (0.0, tf)
ν = 0.4  # viscosity
dt = 0.01

## Define the ODE problem
prob = ODEProblem((u,p,t) -> burgers_equation(dx, u, ν), u0, tspan)

## Solve the problem
sol = solve(prob, Tsit5())

## Extract the solution snapshots
# snapshots = reshape(sol, (N, length(sol[2])))
snapshots = [sol(u) for u in range(tspan[1], tspan[2], length=size(sol)[2])]
snapshots = reduce(hcat, snapshots)

## Perform spectral POD
function spod(data, m)
    # Compute covariance matrix
    R = data * data' / (size(data, 2)-1)
    # Eigen decomposition
    λ, Φ = eigen(R)
    # Sort and select modes
    modes = Φ[:, sortperm(λ, rev=true)]
    # Extract the first m modes
    modes = modes[:, 1:m]
    # Compute the modal amplitudes
    a = modes' * data
    # Compute the frequency spectrum
    f = fft(a, (1,))
    return modes, a, f
end

## Define the number of POD modes
m = 10

## Perform SPOD
phi, a, f = spod(snapshots, m)

## Reconstruct the solution from the POD modes
function reconstruct(phi, a)
    u_pod = phi * a
    return u_pod
end

## Compare the full-order and reduced-order solutions
u_pod = reconstruct(phi, a)

## Convert the sol into a matrix for convenience
sol_mat = zeros(size(sol))
for i in 1:size(sol)[2]
    sol_mat[:, i] = sol[i]
end

## Plot the results
sep = length(sol) ÷ 5
idx = 1:sep:length(sol)
l = @layout [[grid(length(idx)÷2,2)]]
p = plot(layout=l, size=(800, 1000))
for (i,j) in enumerate(idx)
    println(i,j)
    plot!(p[i], sol_mat[:, j], x, lw=2, label="Full-order")
    plot!(p[i], u_pod[:, j], x, ls=:dash, lw=2, label="Reduced-order")
    plot!(p[i], legend=:topright, title="t = $(round(sol.t[j], digits=2))")
end
display(p)
# plot(sol[end], x, lw=2, label="Full-order")
# plot!(u_pod[:, end], x, ls=:dash, lw=2, label="Reduced-order")
# plot!(legend=:bottomright)

## Calculate the relative error
error = norm(sol_mat - u_pod) / norm(sol_mat)
println("Relative error:", error)
