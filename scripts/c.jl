using ChaosTools
using DynamicalSystems
using FileIO
using JLD2
using LinearAlgebra
using Random
using StaticArrays

include("../src/model/KS.jl")
include("../src/LiftAndLearn.jl")
const LnL = LiftAndLearn

DATA = load("examples/data/kse_data_L22.jld2")
OPS = load("examples/data/kse_operators_L22.jld2")

# @inline @inbounds function kse(x, p, t)
#     A = p[1]
#     H = p[2]
#     N = size(A,1)
#     xdot = A*x + H * kron(x,x)
#     return SVector{N}(xdot...)
# end

# @inline @inbounds function kse_jacobian(x, p, t)
#     A = p[1]
#     H = p[2]
#     N = size(A,1)
#     J = A + H * kron(1.0I(N), x) + H * kron(x, 1.0I(N))
#     return SMatrix{N,N}(J)
# end

@inline @inbounds function kse(x, p, t)
    A = p[1]
    F = p[2]
    N = size(A,1)
    xdot = A*x + F * LnL.vech(x*x')
    return SVector{N}(xdot...)
end

@inline @inbounds function kse_jacobian(x, p, t)
    A = p[1]
    F = p[2]
    N = size(A,1)
    J = A + F * LnL.elimat(N) * kron(1.0I(N), x) + F * LnL.elimat(N) * kron(x, 1.0I(N))
    return SMatrix{N,N}(J)
end

# Settings for the KS equation
KSE = KS(
    [0.0, 22.0], [0.0, 300.0], [1.0, 1.0],
    512, 0.001, 1, "ep"
)

ic_a = [0.8, 1.0, 1.2]
ic_b = [0.2, 0.4, 0.6]

L = KSE.Omega[2] - KSE.Omega[1]  # length of the domain
u0 = (a,b) -> a * cos.((2*π*KSE.x)/L) .+ b * cos.((4*π*KSE.x)/L)  # initial condition

ds = DeterministicIteratedMap(kse, DATA["Vr"][1][:,1:DATA["ro"][end]]' * u0(ic_a[1],ic_b[1]), [OPS["op_int"][1].A, OPS["op_int"][1].F])
tands = TangentDynamicalSystem(ds; J=kse_jacobian, k=10)

λs = lyapunovspectrum(tands, 1000; Ttr=300, Δt=0.1, show_progress=true)