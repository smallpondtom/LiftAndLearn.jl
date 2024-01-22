using LiftAndLearn
using Test

const LnL = LiftAndLearn

# 1D Heat equation setup
heat1d = LnL.heat1d(
    [0.0, 1.0], [0.0, 1.0], [0.1, 10],
    2^(-7), 1e-3, 10
)





@testset "Forward Euler" begin
    Us = Vector{Matrix{Float64}}(undef, heat1d.Pdim)
    for (i, μ) in enumerate(heat1d.μs)
        A, B = heat1d.generateABmatrix(heat1d.Xdim,μ,heat1d.Δx)
        U = LnL.forwardEuler(A,B,heat1d.Ubc,heat1d.t,heat1d.IC)
        Us[i] = U
    end
    @test any(isnan.(Us[2]))
end

@testset "Crank Nicolson" begin
    Uf = Vector{Matrix{Float64}}(undef, heat1d.Pdim)
    r = 10  # order of the reduced form

    for (i, μ) in enumerate(heat1d.μs)
        A, B = heat1d.generateABmatrix(heat1d.Xdim,μ,heat1d.Δx)
        Uf[i] = LnL.crankNicolson(A,B,heat1d.Ubc,heat1d.t,heat1d.IC)
    end
    @test !any(isnan.(Uf[2]))
end

@testset "Backward Euler" begin
    Ub = Vector{Matrix{Float64}}(undef, heat1d.Pdim)
    r = 10  # order of the reduced form

    for (i, μ) in enumerate(heat1d.μs)
        A, B = heat1d.generateABmatrix(heat1d.Xdim,μ,heat1d.Δx)
        Ub[i] = LnL.backwardEuler(A,B,heat1d.Ubc,heat1d.t,heat1d.IC)
    end
    @test !any(isnan.(Ub[2]))
end
