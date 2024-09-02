using Kronecker
using LiftAndLearn 
using Test
using Random

const LnL = LiftAndLearn

@testset "Duplication Matrix" begin
    n = 2
    D = LnL.dupmat(n, 2)
    @test all(D .== [1 0 0; 0 1 0; 0 1 0; 0 0 1])
end

@testset "Elimination Matrix" begin
    n = 2
    L = LnL.elimat(n, 2)
    @test all(L .== [1 0 0 0; 0 1 0 0; 0 0 0 1])
end

@testset "Symmetric Elimination Matrix" begin
    n = 2
    L = LnL.elimat(n, 2) * LnL.symmtzrmat(n, 2)
    @test all(L .== [1 0 0 0; 0 0.5 0.5 0; 0 0 0 1])
end

@testset "Commutation matrix" begin
    n = 2
    K = LnL.commat(n, 2)
    @test all(K .== [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1])
end


@testset "vech" begin
    A = [1 2; 3 4]
    a = [1; 3; 4]
    @test all(a .== LnL.vech(A))
end


@testset "inverse vectorization" begin
    A = [1 2; 3 4]
    a = vec(A)
    @test all(A .== LnL.invec(a, 2, 2))
end


@testset "Quadratic matrix conversion" begin    # Settings for the KS equation
    N = 32
    KSE = LnL.KuramotoSivashinskyModel(
        spatial_domain=(0.0, 22.0), time_domain=(0.0, 100.0), diffusion_coeffs=1.0,
        Δx=(22 - 1/N)/N, Δt=0.01, conservation_type=:EP
    )
    _, F = KSE.spectral_galerkin_model(KSE, KSE.diffusion_coeffs)
    n = size(F, 1)

    # @test all(F .== LnL.H2F(LnL.F2H(F)))
    # @test all(F .== LnL.H2F(LnL.F2Hs(F)))
    @test all(F .== LnL.eliminate(LnL.duplicate(F, 2), 2))
    @test all(F .== LnL.eliminate(LnL.duplicate_symmetric(F, 2), 2))

    # H = LnL.F2H(F)
    H = LnL.duplicate(F,2)
    Q = LnL.H2Q(H)
    Hnew = Matrix(LnL.Q2H(Q))
    @test all(H .== Hnew)
end


@testset "Insert and Extract" begin
    H = [
        0.0  1.0  0.0  0.0  1.0  0.0  0.0  0.0  1.0
        0.0  0.0  1.0  0.0  1.0  0.0  0.0  0.0  1.0
        1.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0
    ]
    H2 = LnL.insert2H(H, 4)
    H2test = [
        0.0  1.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0
        0.0  0.0  1.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0
        1.0  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
    ]
    @test all(H2 .== H2test)
    @test all(H .== LnL.extractH(H2, 3))

    F = LnL.eliminate(H, 2)
    F2 = LnL.insert2F(F, 4)
    F2test = [
        0.0  1.0  0.0  0.0  1.0  0.0  0.0  1.0  0.0  0.0
        0.0  0.0  1.0  0.0  1.0  0.0  0.0  1.0  0.0  0.0
        1.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0
        0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
    ]
    @test all(F2 .== F2test)
    @test all(F .== LnL.extractF(F2, 3))
end


@testset "3rd order matrix conversions" begin
    for n in [2, 3, 4, 5, 6, 10]
        x = 1:n # Julia creates Double by default
        x3e = zeros(Int, div(n*(n+1)*(n+2), 6))
        l = 1
        for i = 1:n
            for j = i:n
                for k = j:n
                    x3e[l] = x[i] * x[j] * x[k]
                    l += 1
                end
            end
        end

        L = LnL.elimat(n, 3)
        x3 = (x ⊗ x ⊗ x)
        x3e_2 = L * x3
        @test all(x3e_2 .== x3e)

        D = LnL.dupmat(n, 3)
        x3_2 = D * x3e
        @test all(x3_2 .== x3)

        G = zeros(n, n^3)
        for i = 1:n
            y = rand(n)
            G[i, :] = y ⊗ y ⊗ y
        end
        E = LnL.eliminate(G, 3)
        @test E * x3e ≈ G * x3

        Gs = LnL.duplicate_symmetric(E, 3)
        @test Gs * x3 ≈ G * x3

        G2 = LnL.duplicate(E, 3)
        @test G2 * x3_2 ≈ G * x3
    end
end


@testset "Run legacy code 1" begin
    n, j, k = 4, 2, 3
    idx = LnL.fidx(n, j, k)
    @test idx == Int((n - j/2)*(j - 1) + k)

    n, j, k = 4, 3, 2
    idx2 = LnL.fidx(n, j, k)
    @test idx == Int((n - k/2)*(k - 1) + j)

    d1 = LnL.delta(j, k)
    @test d1 == 0.5

    j = k
    d2 = LnL.delta(j, k)
    @test d2 == 1.0

    X = rand(2,2)
    BL = LnL.insert2bilin(X, 3, 1)

    idx = [(1, 1, 1), (1, 1, 2)]
    val = [1.0, 2.0]
    H = LnL.makeQuadOp(3, idx, val, which_quad_term="H", symmetric=true)
    F = LnL.makeQuadOp(3, idx, val, which_quad_term="F", symmetric=false)
    Q = LnL.makeQuadOp(3, idx, val, which_quad_term="Q", symmetric=false)
    @test all(H .== LnL.duplicate_symmetric(F, 2))
    @test all(H .== LnL.Q2H(Q))

    idx = [(1, 1, 1, 1), (1, 1, 1, 2)]
    val = [1.0, 2.0]
    G = LnL.makeCubicOp(3, idx, val, which_cubic_term="G", symmetric=true)
    E = LnL.makeCubicOp(3, idx, val, which_cubic_term="E", symmetric=false)
    @test all(G .== LnL.duplicate_symmetric(E, 3))
end


@testset "Unique Kronecker" begin
    n = 2
    x = rand(n)
    y = x ⊗ x

    z = x ⊘ x 
    @test all(dupmat(n, 2) * z .== y)

    w = ⊘(x, 3)
    y = x ⊗ x ⊗ x
    @test all(dupmat(n, 3) * w ≈ y)
end


@testset "Making polynomial operator" begin
    idx = [(1, 1, 1), (1, 1, 2)]
    val = [1.0, 2.0]
    H = LnL.makeQuadOp(3, idx, val, which_quad_term="H", symmetric=true)
    F = LnL.makeQuadOp(3, idx, val, which_quad_term="F", symmetric=false)

    A2 = LnL.makePolyOp(3, idx, val, nonredundant=false, symmetric=true)
    A2u = LnL.makePolyOp(3, idx, val, nonredundant=true, symmetric=false)
    @test all(H .== A2)
    @test all(F .== A2u)
end


@testset "Creating polynomial snapshot matrices" begin 
    n = 3
    K = 10
    X = rand(n, K)
    X2 = LnL.kron_snapshot_matrix(X, 2)
    X2u = LnL.unique_kron_snapshot_matrix(X, 2)

    @test all(elimat(n,2) * X2 .== X2u)
end


@testset "Matix dimension conversion" begin
    A = [1 2 3; 4 5 6]
    @test all(A .== LnL.tall2fat(A))
    @test all(A' .== LnL.fat2tall(A))
end