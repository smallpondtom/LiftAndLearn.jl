using LiftAndLearn
using Test

const LnL = LiftAndLearn

@testset "POD test 1" begin
    # POD test 1
    n = 6
    m = 2
    l = 2
    p = 1
    r = 4

    A = round.(rand(n, n), digits=2)
    B = round.(rand(n, m), digits=2)
    C = round.(rand(l, n), digits=2)
    K = round.(rand(n, 1), digits=2)
    H = zeros(n, n^2)
    for i in 1:n
        h = round.(rand(n,n), digits=2)
        h *= h'
        H[i,:] = vec(h)
    end
    F = LnL.eliminate(H,2)
    N = round.(rand(n, n), digits=2)
    op = LnL.Operators(A=A, B=B, C=C, K=K, A2u=F, A2=H, N=N)

    Vr = round.(rand(n, r), digits=2)
    Ahat = Vr' * A * Vr
    Bhat = Vr' * B
    Chat = C * Vr
    Khat = Vr' * K
    Hhat = Vr' * H * kron(Vr, Vr)
    Fhat = Vr' * F * LnL.elimat(n,2) * kron(Vr, Vr) * LnL.dupmat(r,2)
    Nhat = Vr' * N * Vr
    op_naive = LnL.Operators(A=Ahat, B=Bhat, C=Chat, K=Khat, A2u=Fhat, A2=Hhat, N=Nhat)

    system = LnL.SystemStructure(
        state=[1,2],
        control=1,
        output=1,
        coupled_input=1,
        constant=1,
    )
    op_rom = LnL.pod(op, Vr, system, nonredundant_operators=true)

    for field in fieldnames(typeof(op_rom))
        if field != :f && field != :A2t && field != :dims
            mat_rom = getfield(op_rom, field)
            mat_naive = getfield(op_naive, field)
            @test all(mat_rom ≈ mat_naive)
        end
    end
end


@testset "POD test 2" begin
    # POD test 1
    n = 6
    m = 2
    l = 2
    p = 3
    r = 4

    A = round.(rand(n, n), digits=2)
    B = round.(rand(n, m), digits=2)
    C = round.(rand(l, n), digits=2)
    K = round.(rand(n, 1), digits=2)
    H = round.(rand(n, n^2), digits=2)
    F = LnL.eliminate(H,2)
    N = round.(rand(n, n, p), digits=2)
    op = LnL.Operators(A=A, B=B, C=C, K=K, A2u=F, A2=H, N=N)

    Vr = round.(rand(n, r), digits=2)
    Ahat = Vr' * A * Vr
    Bhat = Vr' * B
    Chat = C * Vr
    Khat = Vr' * K
    Hhat = Vr' * H * kron(Vr, Vr)
    Fhat = Vr' * F * LnL.elimat(n,2) * kron(Vr, Vr) * LnL.dupmat(r,2)
    Nhat = zeros(r, r, p)
    for i in 1:p
        Nhat[:, :, i] = Vr' * N[:, :, i] * Vr
    end
    op_naive = LnL.Operators(A=Ahat, B=Bhat, C=Chat, K=Khat, A2u=Fhat, A2=Hhat, N=Nhat)

    system = LnL.SystemStructure(
        state=[1,2],
        control=1,
        output=1,
        coupled_input=1,
        constant=1,
    )
    op_rom = LnL.pod(op, Vr, system, nonredundant_operators=false)

    for field in fieldnames(typeof(op_rom))
        if field != :f && field != :A2t && field != :dims
            mat_rom = getfield(op_rom, field)
            mat_naive = getfield(op_naive, field)
            @test all(mat_rom ≈ mat_naive)
        end
    end
end
