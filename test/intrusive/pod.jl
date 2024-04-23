using LiftAndLearn
using Test

const LnL = LiftAndLearn

@testset "POD test 1" begin
    # POD test 1
    n = 10
    m = 3
    l = 2
    p = 1
    r = 5

    A = round.(rand(n, n), digits=2)
    B = round.(rand(n, m), digits=2)
    C = round.(rand(l, n), digits=2)
    K = round.(rand(n, 1), digits=2)
    H = round.(rand(n, n^2), digits=2)
    F = LnL.H2F(H)
    N = round.(rand(n, n), digits=2)
    op = LnL.operators(A=A, B=B, C=C, K=K, F=F, H=H, N=N)

    Vr = round.(rand(n, r), digits=2)
    Ahat = Vr' * A * Vr
    Bhat = Vr' * B
    Chat = C * Vr
    Khat = Vr' * K
    Hhat = Vr' * H * kron(Vr, Vr)
    Fhat = Vr' * F * LnL.elimat(n) * kron(Vr, Vr) * LnL.dupmat(r)
    Nhat = Vr' * N * Vr
    op_naive = LnL.operators(A=Ahat, B=Bhat, C=Chat, K=Khat, F=Fhat, H=Hhat, N=Nhat)

    system = LnL.sys_struct(
        is_lin=true, 
        is_quad=true, 
        is_bilin=true, 
        has_control=true, 
        has_output=true, 
        has_const=true,
    )
    options = LnL.LS_options(system=system)
    op_rom = LnL.intrusiveMR(op, Vr, options)

    for field in fieldnames(typeof(op_rom))
        if field != :f && field != :Q
            mat_rom = getfield(op_rom, field)
            mat_naive = getfield(op_naive, field)
            @test all(mat_rom ≈ mat_naive)
        end
    end
end


@testset "POD test 2" begin
    # POD test 1
    n = 10
    m = 3
    l = 2
    p = 3
    r = 5

    A = round.(rand(n, n), digits=2)
    B = round.(rand(n, m), digits=2)
    C = round.(rand(l, n), digits=2)
    K = round.(rand(n, 1), digits=2)
    H = round.(rand(n, n^2), digits=2)
    F = LnL.H2F(H)
    N = round.(rand(n, n, p), digits=2)
    op = LnL.operators(A=A, B=B, C=C, K=K, F=F, H=H, N=N)

    Vr = round.(rand(n, r), digits=2)
    Ahat = Vr' * A * Vr
    Bhat = Vr' * B
    Chat = C * Vr
    Khat = Vr' * K
    Hhat = Vr' * H * kron(Vr, Vr)
    Fhat = Vr' * F * LnL.elimat(n) * kron(Vr, Vr) * LnL.dupmat(r)
    Nhat = zeros(r, r, p)
    for i in 1:p
        Nhat[:, :, i] = Vr' * N[:, :, i] * Vr
    end
    op_naive = LnL.operators(A=Ahat, B=Bhat, C=Chat, K=Khat, F=Fhat, H=Hhat, N=Nhat)

    system = LnL.sys_struct(
        is_lin=true, 
        is_quad=true, 
        is_bilin=true, 
        has_control=true, 
        has_output=true, 
        has_const=true,
    )
    options = LnL.LS_options(system=system)
    op_rom = LnL.intrusiveMR(op, Vr, options)

    for field in fieldnames(typeof(op_rom))
        if field != :f && field != :Q 
            mat_rom = getfield(op_rom, field)
            mat_naive = getfield(op_naive, field)
            @test all(mat_rom ≈ mat_naive)
        end
    end
end
