## Smoke tests for src/utilities/spline_interpolation.jl

# Non-uniform knots and a deterministic family of matrices A(z) = A0 + z*A1 (+ z^2*A2)
_SPL_Z = [0.0, 0.4, 1.0, 1.5, 2.2, 3.0]
_SPL_A0 = [1.0 -2.0 0.5; 3.0 0.0 -1.0]
_SPL_A1 = [0.3 1.0 -0.7; 2.0 -1.5 0.25]
_SPL_A2 = [0.1 -0.2 0.05; 0.0 0.3 -0.1]

@testset "interpolate_matrix_elements" begin
    z = _SPL_Z
    affine(t) = _SPL_A0 + t * _SPL_A1
    smooth(t) = _SPL_A0 + sin(t) * _SPL_A1 + t^2 * _SPL_A2

    for order in (3, 4, 5)
        mats = [smooth(t) for t in z]
        # reproduces the data at every knot (including the last one)
        for (k, t) in enumerate(z)
            @test LnL.interpolate_matrix_elements(z, mats, t; order=order) ≈ mats[k] atol=1e-10
        end
        out = LnL.interpolate_matrix_elements(z, mats, 1.234; order=order)
        @test size(out) == size(mats[1])
        @test eltype(out) == Float64

        # constant families are reproduced exactly by all spline types
        cmats = [copy(_SPL_A0) for _ in z]
        for t in (0.1, 0.77, 2.6)
            @test LnL.interpolate_matrix_elements(z, cmats, t; order=order) ≈ _SPL_A0 atol=1e-10
        end
    end

    # cubic (natural) and quartic splines reproduce affine families exactly
    amats = [affine(t) for t in z]
    for order in (3, 4), t in (0.1, 0.77, 1.2, 2.6, 3.0)
        @test LnL.interpolate_matrix_elements(z, amats, t; order=order) ≈ affine(t) atol=1e-10
    end

    # cubic/quartic interpolate a smooth family accurately on a fine grid
    zf = collect(range(0.0, 3.0, length=31))
    mats = [smooth(t) for t in zf]
    for order in (3, 4), t in (0.55, 1.37, 2.41)
        @test norm(LnL.interpolate_matrix_elements(zf, mats, t; order=order) - smooth(t)) < 1e-3
    end

    # unsupported order
    @test_throws AssertionError LnL.interpolate_matrix_elements(z, amats, 1.0; order=2)
    @test_throws AssertionError LnL.interpolate_matrix_elements(z, amats, 1.0; order=6)
end

@testset "Cubic spline" begin
    z = _SPL_Z
    # Natural spline: linear data has zero second derivatives and is reproduced exactly
    f = 2.0 .- 0.5 .* z
    M = LnL.cubic_spline_coefficients(z, f)
    @test length(M) == length(z)
    @test norm(M) < 1e-12
    for t in (0.0, 0.2, 0.4, 1.7, 2.99, 3.0)
        @test LnL.cubic_spline_interpolate(z, f, M, t) ≈ 2.0 - 0.5t atol=1e-12
    end

    # General data: knots reproduced, natural BCs M[1] = M[end] = 0
    g = cos.(z) .+ z .^ 3
    M = LnL.cubic_spline_coefficients(z, g)
    @test M[1] == 0 && M[end] == 0
    @test [LnL.cubic_spline_interpolate(z, g, M, t) for t in z] ≈ g atol=1e-12

    # C1 continuity at an interior knot
    ϵ = 1e-6
    k = 3
    dl = (LnL.cubic_spline_interpolate(z, g, M, z[k]) - LnL.cubic_spline_interpolate(z, g, M, z[k] - ϵ)) / ϵ
    dr = (LnL.cubic_spline_interpolate(z, g, M, z[k] + ϵ) - LnL.cubic_spline_interpolate(z, g, M, z[k])) / ϵ
    @test dl ≈ dr atol=1e-4

    # Two knots: reduces to linear interpolation
    z2, f2 = [1.0, 3.0], [2.0, 6.0]
    M2 = LnL.cubic_spline_coefficients(z2, f2)
    @test LnL.cubic_spline_interpolate(z2, f2, M2, 2.0) ≈ 4.0
end

@testset "Quartic spline" begin
    z = _SPL_Z
    n = length(z) - 1
    f = 1.0 .+ 3.0 .* z
    c = LnL.quartic_spline_coefficients(z, f)
    @test length(c) == 5n
    for t in (0.0, 0.3, 1.1, 2.5, 3.0)
        @test LnL.quartic_spline_interpolate(z, c, t) ≈ 1.0 + 3.0t atol=1e-10
    end

    g = sin.(z) .+ 0.2 .* z .^ 2
    c = LnL.quartic_spline_coefficients(z, g)
    @test [LnL.quartic_spline_interpolate(z, c, t) for t in z] ≈ g atol=1e-10
    # boundary conditions at the first knot: s''(z1) = s'''(z1) = 0
    @test abs(c[3]) < 1e-10 && abs(c[4]) < 1e-10
    # first-derivative continuity at interior knot k (local coefficients)
    k = 3
    h = z[k] - z[k-1]
    a = c[5(k-1)-4:5(k-1)]
    @test a[2] + 2a[3]*h + 3a[4]*h^2 + 4a[5]*h^3 ≈ c[5k-3] atol=1e-10
end

@testset "Quintic spline" begin
    z = _SPL_Z
    n = length(z) - 1
    f = fill(-1.5, length(z))
    c = LnL.quintic_spline_coefficients(z, f)
    @test length(c) == 6n
    for t in (0.0, 0.3, 1.1, 2.5, 3.0)
        @test LnL.quintic_spline_interpolate(z, c, t) ≈ -1.5 atol=1e-10
    end

    g = sin.(z) .+ 0.2 .* z .^ 2
    c = LnL.quintic_spline_coefficients(z, g)
    @test [LnL.quintic_spline_interpolate(z, c, t) for t in z] ≈ g atol=1e-10
    # clamped boundary conditions at the first knot: s'(z1) = s''(z1) = 0
    @test abs(c[2]) < 1e-10 && abs(c[3]) < 1e-10
    # ... and at the last knot
    h = z[end] - z[end-1]
    a = c[6n-5:6n]
    @test abs(a[2] + 2a[3]*h + 3a[4]*h^2 + 4a[5]*h^3 + 5a[6]*h^4) < 1e-8
    @test abs(2a[3] + 6a[4]*h + 12a[5]*h^2 + 20a[6]*h^3) < 1e-8
end
