"""
$(SIGNATURES)

Approximate the time derivative of the snapshot data by finite differences, using the
scheme `options.data.deriv_type` and time step `options.data.Δt`.

Same as `time_derivative_approx(X, options.data.deriv_type, options.data.Δt)`; see that
method for the available schemes and the meaning of `idx`.

## Arguments
- `X::VecOrMat`: state snapshot matrix of size ``n \\times K`` (rows = state, columns = time).
- `options::AbstractOption`: OpInf options providing `data.deriv_type` and `data.Δt`.

## Returns
- `dXdt::Matrix`: approximate time derivative of size ``n \\times K'``.
- `idx::UnitRange{Int}`: time indices (columns of `X`) at which `dXdt` approximates the derivative, with `length(idx) == K'`.
"""
function time_derivative_approx(X::VecOrMat, options::AbstractOption)
    N = size(X, 2)
    choice = options.data.deriv_type

    if choice == "FE"  # Forward Euler
        dXdt = (X[:, 2:end] - X[:, 1:end-1]) / options.data.Δt
        idx = 1:N-1
    elseif choice == "BE"  # Backward Euler
        dXdt = (X[:, 2:end] - X[:, 1:end-1]) / options.data.Δt
        idx = 2:N
    elseif choice == "SI"  # Semi-implicit Euler
        dXdt = (X[:, 2:end] - X[:, 1:end-1]) / options.data.Δt
        idx = 2:N
    elseif choice == "FE4"  # 4th order Forward Euler
        dXdt = (- 25 * X[:,1:end-4] 
                + 48 * X[:,2:end-3] 
                - 36 * X[:,3:end-2] 
                + 16 * X[:,4:end-1] 
                -  3 * X[:,5:end]) / (12*options.data.Δt)
        idx = 1:N-4
    elseif choice == "BE4"  # 4th order Backward Euler
        dXdt = (   3 * X[:,1:end-4] 
                - 16 * X[:,2:end-3] 
                + 36 * X[:,3:end-2] 
                - 48 * X[:,4:end-1] 
                + 25 * X[:,5:end]) / (12*options.data.Δt)
        idx = 5:N
    elseif choice == "CTD4"  # 4th order Central
        dXdt = (      X[:,1:end-4] 
                - 8 * X[:,2:end-3] 
                + 8 * X[:,4:end-1] 
                -     X[:,5:end]) / (12*options.data.Δt)
        idx = 3:N-2
    elseif choice == "FBCT4"  # 4th order Forward Backward Central
        dXdt = hcat(
            fwd4(X[:, 1:5], options.data.Δt, true),
            fwd4(X[:, 1:5], options.data.Δt, false),
            (      X[:,1:end-4] 
             - 8 * X[:,2:end-3] 
             + 8 * X[:,4:end-1] 
             -     X[:,5:end]) / (12*options.data.Δt),
            bwd4(X[:, end-4:end], options.data.Δt, false),
            bwd4(X[:, end-4:end], options.data.Δt, true)
        )
        idx = 1:N
    else
        error("Undefined choice of numerical integration. Choose only an " * 
              "accepted method from: FE (Forward Euler), BE (Backward Euler)," *
              "SI (Semi-implicit Euler)")
    end
    return dXdt, idx
end

"""
$(SIGNATURES)

Approximate the time derivative of the snapshot data with the finite-difference scheme
`choice` and time step `Δt`.

Column `j` of `dXdt` approximates ``\\dot{\\mathbf{x}}(t_{\\text{idx}[j]})``, so the state
data aligned with the derivative data is `X[:, idx]`. Available schemes:

| `choice`  | stencil                                                                    | `idx`      | ``K'``    |
|:----------|:---------------------------------------------------------------------------|:-----------|:----------|
| `"FE"`    | 1st-order forward difference ``(x_{k+1}-x_k)/\\Delta t``                   | `1:K-1`    | ``K-1``   |
| `"BE"`    | 1st-order backward difference ``(x_k-x_{k-1})/\\Delta t``                  | `2:K`      | ``K-1``   |
| `"SI"`    | same differences as `"BE"` (semi-implicit Euler)                           | `2:K`      | ``K-1``   |
| `"FE4"`   | 4th-order one-sided forward difference                                     | `1:K-4`    | ``K-4``   |
| `"BE4"`   | 4th-order one-sided backward difference                                    | `5:K`      | ``K-4``   |
| `"CTD4"`  | 4th-order central difference                                               | `3:K-2`    | ``K-4``   |
| `"FBCT4"` | 4th-order central inside, 4th-order one-sided at the two first/last points | `1:K`      | ``K``     |

## Arguments
- `X::VecOrMat`: state snapshot matrix of size ``n \\times K`` (rows = state, columns = time).
- `choice::String`: finite-difference scheme (see table above).
- `Δt::Real`: time step between consecutive snapshots.

## Returns
- `dXdt::Matrix`: approximate time derivative of size ``n \\times K'``.
- `idx::UnitRange{Int}`: time indices (columns of `X`) at which `dXdt` approximates the derivative.

## Notes
- An error is thrown for an unknown `choice`.
- The 4th-order schemes require ``K \\geq 5``; `"FBCT4"` additionally requires `Δt::Float64`
  (it calls `fwd4`/`bwd4`).
- The equivalent linear operator is returned by `finite_diff_matrix`.
"""
function time_derivative_approx(X::VecOrMat, choice::String, Δt::Real)
    N = size(X, 2)

    if choice == "FE"  # Forward Euler
        dXdt = (X[:, 2:end] - X[:, 1:end-1]) / Δt
        idx = 1:N-1
    elseif choice == "BE"  # Backward Euler
        dXdt = (X[:, 2:end] - X[:, 1:end-1]) / Δt
        idx = 2:N
    elseif choice == "SI"  # Semi-implicit Euler
        dXdt = (X[:, 2:end] - X[:, 1:end-1]) / Δt
        idx = 2:N
    elseif choice == "FE4"  # 4th order Forward Euler
        dXdt = (- 25 * X[:,1:end-4] 
                + 48 * X[:,2:end-3] 
                - 36 * X[:,3:end-2] 
                + 16 * X[:,4:end-1] 
                -  3 * X[:,5:end]) / (12*Δt)
        idx = 1:N-4
    elseif choice == "BE4"  # 4th order Backward Euler
        dXdt = (   3 * X[:,1:end-4] 
                - 16 * X[:,2:end-3] 
                + 36 * X[:,3:end-2] 
                - 48 * X[:,4:end-1] 
                + 25 * X[:,5:end]) / (12*Δt)
        idx = 5:N
    elseif choice == "CTD4"  # 4th order Central
        dXdt = (      X[:,1:end-4] 
                - 8 * X[:,2:end-3] 
                + 8 * X[:,4:end-1] 
                -     X[:,5:end]) / (12*Δt)
        idx = 3:N-2
    elseif choice == "FBCT4"  # 4th order Forward Backward Central
        dXdt = hcat(
            fwd4(X[:, 1:5], Δt, true),
            fwd4(X[:, 1:5], Δt, false),
            (      X[:,1:end-4] 
             - 8 * X[:,2:end-3] 
             + 8 * X[:,4:end-1] 
             -     X[:,5:end]) / (12*Δt),
            bwd4(X[:, end-4:end], Δt, false),
            bwd4(X[:, end-4:end], Δt, true)
        )
        idx = 1:N
    else
        error("Undefined choice of numerical integration. Choose only an " * 
              "accepted method from: FE (Forward Euler), BE (Backward Euler)," *
              "SI (Semi-implicit Euler)")
    end
    return dXdt, idx
end


"""
$(SIGNATURES)

Build the sparse finite-difference matrix ``\\mathbf{E}`` of a time-derivative scheme, so
that for a snapshot matrix ``\\mathbf{X}`` with `n` columns (time steps) the product
``\\mathbf{X}\\mathbf{E}`` equals the derivative approximation returned by
`time_derivative_approx(X, fd_method, dt)`.

Column `j` of ``\\mathbf{E}`` holds the stencil weights for the derivative at time index
`idx[j]`. The schemes, index ranges and number of columns ``K'`` are the same as in
`time_derivative_approx`: `"FE"` (`1:n-1`), `"BE"`/`"SI"` (`2:n`), `"FE4"` (`1:n-4`),
`"BE4"` (`5:n`), `"CTD4"` (`3:n-2`) and `"FBCT4"` (`1:n`).

## Arguments
- `fd_method::String`: finite-difference scheme (`"FE"`, `"BE"`, `"SI"`, `"FE4"`, `"BE4"`, `"CTD4"` or `"FBCT4"`).
- `n::Int`: number of time steps (columns of the snapshot matrix).
- `dt::Float64`: time step.

## Returns
- `E::SparseMatrixCSC{Float64,Int}`: finite-difference matrix of size ``n \\times K'``.
- `idx::UnitRange{Int}`: time indices at which ``\\mathbf{X}\\mathbf{E}`` approximates the derivative.

## Notes
- Throws an error for an unknown `fd_method` or if `n` is too small (``n < 2`` for the
  first-order schemes, ``n < 5`` for the fourth-order schemes).
"""
function finite_diff_matrix(fd_method::String, n::Int, dt::Float64)
    if fd_method == "FE"
        if n < 2
            error("Need at least 2 time points for 1st order finite differences.")
        end
        D = spzeros(n, n-1)
        for i in 1:n-1
            D[i, i]   = -1 / dt
            D[i+1, i] =  1 / dt
        end
        idx = 1:n-1
    elseif fd_method == "BE"
        if n < 2
            error("Need at least 2 time points for 1st order finite differences.")
        end
        D = spzeros(n, n-1)
        for i in 2:n
            D[i-1, i-1] = -1 / dt
            D[i, i-1]   =  1 / dt
        end
        idx = 2:n
    elseif fd_method == "SI"
        if n < 2
            error("Need at least 2 time points for 1st order finite differences.")
        end
        D = spzeros(n, n-1)
        for i in 2:n
            D[i-1, i-1] = -1 / dt
            D[i, i-1]   =  1 / dt
        end
        idx = 2:n
    elseif fd_method == "FE4"
        if n < 5
            error("Need at least 5 time points for 4th order finite differences.")
        end
        D = spzeros(n, n-4)
        for i in 1:n-4
            D[i, i]   = -25 / (12*dt)
            D[i+1, i] =  48 / (12*dt)
            D[i+2, i] = -36 / (12*dt)
            D[i+3, i] =  16 / (12*dt)
            D[i+4, i] =  -3 / (12*dt)
        end
        idx = 1:n-4
    elseif fd_method == "BE4"
        if n < 5
            error("Need at least 5 time points for 4th order finite differences.")
        end
        D = spzeros(n, n-4)
        for i in 1:n-4
            D[i, i]   =   3 / (12*dt)
            D[i+1, i] = -16 / (12*dt)
            D[i+2, i] =  36 / (12*dt)
            D[i+3, i] = -48 / (12*dt)
            D[i+4, i] =  25 / (12*dt)
        end
        idx = 5:n
    elseif fd_method == "CTD4"
        if n < 5
            error("Need at least 5 time points for 4th order finite differences.")
        end
        D = spzeros(n, n-4)
        for i in 3:n-2
            D[i-2, i-2] =  1 / (12*dt)
            D[i-1, i-2] = -8 / (12*dt)
            D[i+1, i-2] =  8 / (12*dt)
            D[i+2, i-2] = -1 / (12*dt)
        end
        idx = 3:n-2
    elseif fd_method == "FBCT4"
        if n < 5
            error("Need at least 5 time points for 4th order finite differences.")
        end
        D = spzeros(n, n)
        # Forward difference for first two points
        D[1, 1] = -25 / (12*dt)
        D[2, 1] =  48 / (12*dt)
        D[3, 1] = -36 / (12*dt)
        D[4, 1] =  16 / (12*dt)
        D[5, 1] =  -3 / (12*dt)

        D[1, 2] =  -3 / (12*dt)
        D[2, 2] = -10 / (12*dt)
        D[3, 2] =  18 / (12*dt)
        D[4, 2] =  -6 / (12*dt)
        D[5, 2] =   1 / (12*dt)

        # central difference for middle points
        for i in 3:n-2
            D[i-2, i] =  1 / (12*dt)
            D[i-1, i] = -8 / (12*dt)
            D[i+1, i] =  8 / (12*dt)
            D[i+2, i] = -1 / (12*dt)
        end

        # Backward difference for last two points
        D[n-4, n-1] =  -1 / (12*dt)
        D[n-3, n-1] =   6 / (12*dt)
        D[n-2, n-1] = -18 / (12*dt)
        D[n-1, n-1] =  10 / (12*dt)
        D[n, n-1]   =   3 / (12*dt)

        D[n-4, n] =   3 / (12*dt)
        D[n-3, n] = -16 / (12*dt)
        D[n-2, n] =  36 / (12*dt)
        D[n-1, n] = -48 / (12*dt)
        D[n, n]   =  25 / (12*dt)

        idx = 1:n
    else
        error("Undefined choice of numerical integration. Choose only an " * 
              "accepted method from: FE (Forward Euler), BE (Backward Euler)," *
              "SI (Semi-implicit Euler), FE4 (4th order Forward Euler), " *
              "BE4 (4th order Backward Euler), CTD4 (4th order Central), " *
              "FBCT4 (4th order Forward Backward Central)")
    end
    return D, idx
end




"""
$(SIGNATURES)

Estimate the time derivative at the first or second of five consecutive snapshots with a
4th-order one-sided (forward) finite difference. Internal helper.

## Arguments
- `xs::Matrix`: snapshot matrix with at least 5 columns; only the first 5 are used.
- `dt::Float64`: time step.
- `is_init::Bool`: `true` for the derivative at the first snapshot (stencil
  ``(-25, 48, -36, 16, -3)/(12\\Delta t)``), `false` for the second snapshot (stencil
  ``(-3, -10, 18, -6, 1)/(12\\Delta t)``).

## Returns
- `dxdt::Vector`: derivative estimate.
"""
function fwd4(xs::Matrix, dt::Float64, is_init::Bool)
    if size(xs,2) < 5
        error("Need at least 5 time points for 4th order finite differences.")
    end
    if is_init
        # Initial point
        dxdt = (-25 * xs[:,1] + 48 * xs[:,2] - 36 * xs[:,3] + 16 * xs[:,4] - 3 * xs[:,5]) / (12*dt)
    else
        # Second point due to using the initial point to the left making it an
        # asymmetric stencil
        dxdt = (-3 * xs[:,1] - 10 * xs[:,2] + 18 * xs[:,3] - 6 * xs[:,4] + xs[:,5]) / (12*dt)
    end
    return dxdt
end

"""
$(SIGNATURES)

Same as the matrix method of `fwd4`, for snapshots given as a vector of vectors.
Internal helper.

## Arguments
- `xs::Array{<:Vector}`: at least 5 consecutive snapshots; only the first 5 are used.
- `dt::Float64`: time step.
- `is_init::Bool`: `true` for the derivative at the first snapshot, `false` for the second.

## Returns
- `dxdt::Vector`: derivative estimate.
"""
function fwd4(xs::Array{<:Vector}, dt::Float64, is_init::Bool)
    if length(xs) < 5
        error("Need at least 5 time points for 4th order finite differences.")
    end
    if is_init
        # Initial point
        dxdt = (-25 * xs[1] + 48 * xs[2] - 36 * xs[3] + 16 * xs[4] - 3 * xs[5]) / (12*dt)
    else
        # Second point due to using the initial point to the left making it an
        # asymmetric stencil
        dxdt = (-3 * xs[1] - 10 * xs[2] + 18 * xs[3] - 6 * xs[4] + xs[5]) / (12*dt)
    end
    return dxdt
end


"""
$(SIGNATURES)

Estimate the time derivative at the last or second-to-last of the final five snapshots
with a 4th-order one-sided (backward) finite difference. Internal helper.

## Arguments
- `xs::Matrix`: snapshot matrix with at least 5 columns; only the last 5 are used.
- `dt::Float64`: time step.
- `is_final::Bool`: `true` for the derivative at the last snapshot (stencil
  ``(3, -16, 36, -48, 25)/(12\\Delta t)``), `false` for the second-to-last snapshot
  (stencil ``(-1, 6, -18, 10, 3)/(12\\Delta t)``).

## Returns
- `dxdt::Vector`: derivative estimate.
"""
function bwd4(xs::Matrix, dt::Float64, is_final::Bool)
    if size(xs,2) < 5
        error("Need at least 5 time points for 4th order finite differences.")
    end
    if is_final
        # Final point
        dxdt = (3 * xs[:,end-4] - 16 * xs[:,end-3] + 36 * xs[:,end-2] - 48 * xs[:,end-1] + 25 * xs[:,end]) / (12*dt)
    else
        # Second to last point due to using the final point to the right making it an
        # asymmetric stencil
        dxdt = (-xs[:,end-4] + 6 * xs[:,end-3] - 18 * xs[:,end-2] + 10 * xs[:,end-1] + 3 * xs[:,end]) / (12*dt)
    end
    return dxdt
end

"""
$(SIGNATURES)

Same as the matrix method of `bwd4`, for snapshots given as a vector of vectors.
Internal helper.

## Arguments
- `xs::Array{<:Vector}`: at least 5 consecutive snapshots; only the last 5 are used.
- `dt::Float64`: time step.
- `is_final::Bool`: `true` for the derivative at the last snapshot, `false` for the second-to-last.

## Returns
- `dxdt::Vector`: derivative estimate.
"""
function bwd4(xs::Array{<:Vector}, dt::Float64, is_final::Bool)
    if length(xs) < 5
        error("Need at least 5 time points for 4th order finite differences.")
    end
    if is_final
        # Final point
        dxdt = (3 * xs[end-4] - 16 * xs[end-3] + 36 * xs[end-2] - 48 * xs[end-1] + 25 * xs[end]) / (12*dt)
    else
        # Second to last point due to using the final point to the right making it an
        # asymmetric stencil
        dxdt = (-xs[end-4] + 6 * xs[end-3] - 18 * xs[end-2] + 10 * xs[end-1] + 3 * xs[end]) / (12*dt)
    end
    return dxdt
end


"""
$(SIGNATURES)

Estimate the time derivative at the middle (third) of five consecutive snapshots with the
4th-order central difference ``(x_1 - 8x_2 + 8x_4 - x_5)/(12\\Delta t)``. Internal helper.

## Arguments
- `xs::Matrix`: snapshot matrix with at least 5 columns; only the first 5 are used.
- `dt::Float64`: time step.

## Returns
- `dxdt::Vector`: derivative estimate at the third snapshot.
"""
function ctd4(xs::Matrix, dt::Float64)
    if size(xs,2) < 5
        error("Need at least 5 time points for 4th order finite differences.")
    end
    dxdt = (xs[:,1] - 8 * xs[:,2] + 8 * xs[:,4] - xs[:,5]) / (12*dt)
    return dxdt
end

"""
$(SIGNATURES)

Same as the matrix method of `ctd4`, for snapshots given as a vector of vectors.
Internal helper.

## Arguments
- `xs::Array{<:Vector}`: at least 5 consecutive snapshots; only the first 5 are used.
- `dt::Float64`: time step.

## Returns
- `dxdt::Vector`: derivative estimate at the third snapshot.
"""
function ctd4(xs::Array{<:Vector}, dt::Float64)
    if length(xs) < 5
        error("Need at least 5 time points for 4th order finite differences.")
    end
    dxdt = (xs[1] - 8 * xs[2] + 8 * xs[4] - xs[5]) / (12*dt)
    return dxdt
end
