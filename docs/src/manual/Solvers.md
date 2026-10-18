```@meta
CurrentModule = LiftAndLearn
```

# Least-Squares Solvers and Data Preparation

This page documents the building blocks used by [`opinf`](@ref) and the streaming methods:
time-derivative approximation, data-matrix assembly, and the least-squares and Tikhonov
solvers. You rarely need to call them directly, but they are useful for custom workflows (e.g.,
Streaming OpInf, regularization studies, or very large regression problems).

## Time derivatives

If time-derivative data are not available from the simulator, they are approximated from the
snapshots with finite differences. The scheme is selected by `DataStructure.deriv_type`:

| `deriv_type` | Scheme | Order | Snapshots used (`idx`) |
|:--|:--|:--|:--|
| `"FE"` | forward Euler | 1 | `1:K-1` |
| `"BE"` | backward Euler | 1 | `2:K` |
| `"SI"` | semi-implicit Euler (backward difference) | 1 | `2:K` |
| `"FE4"` | forward difference | 4 | `1:K-4` |
| `"BE4"` | backward difference | 4 | `5:K` |
| `"CTD4"` | central difference | 4 | `3:K-2` |
| `"FBCT4"` | forward/central/backward difference | 4 | `1:K` |

[`time_derivative_approx`](@ref) applies the scheme to a snapshot matrix, whereas
[`finite_diff_matrix`](@ref) returns the equivalent sparse matrix ``\mathbf{E}`` such that
``\dot{\mathbf{X}} \approx \mathbf{X}\mathbf{E}``. The matrix form is what
[One-pass Streaming OpInf](streaming/OnePass.md) uses, because it can be applied to the
compressed snapshot factors.

```@docs
time_derivative_approx
finite_diff_matrix
```

## Data matrix and operator unpacking

```@docs
get_data_matrix
unpack_operators!
leastsquares_solve
reproject
```

## Least-squares solvers

Unregularized OpInf problems ``\min_{\mathbf{O}}\|\mathbf{D}\mathbf{O}^\top - \mathbf{R}\|_F``
are solved by [`standard_least_squares`](@ref). Depending on the options in
[`LSOpInfOption`](@ref) it uses backslash (QR), the normal equations, or a memory-aware batched
QR/LinearSolve.jl solve, optionally on the GPU.

```@docs
standard_least_squares
LeastSquaresSolver
estimate_memory_usage
```

## Tikhonov-regularized solvers

With `with_reg = true`, OpInf solves

```math
\min_{\mathbf{O}} \|\mathbf{D}\mathbf{O}^\top - \mathbf{R}\|_F^2 + \operatorname{tr}\!\left(\mathbf{O}\,\boldsymbol{\Gamma}\,\mathbf{O}^\top\right),
\qquad
\mathbf{O}^\top = (\mathbf{D}^\top\mathbf{D} + \boldsymbol{\Gamma})^{-1}\mathbf{D}^\top\mathbf{R},
```

where the diagonal Tikhonov matrix ``\boldsymbol{\Gamma}`` is assembled by
[`tikhonov_matrix!`](@ref) from the per-operator weights in [`TikhonovParameter`](@ref). The
solver [`tikhonov`](@ref) chooses between an augmented least-squares system, the normal
equations, an iterative method, or an SVD with truncation of small singular values.

```@docs
tikhonov
tikhonov_matrix!
TikhonovSolver
RegularizationMethod
AugmentedSystem
NormalForm
SVDTruncation
tikhonov_regularization_path
optimize_regularization_parameter
estimate_tikhonov_memory
```
