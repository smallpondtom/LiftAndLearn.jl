```@meta
CurrentModule = LiftAndLearn
```

# One-Pass Streaming OpInf

The one-pass workflow reads every full-order snapshot exactly once. Each snapshot updates a
low-rank SVD of the snapshot matrix,

```math
\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_K] \approx \mathbf{V}\boldsymbol{\Sigma}\mathbf{W}^\top,
```

and is then discarded. The left singular vectors ``\mathbf{V}`` are the POD basis, and the
reduced snapshots come for free from the right factor,

```math
\hat{\mathbf{X}}^\top = \mathbf{X}^\top\mathbf{V} \approx \mathbf{W}\boldsymbol{\Sigma},
```

so the OpInf data matrix ``\mathbf{D}`` can be formed without revisiting the full-order data.
Reduced time derivatives are obtained either

- by applying a finite-difference matrix ``\mathbf{E}`` (see [`finite_diff_matrix`](@ref)) to
  the reduced snapshots, ``\dot{\hat{\mathbf{X}}}^\top \approx \mathbf{E}^\top\mathbf{W}\boldsymbol{\Sigma}``
  (`finite_diff=true`, the default), or
- by streaming the derivative snapshots as well and compressing them with a second iSVD
  ``\dot{\mathbf{X}} \approx \mathbf{P}\mathbf{S}\mathbf{Q}^\top`` (`finite_diff=false`).

The final least-squares (optionally Tikhonov-regularized) problem is only of size
``K \times d`` and is solved by [`compute_stream_operators`](@ref).

## SVD methods

| `isvd_method` | Method | Notes |
|:--|:--|:--|
| `:baker` | deterministic incremental SVD (Baker, Gallivan & Van Dooren) | rank grows up to `rank` and is truncated thereafter; requires the first snapshot `x_init` |
| `:sketchy` | randomized streaming sketch (Tropp et al.) | requires the total number of snapshots `n_snapshots`; the SVD is recovered at the end with [`compute_svd_sketchy!`](@ref) |

## Typical usage

```julia
using LiftAndLearn
const LnL = LiftAndLearn

options = LSOpInfOption(
    system=SystemStructure(state=[1, 2], control=1),
    data=DataStructure(Δt=Δt, deriv_type="SI"),
    with_reg=true, λ=TikhonovParameter(A=1e-9, A2=1e-9, B=1e-9),
)

# Initialize with the first snapshot, then stream the rest
stream = OnePassStreamingOpInf(X[:, 1]; options=options, n_state=n, n_input=1, rank=rmax)
for k in 2:K
    LnL.stream!(stream, X[:, k])        # e.g., read from disk or a running solver
end

# Reduced operators of any dimension r ≤ rmax
E, idx = LnL.finite_diff_matrix(options.data.deriv_type, K, Δt)
op = LnL.compute_stream_operators(stream, E, idx; U=U, rank=r)
Vr = stream.V[:, 1:r]                   # matching POD basis
```

!!! note "Several trajectories"
    When the snapshots come from several trajectories streamed back to back, build one
    finite-difference matrix per trajectory and pass their block-diagonal concatenation
    (e.g., `BlockDiagonal(Es)` from BlockDiagonals.jl) together with the concatenated, shifted
    index vectors.

!!! warning "Truncation"
    The Baker iSVD with `rank < n` is a single-pass approximation: once the rank cap is reached,
    discarded directions cannot be recovered later. Choose `rank` somewhat larger than the
    reduced dimension ``r`` you intend to use.

## API

```@docs
OnePassStreamingOpInf
OnePassStreamingOpInf1
OnePassStreamingOpInf2
OnePassStreamingOpInfSketchy
stream!(::OnePassStreamingOpInf1, ::AbstractVector{T}) where {T<:Real}
stream!(::OnePassStreamingOpInf2, ::AbstractVector{T}, ::AbstractVector{T}) where {T<:Real}
stream!(::OnePassStreamingOpInfSketchy, ::AbstractVector{T}) where {T<:Real}
stream!(::OnePassStreamingOpInfSketchy, ::AbstractMatrix{T}) where {T<:Real}
compute_stream_operators
compute_svd_sketchy!
purge!
```
