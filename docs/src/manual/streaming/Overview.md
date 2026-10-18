```@meta
CurrentModule = LiftAndLearn
```

# Streaming Operator Inference

Standard Operator Inference (see [Standard OpInf](../nonintrusive/LS.md)) assembles the full
snapshot matrix, computes a POD basis from it, and then solves one least-squares problem with
all of the reduced data at once. For large-scale simulations (e.g., 3D turbulent flows) the
snapshot matrix may not fit in memory, or the data may arrive sequentially from a running
simulation or an experiment. **Streaming OpInf** learns the same reduced model while touching
each piece of data only once (or twice), with memory that does not grow with the number of
snapshots.

## Why streaming works

With reduced snapshots ``\hat{\mathbf{x}}_k = \mathbf{V}_r^\top\mathbf{x}_k``, the OpInf
problem with Tikhonov regularization reads

```math
\min_{\mathbf{O}} \sum_{k=1}^{K} \left\| \mathbf{O}\,\mathbf{d}_k - \dot{\hat{\mathbf{x}}}_k \right\|_2^2
+ \mathbf{o}^\top \boldsymbol{\Gamma}\, \mathbf{o},
\qquad
\mathbf{O}^\top = \left(\mathbf{D}^\top\mathbf{D} + \boldsymbol{\Gamma}\right)^{-1}\mathbf{D}^\top\dot{\hat{\mathbf{X}}}^\top,
```

where ``\mathbf{d}_k`` collects the polynomial terms of the ``k``-th snapshot (e.g.,
``[\hat{\mathbf{x}}_k;\ \mathbf{u}_k;\ \hat{\mathbf{x}}_k \oslash \hat{\mathbf{x}}_k]``) and
``\mathbf{D}`` stacks the rows ``\mathbf{d}_k^\top``. Both ``\mathbf{D}^\top\mathbf{D} =
\sum_k \mathbf{d}_k\mathbf{d}_k^\top`` and ``\mathbf{D}^\top\dot{\hat{\mathbf{X}}}^\top`` are sums
over snapshots, so the solution can be updated recursively as new data arrives. This is
exactly what **Recursive Least-Squares (RLS)** does: it maintains the inverse
``\mathbf{P}_k = (\mathbf{D}_k^\top\mathbf{D}_k + \boldsymbol{\Gamma})^{-1}`` and updates the
operator with the newest residual,

```math
\mathbf{K}_k = \frac{\mathbf{P}_{k-1}\mathbf{d}_k}{1 + \mathbf{d}_k^\top\mathbf{P}_{k-1}\mathbf{d}_k},
\qquad
\mathbf{O}_k^\top = \mathbf{O}_{k-1}^\top + \mathbf{K}_k\left(\dot{\hat{\mathbf{x}}}_k^\top - \mathbf{d}_k^\top\mathbf{O}_{k-1}^\top\right),
\qquad
\mathbf{P}_k = \mathbf{P}_{k-1} - \mathbf{K}_k\mathbf{d}_k^\top\mathbf{P}_{k-1},
```

initialized with ``\mathbf{P}_0 = \boldsymbol{\Gamma}^{-1}`` and ``\mathbf{O}_0 = \mathbf{0}``.
After the last snapshot, ``\mathbf{O}_K`` equals the batch Tikhonov-regularized OpInf solution.

The POD basis can likewise be computed in a streaming fashion with an **incremental SVD
(iSVD)** of the snapshot matrix ``\mathbf{X} \approx \mathbf{V}\boldsymbol{\Sigma}\mathbf{W}^\top``.

## Two workflows

LiftAndLearn.jl provides two Streaming OpInf workflows:

| | [Two-pass](TwoPass.md) | [One-pass](OnePass.md) |
|:--|:--|:--|
| Passes over full-order data | 2 (basis, then reduced data) | 1 |
| Basis | computed beforehand by any method (POD, iSVD, ...) | incremental SVD computed while streaming |
| Operator learning | RLS / QRRLS / iQRRLS on reduced data | one small least-squares solve at the end |
| Entry point | [`TwoPassStreamingOpInf`](@ref) | [`OnePassStreamingOpInf`](@ref) |
| Best when | the basis is available and the reduced data stream is long | the full-order data can be read only once |

Both workflows reuse the standard [`LSOpInfOption`](@ref) to define the model structure (via
[`SystemStructure`](@ref)) and the regularization (via `with_reg` and
[`TikhonovParameter`](@ref)), and both return an [`Operators`](@ref) object, so learned models
plug into the same simulation and analysis code as batch OpInf.

## Splitting data into streams

When the data are already in memory (e.g., to emulate streaming or to process reduced data in
chunks), [`streamify`](@ref) splits a matrix into a vector of consecutive blocks along its
snapshot (longer) dimension.

```@docs
streamify
```

## References

- B. Peherstorfer and K. Willcox, *Data-driven operator inference for nonintrusive
  projection-based model reduction*, CMAME, 2016.
- C. G. Baker, K. A. Gallivan, and P. Van Dooren, *Low-rank incremental methods for computing
  dominant singular subspaces*, Linear Algebra and its Applications, 2012.
- J. A. Tropp, A. Yurtsever, M. Udell, and V. Cevher, *Streaming low-rank matrix approximation
  with an application to scientific simulation*, SIAM J. Sci. Comput., 2019.
- A. H. Sayed, *Adaptive Filters*, Wiley, 2008 (RLS, QR-RLS, and inverse QR-RLS algorithms).
