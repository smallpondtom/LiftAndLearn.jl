```@meta
CurrentModule = LiftAndLearn
```

# Two-Pass Streaming OpInf

In the two-pass workflow the reduced basis ``\mathbf{V}_r`` is obtained first (pass 1), e.g.,
with POD on a subset of the data or with an incremental SVD. In pass 2 the full-order data are
projected snapshot by snapshot, ``\hat{\mathbf{x}}_k = \mathbf{V}_r^\top\mathbf{x}_k``, and the
reduced data are streamed into a recursive least-squares solver that updates the operators.
Only reduced quantities (of size ``r`` and ``d``) are stored, never the full snapshot matrix.

## Algorithms

Three recursive solvers are available through the `algorithm` keyword of
[`TwoPassStreamingOpInf`](@ref):

| `algorithm` | Method | Updates | Notes |
|:--|:--|:--|:--|
| `:RLS` | standard (covariance-form) RLS | rank-1 or block (`rank=k`) | fastest; supports a forgetting factor and noise weighting `Q`; works without regularization if the first block has full column rank |
| `:QRRLS` | QR-decomposition RLS | rank-1 | propagates a square root of the correlation matrix; numerically robust |
| `:iQRRLS` | inverse QR-decomposition RLS | rank-1 | propagates a square root of the inverse correlation matrix; robust and avoids a triangular solve per step |

The QR-based variants require a positive regularization (`Γs > 0` or `options.with_reg = true`)
and accept `qr_method = :qr` (Householder QR) or `:givens` (structured Givens rotations, cheaper
for large ``d``). Setting `use_gpu=true` stores the solver state on a CUDA device.

## Typical usage

```julia
using LiftAndLearn
const LnL = LiftAndLearn

# Model structure: linear-quadratic state with linear input
options = LSOpInfOption(
    system=SystemStructure(state=[1, 2], control=1),
    with_reg=true,
    λ=TikhonovParameter(A=1e-8, A2=1e-6, B=1e-8),
)

# Pass 1: basis (here from batch POD on available data)
Vr = svd(X).U[:, 1:r]

# Pass 2: reduced data, streamed one snapshot at a time
Xhat    = Vr' * X                       # or project each snapshot as it arrives
Xhatdot = Vr' * Xdot                    # or use LnL.time_derivative_approx on reduced data
solver = TwoPassStreamingOpInf(options=options, n=r, m=1, algorithm=:iQRRLS)
LnL.stream_all!(solver, LnL.streamify(Xhat, 1), LnL.streamify(Xhatdot, 1);
                U=LnL.streamify(U, 1))

op = LnL.terminate_stream(solver)      # Operators with A, B, A2u, ...
```

Calling [`stream!`](@ref) inside your own loop is equivalent to `stream_all!` and lets you
monitor the operator estimate (`solver.cache.O`) during streaming. To also learn a linear output
map ``\mathbf{y} = \hat{\mathbf{C}}\hat{\mathbf{x}}``, pass `l` (output dimension) and `Γo` to
the constructor, which then returns a `(state_solver, output_solver)` pair; stream outputs with
[`stream_output!`](@ref)/[`stream_output_all!`](@ref) and finish with
`terminate_stream(state_solver, output_solver)`.

!!! tip "Choosing the regularization"
    `Γs` may be a scalar ``\gamma`` (meaning ``\gamma\mathbf{I}``) or a full ``d\times d``
    matrix. If `Γs` is left at zero and `options.with_reg = true`, a diagonal Tikhonov matrix is
    built from `options.λ`, exactly as in batch OpInf, so the streamed solution matches the
    batch Tikhonov solution.

## API

```@docs
TwoPassStreamingOpInf
RLSOpInf
QRRLSOpInf
iQRRLSOpInf
stream!(::RLSOpInf, ::AbstractArray{T}, ::AbstractArray{T}) where T<:Number
stream!(::QRRLSOpInf, ::AbstractArray{T}, ::AbstractArray{T}) where T<:Number
stream!(::iQRRLSOpInf, ::AbstractArray{T}, ::AbstractArray{T}) where T<:Number
stream_all!
stream_output!
stream_output_all!
terminate_stream
```

### Solver internals

```@docs
RLSCache
QRRLSCache
iQRRLSCache
rls!
qrrls!
iqrrls!
```
