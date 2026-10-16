```@meta
CurrentModule = LiftAndLearn
```

# Utilities

## System operators
The struct `Operators` provides a convenient way to organize and store the full and reduced operators.

```@docs
Operators
```

## Math Operations
This package heavily relies on the following math operations
- Kronecker Products ``\otimes``
- Vectorization ``\text{vec}(\cdot)``

For Kronecker products we are very thankful to the [Kronecker.jl](https://github.com/MichielStock/Kronecker.jl) package allowing fast Kronecker product arithmetic. We also, use what is called the **unique Kronecker product** which eliminates the redundant terms arising from the symmetry of Kronecker products. For details on the unique Kronecker product please refer to the package [UniqueKronecker.jl](https://github.com/smallpondtom/UniqueKronecker.jl).

## Error Analysis
Functions to quantify the accuracy of a reduced model and to check the energy-preserving
structure of quadratic operators.

```@docs
proj_error
rel_state_error
rel_output_error
compute_all_errors
choose_ro
ep_constraint_residual
ep_constraint_violation
isenergypreserving
```

## Matrix Orientation
Data matrices may be given as "fat" (state × time) or "tall" (time × state). These helpers
normalize the orientation.

```@docs
checksize
tall2fat
fat2tall
```

## Orthogonalization and QR
Low-level routines used by the incremental SVD of [One-pass Streaming OpInf](streaming/OnePass.md)
and by the QR-based recursive least-squares solvers.

```@docs
qrf!
givens_rotation
qr_givens!
reorthogonalize!
```

## Parametric Interpolation of Operators
For parametric problems, reduced operators learned at a few parameter values ``z_i`` can be
interpolated entry-wise with splines to obtain an operator at a new parameter value ``\hat z``.

```@docs
interpolate_matrix_elements
cubic_spline_coefficients
cubic_spline_interpolate
quartic_spline_coefficients
quartic_spline_interpolate
quintic_spline_coefficients
quintic_spline_interpolate
```

## Partial Differential Equation Models

Please also refer to the package [PolynomialModelReductionDataset.jl](https://github.com/smallpondtom/PolynomialModelReductionDataset.jl) providing a suite of polynomial-based systems arising from PDEs. We extensively use these models in our examples.
