# Intrusive Model Reduction

## POD

Proper orthogonal decomposition originated from the analysis of turbulent flows in aerodynamics, and it has become one of the most widespread projection-based model reduction methods. POD reduces the model by projecting it onto a reduced subspace defined to be the span of basis vectors that optimally represent a set of simulation or experimental data. See the original literatures on POD [lumley1967structure](@cite), [sirovich1987turbulence](@cite), [berkooz1993proper](@cite).

In POD, we begin by collecting snapshots of state trajectory time series data by simulating the original full model ODE with ``K`` timesteps. We define the state snapshot data matrix as follows: 
```math
    \mathbf{X} = \begin{bmatrix} 
        | & | &  & | \\
        \boldsymbol{\mathbf x}(t_1) & \mathbf{x}(t_2) & \cdots & \mathbf{x}(t_K) \\
        | & | &  & | \\
    \end{bmatrix} \in \mathbb{R}^{n\times K} ~.
```
More generally, the state snapshot matrix can contain state data from multiple simulations, e.g., from different initial conditions or using different parameters.
Let ``\mathbf{X} = \mathbf{V\Sigma W}^\top`` denote the singular value decomposition of the state snapshot. To reduce the dimension of the large-scale model, we denote by ``\mathbf{V}_r\in \mathbb R^{n\times r}`` the first ``r \ll n`` columns of ``\mathbf V``; this is called the *POD basis*. Then, we approximate the state ``\mathbf{x}`` in the subspace spanned by the POD basis, ``\mathbf x \approx \mathbf V_r \hat{\mathbf x}`` where ``\hat{\mathbf x}\in\mathbb{R}^r`` is called the _reduced state_. If we substitute this approximation into a linear-quadratic system and enforce the Galerkin orthogonality condition that the approximation residual be orthogonal to the span of ``\mathbf V_r``, we arrive at a POD-Galerkin reduced model of the form
```math
    \dot{\hat{\mathbf x}}(t) = \mathbf{\hat A}\hat{\mathbf x}(t) + \hat{\mathbf{H}}(\hat{\mathbf{x}}(t) \otimes \hat{\mathbf{x}}(t)),
```
where the reduced operators are ``\mathbf{\hat{A}} = \mathbf{V}^\top_r \mathbf{AV}_r \in \mathbb{R}^{r\times r}`` and  ``\hat{\mathbf{H}} = \mathbf{V}^\top_r \mathbf{H}(\mathbf{V}_r \otimes \mathbf{V}_r) \in \mathbb R^{r\times r^2}``.


## Implementation
The implementation of this corresponds to the following function:

```@docs
intrusiveMR
```

!!! note
    In the next release, we will probably replace the function name `intrusiveMR` with `pod` for precision.
