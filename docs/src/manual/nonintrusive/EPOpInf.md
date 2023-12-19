# Energy-Preserving Operator Inference

This is a modified version of Operator Inference. Namely, it solves a constrained optimization for the reduced operators.

## Energy Preserving Quadratic Nonlinearities in PDEs

We consider an $n$-dimensional ordinary differential equation (ODE) which is linear and quadratic in the state $\mathbf x$. Such an ODE often arises from spatially discretizing a PDE and is given by

```math
    \dot{\mathbf x}(t) = \mathbf A \mathbf x(t) + \mathbf H \left(\mathbf x(t) \otimes \mathbf x(t)\right) 
```

where ``\mathbf x(t) \in \mathbb{R}^n`` is the system state vector over ``t \in [0, T_{\text{final}}]``, and ``\otimes`` denotes the Kronecker product. The operators ``\mathbf A \in \mathbb{R}^{n\times n}`` and ``\mathbf H \in \mathbb{R}^{n\times n^2}`` are the linear and quadratic operators, respectively. In our setting, ``n`` is large, so simulating the system is computationally expensive.

The quadratic operator ``\mathbf H`` is called `energy-preserving' if

```math
    \langle \mathbf x, \mathbf H (\mathbf x \otimes \mathbf x)\rangle = \mathbf x^\top \mathbf H(\mathbf x \otimes \mathbf x) = 0, \qquad \text{ for all } \mathbf x \in \mathbb R^n.
```

This condition is derived by setting the quadratic term in the time derivative of the energy, ``\frac12\|\mathbf{x}\|^2``, to zero.


## Energy-Preserving Operator Inference (EP-OpInf)

To impose this energy-preserving structure on the operator, we propose EP-OpInf. For this method, we incorporate the constraint into the standard OpInf optimization and formulate a constrained minimization as follows:

```math
    \textbf{EP-OpInf}: \qquad 
    \min_{\mathbf{O}\in\mathbb R^{r\times(r+r^2)}} \|\mathbf{D}\mathbf{O}^\top - \dot{\hat{\mathbf{X}}}\|_F^2 \quad \text{ subject to } \quad \hat h_{ijk} + \hat h_{jik} + \hat h_{kij} = 0, \quad  1 \leq i,j,k \leq r ~ .
```

!!! note "Multiple Optimization Methods"
    For EP-OpInf, we implement multiple variations of the EP constraint.
    1. A hard equality constraint.
    2. A soft inequality constraint.
    3. A Penalty method where the constraint is indirectly applied to the problem.
    Note that the hard equality constraint is the enforces the constraint in the strictest manner and the penalty method is the weakest. The inequality constraint lies between the two. However, depending on the parameter settings the constraint violations may vary between the inequality and penalty methods.

```@docs
EPHEC_Optimize
EPSIC_Optimize
EPP_Optimize
```