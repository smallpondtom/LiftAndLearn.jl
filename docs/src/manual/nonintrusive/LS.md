# Standard Operator Inference (OpInf)

## Theory 
If we consider a __linear-quadratic system__, the goal of Operator Inference is to non-intrusively obtain a reduced model of the form

```math
    \dot{\hat{\mathbf x}}(t) = \hat{\mathbf A}\hat{\mathbf x}(t) + \hat{\mathbf{H}}(\hat{\mathbf{x}}(t) \otimes \hat{\mathbf{x}}(t)),
```

To do so, we will fit reduced operators ``\hat{\mathbf{A}}`` and ``\hat{\mathbf{H}}`` to the reduced data in a least-squares sense. In addition to the state trajectory data 

```math
    \mathbf{X} = \begin{bmatrix} 
        | & | &  & | \\
        \boldsymbol{\mathbf x}(t_1) & \mathbf{x}(t_2) & \cdots & \mathbf{x}(t_K) \\
        | & | &  & | \\
    \end{bmatrix} \in \mathbb{R}^{n\times K} ~,
```

we also require paired state time derivative data: 

```math
\begin{aligned}
    \dot{\mathbf{X}} = \begin{bmatrix}
        | & | & & | \\
        \dot{\mathbf{x}}(t_1) & \dot{\mathbf{x}}(t_2) & \cdots & \dot{\mathbf{x}}(t_K) \\
        | & | & & | 
    \end{bmatrix}\in\mathbb{R}^{n\times K} \quad \text{ where ~~} 
    \dot{\mathbf x}(t_i)=\mathbf A\mathbf x(t_i)+ \mathbf H (\mathbf x(t_i) \otimes \mathbf x(t_i))~.
\end{aligned}
```

This time derivative data can come directly from the simulation of the high-dimensional ODE or can be approximated numerically from the snapshot data. We use the POD basis ``\mathbf{V}_r`` to compute reduced state and time derivative data as follows: let ``\hat{\mathbf{x}}_i = \mathbf{V}_r^\top\mathbf{x}(t_i)`` as an ansatz, and ``\dot{\hat{\mathbf{x}}}_i = \mathbf{V}_r^\top\dot{\mathbf{x}}(t_i)`` for ``i=1,\ldots,K``. Then, define

```math
    \hat{\textbf X} = \begin{bmatrix}
        | & | &  & | \\
        \boldsymbol{\hat{\mathbf x}}_1 & \boldsymbol{ \hat{\mathbf x}}_2 & \cdots & \boldsymbol{\hat{\mathbf x}}_K \\
        | & | &  & | \\
    \end{bmatrix} \in \mathbb R^{r\times K}, \qquad \text{and} \qquad 
    \dot{\hat{\textbf X}} = \begin{bmatrix}
        | & | &  & | \\
        \dot{\hat{\mathbf x}}_1 & \dot{\hat{\mathbf x}}_2 & \cdots & \dot{\hat{\mathbf x}}_K \\
        | & | &  & | \\
    \end{bmatrix} \in \mathbb R^{r\times K}.
```

Additionally, we define the matrix formed by the quadratic terms of the state data

```math
    \hat{\textbf X}_\otimes = \begin{bmatrix}
        | & | &  & | \\
        (\hat{\mathbf x}_1 \otimes \hat{\mathbf x}_1) & (\hat{\mathbf x}_2 \otimes \hat{\mathbf x}_2) & \cdots & (\hat{\mathbf x}_K \otimes \hat{\mathbf x}_K) \\
        | & | &  & | \\
    \end{bmatrix} \in \mathbb R^{r^2\times K}.
```

This allows us to formulate the following minimization for finding the reduced operators ``\hat{\mathbf{A}}`` and ``\hat{\mathbf{H}}``:   

```math
    \textbf{Standard OpInf}: \qquad 
    \min_{\mathbf{\hat{A}}\in\mathbb R^{r\times r},~\hat{\mathbf{H}}\in\mathbb R^{r\times r^2}} \sum_{i=1}^K \left\| \dot{\hat{\mathbf x}}_i - \mathbf{\hat{A}}\hat{\mathbf{x}}_i - \hat{\mathbf H}(\hat{\mathbf x}_i \otimes \hat{\mathbf x}_i) \right \|^2_2 = \min_{\mathbf{O}\in\mathbb R^{r\times(r+r^2)}} \|\mathbf{D}\mathbf{O}^\top - \dot{\hat{\mathbf X}}\|_F^2,
```

where ``\mathbf D = [\hat{\mathbf X}^\top, ~\hat{\mathbf X}_\otimes^\top] \in \mathbb R^{K\times(r+r^2)}`` and ``\mathbf O = [\mathbf{\hat A}, ~\hat{\mathbf H}] \in \mathbb R^{r\times(r+r^2)}``.


## Implementation
In this package we implement the standard OpInf along with Tikhonov regularized version with the function `opinf`.

There are many things going on under the hood when the function: 
- constructing the data matrix
- construting the Tikhonov matrix
- __reprojection__

But all of those operations are taken care of automatically. For full details please see the [source code](https://github.com/smallpondtom/LiftAndLearn.jl/blob/main/src/learn.jl).


```@docs
opinf
```


### Optimization Implementation

There is a function that solves the least squares problem using `Ipopt` as well.

```@docs
NC_Optimize
NC_Optimize_output
```