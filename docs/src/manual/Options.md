# Options for Model Learning

## General Setup Options

### System Structure
This package works with model reduction for polynomial systems with affine control taking a generel form of 

```math
\dot{\mathbf{x}} = \mathbf{A}_1\mathbf{x} + \mathbf{A}_2(\mathbf{x}\otimes\mathbf{x}) + \cdots + \mathbf{A}_n(\underbrace{\mathbf{x}\otimes\cdots\otimes\mathbf{x}}_{n-\text{times}}) + \sum_{i=1}^m N_{1,i}\mathbf{x}\mathbf{u}_i + \cdots + \sum_{i=1}^m N_{k,i}(\underbrace{\mathbf{x}\otimes\cdots\otimes\mathbf{x}}_{k-\text{times}})\mathbf{u}_i + \mathbf{B}\mathbf{u} +\mathbf{K}~.
```

!!! note "Current Implementations"
    With the current version we have only implemented up to quartic states and bilinear controls. We plan to implement linear-quadratic controls and other-higher order systems in the future, as these types of systems are observed in many applications. 

For such structures, use `SystemStructure` struct to define the system.

```@docs
SystemStructure
```

### Variable info
This structure allows you to input information about the variables in the system. For example, the Fitzhugh-Nagumo system has two main variables: a fast-recovery variable (usually denoted as v) and a slow-activating variable (usually denoted as w). The Fitzhugh-Nagumo equation is given by:
```math
\begin{align*}
    \frac{dv}{dt} &= v - \frac{v^3}{3} - w + I \\
    \frac{dw}{dt} &= \epsilon (v + a - bw)
\end{align*}
```
Additionally, if we lift this system, it becomes a lifted system of 3 variables. For the `VariableStructure` you can define the number of unlifted state variables `N=2` and the number of lifted state variables `Nl=3`.

```@docs
VariableStructure
```

### Data Info
With the `DataStructure` struct, the user will define the time-step ``\Delta t`` and optionally the down-sampling and type of numerical scheme used for the Partial Differential Equation.

```@docs
DataStructure
```

### Optimization Settings
This options is only required if you are dealing with the optimization based model reduction methods (e.g., Energy-Preserving Operator Inference) in which you must select some options for the optimization. 

```@docs
OptimizationSetting
```

#### Reprojection
Reprojection is a sampling scheme used to increase the accuracy of the recovered reduced model. For further details and mathematical proofs refer to [Uy2023active](@cite) and [Peherstorfer2020sampling](@cite).

#### Successive Initial Guess Estimation (SIGE)
The `SIGE` option is used to run the optimization successive from a lower reduced dimension and increasing the dimensions sequentially by using the solution of the lower dimensional operator as the initial guess for the next optimization. For example, you will solve the optimization for `r=2` and then use the solution to solve for the optimization of `r=3`.

```julia
# r = 2
options.optim.initial_guess = false  # turn off initial guess for the first iteration
op_tmp = LnL.opinf(Xdata, Vr[:,1:2], options; U=zeros(100), Y=zeros(100), Xdot=Xdotdata)  # compute the first operator
op[1] = op_tmp  # store the first operator

# r =3
options.optim.initial_guess = true  # turn on initial guess for the next step
op_tmp = LnL.opinf(Xdata, Vr[:,1:3], options; U=zeros(100), Y=zeros(100), Xdot=Xdotdata, IG=LnL.Operators(A=op_tmp.A, F=op_tmp.F)) # compute the second operator
op[2] = op_tmp
```

#### Linear Solvers
For the optimization we use [Ipopt](https://coin-or.github.io/Ipopt/), and for its linear solvers it is possible to use the default solvers like `MUMPS` and HSL solvers (e.g., `ma77`, `ma86`, and `ma97`).

!!! warning
    To use HSL linear solvers you will need to obtain a license from [here](https://licences.stfc.ac.uk/product/libhsl). Then set the path to the solvers using the option `HSL_lib_path`:
    ```julia
    import HSL_jll
    OptimizationSetting.HSL_lib_path = HSL_jll.libhsl_path
    ```

### Regularization
Using the `TikhonovParameter` struct you will define the Tikhonov regularization matrix. This will be a diagonal matrix with diagonal entries having different values corresponding to the operators that it is regulating (e.g., linear, quadratic, bilinear). 

The Tikhonov regulated optimization problem is defined as 
```math
\text{min}~\|\|\mathbf{D}\mathbf{O}^\top - \dot{\mathbf{\hat{X}}}\|\|^2_F + \|\|\mathbf{\Gamma}\mathbf{O}^\top\|\|^2_F
```
where ``\mathbf{D}``, ``\mathbf{O}^\top``, ``\dot{\hat{\mathbf{X}}}`` are the data matrix, operator matrix (with minimizers), and time derivative snapshot matrix, respectively. The linear least-squares solution of this becomes
```math
\mathbf{O}^\top = (\mathbf{D}^\top\mathbf{D} + \mathbf{\Gamma}^\top\mathbf{\Gamma})^\dag \mathbf{D}\dot{\mathbf{\hat{X}}}
```

```@docs
TikhonovParameter
```

## Model Reduction Specific Options

These options are all specific to each solution method of Operator Inference. All of the options below are a subtype of `AbstractOption`.

```@docs
LiftAndLearn.AbstractOption
```

### Standard Operator Inference
This option is required when using the standard Operator Inference method.

```@docs 
LSOpInfOption
```

!!! note
    The option `with_tol` turns on/off the settings to truncate ill-posed singular values with order of `pinv_tol`.

### Energy-Preserving Operator Inference Options
The three options below are for the energy-preserving Operator Inference approaches: hard equality constraint, soft inequality constraint, and penalty. For details of each parameter please check out the documentation of [EP-OpInf Manual](nonintrusive/EPOpInf.md)


```@docs
EPHECOpInfOption
EPSICOpInfOption
EPPOpInfOption
```