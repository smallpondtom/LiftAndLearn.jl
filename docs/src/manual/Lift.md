# Lifting

This page gives an explanation for the mathematical concept of lifting and provides example code for its implementation.

## Lift Map Structure

```@docs
lifting
```

!!! tip
    For a simple pendulum we have 
    ```math
    \begin{bmatrix}
    \dot{x}_1 \\
    \dot{x}_2 
    \end{bmatrix} = \begin{bmatrix}
    x_2 \\
    -\frac{g}{l} \sin(x_1)
    \end{bmatrix}
    ```
    The lifted system becomes 
    ```math
    \begin{bmatrix}
    \dot{x}_1 \\
    \dot{x}_2 \\
    \dot{x}_3 \\
    \dot{x}_4
    \end{bmatrix} = \begin{bmatrix}
    x_2 \\
    -\frac{g}{l} x_3 \\
    x_2 x_4 \\
    -x_2 x_3
    \end{bmatrix}
    ```
    when ``x_3 = \sin(x_1)`` and ``x_4 = \cos(x_1)``. Which if coded, would look like this:
    ```julia
    lifter = LnL.lifting(2, 4, [x -> sin.(x[1]), x -> cos.(x[1])])
    ```

## Construct Lifted POD Basis from Data

```@docs
liftedBasis
```

An example implementation would be:

```@example
using LiftAndLearn
using Random
LnL = LiftAndLearn
W = round.(rand(30,100), digits=4)
LnL.liftedBasis(W, 3, 10, [2,3,4])
```