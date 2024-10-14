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

## Partial Differential Equation Models

Please also refer to the package [PolynomialModelReductionDataset.jl](https://github.com/smallpondtom/PolynomialModelReductionDataset.jl) providing a suite of polynomial-based systems arising from PDEs. We extensively use these models in our examples.
