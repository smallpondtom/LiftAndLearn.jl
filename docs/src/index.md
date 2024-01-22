# LiftAndLearn

**LiftAndLearn.jl** is an implementation of the Lift and Learn as well as the operator inference algorithm proposed in the papers listed in [Key References](@ref).

### Operator Inference (OpInf)
Operator Inference is a scientific machine-learning framework used in data-driven modeling of dynamical systems that aims to learn the governing equations or operators from observed data without explicit knowledge of the underlying physics or dynamics (but with some information such as the structure, e.g., linear, quadratic, bilinear, etc.). To know more about OpInf, please refer to these resources by [Willcox Research Group](https://kiwi.oden.utexas.edu/research/operator-inference) and [ACE Lab](https://github.com/elizqian/operator-inference/tree/master). __Or__ you can head over to the documentation page of this package about [OpInf](manual/nonintrusive/LS.md).

### Lift and Learn (LnL)
Lift and Learn is a physics-informed method for learning low-dimensional models for large-scale dynamical systems. Lifting refers to the transformation of the original nonlinear system to a linear, quadratic, bilinear, or polynomial system by mapping the original state space to a new space with additional auxiliary variables. After lifting the system to a more approachable form, we can learn a reduced model using the OpInf approach. For more info, head over to the documentation on [LnL](manual/nonintrusive/LnL.md).

## Requirements
- julia versions 1.8.5 >
- We use [Ipopt](https://github.com/jump-dev/Ipopt.jl) for the optimization (e.g., EP-OpInf)
    - This requires additional proprietary linear-solvers including `ma86` and `ma97`. 
    - You can run the code without it by changing the options. By default Ipopt will use `MUMPS` but we recommend you obtain and download `HSL_jll.jl`. You can find the instructions [here](https://licences.stfc.ac.uk/product/libhsl).

## Installation

To use LiftAndLearn, install Julia, then at the Julia REPL, type:

```julia
using Pkg
Pkg.add("LiftAndLearn")
using LiftAndLearn
```

## Features

Features included in this package are the following:

- Intrusive model reduction using Proper Orthogonal Decomposition (POD)
- Non-intrusive model reduction using the standard Operator Inference
- Non-intrusive model reduction for non-polynomial systems using Lift And Learn
- Physics-informed Operator Inference approaches
    - Energy-preserving
    - More to come in the future ...

!!! note
    We are actively working to incorporate new features into this package.

## Examples 
If you wish to give this package a try see our [Jupyter Notebook examples](https://github.com/smallpondtom/LiftAndLearn.jl/tree/main/examples), where you will find a variety of examples:
- 1-dimensional heat equation
- Viscous Burgers' equation
- Fitzhugh-Nagumo equation
- Kuramoto-Sivashinksy equation (chaotic system)

If you prefer running scripts rather then notebooks, then see the [example scripts](https://github.com/smallpondtom/LiftAndLearn.jl/tree/main/scripts).

## Contributing

If you find any bugs or issues please follow the instructions:

- Open an issue with clear explanation of bug. Recommended to have minimal reproduction example.
- If you have patched the bug on your own, then create a pull request.
- For further inquiries please contact [tkoike3@gatech.edu](mailto:tkoike3@gatech.edu).

## License

The source code is distributed under [MIT License](https://github.com/smallpondtom/LiftAndLearn.jl/blob/main/LICENSE).

## Key References

(1) Peherstorfer, B. and Willcox, K. 
[Data-driven operator inference for non-intrusive projection-based model reduction.](https://www.sciencedirect.com/science/article/pii/S0045782516301104)
Computer Methods in Applied Mechanics and Engineering, 306:196-215, 2016. ([Download](https://cims.nyu.edu/~pehersto/preprints/Non-intrusive-model-reduction-Peherstorfer-Willcox.pdf))
```
@article{Peherstorfer16DataDriven,
    title   = {Data-driven operator inference for nonintrusive projection-based model reduction},
    author  = {Peherstorfer, B. and Willcox, K.},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume  = {306},
    pages   = {196-215},
    year    = {2016},
}
```

(2) Qian, E., Kramer, B., Marques, A., and Willcox, K. 
[Transform & Learn: A data-driven approach to nonlinear model reduction](https://arc.aiaa.org/doi/10.2514/6.2019-3707).
In the AIAA Aviation 2019 Forum, June 17-21, Dallas, TX. ([Download](https://www.dropbox.com/s/5znea6z1vntby3d/QKMW_aviation19.pdf?dl=0))
```
@inbook{QKMW2019aviation,
    author = {Qian, E. and Kramer, B. and Marques, A. N. and Willcox, K. E.},
    title = {Transform \&amp; Learn: A data-driven approach to nonlinear model reduction},
    booktitle = {AIAA Aviation 2019 Forum},
    doi = {10.2514/6.2019-3707},
    URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2019-3707},
    eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2019-3707}
}
```

(3) Qian, E., Kramer, B., Peherstorfer, B., and Willcox, K. [Lift & Learn: Physics-informed machine learning for large-scale nonlinear dynamical systems](https://www.sciencedirect.com/science/article/pii/S0167278919307651), Physica D: Nonlinear Phenomena, 2020.
```
@article{qian2020lift,
    title={Lift \& {L}earn: {P}hysics-informed machine learning for large-scale nonlinear dynamical systems},
    author={Qian, E. and Kramer, B. and Peherstorfer, B. and Willcox, K.},
    journal={Physica D: Nonlinear Phenomena},
    volume={406},
    pages={132401},
    year={2020},
    publisher={Elsevier}
}
```

(4) Qian, E., Farcas, I.-G., and Willcox, K. [Reduced operator inference for nonlinear partial differential equations](https://epubs.siam.org/doi/10.1137/21M1393972), SIAM Journal of Scientific Computing, 2022.
```
@article{doi:10.1137/21M1393972,
    author = {Qian, Elizabeth and Farca\c{s}, Ionu\c{t}-Gabriel and Willcox, Karen},
    title = {Reduced Operator Inference for Nonlinear Partial Differential Equations},
    journal = {SIAM Journal on Scientific Computing},
    volume = {44},
    number = {4},
    pages = {A1934-A1959},
    year = {2022},
    doi = {10.1137/21M1393972},
    URL = {https://doi.org/10.1137/21M1393972},
    eprint = {https://doi.org/10.1137/21M1393972},
}
```