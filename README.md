# Lift & Learn/Operator Inference with Julia

<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="docs/src/assets/logo-dark.png">
        <source media="(prefers-color-scheme: light)" srcset="docs/src/assets/logo.png">
        <img alt="logo" src="docs/src/assets/logo.png" width="250" height="250">
    </picture>
</div>

<div align="center">

[![Powered by ACE Lab](https://img.shields.io/badge/powered%20by-ACE%20Lab-pink)](https://sites.google.com/view/elizabeth-qian/research/ace-group)
[![CI](https://github.com/smallpondtom/LiftAndLearn.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/smallpondtom/LiftAndLearn.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/smallpondtom/LiftAndLearn.jl/graph/badge.svg?token=4MJJ4716UA)](https://codecov.io/gh/smallpondtom/LiftAndLearn.jl)
[![Doc](https://img.shields.io/badge/docs-stable-blue.svg)](https://smallpondtom.github.io/LiftAndLearn.jl/stable)
[![Doc](https://img.shields.io/badge/docs-dev-green.svg)](https://smallpondtom.github.io/LiftAndLearn.jl/dev)
[![DOI](https://zenodo.org/badge/657587865.svg)](https://zenodo.org/doi/10.5281/zenodo.10826114)
</div>

LiftAndLearn.jl is an implementation of the Lift and Learn as well as the operator inference algorithm proposed in the papers listed in [References](#references). 

### Operator Inference (OpInf)
Operator Inference is a scientific machine-learning framework used in data-driven modeling of dynamical systems that aims to learn the governing equations or operators from observed data without explicit knowledge of the underlying physics or dynamics (but with some information such as the structure, e.g., linear, quadratic, bilinear, etc.). To know more about OpInf, please refer to these resources by  [ACE Lab](https://github.com/elizqian/operator-inference/tree/master) and [Willcox Research Group](https://kiwi.oden.utexas.edu/research/operator-inference).

### Lift and Learn (LnL)
Lift and Learn is a physics-informed method for learning low-dimensional models for large-scale dynamical systems. Lifting refers to the transformation of the original nonlinear system to a linear, quadratic, bilinear, or polynomial system by mapping the original state space to a new space with additional auxiliary variables. After lifting the system to a more approachable form we can learn a reduced model using the OpInf approach. 

### Requirements
- julia versions 1.8.5 >
- We use [Ipopt](https://github.com/jump-dev/Ipopt.jl) for the optimization (e.g., EP-OpInf)
    - This requires additional proprietary linear-solvers including `ma86` and `ma97`. 
    - You can run the code without it by changing the options. By default Ipopt will use `MUMPS` but we recommend you obtain and download `HSL_jll.jl`. You can find the instructions [here](https://licences.stfc.ac.uk/product/libhsl).
- As this package is designed for model reduction of large-scale models, we recommend using a high-memory machine to run the examples, ideally with more than 32GB of RAM.

### Installation
```julia-repl
(@v1.10) pkg> add LiftAndLearn
```

### Get Started
Clone this repository and try the 1D heat equation example out
$$\frac{\partial u}{\partial t} = \mu\frac{\partial^2 u}{\partial x^2}$$
from the command line using the following command
```
> julia --project="./scripts" scripts/OpInf/heat1d_OpInf_example.jl
```

### Similar Works
- [Python Operator Inference Code](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3?tab=readme-ov-file) by [Willcox Research Group](https://oden.utexas.edu/research/centers-and-groups/willcox-research-group/)
- Julia SciML [ModelReduction.jl](https://github.com/SciML/ModelOrderReduction.jl)

### References

1. Peherstorfer, B. and Willcox, K. 
[Data-driven operator inference for non-intrusive projection-based model reduction.](https://www.sciencedirect.com/science/article/pii/S0045782516301104)
Computer Methods in Applied Mechanics and Engineering, 306:196-215, 2016. ([Download](https://cims.nyu.edu/~pehersto/preprints/Non-intrusive-model-reduction-Peherstorfer-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{Peherstorfer16DataDriven,
    title   = {Data-driven operator inference for nonintrusive projection-based model reduction},
    author  = {Peherstorfer, B. and Willcox, K.},
    journal = {Computer Methods in Applied Mechanics and Engineering},
    volume  = {306},
    pages   = {196-215},
    year    = {2016},
}</pre></details>


2. Qian, E., Kramer, B., Marques, A., and Willcox, K. 
[Transform & Learn: A data-driven approach to nonlinear model reduction](https://arc.aiaa.org/doi/10.2514/6.2019-3707).
In the AIAA Aviation 2019 Forum, June 17-21, Dallas, TX. ([Download](https://www.dropbox.com/s/5znea6z1vntby3d/QKMW_aviation19.pdf?dl=0))<details><summary>BibTeX</summary><pre>
@inbook{QKMW2019aviation,
    author = {Qian, E. and Kramer, B. and Marques, A. N. and Willcox, K. E.},
    title = {Transform \&amp; Learn: A data-driven approach to nonlinear model reduction},
    booktitle = {AIAA Aviation 2019 Forum},
    doi = {10.2514/6.2019-3707},
    URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2019-3707},
    eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2019-3707}
}</pre></details>

3. Qian, E., Kramer, B., Peherstorfer, B., and Willcox, K. [Lift & Learn: Physics-informed machine learning for large-scale nonlinear dynamical systems](https://www.sciencedirect.com/science/article/pii/S0167278919307651), Physica D: Nonlinear Phenomena, 2020.<details><summary>BibTeX</summary><pre>
@article{qian2020lift,
    title={Lift \& {L}earn: {P}hysics-informed machine learning for large-scale nonlinear dynamical systems},
    author={Qian, E. and Kramer, B. and Peherstorfer, B. and Willcox, K.},
    journal={Physica D: Nonlinear Phenomena},
    volume={406},
    pages={132401},
    year={2020},
    publisher={Elsevier}
}</pre></details>

4. Qian, E., Farcas, I.-G., and Willcox, K. [Reduced operator inference for nonlinear partial differential equations](https://epubs.siam.org/doi/10.1137/21M1393972), SIAM Journal of Scientific Computing, 2022.<details><summary>BibTeX</summary><pre>
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
}</pre></details>


5. Koike, T., Qian, E. [Energy-Preserving Reduced Operator Inference for Efficient Design and Control](https://arc.aiaa.org/doi/10.2514/6.2024-1012), AIAA SCITECH 2024 Forum. 2024.<details><summary>BibTeX</summary><pre>
@inproceedings{koike2024energy,
  title={Energy-Preserving Reduced Operator Inference for Efficient Design and Control},
  author={Koike, Tomoki and Qian, Elizabeth},
  booktitle={AIAA SCITECH 2024 Forum},
  pages={1012},
  year={2024},
  doi={https://doi.org/10.2514/6.2024-1012}
}
</pre></details>


### Citing this Project
If you have used this code for your research, we would be grateful if you could cite us using the following BibTeX
```
@software{smallpondtom_LnL,
  author       = {Tomoki Koike},
  title        = {LiftAndLearn.jl: Julia Implementation of Lift and Learn and Operator Inference},
  month        = Dec,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.10826115},
  url          = {https://zenodo.org/doi/10.5281/zenodo.10826114}
}
```
