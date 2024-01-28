# Utilities

## System operators
The struct `operators` provides a convenient way to organize and store the full and reduced operators.

```@docs
operators
```

## Linear Algebra
Many of the utility functions revolve around the properties of the following concepts
- Kronecker Products ``\otimes``
- Vectorization ``\text{vec}(\cdot)``
- Half-vectorization ``\text{vech}(\cdot)``

The utility functions are based on the following key references 

(1) Brewer, J. W.
[Kronecker Products and Matrix Calculus in System Theory](http://ieeexplore.ieee.org/document/1084534/)
IEEE Transactions on Circuits and Systems, 25(9) 772-781, 1978.
```
@article{Brewer1978,
  title = {Kronecker products and matrix calculus in system theory},
  volume = {25},
  ISSN = {0098-4094},
  url = {http://dx.doi.org/10.1109/TCS.1978.1084534},
  DOI = {10.1109/tcs.1978.1084534},
  number = {9},
  journal = {IEEE Transactions on Circuits and Systems},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  author = {Brewer,  J.},
  year = {1978},
  month = sep,
  pages = {772–781}
}
```

(2) Magnus, J. R. and Neudecker, H.
[The Elimination Matrix: Some Lemmas and Applications](https://epubs.siam.org/doi/10.1137/0601049).
SIAM Journal on Algebraic Discrete Methods, 1(4) 422-449, 1980-12.
```
@article{Magnus1980,
  title = {The Elimination Matrix: Some Lemmas and Applications},
  volume = {1},
  ISSN = {2168-345X},
  url = {http://dx.doi.org/10.1137/0601049},
  DOI = {10.1137/0601049},
  number = {4},
  journal = {SIAM Journal on Algebraic Discrete Methods},
  publisher = {Society for Industrial & Applied Mathematics (SIAM)},
  author = {Magnus,  Jan R. and Neudecker,  H.},
  year = {1980},
  month = dec,
  pages = {422–449}
}
```

## Utility APIs

```@docs
vech
⊘
invec
dupmat
elimat
commat
nommat
F2H
H2F
F2Hs
Q2H
H2Q
squareMatStates
kronMatStates
extractF
insert2F
insert2randF
extractH
insert2H
insert2bilin
```