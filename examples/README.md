# Examples

Jupyter notebooks demonstrating LiftAndLearn.jl. They are up to date with LiftAndLearn v2 and
run from a clean checkout.

| Notebook | Topic | Runtime* |
|:--|:--|:--|
| `01_heat1D_OpInf_LS.ipynb` | 1D heat equation: time integrators, intrusive POD vs. Operator Inference (with and without re-projection) | ~1 min |
| `02_burgers_OpInf_LS.ipynb` | Viscous Burgers' equation with boundary control: intrusive vs. OpInf vs. OpInf with re-projection | ~3 min |
| `03_fhn_LnL_LS.ipynb` | FitzHugh–Nagumo: Lift & Learn vs. intrusive POD on the lifted system | ~4 min |
| `04_ep-burgers_EP-OpInf.ipynb` | Energy-preserving OpInf (EPHEC / EPSIC / EPP) on periodic viscous Burgers | ~5 min |
| `05_KSE_model.ipynb` | Kuramoto–Sivashinsky equation: finite-difference, pseudo-spectral, and spectral-Galerkin models | ~1 min |
| `06_KSE_EP-OpInf.ipynb` | Energy-preserving OpInf on the chaotic KSE: training, errors, autocorrelation, Lyapunov exponents | ~9 min |
| `07_KSE_EP-OpInf_test.ipynb` | Testing the KSE models from `06` on new initial conditions (run `06` first) | ~5 min |
| `08_burgers_Streaming-OpInf.ipynb` | Two-pass and one-pass Streaming Operator Inference | ~1 min |

\*Single-threaded, after precompilation. Notebooks `04`, `06`, and `07` default to a demo
scale; set `PAPER_SCALE = true` near the top to reproduce the full-size experiments from the
papers (hours).

## Setup

The notebooks use the environment in this folder, which points at the local LiftAndLearn.jl
checkout (`[sources]`, requires Julia ≥ 1.11). From the repository root:

```julia
julia> using Pkg
julia> Pkg.activate("examples")
julia> Pkg.instantiate()
julia> using IJulia; installkernel("Julia")   # once, to register a Jupyter kernel for this Julia
```

Then open a notebook in Jupyter or VS Code and select the Julia kernel.

The PDE models come from
[PolynomialModelReductionDataset.jl](https://github.com/smallpondtom/PolynomialModelReductionDataset.jl).
The energy-preserving notebooks use Ipopt; they automatically use the HSL `ma97` linear solver if a
licensed HSL library is installed (see the [Ipopt HSL notes](https://licences.stfc.ac.uk/product/libhsl)),
and Ipopt's default MUMPS solver otherwise. Generated data and figures are written to `data/` and
`figures/` in this folder.

For script versions of these experiments see [`scripts`](../scripts).
