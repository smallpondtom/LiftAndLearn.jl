# Analyzing Chaos

## Lyapunov Exponent

We implement two different algorithms to compute the Lyapunov exponent. The first implementation `lyapunovExponent` requires numerical integration of the model with unperturbed initial conditions and perturbed initial conditions to compute the Lyapunov exponent. The second implementation `lyapunovExponentJacobian` leverages the tangent map or Jacobian. 

```@docs
LiftAndLearn.ChaosGizmo.lyapunovExponent
LiftAndLearn.ChaosGizmo.lyapunovExponentJacobian
```

## Kaplan-Yorke dimension

The Kaplan-Yorke dimension represents the effective dimension of the chaotic attractor. For more details see this [explanation](https://www.wikiwand.com/en/Kaplan%E2%80%93Yorke_conjecture). In ChaosGizmo, the Kaplan-Yorke dimension is computed for given Lyapunov exponents.

```@docs
LiftAndLearn.ChaosGizmo.kaplanYorkeDim
```

## Implementation Details

### Integration step of the tangent map

To integrate the tangent map/Jacobian further in time we employ multiple integration schemes ranging from Euler method, 4th order Runge-Kutta, and strong stability preserving Runge-Kutta (SSPRK3).

```@docs
LiftAndLearn.ChaosGizmo.EULER
LiftAndLearn.ChaosGizmo.RK2
LiftAndLearn.ChaosGizmo.RK4
LiftAndLearn.ChaosGizmo.RALSTON4
LiftAndLearn.ChaosGizmo.SSPRK3
```

### Adjusting the QR decomposition

In every step of computing the Lyapunov exponent, it is necessary to reorthogonalize the tangent map. And, in this process we must adjust the signs in the `Q` and `R` matrices using the following routine:

```@docs
LiftAndLearn.ChaosGizmo.posDiag_QR!
```

### Options for Lyapunov exponent computation

To compute the Lyapunov exponents, the user must define the `LE_options` with necessary parameters. It is crucial to tune the parameters based on your system at hand.

```@docs
LiftAndLearn.ChaosGizmo.LE_options
```

## Example

We use a 9-dimensional Lorenz system to analyze the Lyapunov exponents and Kaplan-Yorke dimension of a reduced model. This model is introduced in [Reiterer et al.](https://iopscience.iop.org/article/10.1088/0305-4470/31/34/015). We reduce this system to an order of 7 from 9 for the analysis. Here are some cool visualizations of the model's dynamics.


```@raw html
<table border="0">
    <tr>
        <td>
            <figure>
                <img src='../../../images/ChaosGizmo/9dim_lorenz_frame1.png' alt='missing'><br>
                <figcaption><em></em></figcaption>
            </figure>
        </td>
        <td>
            <figure>
                <img src='../../../images/ChaosGizmo/9dim_lorenz_frame2.png' alt='missing'><br>
                <figcaption><em></em></figcaption>
            </figure>
        </td> 
    </tr>
    <tr>
        <td>
            <figure>
                <img src='../../../images/ChaosGizmo/9dim_lorenz_3d.png' alt='missing'><br>
                <figcaption><em></em></figcaption>
            </figure>
        </td>
        <td>
            <figure>
                <img src='../../../images/ChaosGizmo/9dim_lorenz_frame3.png' alt='missing'><br>
                <figcaption><em></em></figcaption>
            </figure>
        </td> 
    </tr>
</table>
```

```@example lorenz9
using LinearAlgebra: svd, I
using LiftAndLearn
const LnL = LiftAndLearn

# Define the 9-dimensional Lorenz system
a = 0.5  # wave number in the horizontal direction
b1 = 4*(1+a^2) / (1+2*a^2)
b2 = (1+2*a^2) / (2*(1+a^2))
b3 = 2*(1-a^2) / (1+a^2)
b4 = a^2 / (1+a^2)
b5 = 8*a^2 / (1+2*a^2)
b6 = 4 / (1+2*a^2)

sigma = 0.5  # Prandtl number
r = 15.10  # reduced Rayleigh number

n = 9  # dimension of the system
# Linear operator
A = zeros(n,n)
A[1,1] = -sigma * b1
A[1,7] = -sigma * b2
A[2,2] = -sigma
A[2,9] = -sigma / 2
A[3,3] = -sigma * b1
A[3,8] = sigma * b2
A[4,4] = -sigma
A[4,9] = sigma / 2
A[5,5] = -sigma * b5
A[6,6] = -b6
A[7,1] = -r
A[7,7] = -b1
A[8,3] = r
A[8,8] = -b1
A[9,2] = -r
A[9,4] = r
A[9,9] = -1

# Quadratic operator
indices = [
    (2,4,1), (4,4,1), (3,5,1),
    (1,4,2), (2,5,2), (4,5,2),
    (2,4,3), (2,2,3), (1,5,3),
    (2,3,4), (2,5,4), (4,5,4),
    (2,2,5), (4,4,5),
    (2,9,6), (4,9,6),
    (5,8,7), (4,9,7),
    (5,7,8), (2,9,8),
    (2,6,9), (4,6,9), (4,7,9), (2,8,9)
]
values = [
    -1, b4, b3,
    1, -1, 1,
    1, -b4, -b3,
    -1, -1, 1,
    0.5, -0.5,
    1, -1,
    2, -1,
    -2, 1,
    -2, 2, 1, -1
]
H = LnL.makeQuadOp(n, indices, values; which_quad_term="H")
F = LnL.makeQuadOp(n, indices, values; which_quad_term="F")
lorenz9_ops = LnL.operators(A=A, H=H, F=F)


# Define some helper functions
function lorenz_jacobian(ops::LnL.operators, x::AbstractArray)
    n = size(x,1)
    return ops.A + ops.H * kron(I(n),x) + ops.H*kron(x,I(n))
end

function lorenz_integrator(ops::LnL.operators, tspan::AbstractArray, IC::Array; params...)
    K = length(tspan)
    N = size(IC,1)
    f = let A = ops.A, H = ops.H, F = ops.F
        if ops.H == 0
            (x, t) -> A*x + F*LnL.vech(x*x')
        else
            (x, t) -> A*x + H*kron(x,x)
        end
    end
    xk = zeros(N,K)
    xk[:,1] = IC

    for k in 2:K
        timestep = tspan[k] - tspan[k-1]
        k1 = f(xk[:,k-1], tspan[k-1])
        k2 = f(xk[:,k-1] + 0.5 * timestep * k1, tspan[k-1] + 0.5 * timestep)
        k3 = f(xk[:,k-1] + 0.5 * timestep * k2, tspan[k-1] + 0.5 * timestep)
        k4 = f(xk[:,k-1] + timestep * k3, tspan[k-1] + timestep)
        xk[:,k] = xk[:,k-1] + (timestep / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    end
    return xk
end

# Then construct the intrusive POD-Galerkin reduced model
# Lyapunov exponent of POD reduced model
data1 = lorenz_integrator(lorenz9_ops, 0:1e-2:1e3, 2*rand(9).-1)
data2 = lorenz_integrator(lorenz9_ops, 0:1e-2:1e3, 2*rand(9).-1)
data3 = lorenz_integrator(lorenz9_ops, 0:1e-2:1e3, 2*rand(9).-1)
data4 = lorenz_integrator(lorenz9_ops, 0:1e-2:1e3, 2*rand(9).-1)
data5 = lorenz_integrator(lorenz9_ops, 0:1e-2:1e3, 2*rand(9).-1)
data = hcat(data1, data2, data3, data4, data5)
rmax = 7
Vr = svd(data).U[:,1:rmax]   # choose rmax columns
rom_option = LnL.LSOpInfOption(
    system=LnL.SystemStructure(is_lin=true, is_quad=true),
)
oprom = LnL.pod(lorenz9_ops, Vr, rom_option)


# Now, we can compute the Lyapunov spectrum and the Kaplan-Yorke dimension using the method without the Tangent map
const CG = LnL.ChaosGizmo
x0 = [0.01, 0, 0.01, 0.0, 0.0, 0.0, 0, 0, 0.01] 
options = CG.LE_options(N=1e4, τ=1e2, τ0=0.0, Δt=1e-2, m=rmax, T=1e-2, verbose=false, history=true)
λr1, λr1_all = CG.lyapunovExponent(oprom, lorenz_integrator, Vr, x0, options)
dkyr1 = CG.kaplanYorkeDim(λr1; sorted=false)
println("Kaplan-Yorke dimension (without Jacobian): ", dkyr1)
```

```@example lorenz9
using Plots
p = plot()
for i in 1:rmax
    plot!(p, λr1_all[i,:], label="λ$i", lw=2)
end
plot!(p, xlabel="reorthonormalization steps", ylabel="Lyapunov exponent", legend=:right, fontfamily="Computer Modern")
p
```

```@example lorenz9
# You can also compute them using the method with the Tangent map.
λr2, λr2_all = CG.lyapunovExponentJacobian(oprom, lorenz_integrator, lorenz_jacobian, Vr' * x0, options)
dkyr2 = CG.kaplanYorkeDim(λr2; sorted=false)
println("Kaplan-Yorke dimension (with Jacobian): ", dkyr2)
```

```@example lorenz9
p = plot()
for i in 1:rmax
    plot!(p, λr2_all[i,:], label="λ$i", lw=2)
end
plot!(p, xlabel="reorthonormalization steps", ylabel="Lyapunov exponent", legend=:right, fontfamily="Computer Modern")
p
```