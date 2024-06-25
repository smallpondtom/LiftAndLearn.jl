"""
    Streaming-OpInf example of the Allen-Cahn equation.
"""

#############
## Packages
#############
using CairoMakie
using LinearAlgebra
using ProgressMeter


###############
## My modules
###############
using LiftAndLearn
const LnL = LiftAndLearn


###################
## Global Settings
###################
SAVEFIG = true


###############################
## Include functions and files
###############################
include("utilities/plot_theme.jl")
include("utilities/analysis.jl")
include("utilities/plotting.jl")



#############################
## Fisher-KPP equation setup
#############################
Ω = (0.0, 1.0)
T = (0.0, 0.1)
Nx = 2^7
dt = 1e-4
allencahn = LnL.AllenCahnModel(
    spatial_domain=Ω, time_domain=T, Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt, 
    diffusion_coeffs=1.0, nonlin_coeffs=1.0, BC=:mixed
)
options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        is_lin=true,
        is_cubic=true,
        has_control=true,
        has_output=true,
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=dt,
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)
num_of_inputs = 10
rmax = 10
DS = 1 # downsampling factor


##########################
## Generate training data
##########################
A, B, E = allencahn.finite_diff_model(allencahn, allencahn.μ, allencahn.ϵ)
C = ones(1, allencahn.spatial_dim) / allencahn.spatial_dim
op_allencahn = LnL.Operators(A=A, B=B, C=C, E=E)
allencahn.IC = 1 .+ 5*sin.((2*allencahn.xspan.+1)*π).^2

# Reference solution
g = (t) -> 10*(sin.(π * t) .+ 1)
Uref = g(allencahn.tspan)  # boundary condition → control input
Uref = hcat(Uref, 10*ones(allencahn.time_dim))'
Xref = allencahn.integrate_model(A, B, E, Uref, allencahn.tspan, allencahn.IC; const_stepsize=true)
Yref = C * Xref

# Generate training data
Xall = Vector{Matrix{Float64}}(undef, num_of_inputs)
Xdotall = Vector{Matrix{Float64}}(undef, num_of_inputs)
Uall = Vector{Matrix{Float64}}(undef, num_of_inputs)
Xstore = nothing  # store one data trajectory for plotting
for j in 1:num_of_inputs
    @info "Generating data for input $j"
    Urand = hcat(10*rand(allencahn.time_dim), 10*ones(allencahn.time_dim))'  # boundary condition → control input
    Uall[j] = Urand[:, 2:DS:end]
    states = allencahn.integrate_model(A, B, E, Urand, allencahn.tspan, allencahn.IC; const_stepsize=true)
    Xall[j] = states[:, 2:DS:end]
    tmp = (states[:, 2:end] - states[:, 1:end-1]) / allencahn.Δt
    Xdotall[j] = tmp[:, 1:DS:end]
    if j == 1
        Xstore = states
    end
end
X = reduce(hcat, Xall)
Xdot = reduce(hcat, Xdotall)
U = reduce(hcat, Uall)
Y = C * X

# compute the POD basis from the training data
Vrmax = svd(X).U[:, 1:rmax]


######################
## Plot Data to Check
######################
with_theme(theme_latexfonts()) do
    fig0 = Figure(fontsize=20, size=(1300,500))
    ax1 = Axis3(fig0[1, 1], xlabel="x", ylabel="t", zlabel="u(x,t)")
    surface!(ax1, chafeeinfante.xspan, chafeeinfante.tspan, Xref)
    ax2 = Axis(fig0[1, 2], xlabel="x", ylabel="t")
    hm = heatmap!(ax2, chafeeinfante.xspan, chafeeinfante.tspan, Xref)
    Colorbar(fig0[1, 3], hm)
    display(fig0)
end

