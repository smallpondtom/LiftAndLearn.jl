"""
Gardner equation iOpInf example
"""

#################
## Load Packages
#################
using CairoMakie
using IncrementalPOD
using LinearAlgebra
using LiftAndLearn
const LnL = LiftAndLearn


###################
## Global Settings
###################
TOL = 1e-10


###############################
## Include functions and files
###############################
include("utilities/plot_theme.jl")
include("utilities/analysis.jl")
include("utilities/plotting.jl")


##########################
## Burgers equation setup
##########################
Ω = (0.0, 1.0)
Nx = 2^7; dt = 1e-4
gardner = LnL.GardnerModel(
    spatial_domain=Ω, time_domain=(0.0, 1.0), Δx=(Ω[2] + 1/Nx)/Nx, Δt=dt,
    params=Dict(:a => 1, :b => 2, :c => 3), BC=:periodic,
)

options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        is_lin=true,
        is_quad=true,
        is_cubic=true,
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=dt,
        deriv_type="SI"
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)
rmax = 15
num_of_ic = 1

#################
## Generate Data
#################
parameters = gardner.params
A, F, E = gardner.finite_diff_model(gardner, parameters)
##
C = ones(1, gardner.spatial_dim) / gardner.spatial_dim
op_gardner = LnL.Operators(A=A, F=F, E=E)

# Reference solution
Xref = gardner.integrate_model(A, F, E, gardner.tspan, gardner.IC)
Yref = C * Xref

Xall = Vector{Matrix{Float64}}(undef, num_of_ic)
Xdotall = Vector{Matrix{Float64}}(undef, num_of_ic)
Xstore = nothing  # store one data trajectory for plotting
for j in 1:num_of_ic
    @info "Generating data for input $j"
    IC = sin.(2π * gardner.xspan) + cos.(4π * gardner.xspan) + 0.5 * randn(gardner.spatial_dim)
    states = gardner.integrate_model(A, B, F, Urand[:, j], gardner.tspan, IC)
    Xall[j] = states[:, 2:end]
    Xdotall[j] = (states[:, 2:end] - states[:, 1:end-1]) / gardner.Δt
    if j == 1
        Xstore = states
    end
end
X = reduce(hcat, Xall)
Xdot = reduce(hcat, Xdotall)
Y = C * X



####################################
## Compute the SVD for the POD basis
####################################
r = rmax  # order of the reduced form
V, Σ, _ = svd(X)
Vr = V[:, 1:r]
Σr = Σ[1:r]


#########################
## Compute the iPOD basis
#########################
ipod = iPOD(x1=X[:,1], algo=:baker, kselect=r)
ipod.full_increment!(ipod, X[:,2:end], tol=1e-12, verbose=true)
iVr = ipod.Q[:,1:r]
iΣ = ipod.Σ
iΣr = sort(iΣ, rev=true)[1:r]

# Match the signs 
for i in 1:r
    if abs(Vr[1,i] + iVr[1,i]) < TOL
        iVr[:,i] .*= -1
    end
end

# Error
error = norm(Vr - iVr) / norm(Vr)


######################
## Plot Data to Check
######################
with_theme(theme_latexfonts()) do
    fig0 = Figure(fontsize=20, size=(1300,500), backgroundcolor="#FFFFFF")
    ax1 = Axis3(fig0[1, 1], xlabel="x", ylabel="t", zlabel="u(x,t)")
    surface!(ax1, burgers.xspan, burgers.tspan, Xstore)
    ax2 = Axis(fig0[1, 2], xlabel="x", ylabel="t")
    hm = heatmap!(ax2, burgers.xspan, burgers.tspan, Xstore)
    Colorbar(fig0[1, 3], hm)
    display(fig0)
end
