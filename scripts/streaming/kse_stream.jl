"""
    Streaming-OpInf example of the Kuramoto-Sivashinsky equation.
"""

#############
## Packages
#############
using CairoMakie
using DelimitedFiles
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
CONST_STREAM = true
SAVEDATA = false
LOADDATA = SAVEDATA ? false : true
SAVEFIG = true


###############################
## Include functions and files
###############################
include("utilities/plot_theme.jl")
include("utilities/analysis.jl")
include("utilities/plotting.jl")


######################
## KSE equation setup
######################
Ω = (0.0, 60.0); dt = 1e-3; L = Ω[2] - Ω[1]; N = 2^9
kse = LnL.KuramotoSivashinskyModel(
    spatial_domain=Ω, time_domain=(0.0, 300.0), diffusion_coeffs=1.0,
    Δx=(Ω[2] - 1/N)/N, Δt=dt, conservation_type=:NC
)

options = LnL.LSOpInfOption(
    system=LnL.SystemStructure(
        is_lin=true,
        is_quad=true,
        has_output=true,
    ),
    vars=LnL.VariableStructure(
        N=1,
    ),
    data=LnL.DataStructure(
        Δt=dt,
        DS=100,
    ),
    optim=LnL.OptimizationSetting(
        verbose=true,
    ),
)
DS = 100  # downsample rate of data
# Parameterized function for the initial condition
u0 = (a,b) -> a * cos.((2*π*kse.xspan)/L) .+ b * cos.((4*π*kse.xspan)/L)  # initial condition


##########################
## Generate training data
##########################
A, F = kse.finite_diff_model(kse, kse.diffusion_coeffs)
C = ones(1, kse.spatial_dim) / kse.spatial_dim
op_kse = LnL.Operators(A=A, C=C, F=F)

# Initial condition parameters
a = [0.8, 1.0, 1.2]
b = [0.2, 0.4, 0.6]
num_ic_params = Int(length(a) * length(b))


if LOADDATA
    X = readdlm("scripts/streaming/data/kse/X.csv", ',')
    Y = readdlm("scripts/streaming/data/kse/Y.csv", ',')
    Xdot = readdlm("scripts/streaming/data/kse/Xdot.csv", ',')
    Xref = readdlm("scripts/streaming/data/kse/Xref.csv", ',')
    Yref = readdlm("scripts/streaming/data/kse/Yref.csv", ',')
    Vrmax = readdlm("scripts/streaming/data/kse/Vrmax.csv", ',')
    Σr = readdlm("scripts/streaming/data/kse/Σr.csv", ',')
else
    @info "Generate reference solution data..."
    # Reference solution
    a_ref = sum(a) / length(a)
    b_ref = sum(b) / length(b)
    Xref = kse.integrate_model(A, F, kse.tspan, u0(a_ref,b_ref))
    Yref = C * Xref
    @info "Done"

    @info "Generate training data..."
    # Store the training data 
    Xall = Vector{Matrix{Float64}}(undef, num_ic_params)
    Yall = Vector{Matrix{Float64}}(undef, num_ic_params)
    Xdotall = Vector{Matrix{Float64}}(undef, num_ic_params)

    # Generate the data for all combinations of the initial condition parameters
    ab_combos = collect(Iterators.product(a, b))
    prog = Progress(length(ab_combos))
    Threads.@threads for (i, ic) in collect(enumerate(ab_combos))
        ai, bi = ic
        states = kse.integrate_model(A, F, kse.tspan, u0(ai,bi))
        tmp = states[:, 2:end]
        Yall[i] = C * tmp[:, 1:DS:end]  # downsample data
        Xall[i] = tmp[:, 1:DS:end]  # downsample data
        tmp = (states[:, 2:end] - states[:, 1:end-1]) / kse.Δt
        Xdotall[i] = tmp[:, 1:DS:end]  # downsample data
        next!(prog)
    end
    # Combine all initial condition data to form on big training data matrix
    X = reduce(hcat, Xall) 
    Y = reduce(hcat, Yall)
    Xdot = reduce(hcat, Xdotall)
    @info "Done"

    # compute the POD basis from the training data
    rmax = 150
    Vrmax = svd(X).U[:, 1:rmax]
    Σr = svd(X).S[1:rmax]

    # Discard unnecessary variables
    Xall = nothing
    Yall = nothing
    Xdotall = nothing
    GC.gc()
end


#############
## Save data
#############
if SAVEDATA
    writedlm("scripts/streaming/data/kse/X.csv", X, ",")
    writedlm("scripts/streaming/data/kse/Y.csv", Y, ",")
    writedlm("scripts/streaming/data/kse/Xdot.csv", Xdot, ",")
    writedlm("scripts/streaming/data/kse/Xref.csv", Xref, ",")
    writedlm("scripts/streaming/data/kse/Yref.csv", Yref, ",")
    writedlm("scripts/streaming/data/kse/Vrmax.csv", Vrmax, ",")
    writedlm("scripts/streaming/data/kse/Σr.csv", Σr, ",")
end


######################
## Plot Data to Check
######################
with_theme(theme_latexfonts()) do
    tmp = 100
    fig0 = Figure(fontsize=20, size=(1300,500), backgroundcolor="#FFFFFF")
    ax1 = Axis3(fig0[1, 1], xlabel="x", ylabel="t", zlabel="u(x,t)")
    surface!(ax1, kse.xspan, kse.tspan[1:tmp:end], Xref[:,1:tmp:end])
    ax2 = Axis(fig0[1, 2], xlabel="x", ylabel="t")
    hm = heatmap!(ax2, kse.xspan, kse.tspan[1:tmp:end], Xref[:,1:tmp:end])
    Colorbar(fig0[1, 3], hm)
    display(fig0)
end


################################################
## Check the energy levels from singular values
################################################
nice_orders, energy_level = LnL.choose_ro(Σr; en_low=-12)
with_theme(theme_latexfonts()) do
    fig0 = Figure()
    ax1 = Axis(fig0[1, 1], 
        xlabel="number of retained modes", 
        ylabel="relative energy loss from truncation",
        yscale=log10
    )
    scatterlines!(ax1, nice_orders, energy_level[nice_orders], label="energy loss", lw=2)
    display(fig0)
end


##############################
## Assign selected basis size
##############################
r_select = nice_orders[1:8]
Vr = Vrmax[:, 1:maximum(r_select)]


#######################
## Intrusive-POD model
#######################
op_int = LnL.pod(op_kse, Vr, options)


###############
## OpInf model
###############
# Choose the correct indices corresponding to the derivative data
op_inf = LnL.opinf(X, Vr, options; Y=Y, Xdot=Xdot)


##############################
## Tikhonov Regularized OpInf
##############################
options.with_reg = true
options.λ = LnL.TikhonovParameter(
    lin = 1e-7,
    quad = 1e-7,
    output = 1e-5
)
op_inf_reg = LnL.opinf(X, Vr, options; Y=Y, Xdot=Xdot)


###################
## Streaming-OpInf
###################
# Construct batches of the training data
if CONST_STREAM  # using a single constant batchsize
    global streamsize = 1
else  # initial batch updated with smaller batches
    init_streamsize = 200
    update_size = 1
    global streamsize = vcat([init_streamsize], [update_size for _ in 1:((size(X,2)-init_streamsize)÷update_size)])
end

# Streamify the data based on the selected streamsizes
# INFO: Remember to make data matrices a tall matrix except X matrix
Xhat_stream = LnL.streamify(Vr' * X, streamsize)
Y_stream = LnL.streamify(Y', streamsize)
R_stream = LnL.streamify((Vr' * Xdot)', streamsize)
num_of_streams = length(Xhat_stream)

# Initialize the stream
γs = 1e-7
γo = 1e-5
algo=:iQRRLS
stream = LnL.StreamingOpInf(options, maximum(r_select), 0, size(Y,1); variable_regularize=false, γs_k=γs, γo_k=γo, algorithm=algo)

# Stream all at once
stream.stream!(stream, Xhat_stream, R_stream)
stream.stream_output!(stream, Xhat_stream, Y_stream)

# Unpack solution operators
op_stream = stream.unpack_operators(stream)

# Remove unnecessary variables
Xhat_stream = nothing
Y_stream = nothing
R_stream = nothing
GC.gc()


###############################
## (Analysis 1) Relative Error 
###############################
# Collect all operators into a dictionary
op_dict = Dict(
    "POD" => op_int,
    "OpInf" => op_inf,
    "TR-OpInf" => op_inf_reg,
    "TR-Streaming-OpInf" => op_stream
)
kse.IC = u0(a_ref, b_ref)  # make sure to modify the initial condition
rse, roe = analysis_1(op_dict, kse, Vr, Xref, [], Yref, [:A, :F], kse.integrate_FD; r_select=r_select)

## Plot
fig1 = plot_rse(rse, roe, rmax, ace_light; provided_keys=["POD", "OpInf", "TR-OpInf", "TR-Streaming-OpInf"])
# fig1 = plot_rse(rse, roe, maximum(r_select), ace_light; provided_keys=["POD", "OpInf"])
display(fig1)


######################
## Qualitative Result
######################
Xtmp = kse.integrate_FD(op_stream.A, op_stream.F, kse.t, Vr' * u0(a_ref,b_ref))

with_theme(theme_latexfonts()) do
    tmp = 100
    fig0 = Figure(fontsize=20, size=(1300,500))
    ax1 = Axis3(fig0[1, 1], xlabel="x", ylabel="t", zlabel="u(x,t)")
    surface!(ax1, kse.xspan, kse.tspan[1:tmp:end], (Vr * Xtmp)[:,1:tmp:end])
    ax2 = Axis(fig0[1, 2], xlabel="x", ylabel="t")
    hm = heatmap!(ax2, kse.xspan, kse.tspan[1:tmp:end], (Vr * Xtmp)[:,1:tmp:end])
    Colorbar(fig0[1, 3], hm)
    display(fig0)
end

with_theme(theme_latexfonts()) do
    tmp = 100
    fig0 = Figure(fontsize=20, size=(1300,500))
    ax1 = Axis3(fig0[1, 1], xlabel="x", ylabel="t", zlabel="u(x,t)")
    surface!(ax1, kse.xspan, kse.tspan[1:tmp:end], (Vr * Xtmp - Xref)[:,1:tmp:end], colormap=:roma)
    ax2 = Axis(fig0[1, 2], xlabel="x", ylabel="t")
    hm = heatmap!(ax2, kse.xspan, kse.tspan[1:tmp:end], (Vr * Xtmp - Xref)[:,1:tmp:end], colormap=:roma)
    Colorbar(fig0[1, 3], hm)
    display(fig0)
end