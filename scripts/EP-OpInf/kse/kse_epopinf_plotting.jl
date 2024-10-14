"""
Plotting all results for KSE EP-OpInf example
"""

#================#
## Load packages
#================#
using ChaosGizmo: kaplan_yorke_dim
using FileIO
using JLD2
using Plots
using LaTeXStrings
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using Random
using Statistics
using UniqueKronecker

#========================#
## Load the model module
#========================#
using PolynomialModelReductionDataset: KuramotoSivashinskyModel, AbstractModel

#====================#
## Load LiftAndLearn
#====================#
using LiftAndLearn
const LnL = LiftAndLearn

#================================#
## Configure filepath for saving
#================================#
FILEPATH = occursin("scripts", pwd()) ? joinpath(pwd(),"EP-OpInf/") : joinpath(pwd(), "scripts/EP-OpInf/")

#======================#
## Load all data files
#======================#
KSE = load(joinpath(FILEPATH, "data/kse_epopinf_model_setting.jld2"), "KSE")
OPS = load(joinpath(FILEPATH, "data/kse_epopinf_ops.jld2"), "OPS")
REDUCTION_INFO = load(joinpath(FILEPATH, "data/kse_epopinf_reduction_info.jld2"), "REDUCTION_INFO")
TRAIN_RES = load(joinpath(FILEPATH, "data/kse_epopinf_training_results.jld2"))
TEST1_RES = load(joinpath(FILEPATH, "data/kse_epopinf_test1_results.jld2"))
TEST2_RES = load(joinpath(FILEPATH, "data/kse_epopinf_test2_results.jld2"))
DS = 100
rng = Xoshiro(123)

#=================#
## Color settings
#=================#
# mycolors = cgrad(:Set1_5, 5, categorical = true)
color_choices = Dict(
    :fom => :black,
    :int => :orange,
    :LS => :firebrick3,
    :ephec => :blue3,
    :epsic => :purple,
    :epp => :purple,
    :cvita => :black,
    :edson => :magenta2,
)
# color_choices = Dict(
#     :fom => :black,
#     :int => mycolors[1],
#     :LS => mycolors[2],
#     :ephec => mycolors[3],
#     :epsic => :purple,
#     :epp => :brown,
#     :cvita => :black,
#     :edson => mycolors[4],
# )
marker_choices = Dict(
    :fom => :circle,
    :int => :cross,
    :LS => :rect,
    :ephec => :star5,
    :epsic => :dtriangle,
    :epp => :diamond,
    :cvita => :diamond,
    :edson => :pentagon,
)
markersize_choices = Dict(
    :fom => 5,
    :int => 5,
    :LS => 5,
    :ephec => 5,
    :epsic => 5,
    :epp => 5,
    :cvita => 5,
    :edson => 5,
)
linestyle_choices = Dict(
    :fom => :solid,
    :int => :dot,
    :LS => :dash,
    :ephec => :dashdot,
    :epsic => :dashdot,
    :epp => :dashdot,
    :cvita => :dashdot,
    :edson => :dashdot,
)
linewidth_choices = Dict(
    :fom => 2,
    :int => 2,
    :LS => 2,
    :ephec => 2,
    :epsic => 2,
    :epp => 2,
    :cvita => 2,
    :edson => 2,
)

#=========================#
## Plot projection errors
#=========================#
# Use default colors and custom settings
mean_train_proj_err = mean(TRAIN_RES["proj_err"], dims=2)
plot(REDUCTION_INFO["ro"], mean_train_proj_err, marker=(:rect), fontfamily="Computer Modern")
plot!(yscale=:log10, majorgrid=true, legend=false)
yticks!([1e-1, 1e-2, 1e-3, 1e-4])
xticks!(REDUCTION_INFO["ro"][1]:2:REDUCTION_INFO["ro"][end])
xlabel!("reduced model dimension " * L"r")
ylabel!("projection error")
plot!(guidefontsize=16, tickfontsize=13,  legendfontsize=13)
savefig(joinpath(FILEPATH, "plots/kse/kse_proj_err.pdf"))

#=============================#
## Plot relative state errors
#=============================#
mean_LS_state_err    = mean(TRAIN_RES["state_err"][:LS],    dims=2)
mean_int_state_err   = mean(TRAIN_RES["state_err"][:int],   dims=2)
mean_ephec_state_err = mean(TRAIN_RES["state_err"][:ephec], dims=2)
# mean_epsic_state_err = mean(TRAIN_RES["state_err"][:epsic], dims=2)
# mean_epp_state_err   = mean(TRAIN_RES["state_err"][:epp],   dims=2)

plot( REDUCTION_INFO["ro"], mean_int_state_err,   c=color_choices[:int],   markerstrokecolor=color_choices[:int],   marker=(marker_choices[:int],   markersize_choices[:int],   color_choices[:int]),   ls=linestyle_choices[:int],   lw=linewidth_choices[:int],   label="Intrusive")
plot!(REDUCTION_INFO["ro"], mean_LS_state_err,    c=color_choices[:LS],    markerstrokecolor=color_choices[:LS],    marker=(marker_choices[:LS],    markersize_choices[:LS],    color_choices[:LS]),    ls=linestyle_choices[:LS],    lw=linewidth_choices[:LS],    label="OpInf")
plot!(REDUCTION_INFO["ro"], mean_ephec_state_err, c=color_choices[:ephec], markerstrokecolor=color_choices[:ephec], marker=(marker_choices[:ephec], markersize_choices[:ephec], color_choices[:ephec]), ls=linestyle_choices[:ephec], lw=linewidth_choices[:ephec], label="EP-OpInf")
# plot!(REDUCTION_INFO["ro"], mean_epsic_state_err, c=color_choices[:epsic], markerstrokecolor=color_choices[:epsic], marker=(marker_choices[:epsic], markersize_choices[:epsic], color_choices[:epsic]), ls=linestyle_choices[:epsic], lw=linewidth_choices[:epsic], label="EPSIC-OpInf")
# plot!(REDUCTION_INFO["ro"], mean_epp_state_err,   c=color_choices[:epp],   markerstrokecolor=color_choices[:epp],   marker=(marker_choices[:epp],   markersize_choices[:epp],   color_choices[:epp]),   ls=linestyle_choices[:epp],   lw=linewidth_choices[:epp],   label="EPP-OpInf")
plot!(majorgrid=true, legend=:topright)
# yticks!([1e-0, 1e-1])
xticks!(REDUCTION_INFO["ro"][1]:2:REDUCTION_INFO["ro"][end])
xlabel!("reduced model dimension " * L" r")
ylabel!("average relative state error")
title!("Training")
plot!(guidefontsize=16, tickfontsize=13, legendfontsize=13, fontfamily="Computer Modern")
savefig(joinpath(FILEPATH, "plots/kse/kse_state_err.pdf"))

#============================#
## Plot constraint residuals
#============================#
mean_LS_CR_tr    = mean(TRAIN_RES["CR"][:LS], dims=2)
mean_int_CR_tr   = mean(TRAIN_RES["CR"][:int], dims=2)
mean_ephec_CR_tr = mean(TRAIN_RES["CR"][:ephec], dims=2)
# mean_epsic_CR_tr = mean(TRAIN_RES["CR"][:epsic], dims=2)
# mean_epp_CR_tr   = mean(TRAIN_RES["CR"][:epp], dims=2)

plot( REDUCTION_INFO["ro"], mean_int_CR_tr,   c=color_choices[:int],   markerstrokecolor=color_choices[:int],   marker=(marker_choices[:int],   markersize_choices[:int],   color_choices[:int]),   ls=linestyle_choices[:int],   lw=linewidth_choices[:int],   label="Intrusive")
plot!(REDUCTION_INFO["ro"], mean_LS_CR_tr,    c=color_choices[:LS],    markerstrokecolor=color_choices[:LS],    marker=(marker_choices[:LS],    markersize_choices[:LS],    color_choices[:LS]),    ls=linestyle_choices[:LS],    lw=linewidth_choices[:LS],    label="OpInf")
plot!(REDUCTION_INFO["ro"], mean_ephec_CR_tr, c=color_choices[:ephec], markerstrokecolor=color_choices[:ephec], marker=(marker_choices[:ephec], markersize_choices[:ephec], color_choices[:ephec]), ls=linestyle_choices[:ephec], lw=linewidth_choices[:ephec], label="EP-OpInf")
# plot!(REDUCTION_INFO["ro"], mean_epsic_CR_tr, c=color_choices[:epsic], markerstrokecolor=color_choices[:epsic], marker=(marker_choices[:epsic], markersize_choices[:epsic], color_choices[:epsic]), ls=linestyle_choices[:epsic], lw=linewidth_choices[:epsic], label="EPSIC-OpInf")
# plot!(REDUCTION_INFO["ro"], mean_epp_CR_tr,   c=color_choices[:epp],   markerstrokecolor=color_choices[:epp],   marker=(marker_choices[:epp],   markersize_choices[:epp],   color_choices[:epp]),   ls=linestyle_choices[:epp],   lw=linewidth_choices[:epp],   label="EPP-OpInf")
plot!(yscale=:log10, majorgrid=true, legend=:right, minorgridalpha=0.03)
yticks!(10.0 .^ [-15, -12, -9, -6, -3, 0, 3])
xticks!(REDUCTION_INFO["ro"][1]:2:REDUCTION_INFO["ro"][end])
xlabel!("reduced model dimension " * L" r")
ylabel!("energy-preserving constraint violation")
plot!(xlabelfontsize=16, ylabelfontsize=13, tickfontsize=13, legendfontsize=14, fontfamily="Computer Modern")
savefig(joinpath(FILEPATH, "plots/kse/kse_CR.pdf"))

#=================================#
## Training flow field comparison
#=================================#
training_data_files = readdir(joinpath(FILEPATH, "data/kse_train"), join=true)
chosen_file = rand(rng, training_data_files)

ic = load(chosen_file, "IC")
X_fom = load(chosen_file, "X")
r = REDUCTION_INFO["ro"][end]
Vr = REDUCTION_INFO["Vr"][1][:, 1:r]

# tspan = collect(KSE.time_domain[1]:KSE.Δt:KSE.time_domain[2])
X_int = KSE.integrate_model(
    KSE.tspan, Vr' * ic;
    linear_matrix=OPS["op_int"][1].A[1:r, 1:r], quadratic_matrix=UniqueKronecker.extractF(OPS["op_int"][1].A2u, r), 
    system_input=false, const_stepsize=true
    )
X_LS = KSE.integrate_model(
    KSE.tspan, Vr' * ic;
    linear_matrix=OPS["op_LS"][1].A[1:r, 1:r], quadratic_matrix=UniqueKronecker.extractF(OPS["op_LS"][1].A2u, r),
    system_input=false, const_stepsize=true
    )
X_ephec = KSE.integrate_model(
    KSE.tspan, Vr' * ic;
    linear_matrix=OPS["op_ephec"][1].A[1:r, 1:r], quadratic_matrix=UniqueKronecker.extractF(OPS["op_ephec"][1].A2u, r),
    system_input=false, const_stepsize=true
    )
# X_epsic = KSE.integrate_model(
#     KSE.tspan, Vr' * ic;
#     linear_matrix=OPS["op_epsic"][1].A[1:r, 1:r], quadratic_matrix=UniqueKronecker.extractF(OPS["op_epsic"][1].F, r),
#     system_input=false, const_stepsize=true
#     )
# X_epp = KSE.integrate_model(
#     KSE.tspan, Vr' * ic;
#     linear_matrix=OPS["op_epp"][1].A[1:r, 1:r], quadratic_matrix=UniqueKronecker.extractF(OPS["op_epp"][1].F, r),
#     system_input=false, const_stepsize=true
#     )

# lout = @layout [a{0.3h}; [grid(2,2)]]
lout = @layout [grid(4,2)]
p_fom = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, X_fom[:, 1:DS:end], lw=0, color=:inferno), 
    colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
    #right_margin=-3mm, left_margin=3mm, 
    left_margin=15mm,
    ylabel=L"\textbf{Full}" * "\n" * L"\omega",
    xformatter=_->"",
)
pblank = plot(
    # legend=false,grid=false,foreground_color_subplot=:white, 
    axis=false,legend=false,grid=false, background_color_inside=:transparent, 
    # left_margin=-3mm, right_margin=-3mm, 
)
p_int = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, Vr * X_int[:, 1:DS:end], lw=0, color=:inferno), 
    # contourf(KSE.t[1:DS:end], KSE.xspan, Vr * X_int[:, 1:DS:end], lw=0), 
    colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
    #right_margin=-3mm, left_margin=3mm, top_margin=-2mm,
    left_margin=15mm,
    ylabel=L"\textbf{Intrusive}" * "\n" * L"\omega",
    xformatter=_->"",
)
p_int_err = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, abs.(X_fom[:, 1:DS:end] .- Vr * X_int[:, 1:DS:end]), lw=0, color=:roma), 
    # contourf(KSE.t[1:DS:end], KSE.xspan, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_int[:, 1:DS:end]), lw=0, color=:roma), 
    yticks=(0:5:20), ylims=(0,22), colorbar_ticks=(0:0.5:5), clim=(0,5), 
    # left_margin=-3mm, right_margin=-3mm,
    xformatter=_->"", yformatter=_->"",
)
p_LS = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, Vr * X_LS[:, 1:DS:end], lw=0, color=:inferno), 
    # contourf(KSE.t[1:DS:end], KSE.xspan, Vr * X_LS[:, 1:DS:end], lw=0), 
    colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
    #right_margin=-3mm, left_margin=3mm, top_margin=-2mm,
    left_margin=15mm,
    ylabel=L"\textbf{OpInf}" * "\n" * L"\omega",
    xformatter=_->"",
)
p_LS_err = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, abs.(X_fom[:, 1:DS:end] .- Vr * X_LS[:, 1:DS:end]), lw=0, color=:roma), 
    # contourf(KSE.t[1:DS:end], KSE.xspan, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_LS[:, 1:DS:end]), lw=0, color=:roma), 
    yticks=(0:5:20), ylims=(0,22),  colorbar_ticks=(0:0.5:5), clim=(0,5), 
    # left_margin=-3mm, right_margin=-3mm, 
    xformatter=_->"", yformatter=_->"",
)
p_ephec = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, Vr * X_ephec[:, 1:DS:end], lw=0, color=:inferno), 
    # contourf(KSE.t[1:DS:end], KSE.xspan, Vr * X_ephec[:, 1:DS:end], lw=0), 
    colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
    #right_margin=-3mm, left_margin=3mm,  bottom_margin=7mm, top_margin=-2mm,
    bottom_margin=15mm, left_margin=15mm,
    ylabel=L"\textbf{EP}\rm{-}\textbf{OpInf}" * "\n" * L"\omega", xlabel=L"t" * "\n" * L"\textbf{Predicted}~" * L"x(\omega,t)",
)
p_ephec_err = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, abs.(X_fom[:, 1:DS:end] .- Vr * X_ephec[:, 1:DS:end]), lw=0, color=:roma), 
    # contourf(KSE.t[1:DS:end], KSE.x, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_ephec[:, 1:DS:end]), lw=0, color=:roma), 
    yticks=(0:5:20), ylims=(0,22), colorbar_ticks=(0:0.5:5), clim=(0,5),
    # left_margin=-3mm, right_margin=-3mm, 
    bottom_margin=15mm,
    xlabel=L"t" * "\n" * L"\textbf{Error}",
    yformatter=_->"",
)
# p_epsic = plot(
#     contourf(KSE.tspan[1:DS:end], KSE.xspan, Vr * X_epsic[:, 1:DS:end], lw=0, color=:inferno), 
#     # contourf(KSE.t[1:DS:end], KSE.xspan, Vr * X_epsic[:, 1:DS:end], lw=0), 
#     colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
#     #right_margin=-3mm, left_margin=3mm,  bottom_margin=7mm, top_margin=-2mm,
#     bottom_margin=15mm, left_margin=15mm,
#     ylabel=L"\textbf{EPSIC}\rm{-}\textbf{OpInf}" * "\n" * L"\omega", xlabel=L"t" * "\n" * L"\textbf{Predicted}~" * L"x(\omega,t)",
# )
# p_epsic_err = plot(
#     contourf(KSE.tspan[1:DS:end], KSE.xspan, abs.(X_fom[:, 1:DS:end] .- Vr * X_epsic[:, 1:DS:end]), lw=0, color=:roma), 
#     # contourf(KSE.t[1:DS:end], KSE.x, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_epsic[:, 1:DS:end]), lw=0, color=:roma), 
#     yticks=(0:5:20), ylims=(0,22), colorbar_ticks=(0:0.5:5), clim=(0,5),
#     # left_margin=-3mm, right_margin=-3mm, 
#     bottom_margin=15mm,
#     xlabel=L"t" * "\n" * L"\textbf{Error}",
#     yformatter=_->"",
# )
# p_epp = plot(
#     contourf(KSE.tspan[1:DS:end], KSE.xspan, Vr * X_epp[:, 1:DS:end], lw=0, color=:inferno), 
#     # contourf(KSE.t[1:DS:end], KSE.xspan, Vr * X_epp[:, 1:DS:end], lw=0), 
#     colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
#     #right_margin=-3mm, left_margin=3mm,  bottom_margin=7mm, top_margin=-2mm,
#     bottom_margin=15mm, left_margin=15mm,
#     ylabel=L"\textbf{EPP}\rm{-}\textbf{OpInf}" * "\n" * L"\omega", xlabel=L"t" * "\n" * L"\textbf{Predicted}~" * L"x(\omega,t)",
# )
# p_epp_err = plot(
#     contourf(KSE.tspan[1:DS:end], KSE.xspan, abs.(X_fom[:, 1:DS:end] .- Vr * X_epp[:, 1:DS:end]), lw=0, color=:roma), 
#     # contourf(KSE.t[1:DS:end], KSE.x, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_epp[:, 1:DS:end]), lw=0, color=:roma), 
#     yticks=(0:5:20), ylims=(0,22), colorbar_ticks=(0:0.5:5), clim=(0,5),
#     # left_margin=-3mm, right_margin=-3mm, 
#     bottom_margin=15mm,
#     xlabel=L"t" * "\n" * L"\textbf{Error}",
#     yformatter=_->"",
# )

p = plot(
    p_fom, pblank, 
    p_int, p_int_err,
    p_LS, p_LS_err,
    p_ephec, p_ephec_err,
    # p_epsic, p_epsic_err,
    # p_epp, p_epp_err, 
    fontfamily="Computer Modern", layout=lout, 
    size=(2000, 1080),
    guidefontsize=25, tickfontsize=17, plot_titlefontsize=30, plot_titlefontcolor=:black,
    plot_title="Predicted Flow Fields and Errors of Training",
)
# plot!(p, background_color=:transparent, background_color_inside=:transparent, dpi=600)
plot!(p, dpi=600)
savefig(p, joinpath(FILEPATH, "plots/kse/kse_train_ff.png"))

#===============================#
## Test 1 flow field comparison
#===============================#
test1_data_files = readdir(joinpath(FILEPATH, "data/kse_test1"), join=true)
chosen_file = rand(rng, test1_data_files)

ic = load(chosen_file, "IC")
X_fom = load(chosen_file, "X")
r = REDUCTION_INFO["ro"][end]
Vr = REDUCTION_INFO["Vr"][1][:, 1:r]

# tspan = collect(KSE.time_domain[1]:KSE.Δt:KSE.time_domain[2])
X_int = KSE.integrate_model(
    KSE.tspan, Vr' * ic;
    linear_matrix=OPS["op_int"][1].A[1:r, 1:r], quadratic_matrix=UniqueKronecker.extractF(OPS["op_int"][1].A2u, r), 
    system_input=false, const_stepsize=true
    )
X_LS = KSE.integrate_model(
    KSE.tspan, Vr' * ic;
    linear_matrix=OPS["op_LS"][1].A[1:r, 1:r], quadratic_matrix=UniqueKronecker.extractF(OPS["op_LS"][1].A2u, r),
    system_input=false, const_stepsize=true
    )
X_ephec = KSE.integrate_model(
    KSE.tspan, Vr' * ic;
    linear_matrix=OPS["op_ephec"][1].A[1:r, 1:r], quadratic_matrix=UniqueKronecker.extractF(OPS["op_ephec"][1].A2u, r),
    system_input=false, const_stepsize=true
    )
# X_epsic = KSE.integrate_model(
#     KSE.tspan, Vr' * ic;
#     linear_matrix=OPS["op_epsic"][1].A[1:r, 1:r], quadratic_matrix=UniqueKronecker.extractF(OPS["op_epsic"][1].A2u, r),
#     system_input=false, const_stepsize=true
#     )
# X_epp = KSE.integrate_model(
#     KSE.tspan, Vr' * ic;
#     linear_matrix=OPS["op_epp"][1].A[1:r, 1:r], quadratic_matrix=UniqueKronecker.extractF(OPS["op_epp"][1].A2u, r),
#     system_input=false, const_stepsize=true
#     )

# lout = @layout [a{0.3h}; [grid(2,2)]]
lout = @layout [grid(4,2)]
p_fom = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, X_fom[:, 1:DS:end], lw=0, color=:inferno), 
    colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
    #right_margin=-3mm, left_margin=3mm, 
    left_margin=15mm,
    ylabel=L"\textbf{Full}" * "\n" * L"\omega",
    xformatter=_->"",
)
pblank = plot(
    # legend=false,grid=false,foreground_color_subplot=:white, 
    axis=false,legend=false,grid=false, background_color_inside=:transparent, 
    # left_margin=-3mm, right_margin=-3mm, 
)
p_int = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, Vr * X_int[:, 1:DS:end], lw=0, color=:inferno), 
    # contourf(KSE.t[1:DS:end], KSE.xspan, Vr * X_int[:, 1:DS:end], lw=0), 
    colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
    #right_margin=-3mm, left_margin=3mm, top_margin=-2mm,
    left_margin=15mm,
    ylabel=L"\textbf{Intrusive}" * "\n" * L"\omega",
    xformatter=_->"",
)
p_int_err = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, abs.(X_fom[:, 1:DS:end] .- Vr * X_int[:, 1:DS:end]), lw=0, color=:roma), 
    # contourf(KSE.t[1:DS:end], KSE.xspan, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_int[:, 1:DS:end]), lw=0, color=:roma), 
    yticks=(0:5:20), ylims=(0,22), colorbar_ticks=(0:0.5:5), clim=(0,5), 
    # left_margin=-3mm, right_margin=-3mm,
    xformatter=_->"", yformatter=_->"",
)
p_LS = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, Vr * X_LS[:, 1:DS:end], lw=0, color=:inferno), 
    # contourf(KSE.t[1:DS:end], KSE.xspan, Vr * X_LS[:, 1:DS:end], lw=0), 
    colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
    #right_margin=-3mm, left_margin=3mm, top_margin=-2mm,
    left_margin=15mm,
    ylabel=L"\textbf{OpInf}" * "\n" * L"\omega",
    xformatter=_->"",
)
p_LS_err = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, abs.(X_fom[:, 1:DS:end] .- Vr * X_LS[:, 1:DS:end]), lw=0, color=:roma), 
    # contourf(KSE.t[1:DS:end], KSE.xspan, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_LS[:, 1:DS:end]), lw=0, color=:roma), 
    yticks=(0:5:20), ylims=(0,22),  colorbar_ticks=(0:0.5:5), clim=(0,5), 
    # left_margin=-3mm, right_margin=-3mm, 
    xformatter=_->"", yformatter=_->"",
)
p_ephec = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, Vr * X_ephec[:, 1:DS:end], lw=0, color=:inferno), 
    # contourf(KSE.t[1:DS:end], KSE.xspan, Vr * X_ephec[:, 1:DS:end], lw=0), 
    colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
    #right_margin=-3mm, left_margin=3mm,  bottom_margin=7mm, top_margin=-2mm,
    bottom_margin=15mm, left_margin=15mm,
    ylabel=L"\textbf{EP}\rm{-}\textbf{OpInf}" * "\n" * L"\omega", xlabel=L"t" * "\n" * L"\textbf{Predicted}~" * L"x(\omega,t)",
)
p_ephec_err = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, abs.(X_fom[:, 1:DS:end] .- Vr * X_ephec[:, 1:DS:end]), lw=0, color=:roma), 
    # contourf(KSE.t[1:DS:end], KSE.xspan, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_ephec[:, 1:DS:end]), lw=0, color=:roma), 
    yticks=(0:5:20), ylims=(0,22), colorbar_ticks=(0:0.5:5), clim=(0,5),
    # left_margin=-3mm, right_margin=-3mm, 
    bottom_margin=15mm,
    xlabel=L"t" * "\n" * L"\textbf{Error}",
    yformatter=_->"",
)
# p_epsic = plot(
#     contourf(KSE.tspan[1:DS:end], KSE.xspan, Vr * X_epsic[:, 1:DS:end], lw=0, color=:inferno), 
#     # contourf(KSE.t[1:DS:end], KSE.xspan, Vr * X_epsic[:, 1:DS:end], lw=0), 
#     colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
#     #right_margin=-3mm, left_margin=3mm,  bottom_margin=7mm, top_margin=-2mm,
#     bottom_margin=15mm, left_margin=15mm,
#     ylabel=L"\textbf{EPSIC}\rm{-}\textbf{OpInf}" * "\n" * L"\omega", xlabel=L"t" * "\n" * L"\textbf{Predicted}~" * L"x(\omega,t)",
# )
# p_epsic_err = plot(
#     contourf(KSE.tspan[1:DS:end], KSE.xspan, abs.(X_fom[:, 1:DS:end] .- Vr * X_epsic[:, 1:DS:end]), lw=0, color=:roma), 
#     # contourf(KSE.t[1:DS:end], KSE.x, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_epsic[:, 1:DS:end]), lw=0, color=:roma), 
#     yticks=(0:5:20), ylims=(0,22), colorbar_ticks=(0:0.5:5), clim=(0,5),
#     # left_margin=-3mm, right_margin=-3mm, 
#     bottom_margin=15mm,
#     xlabel=L"t" * "\n" * L"\textbf{Error}",
#     yformatter=_->"",
# )
# p_epp = plot(
#     contourf(KSE.tspan[1:DS:end], KSE.xspan, Vr * X_epp[:, 1:DS:end], lw=0, color=:inferno), 
#     # contourf(KSE.t[1:DS:end], KSE.xspan, Vr * X_epp[:, 1:DS:end], lw=0), 
#     colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
#     #right_margin=-3mm, left_margin=3mm,  bottom_margin=7mm, top_margin=-2mm,
#     bottom_margin=15mm, left_margin=15mm,
#     ylabel=L"\textbf{EPP}\rm{-}\textbf{OpInf}" * "\n" * L"\omega", xlabel=L"t" * "\n" * L"\textbf{Predicted}~" * L"x(\omega,t)",
# )
# p_epp_err = plot(
#     contourf(KSE.tspan[1:DS:end], KSE.xspan, abs.(X_fom[:, 1:DS:end] .- Vr * X_epp[:, 1:DS:end]), lw=0, color=:roma), 
#     # contourf(KSE.t[1:DS:end], KSE.x, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_epp[:, 1:DS:end]), lw=0, color=:roma), 
#     yticks=(0:5:20), ylims=(0,22), colorbar_ticks=(0:0.5:5), clim=(0,5),
#     # left_margin=-3mm, right_margin=-3mm, 
#     bottom_margin=15mm,
#     xlabel=L"t" * "\n" * L"\textbf{Error}",
#     yformatter=_->"",
# )

p = plot(
    p_fom, pblank, 
    p_int, p_int_err,
    p_LS, p_LS_err,
    p_ephec, p_ephec_err,
    # p_epsic, p_epsic_err,
    # p_epp, p_epp_err, 
    fontfamily="Computer Modern", layout=lout, 
    size=(2000, 1080),
    guidefontsize=25, tickfontsize=17, plot_titlefontsize=30, plot_titlefontcolor=:black,
    plot_title="Predicted Flow Fields and Errors of Training",
)
# plot!(p, background_color=:transparent, background_color_inside=:transparent, dpi=600)
plot!(p, dpi=600)
savefig(p, joinpath(FILEPATH, "plots/kse/kse_test1_ff.png"))

#===============================#
## Test 2 flow field comparison
#===============================#
test2_data_files = readdir(joinpath(FILEPATH, "data/kse_test2"), join=true)
chosen_file = rand(rng, test2_data_files)

ic = load(chosen_file, "IC")
X_fom = load(chosen_file, "X")
r = REDUCTION_INFO["ro"][end]
Vr = REDUCTION_INFO["Vr"][1][:, 1:r]

# tspan = collect(KSE.time_domain[1]:KSE.Δt:KSE.time_domain[2])
X_int = KSE.integrate_model(
    KSE.tspan, Vr' * ic;
    linear_matrix=OPS["op_int"][1].A[1:r, 1:r], quadratic_matrix=UniqueKronecker.extractF(OPS["op_int"][1].A2u, r), 
    system_input=false, const_stepsize=true
    )
X_LS = KSE.integrate_model(
    KSE.tspan, Vr' * ic;
    linear_matrix=OPS["op_LS"][1].A[1:r, 1:r], quadratic_matrix=UniqueKronecker.extractF(OPS["op_LS"][1].A2u, r),
    system_input=false, const_stepsize=true
    )
X_ephec = KSE.integrate_model(
    KSE.tspan, Vr' * ic;
    linear_matrix=OPS["op_ephec"][1].A[1:r, 1:r], quadratic_matrix=UniqueKronecker.extractF(OPS["op_ephec"][1].A2u, r),
    system_input=false, const_stepsize=true
    )
# X_epsic = KSE.integrate_model(
#     KSE.tspan, Vr' * ic;
#     linear_matrix=OPS["op_epsic"][1].A[1:r, 1:r], quadratic_matrix=UniqueKronecker.extractF(OPS["op_epsic"][1].A2u, r),
#     system_input=false, const_stepsize=true
#     )
# X_epp = KSE.integrate_model(
#     KSE.tspan, Vr' * ic;
#     linear_matrix=OPS["op_epp"][1].A[1:r, 1:r], quadratic_matrix=UniqueKronecker.extractF(OPS["op_epp"][1].A2u, r),
#     system_input=false, const_stepsize=true
#     )


# lout = @layout [a{0.3h}; [grid(2,2)]]
lout = @layout [grid(4,2)]
p_fom = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, X_fom[:, 1:DS:end], lw=0, color=:inferno), 
    colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
    #right_margin=-3mm, left_margin=3mm, 
    left_margin=15mm,
    ylabel=L"\textbf{Full}" * "\n" * L"\omega",
    xformatter=_->"",
)
pblank = plot(
    # legend=false,grid=false,foreground_color_subplot=:white, 
    axis=false,legend=false,grid=false, background_color_inside=:transparent, 
    # left_margin=-3mm, right_margin=-3mm, 
)
p_int = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, Vr * X_int[:, 1:DS:end], lw=0, color=:inferno), 
    # contourf(KSE.t[1:DS:end], KSE.xspan, Vr * X_int[:, 1:DS:end], lw=0), 
    colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
    #right_margin=-3mm, left_margin=3mm, top_margin=-2mm,
    left_margin=15mm,
    ylabel=L"\textbf{Intrusive}" * "\n" * L"\omega",
    xformatter=_->"",
)
p_int_err = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, abs.(X_fom[:, 1:DS:end] .- Vr * X_int[:, 1:DS:end]), lw=0, color=:roma), 
    # contourf(KSE.t[1:DS:end], KSE.xspan, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_int[:, 1:DS:end]), lw=0, color=:roma), 
    yticks=(0:5:20), ylims=(0,22), colorbar_ticks=(0:0.5:5), clim=(0,5), 
    # left_margin=-3mm, right_margin=-3mm,
    xformatter=_->"", yformatter=_->"",
)
p_LS = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, Vr * X_LS[:, 1:DS:end], lw=0, color=:inferno), 
    # contourf(KSE.t[1:DS:end], KSE.xspan, Vr * X_LS[:, 1:DS:end], lw=0), 
    colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
    #right_margin=-3mm, left_margin=3mm, top_margin=-2mm,
    left_margin=15mm,
    ylabel=L"\textbf{OpInf}" * "\n" * L"\omega",
    xformatter=_->"",
)
p_LS_err = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, abs.(X_fom[:, 1:DS:end] .- Vr * X_LS[:, 1:DS:end]), lw=0, color=:roma), 
    # contourf(KSE.t[1:DS:end], KSE.xspan, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_LS[:, 1:DS:end]), lw=0, color=:roma), 
    yticks=(0:5:20), ylims=(0,22),  colorbar_ticks=(0:0.5:5), clim=(0,5), 
    # left_margin=-3mm, right_margin=-3mm, 
    xformatter=_->"", yformatter=_->"",
)
p_ephec = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, Vr * X_ephec[:, 1:DS:end], lw=0, color=:inferno), 
    # contourf(KSE.t[1:DS:end], KSE.xspan, Vr * X_ephec[:, 1:DS:end], lw=0), 
    colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
    #right_margin=-3mm, left_margin=3mm,  bottom_margin=7mm, top_margin=-2mm,
    bottom_margin=15mm, left_margin=15mm,
    ylabel=L"\textbf{EP}\rm{-}\textbf{OpInf}" * "\n" * L"\omega", xlabel=L"t" * "\n" * L"\textbf{Predicted}~" * L"x(\omega,t)",
)
p_ephec_err = plot(
    contourf(KSE.tspan[1:DS:end], KSE.xspan, abs.(X_fom[:, 1:DS:end] .- Vr * X_ephec[:, 1:DS:end]), lw=0, color=:roma), 
    # contourf(KSE.t[1:DS:end], KSE.xspan, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_ephec[:, 1:DS:end]), lw=0, color=:roma), 
    yticks=(0:5:20), ylims=(0,22), colorbar_ticks=(0:0.5:5), clim=(0,5),
    # left_margin=-3mm, right_margin=-3mm, 
    bottom_margin=15mm,
    xlabel=L"t" * "\n" * L"\textbf{Error}",
    yformatter=_->"",
)
# p_epsic = plot(
#     contourf(KSE.tspan[1:DS:end], KSE.xspan, Vr * X_epsic[:, 1:DS:end], lw=0, color=:inferno), 
#     # contourf(KSE.t[1:DS:end], KSE.xspan, Vr * X_epsic[:, 1:DS:end], lw=0), 
#     colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
#     #right_margin=-3mm, left_margin=3mm,  bottom_margin=7mm, top_margin=-2mm,
#     bottom_margin=15mm, left_margin=15mm,
#     ylabel=L"\textbf{EPSIC}\rm{-}\textbf{OpInf}" * "\n" * L"\omega", xlabel=L"t" * "\n" * L"\textbf{Predicted}~" * L"x(\omega,t)",
# )
# p_epsic_err = plot(
#     contourf(KSE.tspan[1:DS:end], KSE.xspan, abs.(X_fom[:, 1:DS:end] .- Vr * X_epsic[:, 1:DS:end]), lw=0, color=:roma), 
#     # contourf(KSE.t[1:DS:end], KSE.x, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_epsic[:, 1:DS:end]), lw=0, color=:roma), 
#     yticks=(0:5:20), ylims=(0,22), colorbar_ticks=(0:0.5:5), clim=(0,5),
#     # left_margin=-3mm, right_margin=-3mm, 
#     bottom_margin=15mm,
#     xlabel=L"t" * "\n" * L"\textbf{Error}",
#     yformatter=_->"",
# )
# p_epp = plot(
#     contourf(KSE.tspan[1:DS:end], KSE.xspan, Vr * X_epp[:, 1:DS:end], lw=0, color=:inferno), 
#     # contourf(KSE.t[1:DS:end], KSE.xspan, Vr * X_epp[:, 1:DS:end], lw=0), 
#     colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
#     #right_margin=-3mm, left_margin=3mm,  bottom_margin=7mm, top_margin=-2mm,
#     bottom_margin=15mm, left_margin=15mm,
#     ylabel=L"\textbf{EPP}\rm{-}\textbf{OpInf}" * "\n" * L"\omega", xlabel=L"t" * "\n" * L"\textbf{Predicted}~" * L"x(\omega,t)",
# )
# p_epp_err = plot(
#     contourf(KSE.tspan[1:DS:end], KSE.xspan, abs.(X_fom[:, 1:DS:end] .- Vr * X_epp[:, 1:DS:end]), lw=0, color=:roma), 
#     # contourf(KSE.t[1:DS:end], KSE.x, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_epp[:, 1:DS:end]), lw=0, color=:roma), 
#     yticks=(0:5:20), ylims=(0,22), colorbar_ticks=(0:0.5:5), clim=(0,5),
#     # left_margin=-3mm, right_margin=-3mm, 
#     bottom_margin=15mm,
#     xlabel=L"t" * "\n" * L"\textbf{Error}",
#     yformatter=_->"",
# )

p = plot(
    p_fom, pblank, 
    p_int, p_int_err,
    p_LS, p_LS_err,
    p_ephec, p_ephec_err,
    # p_epsic, p_epsic_err,
    # p_epp, p_epp_err, 
    fontfamily="Computer Modern", layout=lout, 
    size=(2000, 1080),
    guidefontsize=25, tickfontsize=17, plot_titlefontsize=30, plot_titlefontcolor=:black,
    plot_title="Predicted Flow Fields and Errors of Training",
)
# plot!(p, background_color=:transparent, background_color_inside=:transparent, dpi=600)
plot!(p, dpi=600)
savefig(p, joinpath(FILEPATH, "plots/kse/kse_test2_ff.png"))

#====================================================================#
## Plot normalized autocorrelation function for training and testing
#====================================================================#
# lag for autocorrelation
lags = TRAIN_RES["AC_lags"]
lags_t = collect(lags) .* KSE.Δt
idx = length(lags_t)

lout = @layout [grid(4,3)]
p = plot(layout=lout, size=(1400, 1000), dpi=3000)

train_indices = [1, 4, 7, 10]
test1_indices = [2, 5, 8, 11]
test2_indices = [3, 6, 9, 12]

for (plot_id, ri) in enumerate([1, 2, 5, 7])
    # Training
    plot!(p[train_indices[plot_id]], lags_t[1:idx], TRAIN_RES["AC"][:fom][1:idx],         c=color_choices[:fom],   ls=linestyle_choices[:fom],   lw=linewidth_choices[:fom],   label="Full")
    plot!(p[train_indices[plot_id]], lags_t[1:idx], TRAIN_RES["AC"][:int][:,ri][1:idx],   c=color_choices[:int],   ls=linestyle_choices[:int],   lw=linewidth_choices[:int],   label="Intrusive")
    plot!(p[train_indices[plot_id]], lags_t[1:idx], TRAIN_RES["AC"][:LS][:,ri][1:idx],    c=color_choices[:LS],    ls=linestyle_choices[:LS],    lw=linewidth_choices[:LS],    label="OpInf")
    plot!(p[train_indices[plot_id]], lags_t[1:idx], TRAIN_RES["AC"][:ephec][:,ri][1:idx], c=color_choices[:ephec], ls=linestyle_choices[:ephec], lw=linewidth_choices[:ephec], label="EP-OpInf")
    # plot!(p[train_indices[plot_id]], lags_t[1:idx], TRAIN_RES["AC"][:epsic][:,ri][1:idx], c=color_choices[:epsic], ls=linestyle_choices[:epsic], lw=linewidth_choices[:epsic], label="EPSIC-OpInf")
    # plot!(p[train_indices[plot_id]], lags_t[1:idx], TRAIN_RES["AC"][:epp][:,ri][1:idx],   c=color_choices[:epp],   ls=linestyle_choices[:epp],   lw=linewidth_choices[:epp],   label="EPP-OpInf")
    plot!(p[train_indices[plot_id]], fontfamily="Computer Modern", tickfontsize=11)
    ylims!(p[train_indices[plot_id]], (-0.2, 1.05))
    yticks!(p[train_indices[plot_id]], [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if ri != 7
        plot!(p[train_indices[plot_id]], xformatter=_->"", bottom_margin=-1mm)
    end

    # Test 1
    plot!(p[test1_indices[plot_id]], lags_t[1:idx], TEST1_RES["AC"][:fom][1:idx],         c=color_choices[:fom],   ls=linestyle_choices[:fom],   lw=linewidth_choices[:fom],   label="Full")
    plot!(p[test1_indices[plot_id]], lags_t[1:idx], TEST1_RES["AC"][:int][:,ri][1:idx],   c=color_choices[:int],   ls=linestyle_choices[:int],   lw=linewidth_choices[:int],   label="Intrusive")
    plot!(p[test1_indices[plot_id]], lags_t[1:idx], TEST1_RES["AC"][:LS][:,ri][1:idx],    c=color_choices[:LS],    ls=linestyle_choices[:LS],    lw=linewidth_choices[:LS],    label="OpInf")
    plot!(p[test1_indices[plot_id]], lags_t[1:idx], TEST1_RES["AC"][:ephec][:,ri][1:idx], c=color_choices[:ephec], ls=linestyle_choices[:ephec], lw=linewidth_choices[:ephec], label="EP-OpInf")
    # plot!(p[test1_indices[plot_id]], lags_t[1:idx], TEST1_RES["AC"][:epsic][:,ri][1:idx], c=color_choices[:epsic], ls=linestyle_choices[:epsic], lw=linewidth_choices[:epsic], label="EPSIC-OpInf")
    # plot!(p[test1_indices[plot_id]], lags_t[1:idx], TEST1_RES["AC"][:epp][:,ri][1:idx],   c=color_choices[:epp],   ls=linestyle_choices[:epp],   lw=linewidth_choices[:epp],   label="EPP-OpInf")
    plot!(p[test1_indices[plot_id]], yformatter=_->"", left_margin=-6mm)
    ylims!(p[test1_indices[plot_id]], (-0.2, 1.05))
    yticks!(p[test1_indices[plot_id]], [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if ri != 7
        plot!(p[test1_indices[plot_id]], xformatter=_->"", bottom_margin=-1mm)
    end

    # Test 2
    plot!(p[test2_indices[plot_id]], lags_t[1:idx], TEST2_RES["AC"][:fom][1:idx],         c=color_choices[:fom],   ls=linestyle_choices[:fom],   lw=linewidth_choices[:fom],   label="Full")
    plot!(p[test2_indices[plot_id]], lags_t[1:idx], TEST2_RES["AC"][:int][:,ri][1:idx],   c=color_choices[:int],   ls=linestyle_choices[:int],   lw=linewidth_choices[:int],   label="Intrusive")
    plot!(p[test2_indices[plot_id]], lags_t[1:idx], TEST2_RES["AC"][:LS][:,ri][1:idx],    c=color_choices[:LS],    ls=linestyle_choices[:LS],    lw=linewidth_choices[:LS],    label="OpInf")
    plot!(p[test2_indices[plot_id]], lags_t[1:idx], TEST2_RES["AC"][:ephec][:,ri][1:idx], c=color_choices[:ephec], ls=linestyle_choices[:ephec], lw=linewidth_choices[:ephec], label="EP-OpInf")
    # plot!(p[test2_indices[plot_id]], lags_t[1:idx], TEST2_RES["AC"][:epsic][:,ri][1:idx], c=color_choices[:epsic], ls=linestyle_choices[:epsic], lw=linewidth_choices[:epsic], label="EPSIC-OpInf")
    # plot!(p[test2_indices[plot_id]], lags_t[1:idx], TEST2_RES["AC"][:epp][:,ri][1:idx],   c=color_choices[:epp],   ls=linestyle_choices[:epp],   lw=linewidth_choices[:epp],   label="EPP-OpInf")
    plot!(p[test2_indices[plot_id]], yformatter=_->"", left_margin=-6mm)
    ylims!(p[test2_indices[plot_id]], (-0.2, 1.05))
    yticks!(p[test2_indices[plot_id]], [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if ri !== 7
        plot!(p[test2_indices[plot_id]], xformatter=_->"", bottom_margin=-1mm)
    end

end
plot!(p, legend=false)
plot!(p[12], legend=:topright,  legend_font_pointsize=15,)

title!(p[1], L"\textbf{Training}")
title!(p[2], L"\textbf{Test 1: Interpolation}")
title!(p[3], L"\textbf{Test 2: Extrapolation}")
plot!(p, titlefontsize=18)

plot!(p[1], left_margin=35mm)
plot!(p[4], left_margin=35mm)
plot!(p[7], left_margin=35mm)

# plot!(p[3], left_margin=30mm)
# plot!(p[6], left_margin=30mm)
# plot!(p[9], left_margin=30mm)

plot!(p[10], bottom_margin=10mm, left_margin=35mm)
plot!(p[11], bottom_margin=10mm)
plot!(p[12], bottom_margin=10mm)

annotate!(p[11], 75, -0.5, "time lag", annotationfontsize=18)
annotate!(p[7], -40, 1.2, Plots.text("average normalized autocorrelation", 18, rotation=90, "Computer Modern", :black))

annotate!(p[1],  -67, 0.4, Plots.text(L"\textbf{r = 9}", 17, "Computer Modern", :black))
annotate!(p[4],  -67, 0.4, Plots.text(L"\textbf{r = 12}", 17, "Computer Modern", :black))
annotate!(p[7],  -67, 0.4, Plots.text(L"\textbf{r = 20}", 17, "Computer Modern", :black))
annotate!(p[10], -67, 0.4, Plots.text(L"\textbf{r = 24}", 17, "Computer Modern", :black))

# plot!(p, background_color=:transparent, background_color_inside="#44546A", dpi=600)
plot!(p, tickfontsize=11, dpi=600)
savefig(p, joinpath(FILEPATH, "plots/kse/kse_ac_all_vert.pdf"))

#======================================================================#
## Normalized autocorrelation function errors for training and testing
#======================================================================#
lout = @layout [grid(1,3)]
p = plot(layout=lout, size=(1840, 500))

# Training
plot!(p[1], REDUCTION_INFO["ro"], TRAIN_RES["AC_ERR"][:int],   c=color_choices[:int],   marker=(marker_choices[:int], markersize_choices[:int], color_choices[:int]),       markerstrokecolor=color_choices[:int],   lw=linewidth_choices[:int],   ls=linestyle_choices[:int],   markerstrokewidth=2.5, label="Intrusive")
plot!(p[1], REDUCTION_INFO["ro"], TRAIN_RES["AC_ERR"][:LS],    c=color_choices[:LS],    marker=(marker_choices[:LS], markersize_choices[:LS], color_choices[:LS]),          markerstrokecolor=color_choices[:LS],    lw=linewidth_choices[:LS],    ls=linestyle_choices[:LS],                           label="OpInf")
plot!(p[1], REDUCTION_INFO["ro"], TRAIN_RES["AC_ERR"][:ephec], c=color_choices[:ephec], marker=(marker_choices[:ephec], markersize_choices[:ephec], color_choices[:ephec]), markerstrokecolor=color_choices[:ephec], lw=linewidth_choices[:ephec], ls=linestyle_choices[:ephec],                        label="EP-OpInf")
# plot!(p[1], REDUCTION_INFO["ro"], TRAIN_RES["AC_ERR"][:epsic], c=color_choices[:epsic], marker=(marker_choices[:epsic], markersize_choices[:epsic], color_choices[:epsic]), markerstrokecolor=color_choices[:epsic], lw=linewidth=choices[:epsic], ls=linestyle_choices[:epsic],                        label="EPSIC-OpInf")
# plot!(p[1], REDUCTION_INFO["ro"], TRAIN_RES["AC_ERR"][:epp],   c=color_choices[:epp],   marker=(marker_choices[:epp], markersize_choices[:epp], color_choices[:epp]),       markerstrokecolor=color_choices[:epp],   lw=linewidth_choices[:epp],   ls=linestyle_choices[:epp],                          label="EPP-OpInf")
plot!(p[1], 
    majorgrid=true, 
    legend=false,
    # xlabel="reduced model dimension " * L" r",
    # ylabel="avg normalized autocorrelation error",
    title=L"\mathbf{Training}",
    titlefontsize=20,
    fontfamily="Computer Modern", tickfontsize=13,
    xticks=REDUCTION_INFO["ro"][1]:2:REDUCTION_INFO["ro"][end],
    ylims=(0.1, 0.75),
    bottom_margin=20mm,
    left_margin=23mm,
)

# Test 1
plot!(p[2], REDUCTION_INFO["ro"], TEST1_RES["AC_ERR"][:int],   c=color_choices[:int],   marker=(marker_choices[:int], markersize_choices[:int], color_choices[:int]),       markerstrokecolor=color_choices[:int],   lw=linewidth_choices[:int],   ls=linestyle_choices[:int],   markerstrokewidth=2.5, label="Intrusive")
plot!(p[2], REDUCTION_INFO["ro"], TEST1_RES["AC_ERR"][:LS],    c=color_choices[:LS],    marker=(marker_choices[:LS], markersize_choices[:LS], color_choices[:LS]),          markerstrokecolor=color_choices[:LS],    lw=linewidth_choices[:LS],    ls=linestyle_choices[:LS],                           label="OpInf")
plot!(p[2], REDUCTION_INFO["ro"], TEST1_RES["AC_ERR"][:ephec], c=color_choices[:ephec], marker=(marker_choices[:ephec], markersize_choices[:ephec], color_choices[:ephec]), markerstrokecolor=color_choices[:ephec], lw=linewidth_choices[:ephec], ls=linestyle_choices[:ephec],                        label="EP-OpInf")
# plot!(p[2], REDUCTION_INFO["ro"], TEST1_RES["AC_ERR"][:epsic], c=color_choices[:epsic], marker=(marker_choices[:epsic], markersize_choices[:epsic], color_choices[:epsic]), markerstrokecolor=color_choices[:epsic], lw=linewidth=choices[:epsic], ls=linestyle_choices[:epsic],                        label="EPSIC-OpInf")
# plot!(p[2], REDUCTION_INFO["ro"], TEST1_RES["AC_ERR"][:epp],   c=color_choices[:epp],   marker=(marker_choices[:epp], markersize_choices[:epp], color_choices[:epp]),       markerstrokecolor=color_choices[:epp],   lw=linewidth_choices[:epp],   ls=linestyle_choices[:epp],                          label="EPP-OpInf")
plot!(p[2], 
    majorgrid=true, 
    legend=false,
    title=L"\mathbf{Test 1: Interpolation}",
    titlefontsize=20,
    xticks=REDUCTION_INFO["ro"][1]:2:REDUCTION_INFO["ro"][end],
    ylims=(0.1, 0.75),
    fontfamily="Computer Modern", guidefontsize=13, tickfontsize=13,
    left_margin=-2mm,
    bottom_margin=20mm,
)

# Test 2
plot!(p[3], REDUCTION_INFO["ro"], TEST2_RES["AC_ERR"][:int],   c=color_choices[:int],   marker=(marker_choices[:int], markersize_choices[:int], color_choices[:int]),       markerstrokecolor=color_choices[:int],   lw=linewidth_choices[:int],   ls=linestyle_choices[:int],   markerstrokewidth=2.5, label="Intrusive")
plot!(p[3], REDUCTION_INFO["ro"], TEST2_RES["AC_ERR"][:LS],    c=color_choices[:LS],    marker=(marker_choices[:LS], markersize_choices[:LS], color_choices[:LS]),          markerstrokecolor=color_choices[:LS],    lw=linewidth_choices[:LS],    ls=linestyle_choices[:LS],                           label="OpInf")
plot!(p[3], REDUCTION_INFO["ro"], TEST2_RES["AC_ERR"][:ephec], c=color_choices[:ephec], marker=(marker_choices[:ephec], markersize_choices[:ephec], color_choices[:ephec]), markerstrokecolor=color_choices[:ephec], lw=linewidth_choices[:ephec], ls=linestyle_choices[:ephec],                        label="EP-OpInf")
# plot!(p[3], REDUCTION_INFO["ro"], TEST2_RES["AC_ERR"][:epsic], c=color_choices[:epsic], marker=(marker_choices[:epsic], markersize_choices[:epsic], color_choices[:epsic]), markerstrokecolor=color_choices[:epsic], lw=linewidth=choices[:epsic], ls=linestyle_choices[:epsic],                        label="EPSIC-OpInf")
# plot!(p[3], REDUCTION_INFO["ro"], TEST2_RES["AC_ERR"][:epp],   c=color_choices[:epp],   marker=(marker_choices[:epp], markersize_choices[:epp], color_choices[:epp]),       markerstrokecolor=color_choices[:epp],   lw=linewidth_choices[:epp],   ls=linestyle_choices[:epp],                          label="EPP-OpInf")
plot!(p[3],
    majorgrid=true, 
    legend=:bottomleft,
    title=L"\mathbf{Test 2: Extrapolation}",
    titlefontsize=20,
    xticks=REDUCTION_INFO["ro"][1]:2:REDUCTION_INFO["ro"][end],
    ylims=(0.1, 0.75),
    fontfamily="Computer Modern", guidefontsize=13, tickfontsize=13,  legendfontsize=16,
    left_margin=-2mm,
    bottom_margin=20mm,
)
annotate!(p[1], 5.0, 0.4, Plots.text("avg normalized \n autocorrelation error", 21, "Computer Modern", rotation=90, color=:black))
annotate!(p[2], 17, -0.02, Plots.text("reduced model dimension " * L"r", 21, "Computer Modern", color=:black))
# plot!(p, background_color=:transparent, background_color_inside="#44546A", dpi=600)
savefig(p, joinpath(FILEPATH, "plots/kse/kse_ac_err_all.pdf"))

#========================================================#
## Lyapunov exponent comparison for training and testing
#========================================================#
# Reference values
edson = [0.043, 0.003, 0.002, -0.004, -0.008, -0.185, -0.253, -0.296, -0.309, -1.965]
cvitanovic = [0.048, 0, 0, -0.003, -0.189, -0.256, -0.290, -0.310, -1.963, -1.967]

lout = @layout [grid(4,3)]
p = plot(layout=lout, size=(1400, 1000), dpi=3000)

train_indices = [1, 4, 7, 10]
test1_indices = [2, 5, 8, 11]
test2_indices = [3, 6, 9, 12]

le_length = 1:size(TRAIN_RES["LE"][:int],1)

for (plot_id, ri) in enumerate([1, 2, 5, 7])
    # Training
    plot!(p[train_indices[plot_id]], le_length, cvitanovic,                    c=color_choices[:cvita],  marker=(marker_choices[:cvita], markersize_choices[:cvita], color_choices[:cvita]), markerstrokecolor=color_choices[:cvita], lw=linewidth_choices[:cvita], ls=linestyle_choices[:cvita], markerstrokewidth=2.5, label="Cvitanovic")
    plot!(p[train_indices[plot_id]], le_length, edson,                         c=color_choices[:edson],  marker=(marker_choices[:edson], markersize_choices[:edson], color_choices[:edson]), markerstrokecolor=color_choices[:edson], lw=linewidth_choices[:edson], ls=linestyle_choices[:edson],                        label="Edson")
    plot!(p[train_indices[plot_id]], le_length, TRAIN_RES["LE"][:int][:,ri],   c=color_choices[:int],    marker=(marker_choices[:int],   markersize_choices[:int],   color_choices[:int]),   markerstrokecolor=color_choices[:int],   lw=linewidth_choices[:int],   ls=linestyle_choices[:int],   markerstrokewidth=2.5, label="Intrusive")
    plot!(p[train_indices[plot_id]], le_length, TRAIN_RES["LE"][:LS][:,ri],    c=color_choices[:LS],     marker=(marker_choices[:LS],    markersize_choices[:LS],    color_choices[:LS]),    markerstrokecolor=color_choices[:LS],    lw=linewidth_choices[:LS],    ls=linestyle_choices[:LS],                           label="OpInf")
    plot!(p[train_indices[plot_id]], le_length, TRAIN_RES["LE"][:ephec][:,ri], c=color_choices[:ephec],  marker=(marker_choices[:ephec], markersize_choices[:ephec], color_choices[:ephec]), markerstrokecolor=color_choices[:ephec], lw=linewidth_choices[:ephec], ls=linestyle_choices[:ephec],                        label="EP-OpInf")
    # plot!(p[train_indices[plot_id]], le_length, TRAIN_RES["LE"][:epsic][:,ri], c=color_choices[:epsic],  marker=(marker_choices[:epsic], markersize_choices[:epsic], color_choices[:epsic]), markerstrokecolor=color_choices[:epsic], lw=linewidth=choices[:epsic], ls=linestyle_choices[:epsic],                        label="EPSIC-OpInf")
    # plot!(p[train_indices[plot_id]], le_length, TRAIN_RES["LE"][:epp][:,ri],   c=color_choices[:epp],    marker=(marker_choices[:epp],   markersize_choices[:epp],   color_choices[:epp]),   markerstrokecolor=color_choices[:epp],   lw=linewidth_choices[:epp],   ls=linestyle_choices[:epp],                          label="EPP-OpInf")
    plot!(p[train_indices[plot_id]], fontfamily="Computer Modern", tickfontsize=11)
    ylims!(p[train_indices[plot_id]], (-2.5, 0.1))
    yticks!(p[train_indices[plot_id]], collect(-2.5:0.5:0.5))
    if ri != 7
        plot!(p[train_indices[plot_id]], xformatter=_->"", bottom_margin=-1mm)
    end

    # Test 1
    plot!(p[test1_indices[plot_id]], le_length, cvitanovic,                    c=color_choices[:cvita],  marker=(marker_choices[:cvita], markersize_choices[:cvita], color_choices[:cvita]), markerstrokecolor=color_choices[:cvita], lw=linewidth_choices[:cvita], ls=linestyle_choices[:cvita], markerstrokewidth=2.5, label="Cvitanovic")
    plot!(p[test1_indices[plot_id]], le_length, edson,                         c=color_choices[:edson],  marker=(marker_choices[:edson], markersize_choices[:edson], color_choices[:edson]), markerstrokecolor=color_choices[:edson], lw=linewidth_choices[:edson], ls=linestyle_choices[:edson],                        label="Edson")
    plot!(p[test1_indices[plot_id]], le_length, TEST1_RES["LE"][:int][:,ri],   c=color_choices[:int],    marker=(marker_choices[:int],   markersize_choices[:int],   color_choices[:int]),   markerstrokecolor=color_choices[:int],   lw=linewidth_choices[:int],   ls=linestyle_choices[:int],   markerstrokewidth=2.5, label="Intrusive")
    plot!(p[test1_indices[plot_id]], le_length, TEST1_RES["LE"][:LS][:,ri],    c=color_choices[:LS],     marker=(marker_choices[:LS],    markersize_choices[:LS],    color_choices[:LS]),    markerstrokecolor=color_choices[:LS],    lw=linewidth_choices[:LS],    ls=linestyle_choices[:LS],                           label="OpInf")
    plot!(p[test1_indices[plot_id]], le_length, TEST1_RES["LE"][:ephec][:,ri], c=color_choices[:ephec],  marker=(marker_choices[:ephec], markersize_choices[:ephec], color_choices[:ephec]), markerstrokecolor=color_choices[:ephec], lw=linewidth_choices[:ephec], ls=linestyle_choices[:ephec],                        label="EP-OpInf")
    # plot!(p[test1_indices[plot_id]], le_length, TEST1_RES["LE"][:epsic][:,ri], c=color_choices[:epsic],  marker=(marker_choices[:epsic], markersize_choices[:epsic], color_choices[:epsic]), markerstrokecolor=color_choices[:epsic], lw=linewidth=choices[:epsic], ls=linestyle_choices[:epsic],                        label="EPSIC-OpInf")
    # plot!(p[test1_indices[plot_id]], le_length, TEST1_RES["LE"][:epp][:,ri],   c=color_choices[:epp],    marker=(marker_choices[:epp],   markersize_choices[:epp],   color_choices[:epp]),   markerstrokecolor=color_choices[:epp],   lw=linewidth_choices[:epp],   ls=linestyle_choices[:epp],                          label="EPP-OpInf")
    plot!(p[test1_indices[plot_id]], yformatter=_->"", left_margin=-6mm)
    ylims!(p[test1_indices[plot_id]], (-2.5, 0.1))
    yticks!(p[test1_indices[plot_id]], collect(-2.5:0.5:0.5))
    if ri != 7
        plot!(p[test1_indices[plot_id]], xformatter=_->"", bottom_margin=-1mm)
    end

    # Test 2
    plot!(p[test2_indices[plot_id]], le_length, cvitanovic,                    c=color_choices[:cvita],  marker=(marker_choices[:cvita], markersize_choices[:cvita], color_choices[:cvita]), markerstrokecolor=color_choices[:cvita], lw=linewidth_choices[:cvita], ls=linestyle_choices[:cvita], markerstrokewidth=2.5, label="Cvitanovic")
    plot!(p[test2_indices[plot_id]], le_length, edson,                         c=color_choices[:edson],  marker=(marker_choices[:edson], markersize_choices[:edson], color_choices[:edson]), markerstrokecolor=color_choices[:edson], lw=linewidth_choices[:edson], ls=linestyle_choices[:edson],                        label="Edson")
    plot!(p[test2_indices[plot_id]], le_length, TEST2_RES["LE"][:int][:,ri],   c=color_choices[:int],    marker=(marker_choices[:int],   markersize_choices[:int],   color_choices[:int]),   markerstrokecolor=color_choices[:int],   lw=linewidth_choices[:int],   ls=linestyle_choices[:int],   markerstrokewidth=2.5, label="Intrusive")
    plot!(p[test2_indices[plot_id]], le_length, TEST2_RES["LE"][:LS][:,ri],    c=color_choices[:LS],     marker=(marker_choices[:LS],    markersize_choices[:LS],    color_choices[:LS]),    markerstrokecolor=color_choices[:LS],    lw=linewidth_choices[:LS],    ls=linestyle_choices[:LS],                           label="OpInf")
    plot!(p[test2_indices[plot_id]], le_length, TEST2_RES["LE"][:ephec][:,ri], c=color_choices[:ephec],  marker=(marker_choices[:ephec], markersize_choices[:ephec], color_choices[:ephec]), markerstrokecolor=color_choices[:ephec], lw=linewidth_choices[:ephec], ls=linestyle_choices[:ephec],                        label="EP-OpInf")
    # plot!(p[test2_indices[plot_id]], le_length, TEST2_RES["LE"][:epsic][:,ri], c=color_choices[:epsic],  marker=(marker_choices[:epsic], markersize_choices[:epsic], color_choices[:epsic]), markerstrokecolor=color_choices[:epsic], lw=linewidth=choices[:epsic], ls=linestyle_choices[:epsic],                        label="EPSIC-OpInf")
    # plot!(p[test2_indices[plot_id]], le_length, TEST2_RES["LE"][:epp][:,ri],   c=color_choices[:epp],    marker=(marker_choices[:epp],   markersize_choices[:epp],   color_choices[:epp]),   markerstrokecolor=color_choices[:epp],   lw=linewidth_choices[:epp],   ls=linestyle_choices[:epp],                          label="EPP-OpInf")
    plot!(p[test2_indices[plot_id]], yformatter=_->"", left_margin=-6mm)
    ylims!(p[test2_indices[plot_id]], (-2.5, 0.1))
    yticks!(p[test2_indices[plot_id]], collect(-2.5:0.5:0.5))
    if ri !== 7
        plot!(p[test2_indices[plot_id]], xformatter=_->"", bottom_margin=-1mm)
    end

end
plot!(p, legend=false)
plot!(p[12], legend=:bottomleft,  legend_font_pointsize=15,)

title!(p[1], L"\textbf{Training}")
title!(p[2], L"\textbf{Test 1: Interpolation}")
title!(p[3], L"\textbf{Test 2: Extrapolation}")
plot!(p, titlefontsize=18)

plot!(p[1], left_margin=35mm)
plot!(p[4], left_margin=35mm)
plot!(p[7], left_margin=35mm)

# plot!(p[3], left_margin=30mm)
# plot!(p[6], left_margin=30mm)
# plot!(p[9], left_margin=30mm)

plot!(p[10], bottom_margin=10mm, left_margin=35mm)
plot!(p[11], bottom_margin=10mm)
plot!(p[12], bottom_margin=10mm)

annotate!(p[11], 5, -3.8, L"Lyapunov Exponent Index, $i$", annotationfontsize=18)
annotate!(p[7], -1.3, 1.2, Plots.text("first $(length(le_length)) Lyapunov exponents averaged over data", 18, rotation=90, "Computer Modern", :black))

annotate!(p[1],  -2.7, -1.6, Plots.text(L"\textbf{r = 9}", 17, "Computer Modern", :black))
annotate!(p[4],  -2.7, -1.6, Plots.text(L"\textbf{r = 12}", 17, "Computer Modern", :black))
annotate!(p[7],  -2.7, -1.6, Plots.text(L"\textbf{r = 20}", 17, "Computer Modern", :black))
annotate!(p[10], -2.7, -1.6, Plots.text(L"\textbf{r = 24}", 17, "Computer Modern", :black))

# plot!(p, background_color=:transparent, background_color_inside="#44546A", dpi=600)
plot!(p, tickfontsize=11, dpi=600)
savefig(p, joinpath(FILEPATH, "plots/kse/kse_le_all.pdf"))


#=============================================================#
## Kaplan-Yorke dimension comparison for training and testing
#=============================================================#
lout = @layout [grid(1,3)]
p = plot(layout=lout, size=(1840, 500))

# Reference values
edson = [0.043, 0.003, 0.002, -0.004, -0.008, -0.185, -0.253, -0.296, -0.309, -1.965]
cvitanovic = [0.048, 0, 0, -0.003, -0.189, -0.256, -0.290, -0.310, -1.963, -1.967]

edson_ky = kaplan_yorke_dim(edson)
cvitanovic_ky = kaplan_yorke_dim(cvitanovic)

rol = length(REDUCTION_INFO["ro"])

# Training
# plot!(p[1], REDUCTION_INFO["ro"], cvitanovic_ky*ones(rol), c=color_choices[:cvita],  marker=(marker_choices[:cvita], markersize_choices[:cvita], color_choices[:cvita]), markerstrokecolor=color_choices[:cvita], lw=linewidth_choices[:cvita], ls=linestyle_choices[:cvita], markerstrokewidth=2.5, label="Cvitanovic")
# plot!(p[1], REDUCTION_INFO["ro"], edson_ky*ones(rol),      c=color_choices[:edson],  marker=(marker_choices[:edson], markersize_choices[:edson], color_choices[:edson]), markerstrokecolor=color_choices[:edson], lw=linewidth_choices[:edson], ls=linestyle_choices[:edson],                        label="Edson")
plot!(p[1], REDUCTION_INFO["ro"], cvitanovic_ky*ones(rol), c=color_choices[:cvita], lw=linewidth_choices[:cvita], ls=:solid, markerstrokewidth=2.5, label="Cvitanovic")
plot!(p[1], REDUCTION_INFO["ro"], edson_ky*ones(rol),      c=color_choices[:edson], lw=linewidth_choices[:edson], ls=:dash,                         label="Edson")
plot!(p[1], REDUCTION_INFO["ro"], TRAIN_RES["KY"][:int],   c=color_choices[:int],    marker=(marker_choices[:int],   markersize_choices[:int],   color_choices[:int]),   markerstrokecolor=color_choices[:int],   lw=linewidth_choices[:int],   ls=linestyle_choices[:int],   markerstrokewidth=2.5, label="Intrusive")
plot!(p[1], REDUCTION_INFO["ro"], TRAIN_RES["KY"][:LS],    c=color_choices[:LS],     marker=(marker_choices[:LS],    markersize_choices[:LS],    color_choices[:LS]),    markerstrokecolor=color_choices[:LS],    lw=linewidth_choices[:LS],    ls=linestyle_choices[:LS],                           label="OpInf")
plot!(p[1], REDUCTION_INFO["ro"], TRAIN_RES["KY"][:ephec], c=color_choices[:ephec],  marker=(marker_choices[:ephec], markersize_choices[:ephec], color_choices[:ephec]), markerstrokecolor=color_choices[:ephec], lw=linewidth_choices[:ephec], ls=linestyle_choices[:ephec],                        label="EP-OpInf")
# plot!(p[1], REDUCTION_INFO["ro"], TRAIN_RES["KY"][:epsic], c=color_choices[:epsic],  marker=(marker_choices[:epsic], markersize_choices[:epsic], color_choices[:epsic]), markerstrokecolor=color_choices[:epsic], lw=linewidth=choices[:epsic], ls=linestyle_choices[:epsic],                        label="EPSIC-OpInf")
# plot!(p[1], REDUCTION_INFO["ro"], TRAIN_RES["KY"][:epp],   c=color_choices[:epp],    marker=(marker_choices[:epp],   markersize_choices[:epp],   color_choices[:epp]),   markerstrokecolor=color_choices[:epp],   lw=linewidth_choices[:epp],   ls=linestyle_choices[:epp],                          label="EPP-OpInf")
plot!(p[1], 
    majorgrid=true, 
    legend=false,
    # xlabel="reduced model dimension " * L" r",
    # ylabel="Kaplan-Yorke Dimension",
    title=L"\mathbf{Training}",
    titlefontsize=20,
    fontfamily="Computer Modern", tickfontsize=13,
    xticks=REDUCTION_INFO["ro"][1]:2:REDUCTION_INFO["ro"][end],
    ylims=(3, 5.5),
    top_margin=5mm,
    bottom_margin=15mm,
    left_margin=18mm,
)

# Test 1
# plot!(p[2], REDUCTION_INFO["ro"], cvitanovic_ky*ones(rol), c=color_choices[:cvita],  marker=(marker_choices[:cvita], markersize_choices[:cvita], color_choices[:cvita]), markerstrokecolor=color_choices[:cvita], lw=linewidth_choices[:cvita], ls=linestyle_choices[:cvita], markerstrokewidth=2.5, label="Cvitanovic")
# plot!(p[2], REDUCTION_INFO["ro"], edson_ky*ones(rol),      c=color_choices[:edson],  marker=(marker_choices[:edson], markersize_choices[:edson], color_choices[:edson]), markerstrokecolor=color_choices[:edson], lw=linewidth_choices[:edson], ls=linestyle_choices[:edson],                        label="Edson")
plot!(p[2], REDUCTION_INFO["ro"], cvitanovic_ky*ones(rol), c=color_choices[:cvita], lw=linewidth_choices[:cvita], ls=:solid, markerstrokewidth=2.5, label="Cvitanovic")
plot!(p[2], REDUCTION_INFO["ro"], edson_ky*ones(rol),      c=color_choices[:edson], lw=linewidth_choices[:edson], ls=:dash,                         label="Edson")
plot!(p[2], REDUCTION_INFO["ro"], TEST1_RES["KY"][:int],   c=color_choices[:int],    marker=(marker_choices[:int],   markersize_choices[:int],   color_choices[:int]),   markerstrokecolor=color_choices[:int],   lw=linewidth_choices[:int],   ls=linestyle_choices[:int],   markerstrokewidth=2.5, label="Intrusive")
plot!(p[2], REDUCTION_INFO["ro"], TEST1_RES["KY"][:LS],    c=color_choices[:LS],     marker=(marker_choices[:LS],    markersize_choices[:LS],    color_choices[:LS]),    markerstrokecolor=color_choices[:LS],    lw=linewidth_choices[:LS],    ls=linestyle_choices[:LS],                           label="OpInf")
plot!(p[2], REDUCTION_INFO["ro"], TEST1_RES["KY"][:ephec], c=color_choices[:ephec],  marker=(marker_choices[:ephec], markersize_choices[:ephec], color_choices[:ephec]), markerstrokecolor=color_choices[:ephec], lw=linewidth_choices[:ephec], ls=linestyle_choices[:ephec],                        label="EP-OpInf")
# plot!(p[2], REDUCTION_INFO["ro"], TEST1_RES["KY"][:epsic], c=color_choices[:epsic],  marker=(marker_choices[:epsic], markersize_choices[:epsic], color_choices[:epsic]), markerstrokecolor=color_choices[:epsic], lw=linewidth=choices[:epsic], ls=linestyle_choices[:epsic],                        label="EPSIC-OpInf")
# plot!(p[2], REDUCTION_INFO["ro"], TEST1_RES["KY"][:epp],   c=color_choices[:epp],    marker=(marker_choices[:epp],   markersize_choices[:epp],   color_choices[:epp]),   markerstrokecolor=color_choices[:epp],   lw=linewidth_choices[:epp],   ls=linestyle_choices[:epp],                          label="EPP-OpInf")
plot!(p[2], 
    majorgrid=true, 
    legend=false,
    title=L"\mathbf{Test 1: Interpolation}",
    titlefontsize=20,
    xticks=REDUCTION_INFO["ro"][1]:2:REDUCTION_INFO["ro"][end],
    ylims=(3, 5.5),
    fontfamily="Computer Modern", guidefontsize=13, tickfontsize=13,
    top_margin=5mm,
    left_margin=-1mm,
    bottom_margin=18mm,
)

# Test 2
# plot!(p[3], REDUCTION_INFO["ro"], cvitanovic_ky*ones(rol), c=color_choices[:cvita],  marker=(marker_choices[:cvita], markersize_choices[:cvita], color_choices[:cvita]), markerstrokecolor=color_choices[:cvita], lw=linewidth_choices[:cvita], ls=linestyle_choices[:cvita], markerstrokewidth=2.5, label="Cvitanovic")
# plot!(p[3], REDUCTION_INFO["ro"], edson_ky*ones(rol),      c=color_choices[:edson],  marker=(marker_choices[:edson], markersize_choices[:edson], color_choices[:edson]), markerstrokecolor=color_choices[:edson], lw=linewidth_choices[:edson], ls=linestyle_choices[:edson],                        label="Edson")
plot!(p[3], REDUCTION_INFO["ro"], cvitanovic_ky*ones(rol), c=color_choices[:cvita], lw=linewidth_choices[:cvita], ls=:solid, markerstrokewidth=2.5, label="Cvitanovic")
plot!(p[3], REDUCTION_INFO["ro"], edson_ky*ones(rol),      c=color_choices[:edson], lw=linewidth_choices[:edson], ls=:dash,                         label="Edson")
plot!(p[3], REDUCTION_INFO["ro"], TEST2_RES["KY"][:int],   c=color_choices[:int],    marker=(marker_choices[:int],   markersize_choices[:int],   color_choices[:int]),   markerstrokecolor=color_choices[:int],   lw=linewidth_choices[:int],   ls=linestyle_choices[:int],   markerstrokewidth=2.5, label="Intrusive")
plot!(p[3], REDUCTION_INFO["ro"], TEST2_RES["KY"][:LS],    c=color_choices[:LS],     marker=(marker_choices[:LS],    markersize_choices[:LS],    color_choices[:LS]),    markerstrokecolor=color_choices[:LS],    lw=linewidth_choices[:LS],    ls=linestyle_choices[:LS],                           label="OpInf")
plot!(p[3], REDUCTION_INFO["ro"], TEST2_RES["KY"][:ephec], c=color_choices[:ephec],  marker=(marker_choices[:ephec], markersize_choices[:ephec], color_choices[:ephec]), markerstrokecolor=color_choices[:ephec], lw=linewidth_choices[:ephec], ls=linestyle_choices[:ephec],                        label="EP-OpInf")
# plot!(p[3], REDUCTION_INFO["ro"], TEST2_RES["KY"][:epsic], c=color_choices[:epsic],  marker=(marker_choices[:epsic], markersize_choices[:epsic], color_choices[:epsic]), markerstrokecolor=color_choices[:epsic], lw=linewidth=choices[:epsic], ls=linestyle_choices[:epsic],                        label="EPSIC-OpInf")
# plot!(p[3], REDUCTION_INFO["ro"], TEST2_RES["KY"][:epp],   c=color_choices[:epp],    marker=(marker_choices[:epp],   markersize_choices[:epp],   color_choices[:epp]),   markerstrokecolor=color_choices[:epp],   lw=linewidth_choices[:epp],   ls=linestyle_choices[:epp],                          label="EPP-OpInf")
plot!(p[3],
    majorgrid=true, 
    legend=:bottomright,
    title=L"\mathbf{Test 2: Extrapolation}",
    titlefontsize=20,
    xticks=REDUCTION_INFO["ro"][1]:2:REDUCTION_INFO["ro"][end],
    ylims=(3, 5.5),
    fontfamily="Computer Modern", guidefontsize=13, tickfontsize=13,  legendfontsize=15,
    top_margin=5mm,
    left_margin=-1mm,
    bottom_margin=18mm,
)
annotate!(p[1], 5.5, 4.1, Plots.text("avg. Kaplan-Yorke dimensions", 21, "Computer Modern", rotation=90, color=:black))
annotate!(p[2], 17, 2.5, Plots.text("reduced model dimension " * L"r", 21, "Computer Modern", color=:black))
# plot!(p, background_color=:transparent, background_color_inside="#44546A", dpi=600)
savefig(p, joinpath(FILEPATH, "plots/kse/kse_ky_all.pdf"))