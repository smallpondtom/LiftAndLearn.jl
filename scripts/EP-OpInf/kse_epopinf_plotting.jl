"""
Plotting all results for KSE EP-OpInf example
"""

#================#
## Load packages
#================#
using FileIO
using JLD2
using Plots
using LaTeXStrings
using LinearAlgebra
using Plots
using Plots.PlotMeasures
using UniqueKronecker

#================#
## Load the data
#================#
DATA = load("./scripts/EP-OpInf/data/kse_epopinf_data.jld2")
RES = load("./scripts/EP-OpInf/data/kse_epopinf_results.jld2")
TEST_RES = load("./scripts/EP-OpInf/data/kse_epopinf_test_results.jld2")
KSE = DATA["KSE"]

#=========================#
## Plot projection errors
#=========================#
mean_train_proj_err = mean(RES["train_proj_err"], dims=2)
plot(DATA["ro"], mean_train_proj_err, marker=(:rect), fontfamily="Computer Modern")
plot!(yscale=:log10, majorgrid=true, legend=false)
yticks!([1e-1, 1e-2, 1e-3, 1e-4])
xticks!(DATA["ro"][1]:2:DATA["ro"][end])
xlabel!("reduced model dimension " * L"r")
ylabel!("projection error")
plot!(guidefontsize=16, tickfontsize=13,  legendfontsize=13)

#=============================#
## Plot relative state errors
#=============================#
mean_LS_state_err = mean(RES["train_state_err"][:LS], dims=2)
mean_int_state_err = mean(RES["train_state_err"][:int], dims=2)
mean_ephec_state_err = mean(RES["train_state_err"][:ephec], dims=2)
mean_epsic_state_err = mean(RES["train_state_err"][:epsic], dims=2)
mean_epp_state_err = mean(RES["train_state_err"][:epp], dims=2)

plot!(DATA["ro"], mean_int_state_err, c=:orange, marker=(:cross, 10, :orange), markerstrokewidth=2.5, label="Intrusive")
plot(DATA["ro"], mean_LS_state_err, c=:crimson, marker=(:circle, 5, :crimson), markerstrokecolor=:red3, label="OpInf")
# plot!(DATA["ro"], mean_ephec_state_err, c=:blue, markerstrokecolor=:blue, marker=(:rect, 3), ls=:dash, lw=2, label="EP-OpInf")
plot!(DATA["ro"], mean_ephec_state_err, c=:blue, markerstrokecolor=:blue, marker=(:rect, 3), ls=:dash, lw=2, label="EPHEC-OpInf")
plot!(DATA["ro"], mean_epsic_state_err, c=:purple, markerstrokecolor=:purple, marker=(:dtriangle, 5), ls=:dot, label="EPSIC-OpInf")
plot!(DATA["ro"], mean_epp_state_err, c=:brown, markerstrokecolor=:brown, marker=(:star, 5), lw=2, ls=:dashdot, label="EPP-OpInf")
plot!(majorgrid=true, legend=:topright)
# yticks!([1e-0, 1e-1])
xticks!(DATA["ro"][1]:2:DATA["ro"][end])
xlabel!("reduced model dimension " * L" r")
ylabel!("average relative state error")
title!("Training")
plot!(guidefontsize=16, tickfontsize=13, legendfontsize=13, fontfamily="Computer Modern")

#============================#
## Plot constraint residuals
#============================#
mean_LS_CR_tr = mean(RES["train_CR"][:LS], dims=2)
mean_int_CR_tr = mean(RES["train_CR"][:int], dims=2)
mean_ephec_CR_tr = mean(RES["train_CR"][:ephec], dims=2)
mean_epsic_CR_tr = mean(RES["train_CR"][:epsic], dims=2)
mean_epp_CR_tr = mean(RES["train_CR"][:epp], dims=2)

plot(DATA["ro"], mean_int_CR_tr, c=:orange, marker=(:cross, 10), markerstrokewidth=2.5, label="Intrusive")
plot!(DATA["ro"], mean_LS_CR_tr, marker=(:circle, 5), c=:crimson, markerstrokecolor=:crimson, lw=2, label="OpInf")
# plot!(DATA["ro"], mean_ephec_CR_tr, c=:blue, markerstrokecolor=:blue, marker=(:rect, 3), lw=2, ls=:dash, label="EP-OpInf")
plot!(DATA["ro"], mean_ephec_CR_tr, c=:blue, markerstrokecolor=:blue, marker=(:rect, 3), lw=2, ls=:dash, label="EPHEC-OpInf")
plot!(DATA["ro"], mean_epsic_CR_tr, c=:purple, markerstrokecolor=:purple, marker=(:dtriangle, 5), ls=:dot, label="EPSIC-OpInf")
plot!(DATA["ro"], mean_epp_CR_tr, c=:brown, markerstrokecolor=:brown, marker=(:star, 5), ls=:dashdot, lw=1, label="EPP-OpInf")
plot!(yscale=:log10, majorgrid=true, legend=:right, minorgridalpha=0.03)
yticks!(10.0 .^ [-15, -12, -9, -6, -3, 0, 3])
xticks!(DATA["ro"][1]:2:DATA["ro"][end])
xlabel!("reduced model dimension " * L" r")
ylabel!("energy-preserving constraint violation")
plot!(xlabelfontsize=16, ylabelfontsize=13, tickfontsize=13, legendfontsize=14, fontfamily="Computer Modern")

#=================================#
## Training flow field comparison
#=================================#
i = 1  # <--- Choose one candidate from training data
r = DATA["ro"][end]
ic = DATA["IC_train"][i]
Vr = DATA["Vr"][1][:, 1:r]

tspan = collect(KSE.time_domain[1]:KSE.Δt:KSE.time_domain[2])
# X_fom = KSE.integrate_FD(DATA["op_fom_tr"][1].A, DATA["op_fom_tr"][1].F, tspan, ic)
X_fom = DATA["Xtr_all"][i];
X_int = KSE.integrate_FD(OPS["op_int"][1].A[1:r, 1:r], UniqueKronecker.extractF(OPS["op_int"][1].F, r), tspan, Vr' * ic)
X_LS = KSE.integrate_FD(OPS["op_LS"][1].A[1:r, 1:r], UniqueKronecker.extractF(OPS["op_LS"][1].F, r), tspan, Vr' * ic)
X_ephec = KSE.integrate_FD(OPS["op_ephec"][1].A[1:r, 1:r], UniqueKronecker.extractF(OPS["op_ephec"][1].F, r), tspan, Vr' * ic)
# X_epsic = KSE.integrate_FD(OPS["op_epsic"][1].A[1:r, 1:r], UniqueKronecker.extractF(OPS["op_epsic"][1].F, r), KSE.t, Vr' * ic)
# X_epp = KSE.integrate_FD(OPS["op_epp"][1].A[1:r, 1:r], UniqueKronecker.extractF(OPS["op_epp"][1].F, r), KSE.t, Vr' * ic)

## 

# lout = @layout [a{0.3h}; [grid(2,2)]]
lout = @layout [grid(4,2)]
p_fom = plot(
    contourf(tspan[1:DS:end], KSE.x, X_fom[:, 1:DS:end], lw=0, color=:inferno), 
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
    contourf(tspan[1:DS:end], KSE.x, Vr * X_int[:, 1:DS:end], lw=0, color=:inferno), 
    # contourf(KSE.t[1:DS:end], KSE.x, Vr * X_int[:, 1:DS:end], lw=0), 
    colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
    #right_margin=-3mm, left_margin=3mm, top_margin=-2mm,
    left_margin=15mm,
    ylabel=L"\textbf{Intrusive}" * "\n" * L"\omega",
    xformatter=_->"",
)
p_int_err = plot(
    contourf(tspan[1:DS:end], KSE.x, abs.(X_fom[:, 1:DS:end] .- Vr * X_int[:, 1:DS:end]), lw=0, color=:roma), 
    # contourf(KSE.t[1:DS:end], KSE.x, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_int[:, 1:DS:end]), lw=0, color=:roma), 
    yticks=(0:5:20), ylims=(0,22), colorbar_ticks=(0:0.5:5), clim=(0,5), 
    # left_margin=-3mm, right_margin=-3mm,
    xformatter=_->"", yformatter=_->"",
)
p_LS = plot(
    contourf(tspan[1:DS:end], KSE.x, Vr * X_LS[:, 1:DS:end], lw=0, color=:inferno), 
    # contourf(KSE.t[1:DS:end], KSE.x, Vr * X_LS[:, 1:DS:end], lw=0), 
    colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
    #right_margin=-3mm, left_margin=3mm, top_margin=-2mm,
    left_margin=15mm,
    ylabel=L"\textbf{OpInf}" * "\n" * L"\omega",
    xformatter=_->"",
)
p_LS_err = plot(
    contourf(tspan[1:DS:end], KSE.x, abs.(X_fom[:, 1:DS:end] .- Vr * X_LS[:, 1:DS:end]), lw=0, color=:roma), 
    # contourf(KSE.t[1:DS:end], KSE.x, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_LS[:, 1:DS:end]), lw=0, color=:roma), 
    yticks=(0:5:20), ylims=(0,22),  colorbar_ticks=(0:0.5:5), clim=(0,5), 
    # left_margin=-3mm, right_margin=-3mm, 
    xformatter=_->"", yformatter=_->"",
)
p_ephec = plot(
    contourf(tspan[1:DS:end], KSE.x, Vr * X_ephec[:, 1:DS:end], lw=0, color=:inferno), 
    # contourf(KSE.t[1:DS:end], KSE.x, Vr * X_ephec[:, 1:DS:end], lw=0), 
    colorbar_ticks=(-3:3), yticks=(0:5:20), ylims=(0,22), 
    #right_margin=-3mm, left_margin=3mm,  bottom_margin=7mm, top_margin=-2mm,
    bottom_margin=15mm, left_margin=15mm,
    ylabel=L"\textbf{EP}\rm{-}\textbf{OpInf}" * "\n" * L"\omega", xlabel=L"t" * "\n" * L"\textbf{Predicted}~" * L"x(\omega,t)",
)
p_ephec_err = plot(
    contourf(tspan[1:DS:end], KSE.x, abs.(X_fom[:, 1:DS:end] .- Vr * X_ephec[:, 1:DS:end]), lw=0, color=:roma), 
    # contourf(KSE.t[1:DS:end], KSE.x, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_ephec[:, 1:DS:end]), lw=0, color=:roma), 
    yticks=(0:5:20), ylims=(0,22), colorbar_ticks=(0:0.5:5), clim=(0,5),
    # left_margin=-3mm, right_margin=-3mm, 
    bottom_margin=15mm,
    xlabel=L"t" * "\n" * L"\textbf{Error}",
    yformatter=_->"",
)
# p_epsic = plot(contourf(KSE.t[1:DS:end], KSE.x, Vr * X_epsic[:, 1:DS:end], lw=0), colorbar_ticks=(-3:3), clim=(-3,3), yticks=(0:5:20), ylims=(0,20))
# p_epsic_err = plot(contourf(KSE.t[1:DS:end], KSE.x, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_epsic[:, 1:DS:end]), lw=0, color=:roma), yticks=(0:5:20), ylims=(0,20), colorbar_ticks=(0:0.5:5), clim=(0,5))
# p_epp = plot(contourf(KSE.t[1:DS:end], KSE.x, Vr * X_epp[:, 1:DS:end], lw=0), colorbar_ticks=(-3:3), clim=(-3,3), yticks=(0:5:20), ylims=(0,20))
# p_epp_err = plot(contourf(KSE.t[1:DS:end], KSE.x, abs.(DATA["Xtr_all"][i][:, 1:DS:end] .- Vr * X_epp[:, 1:DS:end]), lw=0, color=:roma), yticks=(0:5:20), ylims=(0,20), colorbar_ticks=(0:0.5:5), clim=(0,5))

p = plot(
    p_fom, pblank, 
    p_int, p_int_err,
    p_LS, p_LS_err,
    p_ephec, p_ephec_err,
    # p_epsic, p_epsic_err,
    # p_epp, p_epp_err, 
    fontfamily="Computer Modern", layout=lout, 
    size=(2000, 1080),
    guidefontsize=25, tickfontsize=17, plot_titlefontsize=30, plot_titlefontcolor=:white,
    plot_title="Predicted Flow Fields and Errors of Training",
)
plot!(p, background_color=:transparent, background_color_inside=:transparent, dpi=600)
# savefig(p, "figures/kse_train_ff.png")

#====================================================================#
## Plot normalized autocorrelation function for training and testing
#====================================================================#
# lag for autocorrelation
lags = DATA["AC_lags"]
lags_t = collect(lags) .* KSE.Δt
idx = length(lags_t)

lout = @layout [grid(4,3)]
p = plot(layout=lout, size=(1400, 1000), dpi=3000)

train_indices = [1, 4, 7, 10]
test1_indices = [2, 5, 8, 11]
test2_indices = [3, 6, 9, 12]

for (plot_id, ri) in enumerate([1, 2, 5, 7])
    # Training
    plot!(p[train_indices[plot_id]], lags_t[1:idx], RES["train_AC"][:fom][1:idx], c=:black, lw=2)
    plot!(p[train_indices[plot_id]], lags_t[1:idx], RES["train_AC"][:int][:,ri][1:idx], c=:orange, ls=:dot, lw=2)
    plot!(p[train_indices[plot_id]], lags_t[1:idx], RES["train_AC"][:LS][:,ri][1:idx], c=:firebrick3, ls=:dash, lw=2)
    plot!(p[train_indices[plot_id]], lags_t[1:idx], RES["train_AC"][:ephec][:,ri][1:idx], c=:blue3, ls=:dashdot, lw=2)
    # plot!(p[train_indices[plot_id]], lags_t[1:idx], RES["train_AC"][:epsic][:,ri][1:idx], c=:purple, ls=:dashdot, lw=2)
    # plot!(p[train_indices[plot_id]], lags_t[1:idx], RES["train_AC"][:epp][:,ri][1:idx], c=:brown, ls=:dashdot, lw=2)
    plot!(p[train_indices[plot_id]], fontfamily="Computer Modern", tickfontsize=11, legend=false)
    ylims!(p[train_indices[plot_id]], (-0.2, 1.05))
    yticks!(p[train_indices[plot_id]], [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if ri != 7
        plot!(p[train_indices[plot_id]], xformatter=_->"", bottom_margin=-1mm)
    end

    # Test 1
    plot!(p[test1_indices[plot_id]], lags_t[1:idx], TEST_RES["test1_AC"][:fom][1][1:idx], c=:black, lw=2)
    plot!(p[test1_indices[plot_id]], lags_t[1:idx], TEST_RES["test1_AC"][:int][ri][1:idx], c=:orange, ls=:dot, lw=2)
    plot!(p[test1_indices[plot_id]], lags_t[1:idx], TEST_RES["test1_AC"][:LS][ri][1:idx], c=:firebrick3, ls=:dash, lw=2)
    plot!(p[test1_indices[plot_id]], lags_t[1:idx], TEST_RES["test1_AC"][:ephec][ri][1:idx], c=:blue3, ls=:dashdot, lw=2)
    # plot!(p[test1_indices[plot_id]], lags_t[1:idx], TEST_RES["test1_AC"][:epsic][ri][1:idx], c=:purple, ls=:dashdot, lw=2)
    # plot!(p[test1_indices[plot_id]], lags_t[1:idx], TEST_RES["test1_AC"][:epp][ri][1:idx], c=:brown, ls=:dashdot, lw=2)
    plot!(p[test1_indices[plot_id]], yformatter=_->"", left_margin=-6mm)
    ylims!(p[test1_indices[plot_id]], (-0.2, 1.05))
    yticks!(p[test1_indices[plot_id]], [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if ri != 7
        plot!(p[test1_indices[plot_id]], xformatter=_->"", bottom_margin=-1mm)
    end

    # Test 2
    plot!(p[test2_indices[plot_id]], lags_t[1:idx], TEST_RES["test2_AC"][:fom][1][1:idx], c=:black, lw=2, label="Full")
    plot!(p[test2_indices[plot_id]], lags_t[1:idx], TEST_RES["test2_AC"][:int][ri][1:idx], c=:orange, ls=:dot, lw=2, label="Intrusive")
    plot!(p[test2_indices[plot_id]], lags_t[1:idx], TEST_RES["test2_AC"][:LS][ri][1:idx], c=:firebrick3, lw=2, ls=:dash, label="OpInf")
    plot!(p[test2_indices[plot_id]], lags_t[1:idx], TEST_RES["test2_AC"][:ephec][ri][1:idx], c=:blue3, lw=2, ls=:dashdot, label="EP-OpInf")
    # plot!(p[test2_indices[plot_id]], lags_t[1:idx], TEST_RES["test2_AC"][:epsic][ri][1:idx], c=:purple, lw=2, ls=:dashdot, label="EP-OpInf")
    # plot!(p[test2_indices[plot_id]], lags_t[1:idx], TEST_RES["test2_AC"][:epp][ri][1:idx], c=:brown, lw=2, ls=:dashdot, label="EP-OpInf")
    plot!(p[test2_indices[plot_id]], yformatter=_->"", left_margin=-6mm)
    ylims!(p[test2_indices[plot_id]], (-0.2, 1.05))
    yticks!(p[test2_indices[plot_id]], [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if ri !== 7
        plot!(p[test2_indices[plot_id]], xformatter=_->"", bottom_margin=-1mm)
    end

end
plot!(p, legend=false)
plot!(p[12], legend=:topright,  legend_font_pointsize=15)

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
annotate!(p[7], -40, 1.2, Plots.text("average normalized autocorrelation", 18, rotation=90, "Computer Modern", :white))

annotate!(p[1],  -67, 0.4, Plots.text(L"\textbf{r = 9}", 17, "Computer Modern", :white))
annotate!(p[4],  -67, 0.4, Plots.text(L"\textbf{r = 12}", 17, "Computer Modern", :white))
annotate!(p[7],  -67, 0.4, Plots.text(L"\textbf{r = 20}", 17, "Computer Modern", :white))
annotate!(p[10], -67, 0.4, Plots.text(L"\textbf{r = 24}", 17, "Computer Modern", :white))

# plot!(p, background_color=:transparent, background_color_inside="#44546A", dpi=600)
plot!(p, tickfontsize=11)
# savefig(p, "figures/kse_ac4_all_vert.png")

#======================================================================#
## Normalized autocorrelation function errors for training and testing
#======================================================================#
lout = @layout [grid(1,3)]
p = plot(layout=lout, size=(1840, 450))

# Training
plot!(p[1], DATA["ro"], RES["train_AC_ERR"][:int], c=:orange, marker=(:cross, 10, :orange), markerstrokewidth=2.5, label="Intrusive")
plot!(p[1], DATA["ro"], RES["train_AC_ERR"][:LS], c=:firebrick3, marker=(:circle, 5, :firebrick3), markerstrokecolor=:firebrick3, lw=2, label="OpInf")
plot!(p[1], DATA["ro"], RES["train_AC_ERR"][:ephec], c=:blue3, markerstrokecolor=:blue3, marker=(:rect, 3), ls=:dash, lw=2, label="EP-OpInf")
# plot!(p[1], DATA["ro"], RES["train_AC_ERR"][:ephec], c=:blue3, markerstrokecolor=:blue3, marker=(:rect, 3), ls=:dash, lw=2, label="EPHEC-OpInf")
# plot!(p[1], DATA["ro"], RES["train_AC_ERR"][:epsic], c=:purple, markerstrokecolor=:purple, marker=(:dtriangle, 5), ls=:dot, lw=2, label="EPSIC-OpInf")
# plot!(p[1], DATA["ro"], RES["train_AC_ERR"][:epp], c=:brown, markerstrokecolor=:brown, marker=(:star, 5), ls=:dashdot, lw=2, label="EPP-OpInf")
plot!(p[1], 
    majorgrid=true, 
    legend=false,
    # xlabel="reduced model dimension " * L" r",
    # ylabel="avg normalized autocorrelation error",
    title=L"\mathbf{Training}",
    titlefontsize=20,
    fontfamily="Computer Modern", tickfontsize=13,
    xticks=DATA["ro"][1]:2:DATA["ro"][end],
    ylims=(0.2, 0.75),
    bottom_margin=15mm,
    left_margin=20mm,
)

# Test 1
plot!(p[2], DATA["ro"], TEST_RES["test1_AC_ERR"][:int], c=:orange, marker=(:cross, 10, :orange), markerstrokewidth=2.5, label="Intrusive")
plot!(p[2], DATA["ro"], TEST_RES["test1_AC_ERR"][:LS], c=:firebrick3, marker=(:circle, 5, :firebrick3), markerstrokecolor=:firebrick3, lw=2, label="OpInf")
plot!(p[2], DATA["ro"], TEST_RES["test1_AC_ERR"][:ephec], c=:blue3, markerstrokecolor=:blue3, marker=(:rect, 3), ls=:dash, lw=2, label="EP-OpInf", yformatter=_->"")
# plot!(p[2], DATA["ro"], TEST_RES["test1_AC_ERR"][:ephec], c=:blue3, markerstrokecolor=:blue3, marker=(:rect, 3), ls=:dash, lw=2, label="EPHEC-OpInf", yformatter=_->"")
# plot!(p[2], DATA["ro"], TEST_RES["test1_AC_ERR"][:epsic], c=:purple, markerstrokecolor=:purple, marker=(:dtriangle, 5), ls=:dot, lw=2, label="EPSIC-OpInf")
# plot!(p[2], DATA["ro"], TEST_RES["test1_AC_ERR"][:epp], c=:brown, markerstrokecolor=:brown, marker=(:star, 5), ls=:dashdot, lw=2, label="EPP-OpInf")
plot!(p[2], 
    majorgrid=true, 
    legend=false,
    title=L"\mathbf{Test 1: Interpolation}",
    titlefontsize=20,
    xticks=DATA["ro"][1]:2:DATA["ro"][end],
    ylims=(0.2, 0.75),
    fontfamily="Computer Modern", guidefontsize=13, tickfontsize=13,
    left_margin=-8mm,
    bottom_margin=15mm,
)

# Test 2
plot!(p[3], DATA["ro"], TEST_RES["test2_AC_ERR"][:int], c=:orange, marker=(:cross, 10, :orange), markerstrokewidth=2.5, label="Intrusive")
plot!(p[3], DATA["ro"], TEST_RES["test2_AC_ERR"][:LS], c=:firebrick3, marker=(:circle, 5, :firebrick3), markerstrokecolor=:firebrick3, lw=2, label="OpInf")
plot!(p[3], DATA["ro"], TEST_RES["test2_AC_ERR"][:ephec], c=:blue3, markerstrokecolor=:blue3, marker=(:rect, 3), ls=:dash, lw=2, label="EP-OpInf", yformatter=_->"")
# plot!(p[3], DATA["ro"], TEST_RES["test2_AC_ERR"][:ephec], c=:blue3, markerstrokecolor=:blue3, marker=(:rect, 3), ls=:dash, lw=2, label="EPHEC-OpInf", yformatter=_->"")
# plot!(p[3], DATA["ro"], TEST_RES["test2_AC_ERR"][:epsic], c=:purple, markerstrokecolor=:purple, marker=(:dtriangle, 5), ls=:dot, lw=2, label="EPSIC-OpInf")
# plot!(p[3], DATA["ro"], TEST_RES["test2_AC_ERR"][:epp], c=:brown, markerstrokecolor=:brown, marker=(:star, 5), ls=:dashdot, lw=2, label="EPP-OpInf")
plot!(p[3],
    majorgrid=true, 
    legend=:bottomleft,
    title=L"\mathbf{Test 2: Extrapolation}",
    titlefontsize=20,
    xticks=DATA["ro"][1]:2:DATA["ro"][end],
    ylims=(0.2, 0.75),
    fontfamily="Computer Modern", guidefontsize=13, tickfontsize=13,  legendfontsize=16,
    left_margin=-8mm,
    bottom_margin=15mm,
)
annotate!(p[1], 5.8, 0.45, Plots.text("avg normalized \n autocorrelation error", 21, "Computer Modern", rotation=90, color=:white))
annotate!(p[2], 17, 0.1, Plots.text("reduced model dimension " * L"r", 21, "Computer Modern", color=:white))
# plot!(p, background_color=:transparent, background_color_inside="#44546A", dpi=600)
# savefig(p, "figures/kse_ac_err_all.png")