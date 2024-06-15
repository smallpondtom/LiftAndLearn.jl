using CairoMakie
using LinearAlgebra
using LiftAndLearn: ChaosGizmo.kaplanYorkeDim
using Statistics: mean

## First 10 LE data
reduced_models = [0.0520139 0.0456246 0.0478746; 
                  0.0107592	0.00911459 0.0109336; 
                  -0.00634278 -0.00363656 -0.00753194; 
                  -0.0280198 -0.0199012 -0.0278293; 
                  -0.185905 -0.187354 -0.180279; 
                  -0.258883 -0.251936 -0.260319; 
                  -0.292268 -0.290557 -0.292757; 
                  -0.315636 -0.311713 -0.314898;
                  -1.95849 -1.96025 -1.95726; 
                  -1.961 -1.96342 -1.96178]
pod = reduced_models[:, 1]
opinf = reduced_models[:, 2]
epopinf = reduced_models[:, 3]
edson = [0.043, 0.003, 0.002, -0.004, -0.008, -0.185, -0.253, -0.296, -0.309, -1.965]
cvitanovic = [0.048, 0, 0, -0.003, -0.189, -0.256, -0.290, -0.310, -1.963, -1.967]

# Plot
with_theme(theme_latexfonts()) do
    fig1 = Figure(backgroundcolor=:transparent, size=(600, 650))
    ax = Axis(fig1[1, 1], 
            title = "First 10 Lyapunov Exponents", 
            xlabel = "LE Index", ylabel = "LE Value", 
            xticks = 1:10, yticks = -2:0.5:0.1,
            xlabelsize=25, ylabelsize=25,
            xlabelcolor=:white, ylabelcolor=:white, 
            xticklabelsize=20, yticklabelsize=20,
            xtickcolor=:white, ytickcolor=:white,
            xticklabelcolor=:white, yticklabelcolor=:white,
            titlesize=32, titlecolor=:white,
            backgroundcolor="#44546A"
            )
    hot_colors = Makie.resample_cmap(:hot, 5)
    markers = [:circle, :rect, :diamond, :cross, :xcross]
    l1 = scatterlines!(ax, 1:10, edson, linewidth=6, markersize=25, marker=markers[1], color=hot_colors[1])
    l2 = scatterlines!(ax, 1:10, cvitanovic, linewidth=5, markersize=24, marker=markers[2], color=hot_colors[2])
    l3 = scatterlines!(ax, 1:10, pod, linewidth=4, markersize=22, linestyle=:dash, marker=markers[3], color=hot_colors[3])
    l4 = scatterlines!(ax, 1:10, opinf, linewidth=3, markersize=14, linestyle=:dot, marker=markers[4], color=hot_colors[4])
    l5 = scatterlines!(ax, 1:10, epopinf, linewidth=2, markersize=12, linestyle=:dashdot, marker=markers[5], color=hot_colors[5])
    Legend(fig1[1, 2], [l1, l2, l3, l4, l5],
            ["Edson", "Cvitanovic", "POD", "OpInf", "Ep-OpInf"], 
            markersize = 20, labelsize = 25, labelcolor=:white, 
            backgroundcolor=:transparent)
            
    display(fig1)
end

## Kaplan Yorke dimension
edson_ky = kaplanYorkeDim(edson)
cvitanovic_ky = kaplanYorkeDim(cvitanovic)
KY = [
    6.50676	6.31747	5.88233;
    4.8042	0.0	3.855;
    4.71297	4.59819	4.6478;
    5.34132	4.70603	3.37984;
    5.04301	4.64404	4.27807;
    4.13512	5.29635	4.42903;
    4.48699	5.0486	4.69979
]
pod_ky = KY[:, 1]
opinf_ky = KY[:, 2]
epopinf_ky = KY[:, 3]

# Plot
with_theme(theme_latexfonts()) do
    fig2 = Figure(backgroundcolor=:transparent, size=(800, 650))
    rdims = [9,12,15,17,20,22,24]
    ax = Axis(fig2[1, 1], 
            title = L"$D_{KY}$ over reduced dimensions", 
            xlabel = L"$r$, reduced dimensions", ylabel = "KY Dimension", 
            xticks=rdims, yticks = 0:1:7,
            xlabelsize=25, ylabelsize=25,
            xlabelcolor=:white, ylabelcolor=:white, 
            xticklabelsize=20, yticklabelsize=20,
            xtickcolor=:white, ytickcolor=:white,
            xticklabelcolor=:white, yticklabelcolor=:white,
            titlesize=32, titlecolor=:white,
            backgroundcolor="#44546A"
            )
    hot_colors = Makie.resample_cmap(:hot, 5)
    markers = [:diamond, :cross, :xcross]
    l1 = hlines!(ax, edson_ky, color=hot_colors[1], linewidth=3, linestyle=:dash)
    l2 = hlines!(ax, cvitanovic_ky, color=hot_colors[2], linewidth=3, linestyle=:dash)
    l3 = scatterlines!(ax, rdims, pod_ky, linewidth=6, color=hot_colors[3], markersize=25, marker=markers[1])
    l4 = scatterlines!(ax, rdims, opinf_ky, linewidth=5, color=hot_colors[4], markersize=25, marker=markers[2])
    l5 = scatterlines!(ax, rdims, epopinf_ky, linewidth=4, color=hot_colors[5], markersize=25, marker=markers[3])
    # l6 = hlines!(ax, mean(pod_ky), color=hot_colors[3], linewidth=3)
    # l7 = hlines!(ax, mean(opinf_ky), color=hot_colors[4], linewidth=3)
    # l8 = hlines!(ax, mean(epopinf_ky), color=hot_colors[5], linewidth=3)
    Legend(fig2[1, 2], [l1, l2, l3, l4, l5],
            ["Edson", "Cvitanovic", "POD", "OpInf", "Ep-OpInf"], 
            markersize = 20, labelsize = 25, labelcolor=:white, 
            backgroundcolor=:transparent)
    display(fig2)
end
