ace_light = CairoMakie.merge(Theme(
    fontsize = 20,
    backgroundcolor = "#FFFFFF",
    Axis = (
        backgroundcolor = "#FFFFFF",
        xlabelsize = 20, xlabelpaddingg=-5,
        xgridstyle = :dash, ygridstyle = :dash,
        xtickalign = 1, ytickalign = 1,
        xticksize = 10, yticksize = 10,
        rightspinevisible = false, topspinevisible = false,
    ),
    Legend = (
        backgroundcolor = "#FFFFFF",
    ),
), theme_latexfonts())