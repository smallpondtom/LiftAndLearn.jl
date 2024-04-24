ace_light = CairoMakie.merge(Theme(
    fontsize = 20,
    backgroundcolor = "#F2F2F2",
    Axis = (
        backgroundcolor = "#F2F2F2",
        xlabelsize = 20, xlabelpaddingg=-5,
        xgridstyle = :dash, ygridstyle = :dash,
        xtickalign = 1, ytickalign = 1,
        xticksize = 10, yticksize = 10,
        rightspinevisible = false, topspinevisible = false,
    ),
    Legend = (
        backgroundcolor = "#F2F2F2",
    ),
), theme_latexfonts())
