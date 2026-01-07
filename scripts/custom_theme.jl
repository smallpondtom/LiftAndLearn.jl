# Palette 1: High Contrast (Bold & punchy)
palette_1 = [
    "#FDE74C",  # bright yellow (high visibility)
    "#9BFF00",  # lime green (vivid)
    "#2DE2E6",  # cyan (electric)
    "#C77DFF",  # vibrant violet (pop)
    "#FF3C38",  # bright red (alert)
]

# Palette 2: Cool Professional (Blues & Teals)
palette_2 = [
    "#C1DFF0",  # pale blue (backgrounds)
    "#35A7FF",  # bright sky blue (primary)
    "#1C1FFF",  # vivid blue (emphasis)
    "#4ACF1F",  # darker teal (secondary accent)
    "#00C2A8",  # teal (accent)
]

# Palette 3: Warm Energetic (Oranges & Reds)
palette_3 = [
    "#FDE74C",  # bright yellow (highlight)
    "#FFA62B",  # amber-orange (primary)
    "#F29E4C",  # orange-gold (secondary)
    "#F46036",  # deep orange-red (emphasis)
    "#E15554",  # coral red (accent)
]

# Palette 4: Vibrant Mix (Rainbow balanced)
palette_4 = [
    "#35A7FF",  # bright sky blue (cool)
    "#9B6BFF",  # soft violet (cool-neutral)
    "#6BF178",  # bright green (neutral)
    "#F4B942",  # warm amber (warm)
    "#FF5964",  # vivid red-pink (warm-accent)
]

# Palette 5: Soft Pastels (Light & airy)
palette_5 = [
    "#C1DFF0",  # pale blue
    "#C77DFF",  # vibrant violet (lighter end)
    "#FFD670",  # pastel yellow
    "#FF7F50",  # coral orange
    "#EAEAEA",  # soft white (neutral anchor)
]

# All palettes combined (25 colors total)
my_custom_dark_colors = vcat(palette_1, palette_2, palette_3, palette_4, palette_5)

my_custom_dark_theme = Theme(
    Figure = (
        backgroundcolor = CairoMakie.RGB(0, 48/255, 87/255),
    ),
    Axis = (
        backgroundcolor = CairoMakie.RGB(0, 48/255, 87/255),
        xgridcolor = Makie.RGBAf(1.0,1.0,1.0,0.2),
        ygridcolor = Makie.RGBAf(1.0,1.0,1.0,0.2),
        xgridvisible = true,
        ygridvisible = true,
        xticklabelcolor = :white,
        yticklabelcolor = :white,
        xlabelcolor = :white,
        ylabelcolor = :white,
        titlecolor = :white,
        leftspinevisible = true,
        rightspinevisible = false,
        topspinevisible = false,
        bottomspinevisible = true,
        leftspinecolor = :white,
        bottomspinecolor = :white,
        xtickcolor = :white,
        ytickcolor = :white,
    ),
    Legend = (
        backgroundcolor = CairoMakie.RGB(0, 48/255, 87/255),
        labelcolor = :white,
        framevisible = false,
    ),
    palette = (
        color = my_custom_dark_colors,
        patchcolor = my_custom_dark_colors,
    ),
    Lines = (
        cycle = Cycle([:color], covary=true),
    ),
    Scatter = (
        cycle = Cycle([:color, :marker], covary=true),
    ),
    Colorbar = (
        labelcolor = :white,
        ticklabelcolor = :white,
    ),
)

my_custom_dark_theme = merge(theme_latexfonts(), my_custom_dark_theme)

function show_palette(colors; title="Color Palette")
    n = length(colors)
    fig = Figure(size=(800, 100 + 60*n))
    ax = Axis(fig[1,1],
        title=title,
        titlesize=24,
        aspect=DataAspect(),
        yreversed=true
    )
    
    # Draw colored rectangles
    for (i, col) in enumerate(colors)
        poly!(ax, Rect(0, i-1, 4, 0.8), color=col)
        text!(ax, 4.2, i-0.5, text=col * " ($(i))", 
              fontsize=18, align=(:left, :center))
    end
    
    xlims!(ax, -0.5, 10)
    ylims!(ax, -0.5, n-0.2)
    hidedecorations!(ax)
    hidespines!(ax)
    
    return fig
end

"""
    themed_figure(plot_fn; use_custom=true, kwargs...)

Create a figure with automatic theme switching and execute plotting function.

# Arguments
- `plot_fn`: A function that takes a `Figure` as input and performs plotting
- `dark`: Boolean flag to use dark theme (default: false)
- `kwargs...`: Additional keyword arguments passed to `Figure` constructor

# Returns
- The created `Figure` object

# Example
```julia
fig = themed_figure(dark=true, size=(1100,600)) do fig
    ax = Axis(fig[1,1], xlabel="x", ylabel="y", title="My Plot")
    scatter!(ax, 1:10, rand(10))
    Legend(fig[1,2], ax, "Data")
end
```
"""
function themed_figure(plot_fn; dark=true, kwargs...)
    theme = dark ? my_custom_dark_theme : theme_latexfonts()
    
    fig = with_theme(theme) do
        fig = Figure(; 
            backgroundcolor=dark ? 
                            my_custom_dark_theme.Figure.backgroundcolor : 
                            :white,
            kwargs...)
        plot_fn(fig)
        fig
    end
    
    return fig
end

## Show your custom palette
# palette_colors = my_custom_dark_theme.palette.color[]
# fig = show_palette(palette_colors, title="Custom Theme Color Palette")
# display(fig)