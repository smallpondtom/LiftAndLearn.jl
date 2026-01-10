function plot_rse(rse, roe, r, theme; provided_keys=[])
    n = length(rse)
    with_theme(theme) do
        fig = Figure(fontsize=20, size=(1200,600))
        # Relative State Error
        ax1 = Axis(fig[1, 1], 
            xlabel=L"reduced dimension, $r$",
            ylabel="Relative State Error", 
            title="Relative State Error", 
            yscale=log10
        )
        if isempty(provided_keys)
            for val in values(rse)
                scatterlines!(ax1, 1:r, val)
            end
        else
            for key in provided_keys
                scatterlines!(ax1, 1:r, rse[key])
            end
        end
        # Relative Output Error
        lines = []
        labels = []
        ax2 = Axis(fig[1, 2], 
            xlabel=L"reduced dimensions, $r$", 
            ylabel="Relative Output Error", 
            title="Relative Output Error", 
            yscale=log10
        )
        if isempty(provided_keys)
            for (key, values) in roe
                l = scatterlines!(ax2, 1:r, values)
                push!(lines, l)
                push!(labels, key)
            end
        else
            for key in provided_keys
                l = scatterlines!(ax2, 1:r, roe[key], label=key)
                push!(lines, l)
                push!(labels, key)
            end
        end
        Legend(fig[2, 1:2], 
            lines, labels,
            orientation=:horizontal, 
            halign=:center, 
            tellwidth=false, 
            tellheight=true
        )
        return fig
    end
end