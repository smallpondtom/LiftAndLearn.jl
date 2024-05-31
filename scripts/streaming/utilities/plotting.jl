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


function plot_rse_per_stream(rel_state_err_stream, rel_output_err_stream, 
                             streaming_error, streaming_error_output,
                             r_select, num_of_streams; ylimits=([1e-6, 1e1], [1e-6, 1e1]))
    axis_colors = Makie.categorical_colors(:seaborn_bright, 2)
    with_theme(theme_latexfonts()) do
        fig = Figure(size=(1500,900))
        if num_of_streams > 20
            xtick_vals = 0:(num_of_streams÷5):num_of_streams
        else
            xtick_vals = 0:num_of_streams
        end
        lines_ = []
        labels_ = []
        axes = []
        for (j,ri) in enumerate(r_select)
            push!(axes, Axis(fig[1, j], 
                xlabel=L"$k$-th stream", 
                ylabel=L"\Vert \mathbf{X}_{\mathrm{true}}-\mathbf{X}_{\mathrm{recon}}\Vert_F / \Vert\mathbf{X}_{\mathrm{true}}\Vert_F", 
                title=L"Relative State Error & Streaming Error, $r = %$ri$", 
                yscale=log10, xticks=xtick_vals, yticklabelcolor=axis_colors[1]
            ))
            push!(axes, Axis(fig[1, j], 
                ylabel=L"\Vert\mathcal{E}_k\Vert_F=\Vert(\mathbf{I}-\mathbf{K}_k\mathbf{D}_k)\mathcal{E}_{k-1}\Vert_F", 
                yticklabelcolor=axis_colors[2], yaxisposition=:right, yscale=log10, ygridstyle=:dash
            ))
            hidespines!(axes[4*(j-1)+2])
            hidexdecorations!(axes[4*(j-1)+2])
            push!(axes, Axis(fig[2, j], 
                xlabel=L"$k$-th stream", 
                ylabel=L"\Vert\mathbf{Y}_{\mathrm{true}}-\mathbf{Y}_{\mathrm{recon}}\Vert_F / \Vert\mathbf{Y}_{\mathrm{true}}\Vert_F", 
                title=L"Relative Output Error & Streaming Error, $r = %$ri$", 
                yscale=log10, xticks=xtick_vals, yticklabelcolor=axis_colors[1]
            ))
            push!(axes, Axis(fig[2, j],
                ylabel=L"\Vert\mathcal{E}_{y_k}\Vert_F=\Vert(\mathbf{I}-\mathbf{K}_{y_k}\hat{\mathbf{X}}_k^\top)\mathcal{E}_{y_{k-1}}\Vert_F", 
                yticklabelcolor=axis_colors[2], yaxisposition=:right, yscale=log10, ygridstyle=:dash
            ))
            hidespines!(axes[4*(j-1)+4])
            hidexdecorations!(axes[4*(j-1)+4])

            ylims!(axes[4*(j-1)+1], ylimits[1]...)
            ylims!(axes[4*(j-1)+2], ylimits[1]...)
            ylims!(axes[4*(j-1)+3], ylimits[2]...)
            ylims!(axes[4*(j-1)+4], ylimits[2]...)

            l = scatterlines!(axes[4*(j-1)+1], 1:num_of_streams, rel_state_err_stream[ri], color=axis_colors[1])
            scatterlines!(axes[4*(j-1)+2], 1:num_of_streams, streaming_error[ri], color=axis_colors[2])
            scatterlines!(axes[4*(j-1)+3], 1:num_of_streams, rel_output_err_stream[ri], color=axis_colors[1])
            scatterlines!(axes[4*(j-1)+4], 1:num_of_streams, streaming_error_output[ri], color=axis_colors[2])
            push!(lines_, l)
            push!(labels_, "r = $ri")
        end
        # Legend(fig[3, 1:end], lines_, labels_, "Reduced Dimensions", 
        #        orientation=:horizontal, halign=:center, tellwidth=false, tellheight=true)
        return fig
    end
end




function plot_streaming_error(stream_error, stream_error_output, 
                              true_stream_error, true_stream_error_output,
                              r_select, num_of_streams, theme, DS=1)
    with_theme(theme) do 
        fig = Figure(size=(1600,700), fontsize=20)
        if num_of_streams > 20
            xtick_vals = 0:(num_of_streams÷5):num_of_streams
        else
            xtick_vals = 0:num_of_streams
        end
        ax1 = Axis(fig[1,1],
            xlabel=L"$k$-th stream",
            ylabel=L"\Vert\mathcal{E}_k\Vert_F=\Vert(\mathbf{I}-\mathbf{K}_k\mathbf{D}_k)\mathcal{E}_{k-1}\Vert_F",
            title="Streaming Error of States",
            xticks=xtick_vals, yscale=log10
        )
        ax2 = Axis(fig[2,1],
            xlabel=L"$k$-th stream",
            ylabel=L"\Vert\mathcal{E}_{y_k}\Vert_F=\Vert(\mathbf{I}-\mathbf{K}_{y_k}\hat{\mathbf{X}}_k^\top)\mathcal{E}_{y_{k-1}}\Vert_F",
            title="Streaming Error of Outputs",
            xticks=xtick_vals, yscale=log10
        )
        ax3 = Axis(fig[1,2],
            xlabel=L"$k$-th stream",
            ylabel=L"\Vert\mathcal{E}_k\Vert_F=\Vert\mathbf{O}_* - \mathbf{O}_k\Vert_F",
            title="True Streaming Error of States",
            xticks=xtick_vals, yscale=log10
        )
        ax4 = Axis(fig[2,2],
            xlabel=L"$k$-th stream",
            ylabel=L"\Vert\mathcal{E}_{y_k}\Vert_F=\Vert\hat{\mathbf{C}}_* - \hat{\mathbf{C}}_k\Vert_F",
            title="True Streaming Error of Outputs",
            xticks=xtick_vals, yscale=log10
        )
        colors = Makie.resample_cmap(:viridis, length(r_select))
        lines = []
        labels = []
        for (i,r) in enumerate(r_select)
            l = scatterlines!(ax1, 1:DS:num_of_streams, stream_error[r][1:DS:end], color=colors[i])
            scatterlines!(ax2, 1:DS:num_of_streams, stream_error_output[r][1:DS:end], color=colors[i])
            scatterlines!(ax3, 1:DS:num_of_streams, true_stream_error[r][1:DS:end], color=colors[i])
            scatterlines!(ax4, 1:DS:num_of_streams, true_stream_error_output[r][1:DS:end], color=colors[i])
            push!(lines, l)
            push!(labels, "r = $r")
        end
        Legend(fig[1:2, end+1], lines, labels, "Reduced \n Dimension", 
               orientation=:vertical, tellwidth=true, tellheight=true)
        return fig
    end
end


function plot_errorfactor_condition(err_state_cond, err_output_cond, r_select, num_of_streams, theme, DS=1)
    with_theme(theme) do
        fig = Figure(size=(1300,550), fontsize=20)
        if num_of_streams > 20
            xtick_vals = 0:(num_of_streams÷5):num_of_streams
        else
            xtick_vals = 0:num_of_streams
        end
        ax1 = Axis(fig[1, 1], xlabel=L"$k$-th stream", 
                    ylabel=L"\kappa(\mathbf{I}-\mathbf{K}_k\mathbf{D}_k)",
                    title="Condition Number of \n State Streaming Error Factor", xticks=xtick_vals, yscale=log10)
        ax2 = Axis(fig[1, 2], xlabel=L"$k$-th stream", 
                    ylabel=L"\kappa(\mathbf{I}-\mathbf{K}_{y_k}\hat{\mathbf{X}}_k^\top)",
                    title="Condition Number of \n Output Streaming Error Factor", xticks=xtick_vals, yscale=log10)
        colors = Makie.resample_cmap(:viridis, length(r_select))
        lines = []
        labels = []
        for (i, r) in enumerate(r_select)
            l = scatterlines!(ax1, 1:DS:num_of_streams, err_state_cond[r][1:DS:end], color=colors[i])
            scatterlines!(ax2, 1:DS:num_of_streams, err_output_cond[r][1:DS:end], color=colors[i])
            push!(lines, l)
            push!(labels, "r = $r")
        end
        Legend(fig[1, end+1], lines, labels, "Reduced \n Dimension", 
               orientation=:vertical, tellwidth=true, tellheight=true)
        return fig
    end
end



function plot_initial_error(streamsizes, initial_errs, initial_output_errs, theme, r_select)
    with_theme(theme) do
        fig = Figure(fontsize=20, size=(1300,500))
        ax1 = Axis(fig[1, 1], xlabel="stream-size", 
                    ylabel=L"\Vert \mathbf{O}_* - \mathbf{O}_0 \Vert_F ~/~ \Vert \mathbf{O}_* \Vert_F", 
                    title=L"Relative Error of $\mathbf{O}_0$", yscale=log10)
        ax2 = Axis(fig[1, 2], xlabel="stream-size", 
                    ylabel=L"\Vert \hat{\mathbf{C}}_* - \hat{\mathbf{C}}_0\Vert ~/~ \Vert \hat{\mathbf{C}}_* \Vert", 
                    title=L"Relative Error of $\hat{\mathbf{C}}_0$", yscale=log10)

        colors = Makie.resample_cmap(:viridis, length(r_select))
        lines = []
        labels = []
        for (i,r) in enumerate(r_select)
            l = scatterlines!(ax1, streamsizes, initial_errs[:,i], color=colors[i])
            scatterlines!(ax2, streamsizes, initial_output_errs[:,i], color=colors[i])
            push!(labels, "r = $r")
            push!(lines, l)
        end
        Legend(fig[1, end+1], lines, labels, "Reduced \n Dimension", 
               orientation=:vertical, tellwidth=true, tellheight=true)
        return fig
    end
end




# """
#     plot_error_acc_per_stream(err_state_acc_stream::Matrix, err_output_acc_stream::Matrix, 
#         theme::CairoMakie.Attributes; CONST_BATCH::Bool=false)

# Plots the state and output error factors per stream update.

# # Arguments
# - `err_state_acc_stream::Matrix`: The state error factors.
# - `err_output_acc_stream::Matrix`: The output error factors.
# - `theme::CairoMakie.Attributes`: The theme to use for the plot.
# - `CONST_BATCH::Bool`: A boolean indicating if the batch size is constant.

# # Returns
# A figure containing the plot.
# """
# function plot_error_acc_per_stream(err_state_acc_stream::Matrix, err_output_acc_stream::Matrix, 
#         theme::CairoMakie.Attributes, num_of_batches::Int)
#     M = num_of_batches

#     with_theme(theme) do
#         fig4 = Figure(size=(1200,600), fontsize=20)
#         xtick_vals = (M > 30) ? (1:(M÷10):M) : (1:M)
#         ax1 = Axis(fig4[1, 1], xlabel=L"Stream Update, $k$", 
#                     ylabel="Factor",
#                     title="State Error Factor", xticks=xtick_vals, yscale=log10)
#         ax2 = Axis(fig4[1, 2], xlabel=L"Stream Update, $k$", 
#                     ylabel="Factor",
#                     title="Output Error Factor", xticks=xtick_vals, yscale=log10)
#         scatterlines!(ax1, 2:M, err_state_acc_stream[2:end,1])
#         l1 = scatterlines!(ax2, 2:M, err_output_acc_stream[2:end,1])
#         scatterlines!(ax1, 2:M, err_state_acc_stream[2:end,2])
#         l2 = scatterlines!(ax2, 2:M, err_output_acc_stream[2:end,2])
#         scatterlines!(ax1, 2:M, 10 .^ mean(log10.(err_state_acc_stream), dims=2)[2:end,1], linestyle=:dash, linewidth=2)
#         l3 = scatterlines!(ax2, 2:M, 10 .^ mean(log10.(err_output_acc_stream), dims=2)[2:end,1], linestyle=:dash, linewidth=2)
#         Legend(fig4[2, 1:2], [l1, l2, l3], [L"Upper Bound: $\Vert \mathbf{I}-\mathbf{K}_k\mathbf{D}_k\Vert_2$", 
#                 L"Lower Bound: $\sigma_{\text{min}}(\mathbf{I}-\mathbf{K}_k\mathbf{D}_k)$", "Mean"],
#                 orientation=:horizontal, halign=:center, tellwidth=false, tellheight=true)
#         return fig4
#     end
# end