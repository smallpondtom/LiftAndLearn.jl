"""
    plot_rse(rel_state_err::Matrix, rel_output_err::Matrix, r::Int, theme::CairoMakie.Attributes)

Plots the relative state and output errors of the intrusive, inferred, and streaming inferred models.

# Arguments
- `rel_state_err::Matrix`: The relative state errors.
- `rel_output_err::Matrix`: The relative output errors.
- `r::Int`: The number of reduced basis vectors.
- `theme::CairoMakie.Attributes`: The theme to use for the plot.

# Returns
A figure containing the plot.
"""
function plot_rse(rel_state_err::Matrix, rel_output_err::Matrix, r::Int, theme::CairoMakie.Attributes)
    n = size(rel_state_err, 2)
    with_theme(theme) do
        fig2 = Figure(fontsize=20, size=(1200,600))
        ax1 = Axis(fig2[1, 1], xlabel="r", ylabel="Relative Error", title="Relative State Error", yscale=log10)
        scatterlines!(ax1, 1:r, rel_state_err[:, 1], label="Intrusive")
        scatterlines!(ax1, 1:r, rel_state_err[:, 2], label="OpInf")
        if n == 3
            scatterlines!(ax1, 1:r, rel_state_err[:, 3], label="Streaming-OpInf")
        else
            scatterlines!(ax1, 1:r, rel_state_err[:, 3], label="TR-OpInf")
            scatterlines!(ax1, 1:r, rel_state_err[:, 4], label="Streaming-OpInf")
        end
        ax2 = Axis(fig2[1, 2], xlabel="r", ylabel="Relative Error", title="Relative Output Error", yscale=log10)
        l1 = scatterlines!(ax2, 1:r, rel_output_err[:, 1], label="Intrusive")
        l2 = scatterlines!(ax2, 1:r, rel_output_err[:, 2], label="OpInf")
        if n == 3
            l3 = scatterlines!(ax2, 1:r, rel_output_err[:, 3], label="Streaming-OpInf")
            labels = ["Intrusive", "OpInf", "Streaming-OpInf"]
            lines = [l1, l2, l3]
        else
            l3 = scatterlines!(ax2, 1:r, rel_output_err[:, 3], label="TR-OpInf")
            l4 = scatterlines!(ax2, 1:r, rel_output_err[:, 4], label="Streaming-OpInf")
            labels = ["Intrusive", "OpInf", "TR-OpInf", "Streaming-OpInf"]
            lines = [l1, l2, l3, l4]
        end
        Legend(fig2[2, 1:2], lines, labels,
                orientation=:horizontal, halign=:center, tellwidth=false, tellheight=true)
        return fig2
    end
end



"""
    plot_rse_per_stream(rel_state_err_stream::Matrix, rel_output_err_stream::Matrix, r_select::Vector{<:Int},
         theme::CairoMakie.Attributes; CONST_BATCH::Bool=false)

Plots the relative state and output errors of the intrusive, inferred, and streaming inferred models per stream update.

# Arguments
- `rel_state_err_stream::Matrix`: The relative state errors per stream update.
- `rel_output_err_stream::Matrix`: The relative output errors per stream update.
- `r_select::Vector{<:Int}`: The reduced basis vectors to plot.
- `theme::CairoMakie.Attributes`: The theme to use for the plot.
- `CONST_BATCH::Bool`: A boolean indicating if the batch size is constant.

# Returns
A figure containing the plot.
"""
function plot_rse_per_stream(rel_state_err_stream::Matrix, rel_output_err_stream::Matrix, r_select::Vector{<:Int},
         theme::CairoMakie.Attributes, num_of_batches::Int)
    M = num_of_batches

    with_theme(theme) do
        fig3 = Figure(fontsize=20, size=(1200,600))
        xtick_vals = (M > 30) ? (1:(M÷10):M) : (1:M)
        ax1 = Axis(fig3[1, 1], xlabel="Stream Update", ylabel="Relative Error", title="Relative State Error", yscale=log10, xticks=xtick_vals)
        ax2 = Axis(fig3[1, 2], xlabel="Stream Update", ylabel="Relative Error", title="Relative Output Error", yscale=log10, xticks=xtick_vals)
        min1 = minimum(x->isnan(x) ? Inf : x, rel_state_err_stream)
        min2 = minimum(x->isnan(x) ? Inf : x, rel_output_err_stream)
        ylims!(ax1, min1 / 10, 10^5)
        ylims!(ax2, min2 / 10, 10^5)
        lines_ = []
        labels_ = []
        for (j,ri) in enumerate(r_select)
            scatterlines!(ax1, 1:M, rel_state_err_stream[:,j])
            l = scatterlines!(ax2, 1:M, rel_output_err_stream[:,j])
            push!(lines_, l)
            push!(labels_, "r = $ri")
        end
        Legend(fig3[2, 1:2], lines_, labels_, orientation=:horizontal, halign=:center, tellwidth=false, tellheight=true)
        return fig3
    end
end



"""
    plot_error_acc_per_stream(err_state_acc_stream::Matrix, err_output_acc_stream::Matrix, 
        theme::CairoMakie.Attributes; CONST_BATCH::Bool=false)

Plots the state and output error factors per stream update.

# Arguments
- `err_state_acc_stream::Matrix`: The state error factors.
- `err_output_acc_stream::Matrix`: The output error factors.
- `theme::CairoMakie.Attributes`: The theme to use for the plot.
- `CONST_BATCH::Bool`: A boolean indicating if the batch size is constant.

# Returns
A figure containing the plot.
"""
function plot_error_acc_per_stream(err_state_acc_stream::Matrix, err_output_acc_stream::Matrix, 
        theme::CairoMakie.Attributes, num_of_batches::Int)
    M = num_of_batches

    with_theme(theme) do
        fig4 = Figure(size=(1200,600), fontsize=20)
        xtick_vals = (M > 30) ? (1:(M÷10):M) : (1:M)
        ax1 = Axis(fig4[1, 1], xlabel=L"Stream Update, $k$", 
                    ylabel="Factor",
                    title="State Error Factor", xticks=xtick_vals, yscale=log10)
        ax2 = Axis(fig4[1, 2], xlabel=L"Stream Update, $k$", 
                    ylabel="Factor",
                    title="Output Error Factor", xticks=xtick_vals, yscale=log10)
        scatterlines!(ax1, 2:M, err_state_acc_stream[2:end,1])
        l1 = scatterlines!(ax2, 2:M, err_output_acc_stream[2:end,1])
        scatterlines!(ax1, 2:M, err_state_acc_stream[2:end,2])
        l2 = scatterlines!(ax2, 2:M, err_output_acc_stream[2:end,2])
        scatterlines!(ax1, 2:M, 10 .^ mean(log10.(err_state_acc_stream), dims=2)[2:end,1], linestyle=:dash, linewidth=2)
        l3 = scatterlines!(ax2, 2:M, 10 .^ mean(log10.(err_output_acc_stream), dims=2)[2:end,1], linestyle=:dash, linewidth=2)
        Legend(fig4[2, 1:2], [l1, l2, l3], [L"Upper Bound: $\Vert \mathbf{I}-\mathbf{K}_k\mathbf{D}_k\Vert_2$", 
                L"Lower Bound: $\sigma_{\text{min}}(\mathbf{I}-\mathbf{K}_k\mathbf{D}_k)$", "Mean"],
                orientation=:horizontal, halign=:center, tellwidth=false, tellheight=true)
        return fig4
    end
end



"""
    plot_error_cond(err_state_cond::Vector, err_output_cond::Vector, 
        theme::CairoMakie.Attributes; CONST_BATCH::Bool=false)

Plots the condition number of the error factors per stream update.

# Arguments
- `err_state_cond::Vector`: The state error factors.
- `err_output_cond::Vector`: The output error factors.
- `theme::CairoMakie.Attributes`: The theme to use for the plot.
- `CONST_BATCH::Bool`: A boolean indicating if the batch size is constant.

# Returns
A figure containing the plot.
"""
function plot_error_condition(err_state_cond::Vector, err_output_cond::Vector, 
        theme::CairoMakie.Attributes; CONST_BATCH::Bool=false)
    M = CONST_BATCH ? size(X,2)÷batchsize : length(batchsize)
    with_theme(theme) do
        fig4 = Figure(size=(1200,600), fontsize=20)
        xtick_vals = CONST_BATCH ? (1:M) : (1:(M÷10):M)
        ax1 = Axis(fig4[1, 1], xlabel=L"Stream Update, $k$", 
                    ylabel=L"\kappa(\mathbf{I}-\mathbf{K}_k\mathbf{D}_k)",
                    title="State Error Factor", xticks=xtick_vals, yscale=log10)
        ax2 = Axis(fig4[1, 2], xlabel=L"Stream Update, $k$", 
                    ylabel=L"\kappa(\mathbf{I}-\mathbf{K}_{y_k}\hat{\mathbf{X}}_k)",
                    title="Output Error Factor", xticks=xtick_vals, yscale=log10)
        scatterlines!(ax1, 2:M, err_state_cond[2:end])
        scatterlines!(ax2, 2:M, err_output_cond[2:end])
        return fig4
    end
end




"""
    plot_initial_error(batchsizes::Union{AbstractArray{<:Int},Int}, initial_errs::Matrix, initial_output_errs::Matrix, theme::CairoMakie.Attributes)

Plots the initial relative errors of the state and output matrices over different batch sizes.

# Arguments
- `batchsizes::Union{AbstractArray{<:Int},Int}`: The batch sizes to consider.
- `initial_errs::Array`: The initial relative errors of the state matrices.
- `initial_output_errs::Array`: The initial relative errors of the output matrices.
- `theme::CairoMakie.Attributes`: The theme to use for the plot.

# Returns
A figure containing the plot.
"""
function plot_initial_error(batchsizes::Union{AbstractArray{<:Int},Int}, initial_errs::Array, initial_output_errs::Array, theme::CairoMakie.Attributes,
                            orders::Union{Int,AbstractArray{<:Int}})
    with_theme(theme) do
        fig5 = Figure(fontsize=20, size=(1200,600))
        ax1 = Axis(fig5[1, 1], xlabel="batch-size", 
                    ylabel=L"\Vert \mathbf{O}_* - \mathbf{O}_0 \Vert_F ~/~ \Vert \mathbf{O}_* \Vert_F", 
                    title=L"Initial Relative Error of $\mathbf{O}_0$", yscale=log10)
        ax2 = Axis(fig5[1, 2], xlabel="batch-size", 
                    ylabel=L"\Vert \hat{\mathbf{C}}_* - \hat{\mathbf{C}}_0\Vert ~/~ \Vert \hat{\mathbf{C}}_* \Vert", 
                    title=L"Initial Relative Error of $\hat{\mathbf{C}}_0$", yscale=log10)

        colors = Makie.resample_cmap(:viridis, length(orders))
        lines = []
        labels = []
        for (i,r) in enumerate(orders)
            l = scatterlines!(ax1, batchsizes, initial_errs[:,i], color=colors[i])
            scatterlines!(ax2, batchsizes, initial_output_errs[:,i], color=colors[i])
            push!(labels, "r = $r")
            push!(lines, l)
        end
        Legend(fig5[1, end+1], lines, labels, orientation=:vertical, tellwidth=true, tellheight=true)
        return fig5
    end
end