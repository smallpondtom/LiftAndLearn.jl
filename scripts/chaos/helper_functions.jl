## Function to compute the Lyapunov exponent and Kaplan-Yorke dimension for one initial condition
function compute_LE_oneIC!(RES, type, keys, model, op, IC, Vr, ro, integrator, jacobian, options, idx)
    for i in eachindex(model.μs)
        for (j, r) in collect(enumerate(ro))
            Ar = op[i].A[1:r,1:r]
            Hr = LnL.extractH(op[i].H, r)
            Fr = LnL.extractF(op[i].F, r)
            op_tmp = LnL.Operators(A=Ar, H=Hr, F=Fr)
            if options.history
                _, foo = CG.lyapunovExponentJacobian(op_tmp, integrator, jacobian, Vr[i][:,1:r]' * IC, options)
                RES[keys[1]][type][j,i,idx] = foo
                RES[keys[2]][type][j,i,idx] = CG.kaplanYorkeDim(foo[:,end]; sorted=false)
            else
                foo = lyapunovExponentJacobian(op_tmp, integrator, jacobian, Vr[i][:,1:r]' * IC, options)
                RES[keys[1]][type][j,i,idx] = foo[:,end]
                RES[keys[2]][type][j,i,idx] = CG.kaplanYorkeDim(foo; sorted=false)
            end
            @info "Reduced order of $(r) completed..."
        end
        @debug "Loop $(i) out of $(model.Pdim) completed..."
    end
end

function compute_LE_allIC!(RES, type, keys, model, op, ICs, Vr, ro, integrator, jacobian, options)
    for (idx, IC) in collect(enumerate(ICs))
        compute_LE_oneIC!(RES, type, keys, model, op, IC, Vr, ro, integrator, jacobian, options, idx)
        @info "Initial condition $(idx) out of $(length(ICs)) completed..."
    end
end

# FOM dispatch
function compute_LE_oneIC!(RES, type, keys, model, op, IC, integrator, options, idx)
    for i in eachindex(model.μs)    
        if options.history
            _, foo = CG.lyapunovExponent(op[i], integrator, IC, options)
            RES[keys[1]][type][i,idx] = foo
            RES[keys[2]][type] = CG.kaplanYorkeDim(foo[:,end]; sorted=false)
        else
            foo = lyapunovExponent(op[i], integrator, IC, options)
            RES[keys[1]][type][i,idx] = foo
            RES[keys[2]][type] = CG.kaplanYorkeDim(foo; sorted=false)
        end
        @debug "Loop $(i) out of $(model.Pdim) completed..."

    end
end

function compute_LE_allIC!(RES, type, keys, model, op, ICs, integrator, options)
    for (idx, IC) in collect(enumerate(ICs))
        compute_LE_oneIC!(RES, type, keys, model, op, IC, integrator, options, idx)
        @info "Initial condition $(idx) out of $(length(ICs)) completed..."
    end
end


# Normal Distribution
using Distributions: Normal, pdf
using LaTeXStrings
using Plots
using Statistics
using StatsPlots

# Function to plot histogram with bell curve
function plot_dky_distribution(dky_data, fom_dky, ridx, title; bins=30, annote_loc=(3.3, 1.2))
    rom_dky = vec(dky_data[ridx,:,:])
    # Gather some info
    median_rom_dky = median(rom_dky)
    mean_rom_dky = mean(rom_dky)
    std_rom_dky = std(rom_dky)

    p1 = histogram(rom_dky, bins=bins, normed=true, alpha=0.6, label="")
    plot!(Normal(mean_rom_dky, std_rom_dky), label="", lw=2)
    vline!(p1, [median_rom_dky], color=:red, label="Median")
    # vline!(p1, [fom_dky], color=:black, label="Full")
    vline!(p1, [5.198], label="Edson et al.", linestyle=:dash)
    vline!(p1, [4.2381], label="Cvitanovic et al.", linestyle=:dash)
    vspan!(p1, [mean_rom_dky - std_rom_dky, mean_rom_dky + std_rom_dky], color=:green, alpha=0.1, label=L"\pm 1\sigma")
    plot!(p1, fontfamily="Computer Modern", legendfont=9, tickfont=12, guidefontsize=15,
        legend=:topleft, xlabel=L"D_{ky}", ylabel="Normal Distribution", title=title)
    annotate!(p1, annote_loc..., text("r = $(DATA["ro"][ridx])", 14, "Computer Modern"))
    display(p1)
end

function plot_LEmax_distribution(rom_LE, fom_LE, ridx, title; bins=30, annote_loc=(0.01, 30))
    # Gather some info
    LEmax = []
    _, _, n = size(rom_LE)
    for i in 1:n
        push!(LEmax, maximum(rom_LE[ridx,1,i][:,end]))
    end
    LEmax_fom = maximum(fom_LE[1][:,end])

    median_rom_LEmax = median(LEmax)
    mean_rom_LEmax = mean(LEmax)
    std_rom_LEmax = std(LEmax)

    p1 = histogram(LEmax, bins=bins, normed=true, alpha=0.6, label="")
    plot!(Normal(mean_rom_LEmax, std_rom_LEmax), label="", lw=2)
    vline!(p1, [median_rom_LEmax], color=:red, label="Median")
    # vline!(p1, [LEmax_fom], color=:black, label="Full")
    vline!(p1, [0.043], label="Edson et al.", linestyle=:dash)
    vline!(p1, [0.048], label="Cvitanovic et al.", linestyle=:dash)
    vspan!(p1, [mean_rom_LEmax - std_rom_LEmax, mean_rom_LEmax + std_rom_LEmax], color=:green, alpha=0.1, label=L"\pm 1\sigma")
    plot!(p1, fontfamily="Computer Modern", legendfont=9, tickfont=12, guidefontsize=15,
        legend=:topleft, xlabel=L"\lambda_{\text{max}}", ylabel="Normal Distribution", title=title)
    annotate!(p1, annote_loc..., text("r = $(DATA["ro"][ridx])", 14, "Computer Modern"))
    display(p1)
end

function plot_LE_convergence(LE_data, ridx, ICidx, C, title; ylimits=(1e-7, 2e+2), ytickvalues=10.0 .^ (-7:2:2))
    p = plot()
    data = LE_data[ridx,1,ICidx]
    m, n = size(data)
    for i in 1:m
        plot!(p, 
            (1:n-1)[1:100:end], 
            abs.(data[i,1:100:end-1] .- data[i,end]), 
            lw=1.5, label=false
        )
    end
    plot!(p, 1:n, C ./ (1:n), c=:black, ls=:dash, lw=1.5, label=L"C/{i}")
    plot!(p, 1:n, C ./ sqrt.(1:n), c=:red, ls=:dash, lw=1.5, label=L"C/\sqrt{i}")
    plot!(xscale=:log10, yscale=:log10)
    ylims!(ylimits...)
    xticks!(10 .^ (0:floor(Int, log10(n))))
    yticks!(ytickvalues)
    xlabel!(L"i" * "-th reorthonormalization step " * L"\mathrm{log}_{10} " * " scale")
    ylabel!(L"\mathrm{log}_{10}(|\lambda_i - \lambda_N|)")
    plot!(fontfamily="Computer Modern", guidefontsize=13, tickfontsize=13, legendfontsize=13, legend=:bottomleft)
    title!(title)
    annotate!(p, 1e+2, 1e-4, text("r = $(DATA["ro"][ridx])", 14, "Computer Modern"))
    display(p)
end


function plot_LEmax_error(data, ref, ro, title)
    p = plot()
    model_type = [:int, :LS, :ephec]
    r, _, n = size(data[model_type[1]])
    labels = ["Intru", "OpInf", "EP-OpInf"]
    errs = zeros(r,length(model_type))
    for (mi,model) in enumerate(model_type)
        for ri in 1:r
            err = 0.0
            for ni in 1:n
                err += abs(maximum(data[model][ri,1,ni][:,end]) - ref) / abs(ref)
            end
            errs[ri,mi] = err / n
        end
        plot!(p, ro, errs[:,mi], label=labels[mi], marker=true)
    end
    xticks!(ro)
    plot!(p, fontfamily="Computer Modern", legendfont=9, tickfont=12, guidefontsize=15,
        legend=:topright, xlabel=L"r", ylabel="Relative Error", title=title)
    display(p)
end


function plot_dky_error(data, ref, ro, title)
    p = plot()
    model_type = [:int, :LS, :ephec]
    r, _, n = size(data[model_type[1]])
    labels = ["Intru", "OpInf", "EP-OpInf"]
    errs = zeros(r,length(model_type))
    for (mi,model) in enumerate(model_type)
        for ri in 1:r
            err = 0.0
            for ni in 1:n
                err += abs(data[model][ri,1,ni] - ref) / abs(ref)
            end
            errs[ri,mi] = err / n
        end
        plot!(p, ro, errs[:,mi], label=labels[mi], marker=true)
    end
    xticks!(ro)
    ylims!(9e-3, 1e+0)
    yticks!(10.0 .^ (-3:1))
    plot!(p, yscale=:log10, fontfamily="Computer Modern", legendfont=9, tickfont=12, guidefontsize=15,
        legend=:topright, xlabel=L"r", ylabel="Relative Error", title=title)
    display(p)
end