function kse_analyze_proj_err(model, X_all, Vr_all, IC, ro)
    num_ic_params = length(IC)
    PE_all = Array{Float64}(undef, length(ro), model.Pdim) 
    PE = Array{Float64}(undef, num_ic_params)  
    
    for i in eachindex(model.μs)
        prog = Progress(length(ro))
        for (j,r) in enumerate(ro)
            Vr = Vr_all[i][:, 1:r]
            Threads.@threads for (ct, ic) in collect(enumerate(IC))
                PE[ct] = LnL.proj_error(X_all[i,ct], Vr)
            end
            PE_all[j, i] = mean(PE)
        next!(prog)
        end
    end
    return PE_all
end

function kse_analyze_rse(op, model, X_all, Vr_all, IC, ro, DS, integrator)
    num_ic_params = length(IC)
    # Relative state error
    RSE_all = Array{Float64}(undef, length(ro), model.Pdim)
    RSE = Array{Float64}(undef, num_ic_params)

    for i in eachindex(model.μs)
        prog = Progress(length(ro))
        for (j,r) in enumerate(ro)
            Vr = Vr_all[i][:, 1:r]
            Threads.@threads for (ct, ic) in collect(enumerate(IC))
                Fextract = UniqueKronecker.extractF(op[i].A2u, r)
                X = integrator(op[i].A[1:r, 1:r], Fextract, model.t, Vr' * ic)
                RSE[ct] = LnL.rel_state_error(X_all[i,ct][:, 1:DS:end], X[:, 1:DS:end], Vr)
            end
            RSE_all[j, i] = mean(RSE)
            next!(prog)
        end
    end
    return RSE_all
end

function kse_analyze_cr(op, model, IC, ro)
    num_ic_params = length(IC)
    CR_all = Array{Float64}(undef, length(ro), model.Pdim)
    CR = Vector{Float64}(undef, num_ic_params)

    for i in eachindex(model.μs)
        for (j,r) in collect(enumerate(ro))
            for (ct, ic) in enumerate(IC)
                Fextract = UniqueKronecker.extractF(op[i].A2u, r)
                CR[ct] =  LnL.ep_constraint_residual(Fextract, r)
            end
            CR_all[j, i] = mean(CR)
        end
    end
    return CR_all
end

function kse_fom_CR(op, model)
    CR = Array{Float64}(undef, model.Pdim)

    for i in 1:length(model.μs)
        F_full = op[i].A2u
        CR[i] = LnL.EPConstraintResidual(F_full, size(F_full, 1))
    end
    return CR
end

function kse_analyze_autocorr(op, model, Vr_all, IC, ro, integrator, lags)

    # auto_correletion
    auto_correlation = Array{Array{Float64}}(undef, length(ro), model.Pdim)

    for i in eachindex(model.μs)
        Threads.@threads for (j,r) in collect(enumerate(ro))
            Vr = Vr_all[i][:, 1:r]

            Fextract = LnL.extractF(op[i].F, r)
            X = integrator(model.tspan, Vr' * IC, linear_matrix=op[i].A[1:r, 1:r], quadratic_matrix=Fextract, 
                            system_input=false, const_stepsize=true)
            Xrecon = Vr * X
            auto_correlation[j, i] = tmean_autocorr(Xrecon, lags)
        end
    end
    return auto_correlation
end


# For one initial condition for full-order model
function kse_analyze_autocorr(model::AbstractModel, X_all::AbstractArray, IC_idx::Int64, lags::AbstractArray)
    # auto_correletion
    auto_correlation = Array{Array{Float64}}(undef, model.Pdim)

    Threads.@threads for i in eachindex(model.μs)
        auto_correlation[i] = tmean_autocorr(X_all[i,IC_idx], lags)
    end
    return auto_correlation
end


function kse_tmean_autocorr(X::AbstractArray, lags::AbstractVector)
    N, K = size(X)
    M = length(lags)
    Cx = zeros((N, M))
    
    for i in 1:N  # normalzied autocorrelation
        Cx[i,:] = autocor(X[i,:], lags)
    end
    return vec(mean(Cx, dims=1))
end
;