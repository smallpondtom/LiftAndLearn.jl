# function kse_analyze_proj_err(model, X_all, Vr_all, IC, ro)
#     num_ic_params = length(IC)
#     PE_all = Array{Float64}(undef, length(ro), model.param_dim) 
#     PE = Array{Float64}(undef, num_ic_params)  
    
#     for i in eachindex(model.diffusion_coeffs)
#         prog = Progress(length(ro))
#         for (j,r) in enumerate(ro)
#             Vr = Vr_all[i][:, 1:r]
#             Threads.@threads for (ct, ic) in collect(enumerate(IC))
#                 PE[ct] = LnL.proj_error(X_all[i,ct], Vr)
#             end
#             PE_all[j, i] = mean(PE)
#         next!(prog)
#         end
#     end
#     return PE_all
# end

function kse_analyze_proj_err(model, datafiles, Vr_all, ro, num_ic_params)
    PE_all = Array{Float64}(undef, length(ro), model.param_dim) 
    PE = Array{Float64}(undef, num_ic_params)  
    
    for i in eachindex(model.diffusion_coeffs)
        prog = Progress(length(ro))
        for (j,r) in enumerate(ro)
            Vr = Vr_all[i][:, 1:r]

            # Threads.@threads for (ct,datafile) in collect(enumerate(datafiles))
            for (ct,datafile) in collect(enumerate(datafiles))
                jldopen(datafile, "r") do file
                    X = file["X"]
                    PE[ct] = LnL.proj_error(X, Vr)
                end
            end

            PE_all[j, i] = mean(PE)
        next!(prog)
        end
    end
    return PE_all
end

# function kse_analyze_rse(op, model, X_all, Vr_all, IC, ro, DS, integrator)
#     num_ic_params = length(IC)
#     # Relative state error
#     RSE_all = Array{Float64}(undef, length(ro), model.param_dim)
#     RSE = Array{Float64}(undef, num_ic_params)

#     for i in eachindex(model.diffusion_coeffs)
#         prog = Progress(length(ro))
#         for (j,r) in enumerate(ro)
#             Vr = Vr_all[i][:, 1:r]
#             Threads.@threads for (ct, ic) in collect(enumerate(IC))
#                 Fextract = UniqueKronecker.extractF(op[i].A2u, r)
#                 X = integrator(op[i].A[1:r, 1:r], Fextract, model.t, Vr' * ic)
#                 RSE[ct] = LnL.rel_state_error(X_all[i,ct][:, 1:DS:end], X[:, 1:DS:end], Vr)
#             end
#             RSE_all[j, i] = mean(RSE)
#             next!(prog)
#         end
#     end
#     return RSE_all
# end

function kse_analyze_rse(op, model, datafiles, Vr_all, num_ic_params, ro, integrator)
    # Relative state error
    RSE_all = Array{Float64}(undef, length(ro), model.param_dim)
    RSE = Array{Float64}(undef, num_ic_params)

    for i in eachindex(model.diffusion_coeffs)
        prog = Progress(length(ro))
        for (j,r) in enumerate(ro)
            Vr = Vr_all[i][:, 1:r]

            # Threads.@threads for (ct, datafile) in collect(enumerate(datafiles))
            for (ct, datafile) in collect(enumerate(datafiles))
                jldopen(datafile, "r") do file
                    Xtrain = file["X"]
                    IC = file["IC"]
                    DS = file["DS"]

                    Fextract = UniqueKronecker.extractF(op[i].A2u, r)
                    X = integrator(model.tspan, Vr' * IC; linear_matrix=op[i].A[1:r, 1:r], 
                                   quadratic_matrix=Fextract, system_input=false, const_stepsize=true)
                    RSE[ct] = LnL.rel_state_error(Xtrain[:, 1:DS:end], X[:, 1:DS:end], Vr)
                end
            end

            RSE_all[j, i] = mean(RSE)
            next!(prog)
        end
    end
    return RSE_all
end

function kse_analyze_cr(op, model, num_ic_params, ro)
    CR_all = Array{Float64}(undef, length(ro), model.param_dim)
    CR = Vector{Float64}(undef, num_ic_params)

    for i in eachindex(model.diffusion_coeffs)
        for (j,r) in collect(enumerate(ro))
            for ct in 1:num_ic_params
                Fextract = UniqueKronecker.extractF(op[i].A2u, r)
                CR[ct] =  LnL.ep_constraint_residual(Fextract, r)
            end
            CR_all[j, i] = mean(CR)
        end
    end
    return CR_all
end

function kse_fom_CR(op, model)
    CR = Array{Float64}(undef, model.param_dim)

    for i in 1:length(model.diffusion_coeffs)
        F_full = op[i].A2u
        CR[i] = LnL.ep_constraint_residual(F_full, size(F_full, 1))
    end
    return CR
end

function kse_analyze_autocorr(op, model, Vr_all, IC, ro, integrator, lags)

    # auto_correletion
    auto_correlation = Array{Array{Float64}}(undef, length(ro), model.param_dim)

    for i in eachindex(model.diffusion_coeffs)
        Threads.@threads for (j,r) in collect(enumerate(ro))
            Vr = Vr_all[i][:, 1:r]

            Fextract = UniqueKronecker.extractF(op[i].F, r)
            X = integrator(model.tspan, Vr' * IC, linear_matrix=op[i].A[1:r, 1:r], quadratic_matrix=Fextract, 
                            system_input=false, const_stepsize=true)
            Xrecon = Vr * X
            auto_correlation[j, i] = kse_tmean_autocorr(Xrecon, lags)
        end
    end
    return auto_correlation
end


# For one initial condition for full-order model
# function kse_analyze_autocorr(model::AbstractModel, X_all::AbstractArray, IC_idx::Int64, lags::AbstractArray)
#     # auto_correletion
#     auto_correlation = Array{Array{Float64}}(undef, model.Pdim)

#     Threads.@threads for i in eachindex(model.μs)
#         auto_correlation[i] = tmean_autocorr(X_all[i,IC_idx], lags)
#     end
#     return auto_correlation
# end


function kse_analyze_autocorr(model::AbstractModel, X_all::AbstractArray, lags::AbstractArray)
    # auto_correletion
    auto_correlation = Array{Array{Float64}}(undef, model.param_dim)

    Threads.@threads for i in eachindex(model.diffusion_coeffs)
        auto_correlation[i] = kse_tmean_autocorr(X_all, lags)
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