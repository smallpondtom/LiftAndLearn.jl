function ep_analyze(Data,burger,options,X_all,IC,rmin,rmax; is_train=true)

    # Projection error
    PE_all = zeros(rmax - (rmin-1), burger.param_dim) 
    # Relative state error
    SE_all = Dict(
        :int => zeros(rmax-rmin+1, burger.param_dim),
        :LS => zeros(rmax-rmin+1, burger.param_dim),
        :ephec => zeros(rmax-rmin+1, burger.param_dim),
        :epsic => zeros(rmax-rmin+1, burger.param_dim),
        :epp => zeros(rmax-rmin+1, burger.param_dim),
    )
    # Constraint Residual
    CR_all = Dict(
        :int => Matrix{Float64}(undef, rmax - (rmin - 1), burger.param_dim),
        :LS => Matrix{Float64}(undef, rmax - (rmin - 1), burger.param_dim),
        :ephec => Matrix{Float64}(undef, rmax - (rmin - 1), burger.param_dim),
        :epsic => Matrix{Float64}(undef, rmax - (rmin - 1), burger.param_dim),
        :epp => Matrix{Float64}(undef, rmax - (rmin - 1), burger.param_dim),
        :fom => Vector{Float64}(undef, burger.param_dim)
    )

    # Load values
    op_LS = Data["op_LS"]
    op_int = Data["op_int"]
    op_ephec = Data["op_ephec"]
    op_epsic = Data["op_epsic"]
    op_epp = Data["op_epp"]
    Vrmax = Data["Vrmax"]

    num_ic_params = length(IC)

    @info "Analyze the operators with training data..."
    prog = Progress(burger.param_dim)
    for i in 1:length(burger.diffusion_coeffs)

        # Energy, constraint residual, and constraint violation of the FOM
        F_full = Data["op_fom_tr"][i].A2u
        CR_all[:fom][i] = LnL.ep_constraint_residual(F_full, size(F_full, 1), false; with_moment=false)
        
        Threads.@threads for (j,r) in collect(enumerate(rmin:rmax))
            Vr = Vrmax[i][:, 1:r]

            # Temporary data storage
            PE = Array{Float64}(undef, num_ic_params)  # projection error
            SE = Dict(
                :int => Array{Float64}(undef, num_ic_params),
                :LS => Array{Float64}(undef, num_ic_params),
                :ephec => Array{Float64}(undef, num_ic_params),
                :epsic => Array{Float64}(undef, num_ic_params),
                :epp => Array{Float64}(undef, num_ic_params),
            )
            CR = Dict(
                :int => Array{Float64}(undef, num_ic_params),
                :LS => Array{Float64}(undef, num_ic_params),
                :ephec => Array{Float64}(undef, num_ic_params),
                :epsic => Array{Float64}(undef, num_ic_params),
                :epp => Array{Float64}(undef, num_ic_params),
            )

            for (ct, ic) in enumerate(IC)

                # Integrate the LS operator inference model
                Finf_extract_LS = UniqueKronecker.extractF(op_LS[i].A2u, r)
                Xinf_LS = burger.integrate_model(burger.tspan, Vr' * ic; linear_matrix=op_LS[i].A[1:r, 1:r], quadratic_matrix=Finf_extract_LS, system_input=false)

                @assert !any(isnan, Xinf_LS) "NaNs in Xinf_LS!"

                # Integrate the intrusive model
                Fint_extract = UniqueKronecker.extractF(op_int[i].A2u, r)
                Xint = burger.integrate_model(burger.tspan, Vr' * ic; linear_matrix=op_int[i].A[1:r, 1:r], quadratic_matrix=Fint_extract, system_input=false)
                
                # Integrate the energy-preserving hard equality constraint operator inference model
                Finf_extract_ephec = UniqueKronecker.extractF(op_ephec[i].A2u, r)
                Xinf_ephec = burger.integrate_model(burger.tspan, Vr' * ic; linear_matrix=op_ephec[i].A[1:r, 1:r], quadratic_matrix=Finf_extract_ephec, system_input=false)
                
                # Integrate the energy-preserving soft inequality constraint operator inference model
                Finf_extract_epsic = UniqueKronecker.extractF(op_epsic[i].A2u, r)
                Xinf_epsic = burger.integrate_model(burger.tspan, Vr' * ic; linear_matrix=op_epsic[i].A[1:r, 1:r], quadratic_matrix=Finf_extract_epsic, system_input=false)

                # Integrate the energy-preserving unconstrained operator inference model
                Finf_extract_epp = UniqueKronecker.extractF(op_epp[i].A2u, r)
                Xinf_epp = burger.integrate_model(burger.tspan, Vr' * ic; linear_matrix=op_epp[i].A[1:r, 1:r], quadratic_matrix=Finf_extract_epp, system_input=false)

                # Compute the projection error
                PE[ct] = LnL.proj_error(X_all[i,ct], Vr)

                # Compute the relative state error
                SE[:LS][ct] = LnL.rel_state_error(X_all[i,ct][:, 1:DS:end], Xinf_LS[:, 1:DS:end], Vr)
                SE[:int][ct] = LnL.rel_state_error(X_all[i,ct][:, 1:DS:end], Xint[:, 1:DS:end], Vr)
                SE[:ephec][ct] = LnL.rel_state_error(X_all[i,ct][:, 1:DS:end], Xinf_ephec[:, 1:DS:end], Vr)
                SE[:epsic][ct] = LnL.rel_state_error(X_all[i,ct][:, 1:DS:end], Xinf_epsic[:, 1:DS:end], Vr)
                SE[:epp][ct] = LnL.rel_state_error(X_all[i,ct][:, 1:DS:end], Xinf_epp[:, 1:DS:end], Vr)

                # Compute the constraint residual and momentum
                CR[:LS][ct] =  LnL.ep_constraint_residual(Finf_extract_LS, r, false; with_moment=false)
                CR[:int][ct] = LnL.ep_constraint_residual(Fint_extract, r, false; with_moment=false)
                CR[:ephec][ct] = LnL.ep_constraint_residual(Finf_extract_ephec, r, false; with_moment=false)
                CR[:epsic][ct] = LnL.ep_constraint_residual(Finf_extract_epsic, r, false; with_moment=false)
                CR[:epp][ct] = LnL.ep_constraint_residual(Finf_extract_epp, r, false; with_moment=false)
            end

            # Compute errors
            PE_all[j, i] = mean(PE)
            SE_all[:LS][j, i] = mean(SE[:LS])
            SE_all[:int][j, i] = mean(SE[:int])
            SE_all[:ephec][j, i] = mean(SE[:ephec])
            SE_all[:epsic][j, i] = mean(SE[:epsic])
            SE_all[:epp][j, i] = mean(SE[:epp])
            
            # Compute the CR and momentum
            CR_all[:LS][j, i] = mean(CR[:LS])
            CR_all[:int][j, i] = mean(CR[:int])
            CR_all[:ephec][j, i] = mean(CR[:ephec])
            CR_all[:epsic][j, i] = mean(CR[:epsic])
            CR_all[:epp][j, i] = mean(CR[:epp])
        end
        next!(prog)
    end

    if is_train
        Data = Dict{String, Any}(Data)  # convert types to avoid errors
        Data["train_proj_err"] = PE_all
        Data["train_state_err"] = SE_all
        Data["train_CR"] = CR_all
        return Data
    else
        return SE_all
    end
end