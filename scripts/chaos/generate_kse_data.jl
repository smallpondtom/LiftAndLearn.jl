function generate_kse_data(KSE, u0, ic_a, ic_b, num_ic_params, DS, PRUNE_DATA, prune_idx)
    # Store values
    if KSE.Pdim != 1
        Xtr = Vector{Matrix{Float64}}(undef, KSE.Pdim)  # training state data 
        Rtr = Vector{Matrix{Float64}}(undef, KSE.Pdim)  # training derivative data
        V = Vector{Matrix{Float64}}(undef, KSE.Pdim)  # POD basis
        Σ = Vector{Vector{Float64}}(undef, KSE.Pdim)  # singular values 
        Xtr_all = Matrix{Matrix{Float64}}(undef, KSE.Pdim, num_ic_params)  # all training Data
        training_IC = Vector{Vector{Float64}}(undef, num_ic_params)  # all initial conditions 
    else
        Xtr_all = Vector{Matrix{Float64}}(undef, num_ic_params)  # all training Data
        training_IC = Vector{Vector{Float64}}(undef, num_ic_params)  # all initial conditions 
    end

    @info "Generate the FOM system matrices and training data."
    @showprogress for i in eachindex(KSE.μs)
        μ = KSE.μs[i]

        # Generate the FOM system matrices (ONLY DEPENDS ON μ)
        A, F = KSE.model_FD(KSE, μ)
        op_fom_tr[i] = LnL.operators(A=A, F=F)

        # Store the training data 
        Xall = Vector{Matrix{Float64}}(undef, num_ic_params)
        Xdotall = Vector{Matrix{Float64}}(undef, num_ic_params)
        
        # Generate the data for all combinations of the initial condition parameters
        ic_combos = collect(Iterators.product(ic_a, ic_b))
        prog = Progress(length(ic_combos))
        Threads.@threads for (j, ic) in collect(enumerate(ic_combos))
            a, b = ic
            states = KSE.integrate_FD(A, F, KSE.t, u0(a,b))
            if PRUNE_DATA
                if KSE.Pdim != 1
                    Xtr_all[i,j] = states[:, prune_idx-1:end]
                else
                    Xtr_all[j] = states[:, prune_idx-1:end]
                end

                state_tp1 = @view states[:, prune_idx:end]
                state_t = @view states[:, prune_idx-1:end-1]
                Xall[j] = state_tp1[:, 1:DS:end]  # downsample data
                tmp = (state_tp1 - state_t) / KSE.Δt
                Xdotall[j] = tmp[:, 1:DS:end]  # downsample data

                if i == 1
                    training_IC[j] = states[:, prune_idx-1]
                end
            else
                if KSE.Pdim != 1
                    Xtr_all[i,j] = states
                else
                    Xtr_all[j] = states
                end

                state_tp1 = @view states[:, 2:end]
                state_t = @view states[:, 1:end-1]
                Xall[j] = state_tp1[:, 1:DS:end]  # downsample data
                tmp = (state_tp1 - state_t) / KSE.Δt
                Xdotall[j] = tmp[:, 1:DS:end]  # downsample data

                if i == 1
                    training_IC[j] = u0(a, b)
                end
            end

            next!(prog)
        end

        # Combine all initial condition data to form on big training data matrix
        if KSE.Pdim != 1
            Xtr[i] = reduce(hcat, Xall) 
            Rtr[i] = reduce(hcat, Xdotall)    # Compute the POD basis from the training data
            tmp = svd(@views Xtr[i])
            V[i] = tmp.U
            Σ[i] = tmp.S
        else
            Xtr = reduce(hcat, Xall)
            Rtr = reduce(hcat, Xdotall)
            tmp = svd(Xtr)
            V = tmp.U
            Σ = tmp.S
        end
    end

    return Xtr, Rtr, V, Σ, Xtr_all, training_IC
end