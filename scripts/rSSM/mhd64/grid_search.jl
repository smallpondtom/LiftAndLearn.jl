using Statistics
include(joinpath(FILEPATH, "integrate.jl"))

function simulate_opinf(x0, n_time, op, tspan=nothing)
    states, final_idx = rk4_integrate(x0, tspan, op.A, op.A2u, op.A3u, op.K)
    contains_nan = final_idx < n_time ? true : false
    return contains_nan, states, final_idx
end

function find_best_opinf_model(reg_pairs, Xhat, Xhatdot, n_time, n_time_pred, 
                               max_growth, opinf_options, tspan=nothing)

    @assert options.with_reg == true "Regularization must be enabled in options."
    
    best_train_err = 1e20
    best_beta1, best_beta2, best_beta3 = nothing, nothing, nothing
    best_final_idx = 0
    Xtilde_opt = nothing
    eval_time_opt = nothing
    best_model = nothing

    mean_Xhat = mean(Xhat, dims=2)
    max_diff_Xhat = maximum(abs.(Xhat .- mean_Xhat), dims=2)
    ct = 0
    tot = length(reg_pairs)
    
    # Loop over all regularization pairs
    for (beta1, beta2, beta3) in reg_pairs
        ct += 1
        
        # Construct a regularizer that penalizes the linear and constant reduced
        # operators using beta1 and the quadratic operator using beta2
        reg = LnL.TikhonovParameter(A=beta1, A2=beta2, A3=beta3, K=beta1)
        opinf_options.λ = reg
        
        # Solve the regularized OpInf problem
        ops = LnL.opinf(Xhat, opinf_options; Xhatdot=Xhatdot)
        
        # Extract the reduced initial condition from Qhat_1
        xhat0 = Xhat[:,1]
        
        # Compute the reduced solution over the trial time horizon
        start_eval_time = time()
        contains_nans, Xtilde, fidx = simulate_opinf(xhat0, n_time_pred, ops, tspan)
        end_eval_time = time()
        time_opinf_eval = end_eval_time - start_eval_time
        
        # If the model produced an unstable solution, move on to the next
        # regularization candidates
        if contains_nans
            @info "OpInf model with (β1, β2, β3) = ($beta1, $beta2, $beta3) produced NaNs. Skipping."
            @info "Complete pair $ct / $tot"
            ops = nothing
            GC.gc() 
            continue
        end
        
        # If the ratio of the maximum coefficient growth exceeds the allowed
        # threshold, move on to the next regularization candidates
        max_diff_Xhat_trial = maximum(abs.(Xtilde .- mean_Xhat), dims=2)
        max_growth_trial = maximum(max_diff_Xhat_trial) / maximum(max_diff_Xhat)
        if max_growth_trial > max_growth
            @info "Max growth = $max_growth_trial exceeded threshold for \
                   (β1, β2, β3) = ($beta1, $beta2, $beta3)"
            @info "Complete pair $ct / $tot"
            ops = nothing
            GC.gc() 
            continue
        end
        
        # At this point we know the model produced a stable solution without too
        # much growth. Compute the training error and, if it's better than the
        # current best error, save the regularization, reduced solution, and
        # the learning times
        train_err = norm(
                Xhat[:, 1:n_time] - Xtilde[:, 1:n_time]
            )^2 / norm(Xhat[:, 1:n_time])^2
        if train_err < best_train_err
            best_beta1 = beta1
            best_beta2 = beta2
            best_beta3 = beta3
            best_train_err = train_err
            Xtilde_opt = Xtilde
            eval_time_opt = time_opinf_eval
            best_model = ops
        end

        if best_final_idx < fidx
            best_final_idx = fidx
        end

        @info "Regularization pair (β1, β2, β3) = ($beta1, $beta2, $beta3): \
               training error = $train_err, evaluation time = $time_opinf_eval, \
               max growth = $max_growth_trial, final index = $fidx"
        @info "Best so far: (β1, β2, β3) = ($best_beta1, $best_beta2, $best_beta3), \
               training error = $best_train_err"
        @info "Complete pair $ct / $tot"
        ops = nothing
        GC.gc() 
    end

    if isnothing(Xtilde_opt)
        @error "No suitable OpInf model found with the given regularization pairs."
    else
        @info "Best OpInf model found with β1 = $best_beta1, β2 = $best_beta2, \
               β3 = $best_beta3, training error = $best_train_err, evaluation time = $eval_time_opt"
    end
    best_betas = (b1 = best_beta1, b2 = best_beta2, b3 = best_beta3)
    return (best_model, best_betas, best_train_err, 
            Xtilde_opt, eval_time_opt, best_final_idx)
end