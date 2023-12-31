# function opt_zubov(X, Ahat, Fhat, Q, Pi, Ptilde, η, α)
#     n, m = size(X)
    
#     # Construct some values used in the optimization
#     X = n < m ? X : X'  # here we want the row to be the states and columns to be time
#     X2 = squareMatStates(X)'
#     X = X' # now we want the columns to be the states and rows to be time
    
#     model = Model(Ipopt.Optimizer)
#     set_silent(model)
#     @variable(model, P[1:n, 1:n])
#     set_start_value.(P, Pi)
#     @expression(model, Ps, 0.5 * (P + P'))
#     @expression(
#         model, 
#         PDEnorm, 
#         sum((X*Ahat'*Ps*X' + X2*Fhat'*Ps*X' - 0.25*X*Ps*X'*X*Q'*X' + 0.5*X*Q'*X').^2)
#     )
#     @expression(model, Pnorm, sum((Ptilde - (Ps .- α.*I(n))).^2)*η)
#     @constraint(model, c, X*Ps*X' .<= 0.99999)
#     @objective(model, Min, PDEnorm + Pnorm)
#     JuMP.optimize!(model)
#     return value.(P), model
# end


function optimize_P(X, Ahat, Fhat, Q, Pi, Ptilde, η)
    n, m = size(X)
    
    # Construct some values used in the optimization
    X = n < m ? X : X'  # here we want the row to be the states and columns to be time
    X2 = squareMatStates(X)'
    X = X' # now we want the columns to be the states and rows to be time
    
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, P[1:n, 1:n])
    set_start_value.(P, Pi)
    @expression(model, Ps, 0.5 * (P + P'))
    @expression(
        model, 
        PDEnorm, 
        sum((X*Ahat'*Ps*X' + X2*Fhat'*Ps*X' - 0.25*X*Ps*X'*X*Q*X' + 0.5*X*Q*X').^2) 
    )  
    @expression(model, Pnorm, sum((Ptilde - Ps).^2)*η)
    @constraint(model, c, X*Ps*X' .<= 0.99999)
    @objective(model, Min, PDEnorm + Pnorm)
    JuMP.optimize!(model)
    return value.(P), model
end


function optimize_Q(X, Ahat, Fhat, P, Qi, Qtilde, η)
    n, m = size(X)
    
    # Construct some values used in the optimization
    X = n < m ? X : X'  # here we want the row to be the states and columns to be time
    X2 = squareMatStates(X)'
    X = X' # now we want the columns to be the states and rows to be time
    
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, Q[1:n, 1:n])
    set_start_value.(Q, Qi)
    @expression(model, Qs, 0.5 * (Q + Q'))
    @expression(
        model, 
        PDEnorm, 
        sum((X*Ahat'*P*X' + X2*Fhat'*P*X' - 0.25*X*P*X'*X*Qs*X' + 0.5*X*Qs*X').^2)  
    )  
    @expression(model, Qnorm, sum((Qtilde - Qs).^2)*η)
    @objective(model, Min, PDEnorm + Qnorm)
    JuMP.optimize!(model)
    return value.(Q), model
end



function DoA(P::Matrix)
    λ, _ = eigen(P)
    λ_abs = abs.(λ)  # this should be unnecessary since P is symmetric
    rmax = 1 / sqrt(minimum(λ_abs))
    rmin = 1 / sqrt(maximum(λ_abs))
    return [rmin, rmax]
end


function est_stab_rad(Ahat, Hhat, Q)
    P = lyapc(Ahat', 0.5*Q)
    F = svd(0.25*Q)
    σmin = minimum(F.S)
    ρhat = σmin / sqrt(norm(P,2)) / norm(Hhat,2) / 2
    return ρhat
end


function zubov_error(X, A, F, P, Q)
    n, m = size(X)
    # Construct some values used in the optimization
    X = n < m ? X : X'  # here we want the row to be the states and columns to be time
    X2 = squareMatStates(X)'
    X = X' # now we want the columns to be the states and rows to be time
    return norm(X*A'*P*X' + X2*F'*P*X' - 0.25*X*P*X'*X*Q'*X' + 0.5*X*Q'*X', 2)
end


function pdp(A, n, γ_lb; γ_ub=1)
    not_pd = true
    γ_lb_copy = deepcopy(γ_lb)
    while not_pd
        model = Model(SCS.Optimizer)
        set_silent(model)
        @variable(model, D[1:n, 1:n], PSD)
        @variable(model, γ_lb <= γ <= γ_ub)
        @expression(model, d, diag(D))
        @objective(model, Min, sum(d) + γ)
        @constraint(model, (A + D) in PSDCone())
        @constraint(model, (A + D .+ γ*I) .>= γ_lb_copy)
        JuMP.optimize!(model)
        Dopt = value.(D)
        γopt = value(γ)

        Apd = A + Dopt + γopt * I
        not_pd = !isposdef(Apd)
        γ_lb *= 10
        if γ_lb > γ_ub
            return Apd, false
        end

        if !not_pd
            return Apd, true
        end
    end
end


function PR_Zubov_LFInf(
    Xr,                         # Reduced order state trajectory data
    A,                          # Linear system matrix 
    F,                          # Quadratic system matrix
    Pi,                         # Initial P matrix for the optimization
    Qi;                         # Initial Q matrix for the optimization
    γ_lb=1e-8,                  # Control parameter for the Zubov method
    α=0.0,                      # Eigenvalue shift parameter for P
    β=0.0,                      # Eigenvalue shift parameter for Q
    N=size(A,1),                # Size of the system
    Ptilde=1.0I(N),             # Initial target P matrix for the optimization
    Qtilde=1.0I(N),             # Initial target Q matrix for the optimization
    δS=1e-5,                    # Symmetricity tolerance for P
    δJ=1e-5,                    # Objective value tolerance for the optimization
    max_iter=100,               # Maximum number of iterations for the optimization
    η=1,                        # Weighting parameter for the Ptilde term
    extra_iter=3                # Number of extra iterations to run after the optimization has converged
)
    # Convergence metrics
    Jzubov_lm1 = 0  # Jzubov(l-1)
    Zerrbest = 1e+8
    check = 0    # run a few extra iterations to make sure the error is decreasing

    # Initialize Q
    Q = 1.0I(N)

    for l in 1:max_iter
        # Optimize for the P matrix
        P, model_P = optimize_P(Xr, A, F, Q, Pi, Ptilde, η)
        λ_P, _ = eigen(P) 
        λ_P_real = real.(λ_P) 
        
        # Project the P matrix to the positive definite space
        Ps = 0.5 * (P + P') 
        λ_Ps, _ = eigen(Ps)
        λ_Ps_copy = deepcopy(λ_Ps)  

        Ptilde, pd_flag_P = pdp(Ps, N, γ_lb)
        @assert pd_flag_P "Ptidle is not positive definite"
        if (any(λ_Ps .< 0))
            abs_min_λ_Ps = abs(minimum(λ_Ps))
            α = abs_min_λ_Ps + 10^(floor(log10(abs_min_λ_Ps))) 
            Pi = P + α*I
        else
            Pi = P
        end


        # Optimize for the Q matrix
        Q, _ = optimize_Q(Xr, A, F, P, Qi, Qtilde, η)
        λ_Q, _ = eigen(Q)
        λ_Q_real = real.(λ_Q)

        # Project the Q matrix to the positive definite space
        Qs = 0.5 * (Q + Q')
        λ_Qs, _ = eigen(Qs)
        λ_Qs_copy = deepcopy(λ_Qs)

        Qtilde, pd_flag_Q = pdp(Qs, N, γ_lb)
        @assert pd_flag_Q "Qtilde is not positive definite"

        if (any(λ_Qs .< 0))
            abs_min_λ_Qs = abs(minimum(λ_Qs))
            β = abs_min_λ_Qs + 10^(floor(log10(abs_min_λ_Qs))) 
            Qi = Q + β*I
        else
            Qi = Q
        end

        Jzubov = objective_value(model_P) 
        ∇Jzubov = abs(Jzubov - Jzubov_lm1) 
        Jzubov_lm1 = Jzubov 

        # Compute some metrics to check the convergence
        diff_P = norm(P - P', 2)
        diff_Q = norm(Q - Q', 2)
        Zerr = zubov_error(Xr, A, F, P, Q)

        # Save the best one
        if all(λ_P_real .> 0) && all(λ_Q_real .> 0) && (Zerr < Zerrbest)
            Pbest = P
            Qbest = Q
            Zerrbest = Zerr
            ∇Jzubovbest = ∇Jzubov
        end

        # Logging
        @info """[Zubov-LFI Iteration $l]
        Zubov Equation Error:                $(Zerr)
        Gradient of Objective value:         $(∇Jzubov)
        ||P - P'||_F:                        $(diff_P)
        ||Q - Q'||_F:                        $(diff_Q)
        eigenvalues of P:                    $(λ_P)
        eigenvalues of Ps:                   $(λ_Ps_copy)
        eigenvalues of Q:                    $(λ_Q)
        eigenvalues of Qs:                   $(λ_Qs_copy)
        # of Real(λp) <= 0:                  $(count(i->(i <= 0), λ_P_real))
        # of Real(λq) <= 0:                  $(count(i->(i <= 0), λ_Q_real))
        dimension:                           $(N)
        η:                                   $(η)
        α:                                   $(α)
        β:                                   $(β)
        """

        # Check if the resulting P satisfies the tolerance
        if diff_P < δS && diff_Q < δS && all(λ_P_real .> 0) && all(λ_Q_real .> 0) && ∇Jzubov < δJ
            check += 1
            if check == extra_iter
                return P, Q, Zerr, ∇Jzubov
            end
        else
            check = 0  # reset if not converging continuously
        end    
        
        # If the optimization did not end before the maximum iteration assign what we have best for now
        if l == max_iter
            if (@isdefined Pbest)
                return Pbest, Qbest, Zerrbest, ∇Jzubovbest
            else
                return P, Q, Zerr, ∇Jzubov
            end
        end
    end
end




#####################
###### __old__ ######
#####################

# function pp_zqlfi(
#     Xr,                         # Reduced order state trajectory data
#     A,                          # Linear system matrix 
#     F,                          # Quadratic system matrix
#     Q,                          # predefined Quadratic matrix for Zubov method
#     Pi;                         # Initial P matrix for the optimization
#     γ=0.1,                      # Control parameter for the Zubov method
#     α=0.0,                      # Eigenvalue shift parameter
#     Ptilde=γ*1.0I(size(A,1)),   # Initial target P matrix for the optimization
#     δS=1e-3,                    # Symmetricity tolerance for P
#     δJ=1e-3,                    # Objective value tolerance for the optimization
#     δe=1e-1,                    # Error tolerance for the Zubov method
#     max_iter=100,               # Maximum number of iterations for the optimization
#     η=1,                        # Weighting parameter for the Ptilde term
#     Kg=Dict(                    # PID gains for the control parameter γ 
#         "p" => 5.5,            
#         "i" => 0.25, 
#         "d" => 0.05
#     ),
#     extra_iter=3                # Number of extra iterations to run after the optimization has converged
# )
#     # Initialize 
#     Jzubov_lm1 = 0  # Jzubov(l-1)
#     Zerr_lm1 = 0  # Zerr(l-1)
#     Zerrbest = 1e+8
#     γ_err_int = 0  # integral error for γ
#     γ_err_lm1 = 0  # error at l-1 for γ
#     γ_ref = γ/10  # reference γ value for the PID control
#     check = 0    # run a few extra iterations to make sure the error is decreasing
#     not_pos_ct = 0  # count the number of times the Ps matrix does not have all posiive eigenvalues

#     for l in 1:max_iter
#         # Run the optimization of Zubov
#         P, mdl_P = opt_zubov(Xr, A, F, Q, Pi, Ptilde, η, α)
#         λ_P, _ = eigen(P)  # eigenvalue decomposition of P
#         λ_P_real = real.(λ_P)  # real part of the eigenvalues of P
        
#         Jzubov = objective_value(mdl_P)  # cost function value from this iteration
#         ∇Jzubov = abs(Jzubov - Jzubov_lm1)  # gradient of the cost function from this iteration
#         Jzubov_lm1 = Jzubov  # update Jzubov(l-1)

#         # Project the P matrix to the positive definite space
#         Ps = 0.5 * (P + P')  # keep only the symmetric part of P assuming the skew-symmetric part is negligible

#         λ_Ps, V_Ps = eigen(Ps)  # eigenvalue decomposition of Ps
#         λ_Ps_copy = deepcopy(λ_Ps)  # make a copy of the eigenvalues of Ps for later use

#         not_pos_ct += (any(λ_Ps .< 0))  # increment counter if Ps does not have all positive eigenvalues
#         λ_Ps[real.(λ_Ps) .< 0] .= γ  # project the negative eigenvalues to γ
#         Ptilde = V_Ps * Diagonal(λ_Ps) * (V_Ps\I)  # reconstruct Ptilde from the projected eigenvalues

#         if not_pos_ct > 2  # if Ps does not have all positive eigenvalues for more than 5 iterations
#             min_λ_Ps = minimum(λ_Ps)
#             α = abs(min_λ_Ps) + 10^(floor(log10(min_λ_Ps))) / 2  # shift the eigenvalues of P by the minimum eigenvalue of Ps
#         end

#         if any(λ_P_real .< 0) || any(imag.(λ_P) .!= 0)  # if all eigenvalues of P are not positive and real
#             Pi = Ps  # then update the next iteration's initial P matrix
#         end  # if not just use the Pi matrix from the previous iteration

#         # Compute some metrics to check the convergence
#         diff = norm(P - P', 2)
#         Zerr = zubov_error(Xr, A, F, P, Q)
#         ∇Zerr = abs(Zerr - Zerr_lm1)  # gradient of the Zubov error
#         Zerr_lm1 = Zerr  # update Zerr(l-1)
        
#         # Save the best one
#         if all(λ_P_real .> 0) && (Zerr < Zerrbest)
#             Pbest = P
#             Zerrbest = Zerr
#             ∇Jzubovbest = ∇Jzubov
#         end

#         # Logging
#         @info """[Zubov-LFI Iteration $l]
#         Zubov Equation Error:                $(Zerr)
#         Gradient of Zubov Equation Error:    $(∇Zerr)
#         Gradient of Objective value:         $(∇Jzubov)
#         ||P - P'||_F:                        $(diff)
#         eigenvalues of P:                    $(λ_P)
#         eigenvalues of Ps:                   $(λ_Ps_copy)
#         # of Real(λp) <= 0:                  $(count(i->(i <= 0), λ_P_real))
#         dim(P):                              $(size(P))
#         γ:                                   $(γ)
#         α:                                   $(α)
#         """

#         # Check if the resulting P satisfies the tolerance
#         if diff < δS && all(λ_P_real .> 0) && ∇Jzubov < δJ && ∇Zerr < δe
#             check += 1
#             if check == extra_iter
#                 return P, Zerr, ∇Jzubov
#             end
#         end    
        
#         # If the optimization did not end before the maximum iteration assign what we have best for now
#         if l == max_iter
#             if (@isdefined Pbest)
#                 return Pbest, Zerrbest, ∇Jzubovbest
#             else
#                 return P, Zerr, ∇Jzubov
#             end
#         end

#         # PID control of the γ term
#         if any(λ_Ps_copy .< 0) 
#             γ_err = (γ_ref) - minimum(λ_Ps_copy)
#             γ_err_int += γ_err
#             γ_err_der = γ_err - γ_err_lm1
#             γ = Kg["p"] * γ_err + Kg["i"] * γ_err_int + Kg["d"] * γ_err_der
#             γ = 1e+2 < γ ? 1e+2 : (γ < 1e-3 ? 1e-3 : γ)  # add saturation to γ

#             γ_err_lm1 = γ_err
#         end
#     end
# end

    # # TODO: Initialize the positive definite shifting optimization stuff
    # model = Model(SCS.Optimizer)
    # set_silent(model)

    # for l in 1:max_iter
    #     # Run the optimization of Zubov
    #     P, mdl_P = opt_zubov(Xr, A, F, Q, Pi, Ptilde, η, α)
    #     λ_P, _ = eigen(P)  # eigenvalue decomposition of P
    #     λ_P_real = real.(λ_P)  # real part of the eigenvalues of P
        
    #     Jzubov = objective_value(mdl_P)  # cost function value from this iteration
    #     ∇Jzubov = abs(Jzubov - Jzubov_lm1)  # gradient of the cost function from this iteration
    #     Jzubov_lm1 = Jzubov  # update Jzubov(l-1)

    #     # Project the P matrix to the positive definite space
    #     Ps = 0.5 * (P + P')  # keep only the symmetric part of P assuming the skew-symmetric part is negligible

    #     λ_Ps, V_Ps = eigen(Ps)  # eigenvalue decomposition of Ps
    #     λ_Ps_copy = deepcopy(λ_Ps)  # make a copy of the eigenvalues of Ps for later use

    #     if any(λ_Ps .< 0)
    #         # Solve the semi-definite program to find the optimal D matrix
    #         @variable(model, D[1:N, 1:N], PSD)
    #         @expression(model, d, diag(D))
    #         @objective(model, Min, sum(d))
    #         @constraint(model, Ps + D in PSDCone())
    #         JuMP.optimize!(model)
    #         Dopt = value.(D)
    #         JuMP.unregister(model, :D)
    #         JuMP.unregister(model, :d)
            
    #         # Find the shifted D matrix to shift Ps to positive definite
    #         P_psd = Ps + Dopt
    #         λpsd, _ = eigen(P_psd)
    #         λpsd_min = ceil(abs(minimum(λpsd)); sigdigits=4)
    #         D = Dopt + λpsd_min * I
    #         α = λpsd_min

    #         abs_min_λ_Ps = abs(minimum(λ_Ps))
    #         α = abs_min_λ_Ps + 10^(floor(log10(abs_min_λ_Ps)))  # shift the eigenvalues of P by the minimum eigenvalue of Ps
            
    #         # Project the negative eigenvalues to γ
    #         # λ_Ps[real.(λ_Ps) .< 0] .= γ  
    #         # Ptilde = V_Ps * Diagonal(λ_Ps) * (V_Ps\I)  # reconstruct Ptilde from the projected eigenvalues
    #         Ptilde = Ps + D
    #         # Pi = Ptilde_lm1
    #         # Ptilde_lm1 = Ptilde
    #     else
    #         Ptilde = Ps
    #         # Pi = Ptilde_lm1
    #         # Ptilde_lm1 = Ptilde
    #         α = 0
    #     end
    #     Pi = Ps + α*I

# function opt_zubov_Q(X, Ahat, Fhat, P, Qi, Qtilde, η)
#     n, m = size(X)
    
#     # Construct some values used in the optimization
#     X = n < m ? X : X'  # here we want the row to be the states and columns to be time
#     X2 = squareMatStates(X)'
#     X = X' # now we want the columns to be the states and rows to be time
    
#     model = Model(Ipopt.Optimizer)
#     set_silent(model)
#     @variable(model, Q[1:n, 1:n])
#     set_start_value.(Q, Qi)
#     @expression(model, Qs, 0.5 * (Q + Q'))
#     @expression(
#         model, 
#         PDEnorm, 
#         sum((X*Ahat'*P'*X' + X2*Fhat'*P'*X' - 0.25*X*P'*X'*X*Qs*X' + 0.5*X*Qs*X').^2)
#     )
#     @expression(model, Qnorm, sum((Qtilde - Qs).^2)*η)
#     @objective(model, Min, PDEnorm + Qnorm)
#     JuMP.optimize!(model)
#     return value.(Q), model
# end


# FIX! - This function is not working. Should try with Convex.jl and Gurobi
# function opt_zubov(X, Ahat, Fhat, Pi, Qi, Ptilde, Qtilde, η)
#     n, m = size(X)
    
#     # Construct some values used in the optimization
#     X = n < m ? X : X'  # here we want the row to be the states and columns to be time
#     X2 = squareMatStates(X)'
#     X = X' # now we want the columns to be the states and rows to be time
    
#     model = Model(NLopt.Optimizer)
#     set_optimizer_attribute(model, "algorithm", :LD_MMA)
#     @variable(model, P[1:n, 1:n])
#     @variable(model, Q[1:n, 1:n])
#     set_start_value.(P, Pi)
#     set_start_value.(Q, Qi)
#     @expression(model, Ps, 0.5 * (P + P'))
#     @expression(model, Qs, 0.5 * (Q + Q'))
#     @NLexpression(
#         model, 
#         PDEnorm, 
#         sum(abs2(X*Ahat'*Ps*X' + X2*Fhat'*Ps*X' - 0.25*X*Ps*X'*X*Qs*X' + 0.5*X*Qs*X')[i,j] for i in 1:m, j in 1:m)
#     )
#     @expression(model, Pnorm, sum((Ptilde - Ps).^2)*η)
#     @expression(model, Qnorm, sum((Qtilde - Qs).^2)*η)
#     @constraint(model, c, X*Ps*X' .<= 0.9999)
#     @NLobjective(model, Min, PDEnorm + Pnorm + Qnorm)
#     JuMP.optimize!(model)
#     return value.(P), value.(Q), model
# end