function opt_zubov(X, Ahat, Fhat, Q, Pi, Ptilde, η, α)
    n, m = size(X)
    
    # Construct some values used in the optimization
    X = n < m ? X : X'  # here we want the row to be the states and columns to be time
    X2 = squareMatStates(X)'
    X = X' # now we want the columns to be the states and rows to be time
    
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, P[1:n, 1:n])
    set_start_value.(P, Pi)
    @expression(model, Pα, P .- α.*I)
    @expression(model, Ps, 0.5 * (Pα + Pα'))
    @expression(
        model, 
        PDEnorm, 
        sum((X*Ahat'*Ps*X' + X2*Fhat'*Ps*X' - 0.25*X*Ps*X'*X*Q*X' + 0.5*X*Q*X').^2)  # DID change Q' to Q (check if this effects anything)
    )  
    @expression(model, Pnorm, sum((Ptilde - Ps).^2)*η)
    @constraint(model, c, X*Ps*X' .<= 0.99999)
    @objective(model, Min, PDEnorm + Pnorm)
    JuMP.optimize!(model)
    return value.(P), model
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


function pp_zqlfi(
    Xr,                         # Reduced order state trajectory data
    A,                          # Linear system matrix 
    F,                          # Quadratic system matrix
    Q,                          # predefined Quadratic matrix for Zubov method
    Pi;                         # Initial P matrix for the optimization
    γ=0.1,                      # Control parameter for the Zubov method
    α=0.0,                      # Eigenvalue shift parameter
    Ptilde=γ*1.0I(size(A,1)),   # Initial target P matrix for the optimization
    δS=1e-3,                    # Symmetricity tolerance for P
    δJ=1e-3,                    # Objective value tolerance for the optimization
    δe=1e-1,                    # Error tolerance for the Zubov method
    max_iter=100,               # Maximum number of iterations for the optimization
    η=1,                        # Weighting parameter for the Ptilde term
    Kg=Dict(                    # PID gains for the control parameter γ 
        "p" => 5.5,            
        "i" => 0.25, 
        "d" => 0.05
    ),
    extra_iter=3                # Number of extra iterations to run after the optimization has converged
)
    # Initialize 
    Jzubov_lm1 = 0  # Jzubov(l-1)
    diff_lm1 = 0  # ||P(l-1) - P(l-1)'||_F
    Zerr_lm1 = 0  # Zerr(l-1)
    Zerrbest = 1e+8
    γ_err_int = 0  # integral error for γ
    γ_err_lm1 = 0  # error at l-1 for γ
    γ_ref = γ/10  # reference γ value for the PID control
    check = 0    # run a few extra iterations to make sure the error is decreasing
    # not_pos_ct = 0  # count the number of times the Ps matrix does not have all posiive eigenvalues
    symm_no_change_ct = 0  # count the number of times the symmetry error does not change

    for l in 1:max_iter
        # Run the optimization of Zubov
        P, mdl_P = opt_zubov(Xr, A, F, Q, Pi, Ptilde, η, α)
        λ_P, _ = eigen(P)  # eigenvalue decomposition of P
        λ_P_real = real.(λ_P)  # real part of the eigenvalues of P
        
        Jzubov = objective_value(mdl_P)  # cost function value from this iteration
        ∇Jzubov = abs(Jzubov - Jzubov_lm1)  # gradient of the cost function from this iteration
        Jzubov_lm1 = Jzubov  # update Jzubov(l-1)

        # Project the P matrix to the positive definite space
        Ps = 0.5 * (P + P')  # keep only the symmetric part of P assuming the skew-symmetric part is negligible

        λ_Ps, V_Ps = eigen(Ps)  # eigenvalue decomposition of Ps
        λ_Ps_copy = deepcopy(λ_Ps)  # make a copy of the eigenvalues of Ps for later use

        # not_pos_ct += (any(λ_Ps .< 0))  # increment counter if Ps does not have all positive eigenvalues
        # if not_pos_ct > 5  # if Ps does not have all positive eigenvalues for more than 5 iterations
        if (any(λ_Ps .< 0))
            abs_min_λ_Ps = abs(minimum(λ_Ps))
            α = abs_min_λ_Ps + 10^(floor(log10(abs_min_λ_Ps)))  # shift the eigenvalues of P by the minimum eigenvalue of Ps
        end
        λ_Ps[real.(λ_Ps) .< 0] .= γ  # project the negative eigenvalues to γ
        Ptilde = V_Ps * Diagonal(λ_Ps) * (V_Ps\I)  # reconstruct Ptilde from the projected eigenvalues

        # Compute some metrics to check the convergence
        diff = norm(P - P', 2)
        Zerr = zubov_error(Xr, A, F, P, Q)
        ∇Zerr = abs(Zerr - Zerr_lm1)  # gradient of the Zubov error
        Zerr_lm1 = Zerr  # update Zerr(l-1)

        # If the symmetry error does not change for a prolonged period of time, change the η value to make the Ptilde term more important
        if diff - diff_lm1 < 1e-4
            symm_no_change_ct += 1
            if symm_no_change_ct > 5
                η += 1.0
                symm_no_change_ct = 0  # reset the counter
            end
        end
        
        # Save the best one
        if all(λ_P_real .> 0) && (Zerr < Zerrbest)
            Pbest = P
            Zerrbest = Zerr
            ∇Jzubovbest = ∇Jzubov
        end

        # if any(λ_P_real .> 0) && all(imag.(λ_P) .== 0)  # if all eigenvalues of P are positive and real
        #     Pi = Ps  # then update the next iteration's initial P matrix
        # end  # if not just use the Pi matrix from the previous iteration
        # Pi = Ptilde
        Pi = Ps + α*I

        # Logging
        @info """[Zubov-LFI Iteration $l]
        Zubov Equation Error:                $(Zerr)
        Gradient of Zubov Equation Error:    $(∇Zerr)
        Gradient of Objective value:         $(∇Jzubov)
        ||P - P'||_F:                        $(diff)
        eigenvalues of P:                    $(λ_P)
        eigenvalues of Ps:                   $(λ_Ps_copy)
        # of Real(λp) <= 0:                  $(count(i->(i <= 0), λ_P_real))
        dim(P):                              $(size(P))
        γ:                                   $(γ)
        α:                                   $(α)
        η:                                   $(η)
        """

        # Check if the resulting P satisfies the tolerance
        if diff < δS && all(λ_P_real .> 0) && ∇Jzubov < δJ && ∇Zerr < δe
            check += 1
            if check == extra_iter
                return P, Zerr, ∇Jzubov
            end
        end    
        
        # If the optimization did not end before the maximum iteration assign what we have best for now
        if l == max_iter
            if (@isdefined Pbest)
                return Pbest, Zerrbest, ∇Jzubovbest
            else
                return P, Zerr, ∇Jzubov
            end
        end

        # PID control of the γ term
        if any(λ_Ps_copy .< 0) 
            γ_err = (γ_ref) - minimum(λ_Ps_copy)
            γ_err_int += γ_err
            γ_err_der = γ_err - γ_err_lm1
            γ = Kg["p"] * γ_err + Kg["i"] * γ_err_int + Kg["d"] * γ_err_der
            γ = 1e+2 < γ ? 1e+2 : (γ < 1e-3 ? 1e-3 : γ)  # add saturation to γ
            γ_err_lm1 = γ_err
        end
    end
end




#####################
###### __old__ ######
#####################

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

# function opt_zubov(X, Ahat, Fhat, Pi, Qi, Ptilde, Qtilde, η)
#     n, m = size(X)
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
#         sum(sum(sum(abs2(X[:,i]'[j]*Ps[j,k]*Ahat[j,k]*X[:,i][j] + X[:,i]'[j]*Ps[j,k]*(Fhat*LnL.vech(X[:,i]*X[:,i]'))[j]
#                     - 0.25*(X[:,i]'[j])*Qs[j,k]*(X[:,i][j]*X[:,i]'[j])*Ps[j,k]*X[:,i][j] + 0.5*(X[:,i]'[j])*Qs[j,k]*(X[:,i][j])) for j in 1:n, k in 1:n )) for i in 1:m)
#     )
#     @expression(model, Pnorm, sum((Ptilde - Ps).^2)*η)
#     @expression(model, Qnorm, sum((Qtilde - Qs).^2)*η)
#     @NLobjective(model, Min, PDEnorm + Pnorm + Qnorm)
#     JuMP.optimize!(model)
#     println("""
#     termination_status = $(termination_status(model))
#     primal_status      = $(primal_status(model))
#     objective_value    = $(objective_value(model))
#     """)
#     return value.(P), value.(Q), model
# end


# function opt_zubov_P(X, Ahat, Fhat, Q, Pi, Ptilde, η)
#     n, m = size(X)
#     model = Model(Ipopt.Optimizer)
#     set_silent(model)
#     @variable(model, P[1:n, 1:n])
#     set_start_value.(P, Pi)
#     @expression(model, Ps, 0.5 * (P + P'))
#     @expression(
#         model, 
#         PDEnorm, 
#         sum(sum((Ps*Ahat*X[:,i] + Ps*(Fhat*LnL.vech(X[:,i]*X[:,i]'))
#                     - 0.25*(Q*X[:,i]*X[:,i]')*Ps*X[:,i] + 0.5*Q*X[:,i]).^2) for i in 1:m)
#     )
#     @expression(model, Pnorm, sum((Ptilde - Ps).^2)*η)
#     @objective(model, Min, PDEnorm + Pnorm)
#     JuMP.optimize!(model)
#     # println("""
#     # termination_status = $(termination_status(model))
#     # primal_status      = $(primal_status(model))
#     # objective_value    = $(objective_value(model))
#     # """)
#     return value.(P), model
# end

# function opt_zubov_Q(X, Ahat, Fhat, P, Qi, Qtilde, η)
#     n, m = size(X)
#     model = Model(Ipopt.Optimizer)
#     set_silent(model)
#     @variable(model, Q[1:n, 1:n])
#     set_start_value.(Q, Qi)
#     @expression(model, Qs, 0.5 * (Q + Q'))
#     @expression(
#         model, 
#         PDEnorm, 
#         sum(sum((P*Ahat*X[:,i] + P*(Fhat*LnL.vech(X[:,i]*X[:,i]'))
#                     - 0.25*Qs*((X[:,i]*X[:,i]')*P*X[:,i]) + 0.5*Qs*X[:,i]).^2) for i in 1:m)
#     )
#     @expression(model, Qnorm, sum((Qtilde - Qs).^2)*η)
#     @objective(model, Min, PDEnorm + Qnorm)
#     JuMP.optimize!(model)
#     # println("""
#     # termination_status = $(termination_status(model))
#     # primal_status      = $(primal_status(model))
#     # objective_value    = $(objective_value(model))
#     # """)
#     return value.(Q), model
# end

# function opt_zubov_P(X, Ahat, Fhat, Q, Pi, Ptilde, η)
#     n, m = size(X)
#     model = Model(Ipopt.Optimizer)
#     set_silent(model)
#     @variable(model, P[1:n, 1:n])
#     set_start_value.(P, Pi)
#     @expression(model, Ps, 0.5 * (P + P'))
#     @expression(
#         model, 
#         PDEnorm, 
#         sum(sum((X[:,i]'*Ps*Ahat*X[:,i] + X[:,i]'*Ps*(Fhat*LnL.vech(X[:,i]*X[:,i]'))
#                     - 0.25*(X[:,i]'*Q*X[:,i]*X[:,i]')*Ps*X[:,i] + 0.5*X[:,i]'*Q*X[:,i]).^2) for i in 1:m)
#     )
#     @expression(model, Pnorm, sum((Ptilde - Ps).^2)*η)
#     @objective(model, Min, PDEnorm + Pnorm)
#     JuMP.optimize!(model)
#     # println("""
#     # termination_status = $(termination_status(model))
#     # primal_status      = $(primal_status(model))
#     # objective_value    = $(objective_value(model))
#     # """)
#     return value.(P), model
# end

# function opt_zubov_Q(X, Ahat, Fhat, P, Qi, Qtilde, η)
#     n, m = size(X)
#     model = Model(Ipopt.Optimizer)
#     set_silent(model)
#     @variable(model, Q[1:n, 1:n])
#     set_start_value.(Q, Qi)
#     @expression(model, Qs, 0.5 * (Q + Q'))
#     @expression(
#         model, 
#         PDEnorm, 
#         sum(sum((X[:,i]'*P*Ahat*X[:,i] + X[:,i]'*P*(Fhat*LnL.vech(X[:,i]*X[:,i]'))
#                     - 0.25*X[:,i]'*Qs*((X[:,i]*X[:,i]')*P*X[:,i]) + 0.5*X[:,i]'*Qs*X[:,i]).^2) for i in 1:m)
#     )
#     @expression(model, Qnorm, sum((Qtilde - Qs).^2)*η)
#     @objective(model, Min, PDEnorm + Qnorm)
#     JuMP.optimize!(model)
#     # println("""
#     # termination_status = $(termination_status(model))
#     # primal_status      = $(primal_status(model))
#     # objective_value    = $(objective_value(model))
#     # """)
#     return value.(Q), model
# end

# function zubov(X,Ahat,Hhat,Q,ϵ)
#     n, m = size(X)
#     model = Model(Ipopt.Optimizer)
#     @variable(model, P[1:n, 1:n])
#     @constraint(model, con, P + ϵ*I .>= 0)
#     @objective(
#         model, 
#         Min, 
#         sum(sum((X[:,i]'*P*Ahat*X[:,i] + X[:,i]'*P*Hhat*kron(X[:,i],X[:,i]) 
#                     - 0.25*X[:,i]'*Q*X[:,i]*X[:,i]'*P*X[:,i] + 0.5*X[:,i]'*Q*X[:,i]).^2) for i in 1:m),
#     )
#     JuMP.optimize!(model)
#     println("""
#     termination_status = $(termination_status(model))
#     primal_status      = $(primal_status(model))
#     objective_value    = $(objective_value(model))
#     """)
#     return value.(P), objective_value(model), termination_status(model), primal_status(model)
# end