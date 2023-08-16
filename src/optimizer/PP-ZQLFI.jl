function opt_zubov_P(X, Ahat, Fhat, Q, Pi, Ptilde, η)
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
        sum((X*Ahat'*Ps*X' + X2*Fhat'*Ps*X' - 0.25*X*Ps*X'*X*Q'*X' + 0.5*X*Q'*X').^2)
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
    X,
    Vr,
    A, 
    F, 
    Q, 
    Pi;
    γ=0.1,
    Ptilde=γ*1.0I(size(A,1)),
    δS=1e-3,
    δJ=1e-3,
    δe=1.0,
    max_iter=100,
    η=1,
    Ka=Dict("p" => 25.5, "i" => 0.25, "d" => 0.05),
    Kb=Dict("p" => 10.5, "i" => 0.35, "d" => 0.005),
    extra_iter=3
)
    n, r = size(Vr)
    Xr = Vr' * X  # reduced states

    # Initialize Jzubov(l-1)
    Jzubov_lm1 = 0

    # PID gains for the γ control
    # Kgp = 0.8
    # Kgi = 0.09
    # Kgd = 0.005
    # Kgp = 1.5
    # Kgi = 0.1
    # Kgd = 0.05
    # Kgp = 20.5
    # Kgi = 0.1
    # Kgd = 0.05

    # Initialize the errors for γ and β
    γ_err_int = 0  # integral error
    γ_err_lm1 = 0  # error at l-1
    # α = γ
    # β = γ
    # α_err_int = 0  # integral error
    # α_err_lm1 = 0  # error at l-1
    # β_err_int = 0  # integral error
    # β_err_lm1 = 0  # error at l-1
    check = 0    # run a few extra iterations to make sure the error is decreasing

    for l in 1:max_iter
        # Zubov Analysis for P
        P, mdl_P = opt_zubov_P(Xr, A, F, Q, Pi, Ptilde, η)
        Jzubov = objective_value(mdl_P)
        ∇Jzubov = abs(Jzubov - Jzubov_lm1)
        Jzubov_lm1 = Jzubov  # update Jzubov(l-1)

        # Project the P matrix to the positive definite space
        λp, Vp = eigen(P)
        λp_copy = deepcopy(λp)
        λp_copy_real = real.(λp_copy)
        λp_copy_imag = imag.(λp_copy)
        λ_lm1 = λp  # update λ(l-1)
        if l == 1
            Pi = P
        else
            if minimum(real.(λ_lm1)) < minimum(real.(λp)) && maximum(imag.(λ_lm1)) > maximum(imag.(λp))
                Pi = P
            end
        end   
        if any(λp_copy_real .< 0)
            for i in eachindex(λp)
                if real(λp[i]) < γ 
                    λp[i] = γ
                end
            end
            Ptilde = real.(Vp * Diagonal(λp) * (Vp\I))
        elseif any(λp_copy_imag .!= 0)
            Ptilde = real.(P)
        end
        diff = norm(P - P', 2)
        Zerr = zubov_error(Xr, A, F, P, Q)
        
        # Logging
        @info """[PP-ZQLFI RESULTS]
        Error of Zubov Equation:       $(Zerr)
        Gradient of Objective value:   $(∇Jzubov)
        ||P - P'||_F:                  $(diff)
        eigenvalues of P:              $(λp_copy)
        # of Real(λp) <= 0:            $(count(i->(i <= 0), λp_copy_real))
        dim(P):                        $(size(P))
        γ:                             $(γ)
        loop:                          $(l)
        """

        # Check if the resulting P satisfies the tolerance
        if diff < δS && all(λp_copy_real .> 0) && ∇Jzubov < δJ && Zerr < δe
            check += 1
            if check == extra_iter
                return P, Zerr, ∇Jzubov
            end
        end    
        
        # If the optimization did not end before the maximum iteration assign what we have best for now
        if l == max_iter
            return P, Zerr, ∇Jzubov
        end

        # PID control of the γ term
        if any(λp_copy_real .< 0) 
            γ_err = 0.01 - minimum(λp_copy_real)
            γ_err_int += γ_err
            γ_err_der = γ_err - γ_err_lm1
            γ = Kg["p"] * γ_err + Kg["i"] * γ_err_int + Kg["d"] * γ_err_der
            γ = 1e+3 < γ ? 1e+3 : (γ < 1e-3 ? 1e-3 : γ)

            γ_err_lm1 = γ_err
        end
        # if any(λp_copy_real .< 0) 
        #     α_err = 0.01 - minimum(λp_copy_real)
        #     α_err_int += α_err
        #     α_err_der = α_err - α_err_lm1
        #     α = Ka["p"] * α_err + Ka["i"] * α_err_int + Ka["d"] * α_err_der
        #     α = 1e+3 < α ? 1e+3 : (α < 1e-3 ? 1e-3 : α)

        #     α_err_lm1 = α_err
        # end
        # if any(λp_copy_imag .!= 0)
        #     β_err = maximum(λp_copy_imag)
        #     β_err_int +=  β_err
        #     β_err_der =  β_err -  β_err_lm1
        #     β = Kb["p"] *  β_err + Kb["i"] *  β_err_int + Kb["d"] *  β_err_der
        #     β = 1e+3 <  β ? 1e+3 : ( β < 1e-10 ? 1e-10 :  β)

        #     β_err_lm1 =  β_err
        # end
        # γ = 0.5*α + 0.5*β
    end
end




#####################
###### __old__ ######
#####################

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