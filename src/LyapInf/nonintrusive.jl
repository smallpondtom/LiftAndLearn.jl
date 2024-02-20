""" 
    Options for Non-Intrusive Lyapunov Function Inference (NonInt_LyapInf)

# Fields
"""
@with_kw mutable struct NonInt_LyapInf_options 
    α::Real                     = 1e-4     # Eigenvalue shift parameter for P
    β::Real                     = 1e-4     # Eigenvalue shift parameter for Q
    δS::Real                    = 1e-5     # Symmetricity tolerance for P
    δJ::Real                    = 1e-3     # Objective value tolerance for the optimization
    max_iter::Int               = 100000   # Maximum number of iterations for the optimization
    opt_max_iter::Int           = 100      # Maximum number of iterations for the solving for both P and Q in loop
    extra_iter::Int             = 3        # Number of extra iterations to run after the optimization has converged
    optimizer::String           = "ipopt"  # Optimizer to use for the optimization
    ipopt_linear_solver::String = "none"   # Linear solver for Ipopt
    verbose::Bool               = false    # Enable verbose output for the optimization
    optimize_PandQ::String      = "P"      # Optimize both P and Q 
    HSL_lib_path::String        = "none"   # Path to the HSL library

    @assert (optimize_PandQ in ["P", "both", "together"]) "Optimize P and Q options are: P, both, and together."
    @assert (optimizer in ["ipopt", "Ipopt", "SCS"]) "Optimizer options are: Ipopt and SCS."
    @assert !(optimizer == "SCS" && optimize_PandQ in ["both","together"]) "SCS does not support optimizing P and Q together or both."
end


function optimize_P(X::AbstractArray{T}, Xdot::AbstractArray{T}, Q::AbstractArray{T}, 
        options::NonInt_LyapInf_options; Pi=nothing) where {T<:Real}
    n, K = size(X)
    @assert n < K "The number of states must be less than the number of time steps."
    @assert size(Xdot) == size(X) "The state trajectory and its derivative must have the same size."

    if options.optimizer in ["ipopt", "Ipopt"]  # Ipopt prefers constraints
        model = Model(Ipopt.Optimizer)
        set_optimizer_attribute(model, "max_iter", options.max_iter)
        if options.ipopt_linear_solver != "none"
            if options.HSL_lib_path != "none"
                set_attribute(model, "hsllib", options.HSL_lib_path)
            end
            set_attribute(model, "linear_solver", options.ipopt_linear_solver)
        end
        # Set up verbose or silent
        if !options.verbose
            set_silent(model)
        end 
        set_string_names_on_creation(model, false)

        @variable(model, P[1:n, 1:n], Symmetric)
        @variable(model, Ld[1:n, 1:n] >= 0, Symmetric)  # Lower triangular matrix
        L = LinearAlgebra.LowerTriangular(Ld)
        if !isnothing(Pi)
            set_start_value.(P, Pi)  # set initial guess for the quadratic P matrix
        end

        @variable(model, Z[1:n, 1:K])
        @constraint(
            model, 
            Z .== 2.0 .*P*Xdot .- Q*X*X'*P*X .+  Q*X
        )
        @objective(model, Min, sum(Z.^2))

        # Add constraints for positive definiteness:
        # Cholesky decomposition constraint: P = L * L'
        for i in 1:n
            for j in 1:n
                @constraint(model, P[i, j] == sum(L[i, k] * L[j, k] for k in 1:min(i, j)))
            end
        end
    else   # SCS is okay with large objective
        model = Model(SCS.Optimizer)
        set_optimizer_attribute(model, "max_iters", options.max_iter)
        # Set up verbose or silent
        if !options.verbose
            set_silent(model)
        end 
        set_string_names_on_creation(model, false)

        @variable(model, P[1:n, 1:n], PSD)
        if !isnothing(Pi)
            set_start_value.(P, Pi)  # set initial guess for the quadratic P matrix
        end
        @expression(
            model, 
            inside_norm, 
            sum((2.0 .*P*Xdot .- Q*X*X'*P*X .+ Q*X).^2) 
        )  
        @objective(model, Min, inside_norm)

        # Add a constraint to make A positive definite
        for i in 1:n
            @constraint(model, P[i, i] >= options.α)
        end
    end
    
    # @constraint(model, X'*P*X .<= 1 - eps())
    JuMP.optimize!(model)
    P_sol = value.(P)
    return P_sol, JuMP.objective_value(model)
end


function optimize_Q(X::AbstractArray{T}, Xdot::AbstractArray{T}, P::AbstractArray{T}, 
        options::NonInt_LyapInf_options; Qi=nothing) where {T<:Real}
    n, K = size(X)
    @assert n < K "The number of states must be less than the number of time steps."
    @assert size(Xdot) == size(X) "The state trajectory and its derivative must have the same size."
    
    # if options.optimizer in ["ipopt", "Ipopt"]  # Ipopt prefers constraints

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", options.max_iter)
    if options.ipopt_linear_solver != "none"
        if options.HSL_lib_path != "none"
            set_attribute(model, "hsllib", options.HSL_lib_path)
        end
        set_attribute(model, "linear_solver", options.ipopt_linear_solver)
    end
    # Set up verbose or silent
    if !options.verbose
        set_silent(model)
    end
    set_string_names_on_creation(model, false)

    @variable(model, Q[1:n, 1:n], Symmetric)
    @variable(model, Rd[1:n, 1:n] >= 0, Symmetric)  # Lower triangular matrix
    R = LinearAlgebra.LowerTriangular(Rd)
    if !isnothing(Qi)
        set_start_value.(Q, Qi)  # set initial guess for the quadratic P matrix
    end

    @variable(model, Z[1:n, 1:K])
    @constraint(
        model, 
        Z .== 2.0 .*P*Xdot .- Q*X*X'*P*X .+ Q*X
    )
    @objective(model, Min, sum(Z.^2))

    # Add constraints for positive definiteness:
    # Cholesky decomposition constraint: Q = R * R'
    for i in 1:n
        for j in 1:n
            @constraint(model, Q[i, j] == sum(R[i, k] * R[j, k] for k in 1:min(i, j)))
        end
    end

    # else   # SCS is okay with large objective
    #     model = Model(SCS.Optimizer)
    #     set_optimizer_attribute(model, "max_iters", options.max_iter)

    #     @variable(model, Q[1:n, 1:n], PSD)
    #     if !isnothing(Qi)
    #         set_start_value.(Q, Qi)  # set initial guess for the quadratic P matrix
    #     end

    #     # Set up verbose or silent
    #     if !options.verbose
    #         set_silent(model)
    #     end
    #     set_string_names_on_creation(model, false)

    #     @expression(
    #         model, 
    #         inside_norm, 
    #         sum((2.0 .*P*Xdot .- Q*X*X'*P*X .+ Q*X).^2) 
    #     )  
    #     @objective(model, Min, inside_norm)

    #     # Add a constraint to make positive definite
    #     for i in 1:n
    #         @constraint(model, Q[i, i] >= options.β)
    #     end
    # end

    # @constraint(model, X'*P*X .<= 1 - eps())
    JuMP.optimize!(model)
    Q_sol = value.(Q)
    return Q_sol, JuMP.objective_value(model)
end


function optimize_PQ(X::AbstractArray{T}, Xdot::AbstractArray{T}, options::NonInt_LyapInf_options; 
        Pi=nothing, Qi=nothing) where {T<:Real}
    n, K = size(X)
    @assert n < K "The number of states must be less than the number of time steps."
    @assert size(Xdot) == size(X) "The state trajectory and its derivative must have the same size."

    # if options.optimizer in ["ipopt", "Ipopt"]  # Ipopt prefers constraints

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", options.max_iter)
    if options.ipopt_linear_solver != "none"
        if options.HSL_lib_path != "none"
            set_attribute(model, "hsllib", options.HSL_lib_path)
        end
        set_attribute(model, "linear_solver", options.ipopt_linear_solver)
    end
    # Set up verbose or silent
    if !options.verbose
        set_silent(model)
    end 
    set_string_names_on_creation(model, false)
    
    # P matrix
    @variable(model, P[1:n, 1:n], Symmetric)
    @variable(model, Ld[1:n, 1:n] >= 0, Symmetric)  # Lower triangular matrix
    L = LinearAlgebra.LowerTriangular(Ld)
    if !isnothing(Pi)
        set_start_value.(P, Pi)  # set initial guess for the quadratic P matrix
    end

    # Q matrix
    @variable(model, Q[1:n, 1:n], Symmetric)
    @variable(model, Rd[1:n, 1:n] >= 0, Symmetric)  # Lower triangular matrix
    R = LinearAlgebra.LowerTriangular(Rd)
    if !isnothing(Qi)
        set_start_value.(Q, Qi)  # set initial guess for the quadratic P matrix
    end

    # Objective
    @variable(model, Z[1:n, 1:K])
    @constraint(
        model, 
        Z .== 2.0 .*P*Xdot .- Q*X*X'*P*X .+ Q*X
    )
    @objective(model, Min, sum(Z.^2))

    # Add constraints for positive definiteness:
    # Cholesky decomposition constraint: P = L * L' and Q = R * R'
    for i in 1:n
        for j in 1:n
            @constraint(model, P[i, j] == sum(L[i, k] * L[j, k] for k in 1:min(i, j)))
            @constraint(model, Q[i, j] == sum(R[i, k] * R[j, k] for k in 1:min(i, j)))
        end
    end

    # else   # SCS is okay with large objective
    #     model = Model(SCS.Optimizer)
    #     set_optimizer_attribute(model, "max_iters", options.max_iter)
    #     # Set up verbose or silent
    #     if !options.verbose
    #         set_silent(model)
    #     end 
    #     set_string_names_on_creation(model, false)

    #     @variable(model, P[1:n, 1:n], PSD)
    #     @variable(model, Q[1:n, 1:n], PSD)
    #     if !isnothing(Pi)
    #         set_start_value.(P, Pi)  # set initial guess for the quadratic P matrix
    #     end
    #     if !isnothing(Qi)
    #         set_start_value.(Q, Qi)  # set initial guess for the quadratic P matrix
    #     end
    #     @expression(
    #         model, 
    #         inside_norm,
    #         sum((2.0 .*P*Xdot .- Q*X*X'*P*X .+ Q*X).^2) 
    #     )  
    #     @objective(model, Min, inside_norm)

    #     # Add a constraint to make A positive definite
    #     for i in 1:n
    #         @constraint(model, P[i, i] >= options.α)
    #         @constraint(model, Q[i, i] >= options.β)
    #     end
    # end
    
    # @constraint(model, X'*P*X .<= 1 - eps())
    JuMP.optimize!(model)
    P_sol = value.(P)
    Q_sol = value.(Q)
    return P_sol, Q_sol, JuMP.objective_value(model)
end


function NonInt_LyapInf(
    X::AbstractArray{T},  # state trajectory data
    Xdot::AbstractArray{T},  # state trajectory derivative data
    options::NonInt_LyapInf_options;   # Options for the optimization
    Pi=nothing,  # Initial P matrix for the optimization
    Qi=nothing   # Initial Q matrix for the optimization
) where {T<:Real}
    # Convergence metrics
    Jzubov_lm1 = 0  # Jzubov(l-1)
    check = 0    # run a few extra iterations to make sure the error is decreasing

    # Initialize P and Q
    N = size(X,1)
    Q = isnothing(Qi) ? 1.0I(N) : Qi
    P = isnothing(Pi) ? 1.0I(N) : Pi

    if options.optimize_PandQ == "both"
        for l in 1:options.opt_max_iter
            # Optimize for the P matrix
            P, Jzubov = optimize_P(X, Xdot, Q, options; Pi=Pi)
            λ_P = eigen(P).values
            λ_P_real = real.(λ_P) 
            Pi = P 

            # Optimize for the Q matrix
            Q, _ = optimize_Q(X, Xdot, P, options; Qi=Qi)
            λ_Q = eigen(Q).values
            λ_Q_real = real.(λ_Q)
            Qi = Q

            ∇Jzubov = abs(Jzubov - Jzubov_lm1) 
            Jzubov_lm1 = Jzubov 

            # Compute some metrics to check the convergence
            diff_P = norm(P - P', 2)
            diff_Q = norm(Q - Q', 2)

            # Save the best one
            if all(λ_P_real .> 0) && all(λ_Q_real .> 0) # && (Zerr < Zerrbest)
                Pbest = P
                Qbest = Q
                ∇Jzubovbest = ∇Jzubov
            end

            # Zubov Equation Error:                $(Zerr)
            # Logging
            @info """[NonInt_LyapInf Iteration $l: Optimize P then Q]
            Objective value:                     $(Jzubov)
            Gradient of Objective value:         $(∇Jzubov)
            ||P - P'||_F:                        $(diff_P)
            ||Q - Q'||_F:                        $(diff_Q)
            eigenvalues of P:                    $(round.(λ_P; sigdigits=4))
            # of Real(λp) <= 0:                  $(count(i->(i <= 0), λ_P_real))
            eigenvalues of Q:                    $(round.(λ_Q; digits=4))
            # of Real(λq) <= 0:                  $(count(i->(i <= 0), λ_Q_real))
            dimension:                           $(N)
            α:                                   $(options.α)
            β:                                   $(options.β)
            """

            # Check if the resulting P satisfies the tolerance
            if diff_P < options.δS && diff_Q < options.δS && all(λ_P_real .> 0) && all(λ_Q_real .> 0) && ∇Jzubov < options.δJ
                check += 1
                if check == options.extra_iter
                    # return P, Q, Zerr, ∇Jzubov
                    return P, Q, Jzubov, ∇Jzubov
                end
            else
                check = 0  # reset if not converging continuously
            end    
            
            # If the optimization did not end before the maximum iteration assign what we have best for now
            if l == options.opt_max_iter
                if (@isdefined Pbest)
                    # return Pbest, Qbest, Zerrbest, ∇Jzubovbest
                    return Pbest, Qbest, Jzubov, ∇Jzubovbest
                else
                    return P, Q, Jzubov, ∇Jzubov
                end
            end
        end
    elseif options.optimize_PandQ == "P"
        # Optimize for the P matrix
        P, Jzubov = optimize_P(X, Xdot, Q, options; Pi=Pi)
        λ_P = eigen(P).values
        λ_P_real = real.(λ_P) 
        diff_P = norm(P - P', 2)

        # Logging
        @info """[NonInt_LyapInf: Optimize P only]
        Objective value:                     $(Jzubov)
        ||P - P'||_F:                        $(diff_P)
        eigenvalues of P:                    $(round.(λ_P; sigdigits=4))
        # of Real(λp) <= 0:                  $(count(i->(i <= 0), λ_P_real))
        dimension:                           $(N)
        α:                                   $(options.α)
        """
        return P, Q, Jzubov, Jzubov

    elseif options.optimize_PandQ == "together"
        # Optimize for the P matrix
        P, Q, Jzubov = optimize_PQ(X, Xdot, options; Pi=Pi, Qi=Qi)
        λ_P = eigen(P).values
        λ_P_real = real.(λ_P) 
        λ_Q = eigen(Q).values
        λ_Q_real = real.(λ_Q)
        diff_P = norm(P - P', 2)
        diff_Q = norm(Q - Q', 2)

        # Logging
        @info """[NonInt_LyapInf: Optimize P and Q together]
        Objective value:                     $(Jzubov)
        ||P - P'||_F:                        $(diff_P)
        ||Q - Q'||_F:                        $(diff_Q)
        eigenvalues of P:                    $(round.(λ_P; sigdigits=4))
        # of Real(λp) <= 0:                  $(count(i->(i <= 0), λ_P_real))
        eigenvalues of Q:                    $(round.(λ_Q; digits=4))
        # of Real(λq) <= 0:                  $(count(i->(i <= 0), λ_Q_real))
        dimension:                           $(N)
        α:                                   $(options.α)
        β:                                   $(options.β)
        """

        return P, Q, Jzubov, Jzubov
    end
end

