export EPHEC_Optimize, EPSIC_Optimize, EPP_Optimize

"""
    EPHEC_Optimize(D::Matrix, Rt::Union{Matrix,Transpose}, dims::Dict, 
        options::Abstract_Options, IG::operators) → Ahat, Bhat, Fhat, Hhat, Nhat, Khat

Energy preserved (Hard Equality Constraint) operator inference optimization (EPHEC)

## Arguments
- `D`: data matrix
- `Rt`: transpose of the derivative matrix
- `dims`: important dimensions
- `options`: options for the operator inference set by the user
- `IG`: Initial Guesses

## Returns
- Inferred operators
"""
function EPHEC_Optimize(D::Matrix, Rt::Union{Matrix,Transpose},
    dims::Dict, options::Abstract_Options, IG::operators)
    # Some dimensions to unpack for convenience
    n = dims[:n]
    p = dims[:p]
    s = dims[:s]
    v = dims[:v]
    w = dims[:w]

    @info "Initialize optimization model."
    model = Model(Ipopt.Optimizer; add_bridges = false)
    if options.optim.linear_solver != "none"
        set_attribute(model, "hsllib", options.optim.HSL_lib_path)
        set_attribute(model, "linear_solver", options.optim.linear_solver)
    end
    set_optimizer_attribute(model, "max_iter", options.optim.max_iter)
    set_string_names_on_creation(model, false)
    if !options.optim.verbose
        set_silent(model)
    end

    if options.system.is_lin
        @variable(model, Ahat[1:n, 1:n])

        # Set it to be symmetric or nearly symmetric
        # @expression(model, Ahat_s, 0.5 * (Ahat + Ahat'))
        if options.optim.initial_guess 
            if options.optim.SIGE
                ni = size(IG.A, 1)
                set_start_value.(Ahat[1:ni, 1:ni], IG.A)
            else
                set_start_value.(Ahat, IG.A)
            end
        end

        if options.optim.with_bnds
            set_lower_bound.(Ahat, options.A_bnds[1])
            set_upper_bound.(Ahat, options.A_bnds[2])
        end

        tmp = Ahat
    end

    if options.system.has_control
        @variable(model, Bhat[1:n, 1:p])
        if options.optim.initial_guess 
            set_start_value.(Bhat, IG.B)
        end
        tmp = (@isdefined tmp) ? hcat(tmp, Bhat) : Bhat
    end

    if options.system.is_quad
        if options.optim.which_quad_term == "H"
            @variable(model, Hhat[1:n, 1:v])
            if options.optim.initial_guess
                set_start_value.(Hhat, IG.H)
            end
            tmp = (@isdefined tmp) ? hcat(tmp, Hhat) : Hhat
        else
            @variable(model, Fhat[1:n, 1:s])
            
            if options.optim.initial_guess
                if options.optim.SIGE
                    # Insert the lower dimension initial guess into optimization variable F
                    ni = size(IG.F, 1)
                    xsq_idx = [1 + (n + 1) * (k - 1) - k * (k - 1) / 2 for k in 1:n]
                    insert_idx = [collect(x:x+(ni-i)) for (i, x) in enumerate(xsq_idx[1:ni])]
                    idx = Int.(reduce(vcat, insert_idx))
                    set_start_value.(Fhat[1:ni, idx], IG.F)
                else
                    set_start_value.(Fhat, IG.F)
                end
            end

            if options.optim.with_bnds
                set_lower_bound.(Fhat, options.ForH_bnds[1])
                set_upper_bound.(Fhat, options.ForH_bnds[2])
            end

            tmp = (@isdefined tmp) ? hcat(tmp, Fhat) : Fhat
        end
    end

    if options.system.is_bilin
        @variable(model, Nhat[1:n, 1:w])
        if options.optim.initial_guess
            set_start_value.(Nhat, IG.N)
        end
        tmp = (@isdefined tmp) ? hcat(tmp, Nhat) : Nhat
    end

    if options.system.has_const
        @variable(model, Khat[1:n, 1])
        if options.optim.initial_guess
            set_start_value.(Khat, IG.K)
        end
        tmp = (@isdefined tmp) ? hcat(tmp, Khat) : Khat
    end

    @info "Add constraints"
    # Energy preserving Hard equality constraint
    if options.optim.which_quad_term == "H"
        # NOTE: H matrix version
        @constraint(
            model,
            c1[i=1:n, j=1:i, k=1:j],
            Hhat[i, n*(k-1)+j] + Hhat[k, n*(k-1)+i] + Hhat[k, n*(i-1)+j] == 0
        )
    else
        # NOTE: F matrix version
        @constraint(
            model,
            c1[i=1:n, j=1:i, k=1:j],
            delta(j,k)*Fhat[i,fidx(n,j,k)] + delta(i,k)*Fhat[j,fidx(n,i,k)] + delta(j,i)*Fhat[k,fidx(n,j,i)] == 0
        )
    end

    @info "Set objective"
    Ot = @expression(model, tmp')

    # INFO: This is the original objective function
    # REG = @expression(model, sum((D * Ot .- Rt).^2))

    # INFO: This is the constrained version that was recommended online
    n_tmp, m_tmp = size(Rt)
    @variable(model, X[1:n_tmp, 1:m_tmp])
    @constraint(model, X .== D * Ot .- Rt)  # this method answer on: https://discourse.julialang.org/t/write-large-least-square-like-problems-in-jump/35931

    if (options.λ_lin == 0 && options.λ_quad == 0)
        # @objective(model, Min, REG)
        @objective(model, Min, sum(X.^2))
    else
        if options.optim.which_quad_term == "H"
            PEN = @expression(model, options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Hhat.^2))
            @objective(model, Min, sum(X.^2) + PEN)
        else
            PEN = @expression(model, options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Fhat.^2))
            @objective(model, Min, sum(X.^2) + PEN)
        end
    end

    # Symmetry inequality constraint
    # @constraint(model, c2[i=1:n, j=1:n], Ahat[i,j] - Ahat[j,i] .<= 1e-2)
    # @constraint(model, c3[i=1:n, j=1:n], Ahat[i,j] - Ahat[j,i] .>= -1e-2)
    # Eigenvalue on diagonals equality constraint
    # @constraint(model, c4[i=1:n], Ahat[i,i] + 1e-3 <= 0)
    # y = (x) -> 2/x + 1
    # @constraint(model, c4[i=1:n-1], Ahat[i,i] * y(i) >= Ahat[i+1,i+1])
    # @constraint(model, c4[i=1:n, j=1:n], Ahat[i,i] .>= Ahat[i,j])

    @info "Done."

    @info "Optimize model."
    JuMP.optimize!(model)
    @info """[EP-OpInf Results]
    Constraint           = Energy-Preserving Hard Equality Constraint
    Linear Regulation    = $(options.λ_lin)
    Quadratic Regulation = $(options.λ_quad)
    Warm Start           = $(options.optim.initial_guess)
    order                = $(n)
    solve time           = $(solve_time(model))
    termination_status   = $(termination_status(model))
    primal_status        = $(primal_status(model))
    dual_state           = $(dual_status(model))
    dual objective value = $(dual_objective_value(model))
    objective_value      = $(objective_value(model))
    """

    Ahat = value.(Ahat)
    Bhat = options.system.has_control ? value.(Bhat)[:, :] : 0
    Fhat = options.system.is_quad ? options.optim.which_quad_term=="F" ? value.(Fhat) : H2F(value.(Hhat)) : 0
    Hhat = options.system.is_quad ? options.optim.which_quad_term=="H" ? value.(Hhat) : F2Hs(value.(Fhat)) : 0
    Nhat = options.system.is_bilin ? value.(Nhat) : 0
    Khat = options.system.has_const ? value.(Khat) : 0
    @info "Done."
    return Ahat, Bhat, Fhat, Hhat, Nhat, Khat
end


"""
    EPSIC_Optimize(D::Matrix, Rt::Union{Matrix,Transpose}, dims::Dict, 
        options::Abstract_Options, IG::operators) → Ahat, Bhat, Fhat, Hhat, Nhat, Khat

Energy preserved (Soft Inequality Constraint) operator inference optimization (EPSIC)

## Arguments
- `D`: data matrix
- `Rt`: transpose of the derivative matrix
- `dims`: important dimensions
- `options`: options for the operator inference set by the user
- `IG`: Initial Guesses

## Returns
- Inferred operators

"""
function EPSIC_Optimize(D::Matrix, Rt::Union{Matrix,Transpose},
    dims::Dict, options::Abstract_Options, IG::operators)
    # Some dimensions to unpack for convenience
    n = dims[:n]
    p = dims[:p]
    s = dims[:s]
    v = dims[:v]
    w = dims[:w]

    @info "Initialize optimization model."
    model = Model(Ipopt.Optimizer; add_bridges = false)
    if options.optim.linear_solver != "none"
        set_attribute(model, "hsllib", options.optim.HSL_lib_path)
        set_attribute(model, "linear_solver", options.optim.linear_solver)
    end
    set_optimizer_attribute(model, "max_iter", options.optim.max_iter)
    set_string_names_on_creation(model, false)
    if !options.optim.verbose
        set_silent(model)
    end

    if options.system.is_lin
        @variable(model, Ahat[1:n, 1:n])

        # # Set it to be symmetric or nearly symmetric
        # @expression(model, Ahat_s, 0.5 * (Ahat + Ahat'))
        if options.optim.initial_guess 
            if options.optim.SIGE
                ni = size(IG.A, 1)
                set_start_value.(Ahat[1:ni, 1:ni], IG.A)
            else
                set_start_value.(Ahat, IG.A)
            end
        end

        if options.optim.with_bnds
            set_lower_bound.(Ahat, options.A_bnds[1])
            set_upper_bound.(Ahat, options.A_bnds[2])
        end

        tmp = Ahat
    end

    if options.system.has_control
        @variable(model, Bhat[1:n, 1:p])
        if options.optim.initial_guess
            set_start_value.(Bhat, IG.B)
        end
        tmp = (@isdefined tmp) ? hcat(tmp, Bhat) : Bhat
    end

    if options.system.is_quad
        if options.optim.which_quad_term == "H"
            @variable(model, Hhat[1:n, 1:v])
            if options.optim.initial_guess
                set_start_value.(Hhat, IG.H)
            end
            tmp = (@isdefined tmp) ? hcat(tmp, Hhat) : Hhat
        else
            @variable(model, Fhat[1:n, 1:s])

            if options.optim.initial_guess
                if options.optim.SIGE
                    # Insert the lower dimension initial guess into optimization variable F
                    ni = size(IG.F, 1)
                    xsq_idx = [1 + (n + 1) * (k - 1) - k * (k - 1) / 2 for k in 1:n]
                    insert_idx = [collect(x:x+(ni-i)) for (i, x) in enumerate(xsq_idx[1:ni])]
                    idx = Int.(reduce(vcat, insert_idx))
                    set_start_value.(Fhat[1:ni, idx], IG.F)
                else
                    set_start_value.(Fhat, IG.F)
                end
            end

            if options.optim.with_bnds
                set_lower_bound.(Fhat, options.ForH_bnds[1])
                set_upper_bound.(Fhat, options.ForH_bnds[2])
            end

            tmp = (@isdefined tmp) ? hcat(tmp, Fhat) : Fhat
        end
    end

    if options.system.is_bilin
        @variable(model, Nhat[1:n, 1:w])
        if options.optim.initial_guess
            set_start_value.(Nhat, IG.N)
        end
        tmp = (@isdefined tmp) ? hcat(tmp, Nhat) : Nhat
    end

    if options.system.has_const
        @variable(model, Khat[1:n, 1])
        if options.optim.initial_guess
            set_start_value.(Khat, IG.K)
        end
        tmp = (@isdefined tmp) ? hcat(tmp, Khat) : Khat
    end

    # Ot = @expression(model, tmp')
    # if options.λ_lin == 0 && options.λ_quad == 0 
    #     REG = @expression(model, sum((D * Ot .- Rt).^2))
    # else
    #     if options.optim.which_quad_term == "H"
    #         REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Hhat.^2))
    #     else
    #         REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Fhat.^2))
    #     end
    # end

    @info "Set objective"
    Ot = @expression(model, tmp')

    # INFO: This is the original objective function
    # REG = @expression(model, sum((D * Ot .- Rt).^2))

    # INFO: This is the constrained version that was recommended online
    n_tmp, m_tmp = size(Rt)
    @variable(model, X[1:n_tmp, 1:m_tmp])
    @constraint(model, X .== D * Ot .- Rt)  # this method answer on: https://discourse.julialang.org/t/write-large-least-square-like-problems-in-jump/35931

    if (options.λ_lin == 0 && options.λ_quad == 0)
        # @objective(model, Min, REG)
        @objective(model, Min, sum(X.^2))
    else
        if options.optim.which_quad_term == "H"
            PEN = @expression(model, options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Hhat.^2))
            @objective(model, Min, sum(X.^2) + PEN)
        else
            PEN = @expression(model, options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Fhat.^2))
            @objective(model, Min, sum(X.^2) + PEN)
        end
    end

    # Define the objective of the optimization problem
    # @objective(model, Min, REG)

    @info "Add constraints"
    # Energy preserving Soft Inequality constraints
    if options.optim.which_quad_term == "H"
        # NOTE: H matrix version
        @constraint(
            model,
            c1[i=1:n, j=1:i, k=1:j],
            Hhat[i, n*(k-1)+j] + Hhat[j, n*(k-1)+i] + Hhat[k, n*(i-1)+j] .<= options.ϵ
        )
        @constraint(
            model,
            c1[i=1:n, j=1:i, k=1:j],
            Hhat[i, n*(k-1)+j] + Hhat[j, n*(k-1)+i] + Hhat[k, n*(i-1)+j] .>= -options.ϵ
        )
    else
        # NOTE: F matrix version
        @constraint(
            model,
            c1[i=1:n, j=1:i, k=1:j],
            delta(j,k)*Fhat[i,fidx(n,j,k)] + delta(i,k)*Fhat[j,fidx(n,i,k)] + delta(j,i)*Fhat[k,fidx(n,j,i)] .<= options.ϵ
        )
        @constraint(
            model,
            c2[i=1:n, j=1:i, k=1:j],
            delta(j,k)*Fhat[i,fidx(n,j,k)] + delta(i,k)*Fhat[j,fidx(n,i,k)] + delta(j,i)*Fhat[k,fidx(n,j,i)] .>= -options.ϵ
        )
    end

    # Symmetry inequality constraint
    # @constraint(model, c3[i=1:n, j=1:n], Ahat[i,j] - Ahat[j,i] .<= 1e-2)
    # @constraint(model, c4[i=1:n, j=1:n], Ahat[i,j] - Ahat[j,i] .>= -1e-2)
    # Eigenvalue on diagonals equality constraint
    # y = (x) -> 2/x + 1
    # @constraint(model, c5[i=1:n-1], Ahat[i,i] * y(i) >= Ahat[i+1,i+1])

    @info "Done."

    @info "Optimize model."
    JuMP.optimize!(model)
    @info """[EP-OpInf Results]
    Constraint           = Energy-Preserving Soft Inequality Constraint
    Linear Regulation    = $(options.λ_lin)
    Quadratic Regulation = $(options.λ_quad)
    Warm Start           = $(options.optim.initial_guess)
    order                = $(n)
    solve time           = $(solve_time(model))
    termination_status   = $(termination_status(model))
    primal_status        = $(primal_status(model))
    dual_state           = $(dual_status(model))
    dual objective value = $(dual_objective_value(model))
    objective_value      = $(objective_value(model))
    """

    Ahat = value.(Ahat)
    Bhat = options.system.has_control ? value.(Bhat)[:, :] : 0
    Fhat = options.system.is_quad ? options.optim.which_quad_term=="F" ? value.(Fhat) : H2F(value.(Hhat)) : 0
    Hhat = options.system.is_quad ? options.optim.which_quad_term=="H" ? value.(Hhat) : F2Hs(value.(Fhat)) : 0
    Nhat = options.system.is_bilin ? value.(Nhat) : 0
    Khat = options.system.has_const ? value.(Khat) : 0
    @info "Done."
    return Ahat, Bhat, Fhat, Hhat, Nhat, Khat
end


"""
    EPP_Optimize(D::Matrix, Rt::Union{Matrix,Transpose}, dims::Dict, 
        options::Abstract_Options, IG::operators) → Ahat, Bhat, Fhat, Hhat, Nhat, Khat

Energy preserving penalty operator inference optimization (EPP)

## Arguments
- `D`: data matrix
- `Rt`: transpose of the derivative matrix
- `dims`: important dimensions
- `options`: options for the operator inference set by the user
- `IG`: Initial Guesses

## Returns
- Inferred operators

"""
function EPP_Optimize(D::Matrix, Rt::Union{Matrix,Transpose},
    dims::Dict, options::Abstract_Options, IG::operators)
    # Some dimensions to unpack for convenience
    n = dims[:n]
    p = dims[:p]
    s = dims[:s]
    v = dims[:v]
    w = dims[:w]

    @info "Initialize optimization model."
    model = Model(Ipopt.Optimizer; add_bridges = false)
    if options.optim.linear_solver != "none"
        set_attribute(model, "hsllib", options.optim.HSL_lib_path)
        set_attribute(model, "linear_solver", options.optim.linear_solver)
    end
    set_optimizer_attribute(model, "max_iter", options.optim.max_iter)
    set_string_names_on_creation(model, false)
    if !options.optim.verbose
        set_silent(model)
    end

    if options.system.is_lin
        @variable(model, Ahat[1:n, 1:n])

        # # Set it to be symmetric or nearly symmetric
        # @expression(model, Ahat_s = 0.5 * (Ahat + Ahat'))
        
        if options.optim.initial_guess 
            if options.optim.SIGE
                ni = size(IG.A, 1)
                set_start_value.(Ahat[1:ni, 1:ni], IG.A)
            else
                set_start_value.(Ahat, IG.A)
            end
        end

        if options.optim.with_bnds
            set_lower_bound.(Ahat, options.A_bnds[1])
            set_upper_bound.(Ahat, options.A_bnds[2])
        end
        
        tmp = Ahat
    end

    if options.system.has_control
        @variable(model, Bhat[1:n, 1:p])
        if options.optim.initial_guess
            set_start_value.(Bhat, IG.B)
        end
        tmp = (@isdefined tmp) ? hcat(tmp, Bhat) : Bhat
    end

    if options.system.is_quad
        if options.optim.which_quad_term == "H"
            @variable(model, Hhat[1:n, 1:v])
            if options.optim.initial_guess
                set_start_value.(Hhat, IG.H)
            end
            tmp = (@isdefined tmp) ? hcat(tmp, Hhat) : Hhat
        else
            @variable(model, Fhat[1:n, 1:s])

            if options.optim.initial_guess
                if options.optim.SIGE
                    # Insert the lower dimension initial guess into optimization variable F
                    ni = size(IG.F, 1)
                    xsq_idx = [1 + (n + 1) * (k - 1) - k * (k - 1) / 2 for k in 1:n]
                    insert_idx = [collect(x:x+(ni-i)) for (i, x) in enumerate(xsq_idx[1:ni])]
                    idx = Int.(reduce(vcat, insert_idx))
                    set_start_value.(Fhat[1:ni, idx], IG.F)
                else
                    set_start_value.(Fhat, IG.F)
                end
            end

            if options.optim.with_bnds
                set_lower_bound.(Fhat, options.ForH_bnds[1])
                set_upper_bound.(Fhat, options.ForH_bnds[2])
            end

            tmp = (@isdefined tmp) ? hcat(tmp, Fhat) : Fhat
        end
    end

    if options.system.is_bilin
        @variable(model, Nhat[1:n, 1:w])
        if options.optim.initial_guess
            set_start_value.(Nhat, IG.N)
        end
        tmp = (@isdefined tmp) ? hcat(tmp, Nhat) : Nhat
    end

    if options.system.has_const
        @variable(model, Khat[1:n, 1])
        if options.optim.initial_guess
            set_start_value.(Khat, IG.K)
        end
        tmp = (@isdefined tmp) ? hcat(tmp, Khat) : Khat
    end

    # Ot = @expression(model, tmp')
    # if options.λ_lin == 0 && options.λ_quad == 0 
    #     REG = @expression(model, sum((D * Ot .- Rt).^2))
    # else
    #     if options.optim.which_quad_term == "H"
    #         REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Hhat.^2))
    #     else
    #         REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Fhat.^2))
    #     end
    # end

    @info "Set penalty"
    # Unconstrained part for energy-preservation
    if options.optim.which_quad_term == "H"
        EP = @expression(
            model, 
            sum(sum((Hhat[i, n*(k-1)+j] + Hhat[k, n*(k-1)+i] + Hhat[k, n*(i-1)+j]).^2) for i=1:n, j=1:i, k=1:j)
        )
    else
        EP = @expression(
            model,
            sum(sum((delta(j,k)*Fhat[i,fidx(n,j,k)] + delta(i,k)*Fhat[j,fidx(n,i,k)] + delta(j,i)*Fhat[k,fidx(n,j,i)]).^2) for i=1:n, j=1:i, k=1:j)
        )
    end

    @info "Set objective"
    Ot = @expression(model, tmp')

    # INFO: This is the original objective function
    # REG = @expression(model, sum((D * Ot .- Rt).^2))

    # INFO: This is the constrained version that was recommended online
    n_tmp, m_tmp = size(Rt)
    @variable(model, X[1:n_tmp, 1:m_tmp])
    @constraint(model, X .== D * Ot .- Rt)  # this method answer on: https://discourse.julialang.org/t/write-large-least-square-like-problems-in-jump/35931

    if (options.λ_lin == 0 && options.λ_quad == 0)
        # @objective(model, Min, REG)
        @objective(model, Min, sum(X.^2) + options.α * EP)
    else
        if options.optim.which_quad_term == "H"
            PEN = @expression(model, options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Hhat.^2))
            @objective(model, Min, sum(X.^2) + PEN + options.α * EP)
        else
            PEN = @expression(model, options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Fhat.^2))
            @objective(model, Min, sum(X.^2) + PEN + options.α * EP)
        end
    end

    # Symmetry inequality constraint
    # @constraint(model, c1[i=1:n, j=1:n], Ahat[i,j] - Ahat[j,i] .<= 1e-2)
    # @constraint(model, c2[i=1:n, j=1:n], Ahat[i,j] - Ahat[j,i] .>= -1e-2)
    # Eigenvalue on diagonals equality constraint
    # y = (x) -> 2/x + 1
    # @constraint(model, c3[i=1:n-1], Ahat[i,i] * y(i) >= Ahat[i+1,i+1])

    # Define the objective of the optimization problem
    # @objective(model, Min, REG + options.α * EP)
    @info "Done."

    @info "Optimize model."
    JuMP.optimize!(model)
    @info """[EP-OpInf Results]
    Constraint           = Energy-Preserving Unconstrained
    EP Weight            = $(options.α)
    Linear Regulation    = $(options.λ_lin)
    Quadratic Regulation = $(options.λ_quad)
    Warm Start           = $(options.optim.initial_guess)
    order                = $(n)
    solve time           = $(solve_time(model))
    termination_status   = $(termination_status(model))
    primal_status        = $(primal_status(model))
    dual_state           = $(dual_status(model))
    dual objective value = $(dual_objective_value(model))
    objective_value      = $(objective_value(model))
    """

    Ahat = value.(Ahat)
    Bhat = options.system.has_control ? value.(Bhat)[:, :] : 0
    Fhat = options.system.is_quad ? options.optim.which_quad_term=="F" ? value.(Fhat) : H2F(value.(Hhat)) : 0
    Hhat = options.system.is_quad ? options.optim.which_quad_term=="H" ? value.(Hhat) : F2Hs(value.(Fhat)) : 0
    Nhat = options.system.is_bilin ? value.(Nhat) : 0
    Khat = options.system.has_const ? value.(Khat) : 0
    @info "Done."
    return Ahat, Bhat, Fhat, Hhat, Nhat, Khat
end

