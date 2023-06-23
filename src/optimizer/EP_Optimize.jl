"""
Energy preserved (Hard Equality Constraint) operator inference optimization (EPHEC)

# Arguments
- `D`: data matrix
- `Rt`: transpose of the derivative matrix
- `dims`: important dimensions
- `options`: options for the operator inference set by the user

# Returns
- Inferred operators

"""
function EPHEC_Optimize(D::Matrix, Rt::Union{Matrix,Transpose},
    dims::Dict, options::Abstract_Options)
    # Some dimensions to unpack for convenience
    n = dims[:n]
    p = dims[:p]
    s = dims[:s]
    v = dims[:v]
    w = dims[:w]

    @info "Initialize optimization model."
    model = Model(Ipopt.Optimizer; add_bridges=false)
    set_optimizer_attribute(model, "max_iter", options.optim.max_iter)
    if !options.optim.verbose
        set_silent(model)
    end

    if options.system.is_lin
        @variable(model, Ahat[1:n, 1:n])
        tmp = Ahat
    end

    if options.system.has_control
        @variable(model, Bhat[1:n, 1:p])
        tmp = (@isdefined tmp) ? hcat(tmp, Bhat) : Bhat
    end

    if options.system.is_quad
        if options.optim.which_quad_term == "H"
            @variable(model, Hhat[1:n, 1:v])
            tmp = (@isdefined tmp) ? hcat(tmp, Hhat) : Hhat
        else
            @variable(model, Fhat[1:n, 1:s])
            tmp = (@isdefined tmp) ? hcat(tmp, Fhat) : Fhat
        end
    end

    if options.system.is_bilin
        @variable(model, Nhat[1:n, 1:w])
        tmp = (@isdefined tmp) ? hcat(tmp, Nhat) : Nhat
    end

    if options.system.has_const
        @variable(model, Khat[1:n, 1])
        tmp = (@isdefined tmp) ? hcat(tmp, Khat) : Khat
    end

    Ot = @expression(model, tmp')
    if options.λ_lin == 0 && options.λ_quad == 0 
        REG = @expression(model, sum((D * Ot .- Rt).^2))
    else
        if options.which_quad_term == "H"
            REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Hhat.^2))
        else
            REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Fhat.^2))
        end
    end

    # Define the objective of the optimization problem
    @objective(model, Min, REG)

    # Energy preserving Hard equality constraint
    if options.optim.which_quad_term == "H"
        # NOTE: H matrix version
        @constraint(
            model,
            c1[i=1:n, j=1:i, k=1:j],
            Hhat[i, n*(k-1)+j] + Hhat[j, n*(k-1)+i] + Hhat[k, n*(i-1)+j] == 0
        )
    else
        # NOTE: F matrix version
        @constraint(
            model,
            c1[i=1:n, j=1:i, k=1:j],
            delta(j,k)*Fhat[i,fidx(n,j,k)] + delta(i,k)*Fhat[j,fidx(n,i,k)] + delta(j,i)*Fhat[k,fidx(n,j,i)] == 0
        )
    end
    @info "Done."

    @info "Optimize model."
    JuMP.optimize!(model)
    @info """[EP-OpInf Results]
    Constraint           = Energy-Preserving Hard Equality Constraint
    Linear Regulation    = $(options.λ_lin)
    Quadratic Regulation = $(options.λ_quad)
    Warm Start           = $(options.initial_guess_for_opt)
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
Energy preserved (Hard Equality Constraint) operator inference optimization (EPHEC)

# Arguments
- `D`: data matrix
- `Rt`: transpose of the derivative matrix
- `dims`: important dimensions
- `options`: options for the operator inference set by the user
- `IG`: Initial Guesses

# Returns
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
    model = Model(Ipopt.Optimizer; add_bridges=false)
    set_optimizer_attribute(model, "max_iter", options.optim.max_iter)
    if !options.optim.verbose
        set_silent(model)
    end

    if options.system.is_lin
        @variable(model, Ahat[1:n, 1:n])
        set_start_value.(Ahat, IG.A)
        tmp = Ahat
    end

    if options.system.has_control
        @variable(model, Bhat[1:n, 1:p])
        set_start_value.(Bhat, IG.B)
        tmp = (@isdefined tmp) ? hcat(tmp, Bhat) : Bhat
    end

    if options.system.is_quad
        if options.optim.which_quad_term == "H"
            @variable(model, Hhat[1:n, 1:v])
            set_start_value.(Hhat, IG.H)
            tmp = (@isdefined tmp) ? hcat(tmp, Hhat) : Hhat
        else
            @variable(model, Fhat[1:n, 1:s])
            set_start_value.(Fhat, IG.F)
            tmp = (@isdefined tmp) ? hcat(tmp, Fhat) : Fhat
        end
    end

    if options.system.is_bilin
        @variable(model, Nhat[1:n, 1:w])
        set_start_value.(Nhat, IG.N)
        tmp = (@isdefined tmp) ? hcat(tmp, Nhat) : Nhat
    end

    if options.system.has_const
        @variable(model, Khat[1:n, 1])
        set_start_value.(Khat, IG.K)
        tmp = (@isdefined tmp) ? hcat(tmp, Khat) : Khat
    end

    Ot = @expression(model, tmp')
    if options.λ_lin == 0 && options.λ_quad == 0 
        REG = @expression(model, sum((D * Ot .- Rt).^2))
    else
        if options.which_quad_term == "H"
            REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Hhat.^2))
        else
            REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Fhat.^2))
        end
    end

    # Define the objective of the optimization problem
    @objective(model, Min, REG)

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
    @info "Done."

    @info "Optimize model."
    JuMP.optimize!(model)
    @info """[EP-OpInf Results]
    Constraint           = Energy-Preserving Hard Equality Constraint
    Linear Regulation    = $(options.λ_lin)
    Quadratic Regulation = $(options.λ_quad)
    Warm Start           = $(options.initial_guess_for_opt)
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
Energy preserved (Soft Inequality Constraint) operator inference optimization (EPSIC)

# Arguments
- `D`: data matrix
- `Rt`: transpose of the derivative matrix
- `dims`: important dimensions
- `options`: options for the operator inference set by the user

# Returns
- Inferred operators

"""
function EPSIC_Optimize(D::Matrix, Rt::Union{Matrix,Transpose},
    dims::Dict, options::Abstract_Options)
    # Some dimensions to unpack for convenience
    n = dims[:n]
    p = dims[:p]
    s = dims[:s]
    v = dims[:v]
    w = dims[:w]

    @info "Initialize optimization model."
    model = Model(Ipopt.Optimizer; add_bridges = false)
    set_optimizer_attribute(model, "max_iter", options.optim.max_iter)
    if !options.optim.verbose
        set_silent(model)
    end

    if options.system.is_lin
        @variable(model, Ahat[1:n, 1:n])
        tmp = Ahat
    end

    if options.system.has_control
        @variable(model, Bhat[1:n, 1:p])
        tmp = (@isdefined tmp) ? hcat(tmp, Bhat) : Bhat
    end

    if options.system.is_quad
        if options.optim.which_quad_term == "H"
            @variable(model, Hhat[1:n, 1:v])
            tmp = (@isdefined tmp) ? hcat(tmp, Hhat) : Hhat
        else
            @variable(model, Fhat[1:n, 1:s])
            tmp = (@isdefined tmp) ? hcat(tmp, Fhat) : Fhat
        end
    end

    if options.system.is_bilin
        @variable(model, Nhat[1:n, 1:w])
        tmp = (@isdefined tmp) ? hcat(tmp, Nhat) : Nhat
    end

    if options.system.has_const
        @variable(model, Khat[1:n, 1])
        tmp = (@isdefined tmp) ? hcat(tmp, Khat) : Khat
    end

    Ot = @expression(model, tmp')
    if options.λ_lin == 0 && options.λ_quad == 0 
        REG = @expression(model, sum((D * Ot .- Rt).^2))
    else
        if options.which_quad_term == "H"
            REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Hhat.^2))
        else
            REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Fhat.^2))
        end
    end

    # Define the objective of the optimization problem
    @objective(model, Min, REG)

    # Energy preserving Soft Inequality constraints
    if options.optim.which_quad_term == "H"
        # NOTE: H matrix version
        @constraint(
            model,
            c1[i=1:n, j=1:i, k=1:j],
            Hhat[i, n*(k-1)+j] + Hhat[j, n*(k-1)+i] + Hhat[k, n*(i-1)+j] .<= options.ϵ_ep
        )
        @constraint(
            model,
            c2[i=1:n, j=1:i, k=1:j],
            Hhat[i, n*(k-1)+j] + Hhat[j, n*(k-1)+i] + Hhat[k, n*(i-1)+j] .>= -options.ϵ_ep
        )
    else
        # NOTE: F matrix version
        @constraint(
            model,
            c1[i=1:n, j=1:i, k=1:j],
            delta(j,k)*Fhat[i,fidx(n,j,k)] + delta(i,k)*Fhat[j,fidx(n,i,k)] + delta(j,i)*Fhat[k,fidx(n,j,i)] .<= options.ϵ_ep
        )
        @constraint(
            model,
            c2[i=1:n, j=1:i, k=1:j],
            delta(j,k)*Fhat[i,fidx(n,j,k)] + delta(i,k)*Fhat[j,fidx(n,i,k)] + delta(j,i)*Fhat[k,fidx(n,j,i)] .>= -options.ϵ_ep
        )
    end
    @info "Done."

    @info "Optimize model."
    JuMP.optimize!(model)
    @info """[EP-OpInf Results]
    Constraint           = Energy-Preserving Soft Inequality Constraint
    Linear Regulation    = $(options.λ_lin)
    Quadratic Regulation = $(options.λ_quad)
    Warm Start           = $(options.initial_guess_for_opt)
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
Energy preserved (Soft Inequality Constraint) operator inference optimization (EPSIC)

# Arguments
- `D`: data matrix
- `Rt`: transpose of the derivative matrix
- `dims`: important dimensions
- `options`: options for the operator inference set by the user
- `IG`: Initial Guesses

# Returns
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
    model = Model(Ipopt.Optimizer; add_bridges=false)
    set_optimizer_attribute(model, "max_iter", options.optim.max_iter)
    if !options.optim.verbose
        set_silent(model)
    end

    if options.system.is_lin
        @variable(model, Ahat[1:n, 1:n])
        set_start_value.(Ahat, IG.A)
        tmp = Ahat
    end

    if options.system.has_control
        @variable(model, Bhat[1:n, 1:p])
        set_start_value.(Bhat, IG.B)
        tmp = (@isdefined tmp) ? hcat(tmp, Bhat) : Bhat
    end

    if options.system.is_quad
        if options.optim.which_quad_term == "H"
            @variable(model, Hhat[1:n, 1:v])
            set_start_value.(Hhat, IG.H)
            tmp = (@isdefined tmp) ? hcat(tmp, Hhat) : Hhat
        else
            @variable(model, Fhat[1:n, 1:s])
            set_start_value.(Fhat, IG.F)
            tmp = (@isdefined tmp) ? hcat(tmp, Fhat) : Fhat
        end
    end

    if options.system.is_bilin
        @variable(model, Nhat[1:n, 1:w])
        set_start_value.(Nhat, IG.N)
        tmp = (@isdefined tmp) ? hcat(tmp, Nhat) : Nhat
    end

    if options.system.has_const
        @variable(model, Khat[1:n, 1])
        set_start_value.(Khat, IG.K)
        tmp = (@isdefined tmp) ? hcat(tmp, Khat) : Khat
    end

    Ot = @expression(model, tmp')
    if options.λ_lin == 0 && options.λ_quad == 0 
        REG = @expression(model, sum((D * Ot .- Rt).^2))
    else
        if options.which_quad_term == "H"
            REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Hhat.^2))
        else
            REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Fhat.^2))
        end
    end

    # Define the objective of the optimization problem
    @objective(model, Min, REG)

    # Energy preserving Soft Inequality constraints
    if options.optim.which_quad_term == "H"
        # NOTE: H matrix version
        @constraint(
            model,
            c1[i=1:n, j=1:i, k=1:j],
            Hhat[i, n*(k-1)+j] + Hhat[j, n*(k-1)+i] + Hhat[k, n*(i-1)+j] .<= options.ϵ_ep
        )
        @constraint(
            model,
            c1[i=1:n, j=1:i, k=1:j],
            Hhat[i, n*(k-1)+j] + Hhat[j, n*(k-1)+i] + Hhat[k, n*(i-1)+j] .>= -options.ϵ_ep
        )
    else
        # NOTE: F matrix version
        @constraint(
            model,
            c1[i=1:n, j=1:i, k=1:j],
            delta(j,k)*Fhat[i,fidx(n,j,k)] + delta(i,k)*Fhat[j,fidx(n,i,k)] + delta(j,i)*Fhat[k,fidx(n,j,i)] .<= options.ϵ_ep
        )
        @constraint(
            model,
            c2[i=1:n, j=1:i, k=1:j],
            delta(j,k)*Fhat[i,fidx(n,j,k)] + delta(i,k)*Fhat[j,fidx(n,i,k)] + delta(j,i)*Fhat[k,fidx(n,j,i)] .>= -options.ϵ_ep
        )
    end
    @info "Done."

    @info "Optimize model."
    JuMP.optimize!(model)
    @info """[EP-OpInf Results]
    Constraint           = Energy-Preserving Soft Inequality Constraint
    Linear Regulation    = $(options.λ_lin)
    Quadratic Regulation = $(options.λ_quad)
    Warm Start           = $(options.initial_guess_for_opt)
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
Energy preserved unconstrained operator inference optimization (EPUC)

# Arguments
- `D`: data matrix
- `Rt`: transpose of the derivative matrix
- `dims`: important dimensions
- `options`: options for the operator inference set by the user
- `IG`: Initial Guesses

# Returns
- Inferred operators

"""
function EPUC_Optimize(D::Matrix, Rt::Union{Matrix,Transpose},
    dims::Dict, options::Abstract_Options, IG::operators)
    # Some dimensions to unpack for convenience
    n = dims[:n]
    p = dims[:p]
    s = dims[:s]
    v = dims[:v]
    w = dims[:w]

    @info "Initialize optimization model."
    model = Model(Ipopt.Optimizer; add_bridges=false)
    set_optimizer_attribute(model, "max_iter", options.optim.max_iter)
    if !options.optim.verbose
        set_silent(model)
    end

    if options.system.is_lin
        @variable(model, Ahat[1:n, 1:n])
        tmp = Ahat
    end

    if options.system.has_control
        @variable(model, Bhat[1:n, 1:p])
        tmp = (@isdefined tmp) ? hcat(tmp, Bhat) : Bhat
    end

    if options.system.is_quad
        if options.optim.which_quad_term == "H"
            @variable(model, Hhat[1:n, 1:v])
            tmp = (@isdefined tmp) ? hcat(tmp, Hhat) : Hhat
        else
            @variable(model, Fhat[1:n, 1:s])
            tmp = (@isdefined tmp) ? hcat(tmp, Fhat) : Fhat
        end
    end

    if options.system.is_bilin
        @variable(model, Nhat[1:n, 1:w])
        tmp = (@isdefined tmp) ? hcat(tmp, Nhat) : Nhat
    end

    if options.system.has_const
        @variable(model, Khat[1:n, 1])
        tmp = (@isdefined tmp) ? hcat(tmp, Khat) : Khat
    end

    Ot = @expression(model, tmp')
    if options.λ_lin == 0 && options.λ_quad == 0 
        REG = @expression(model, sum((D * Ot .- Rt).^2))
    else
        if options.which_quad_term == "H"
            REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Hhat.^2))
        else
            REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Fhat.^2))
        end
    end

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

    # Define the objective of the optimization problem
    @objective(model, Min, REG + options.α * EP)
    @info "Done."

    @info "Optimize model."
    JuMP.optimize!(model)
    @info """[EP-OpInf Results]
    Constraint           = Energy-Preserving Unconstrained
    EP Weight            = $(options.α)
    Linear Regulation    = $(options.λ_lin)
    Quadratic Regulation = $(options.λ_quad)
    Warm Start           = $(options.initial_guess_for_opt)
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