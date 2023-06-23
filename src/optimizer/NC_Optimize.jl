"""
Non-constrained Optimization of Operator Inference (NC)

# Arguments
- `D`: data matrix
- `Rt`: transpose of the derivative matrix
- `dims`: important dimensions
- `options`: options for the operator inference set by the user

# Returns
- Inferred operators

"""
function NC_Optimize(D::Matrix, Rt::Union{Matrix, Transpose}, 
        dims::Dict, options::Abstract_Options)
    # Some dimensions to unpack for convenience
    n = dims[:n]
    p = dims[:p]
    s = dims[:s]
    v = dims[:v]
    w = dims[:w]

    @info "Initialize optimization model."

    # Model with options
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", options.optim.max_iter)    # maximum number of iterations
    if !options.optim.verbose
        set_silent(model)
    end

    # Linear A matrix
    if options.system.is_lin
        @variable(model, Ahat[1:n, 1:n])
        tmp = Ahat  # concatenate Ahat to objective matrix
    end

    # Input B matrix
    if options.system.has_control
        @variable(model, Bhat[1:n, 1:p])
        tmp = (@isdefined tmp) ? hcat(tmp, Bhat) : Bhat  # concatenate Bhat to objective matrix
    end

    # Quadratic F or H matrix
    if options.system.is_quad
        if options.optim.which_quad_term == "F"
            @variable(model, Fhat[1:n, 1:s])
            tmp = (@isdefined tmp) ? hcat(tmp, Fhat) : Fhat # concatenate Fhat to objective matrix
        else
            @variable(model, Hhat[1:n, 1:v])
            tmp = (@isdefined tmp) ? hcat(tmp, Hhat) : Hhat  # concatenate Hhat to objective matrix
        end
    end

    # Bilinear N matrix
    if options.system.is_bilin
        @variable(model, Nhat[1:n, 1:w])
        tmp = (@isdefined tmp) ? hcat(tmp, Nhat) : Nhat # concatenate Nhat to objective matrix
    end

    # Constant K matrix
    if options.system.has_const
        @variable(model, Khat[1:n, 1])
        tmp = (@isdefined tmp) ? hcat(tmp, Khat) : Khat # concatenate Khat to objective matrix
    end

    # Create objective matrix with JuMP expression
    Ot = @expression(model, tmp')
    if options.λ_lin == 0 && options.λ_quad == 0 
        REG = @expression(model, sum((D * Ot .- Rt).^2))
    else
        if options.optim.which_quad_term == "H"
            REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Hhat.^2))
        else
            REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Fhat.^2))
        end
    end

    # Define the objective of the optimization problem
    @objective(model, Min, REG)
    @info "Done."

    # Logging from the optimizer
    @info "Optimize model."
    JuMP.optimize!(model)
    @info """\n
    Constraint           = Non-Constrained
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

    # Assign the inferred matrices
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
Non-constrained Optimization of Operator Inference (NC)

# Arguments
- `D`: data matrix
- `Rt`: transpose of the derivative matrix
- `dims`: important dimensions
- `options`: options for the operator inference set by the user
- `IG`: Initial Guesses

# Returns
- Inferred operators

"""
function NC_Optimize(D::Matrix, Rt::Union{Matrix, Transpose}, 
        dims::Dict, options::Abstract_Options, IG::operators)
    # Some dimensions to unpack for convenience
    n = dims[:n]
    p = dims[:p]
    s = dims[:s]
    v = dims[:v]
    w = dims[:w]

    @info "Initialize optimization model."

    # Model with options
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", options.optim.max_iter)    # maximum number of iterations
    set_optimizer_attribute(model, "warm_start_init_point", "yes")  # warm start using initial conditions
    if !options.optim.verbose
        set_silent(model)
    end

    # Linear A matrix
    if options.system.is_lin
        @variable(model, Ahat[1:n, 1:n])
        set_start_value.(Ahat, IG.A)  # set initial value of Ahat
        tmp = Ahat  # temporary variable: which is the objective matrix holding linear matrix, Ahat
    end

    # Input B matrix
    if options.system.has_control
        @variable(model, Bhat[1:n, 1:p])
        set_start_value.(Bhat, IG.B) # set initial value of Bhat
        tmp = (@isdefined tmp) ? hcat(tmp, Bhat) : Bhat  # concatenate Bhat to objective matrix
    end

    # Quadratic F or H matrix
    if options.system.is_quad
        if options.optim.which_quad_term == "F"
            @variable(model, Fhat[1:n, 1:s])
            set_start_value.(Fhat, IG.F)  # set initial value of Fhat
            tmp = (@isdefined tmp) ? hcat(tmp, Fhat) : Fhat  # concatenate Fhat to objective matrix
        else
            @variable(model, Hhat[1:n, 1:v])
            set_start_value.(Hhat, IG.H)  # set initial value of Hhat
            tmp = (@isdefined tmp) ? hcat(tmp, Hhat) : Hhat # concatenate Hhat to objective matrix
        end
    end

    # Bilinear N matrix
    if options.system.is_bilin
        @variable(model, Nhat[1:n, 1:w])
        set_start_value.(Nhat, IG.N)  # set initial value of Nhat
        tmp = (@isdefined tmp) ? hcat(tmp, Nhat) : Nhat # concatenate Nhat to objective matrix
    end

    # Constant K matrix
    if options.system.has_const
        @variable(model, Khat[1:n, 1])
        set_start_value.(Khat, IG.K)  # set initial value of Khat
        tmp = (@isdefined tmp) ? hcat(tmp, Khat) : Khat  # concatenate Khat to objective matrix
    end

    # Create objective matrix with JuMP expression
    Ot = @expression(model, tmp')
    if options.λ_lin == 0 && options.λ_quad == 0 
        REG = @expression(model, sum((D * Ot .- Rt).^2))
    else
        if options.optim.which_quad_term == "H"
            REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Hhat.^2))
        else
            REG = @expression(model, sum((D * Ot .- Rt).^2) + options.λ_lin*sum(Ahat.^2) + options.λ_quad*sum(Fhat.^2))
        end
    end
    @info "Done."

    # Logging from the optimizer
    @info "Optimize model."
    JuMP.optimize!(model)
    @info """\n
    Constraint           = Non-Constrained
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

    # Assign the inferred matrices
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
Non-constrained optimization for the operator inference (only for the output)

# Arguments
- `Y`: the output matrix 
- `Xt_hat`: the state matrix

# Return
- the output state matrix `C`
"""
function NC_Optimize_output(Y::Matrix, Xhat_t::Union{Matrix, Transpose}, 
        dims::Dict, options::Abstract_Options)
    # Some dimensions to unpack for convenience
    n = dims[:n]
    q = dims[:q]

    @info "Initialize optimization model."
    Yt = transpose(Y)
    model = Model(Ipopt.Optimizer)
    if !options.optim.verbose
        set_silent(model)
    end

    @variable(model, Chat[1:q, 1:n])
    Chat_t = @expression(model, Chat')
    @objective(model, Min, sum((Matrix(Xhat_t) * Chat_t .- Yt).^2))
    @info "Done."

    @info "Optimize model."
    JuMP.optimize!(model)
    @info """[Output Optimization Results]
    Warm Start           = False
    order                = $(n)
    solve time           = $(solve_time(model))
    termination_status   = $(termination_status(model))
    primal_status        = $(primal_status(model))
    dual_state           = $(dual_status(model))
    dual objective value = $(dual_objective_value(model))
    objective_value      = $(objective_value(model))
    """

    Chat = value.(Chat)
    @info "Done."
    return Chat[:,:]
end
