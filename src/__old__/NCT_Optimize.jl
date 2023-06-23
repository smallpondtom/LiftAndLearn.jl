"""
Non-constrainted Tikhonov Optimization of Operator Inference (NCT)

# Arguments
- `D`: data matrix
- `Rt`: transpose of the derivative matrix
- `λ`: tikhonov constant
- `Xt_hat`: state data
- `dims`: important dimensions
- `options`: options for the operator inference set by the user

# Returns
- Inferred operators

"""
function NCT_Optimize(D::Matrix, Rt::Union{Matrix, Transpose}, dims::Dict, options::OpInf_options)
    # Some dimensions to unpack for convenience
    n = dims[:n]
    p = dims[:p]
    s = dims[:s]
    v = dims[:v]
    w = dims[:w]

    @info "Initialize optimization model."
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", options.max_iter)
    if !options.opt_verbose
        set_silent(model)
    end
    @variable(model, Ahat[1:n, 1:n])
    tmp = Ahat

    if options.has_control
        @variable(model, Bhat[1:n, 1:p])
        tmp = hcat(tmp, Bhat)
    end

    if options.is_quad
        if options.which_quad_term == "F"
            @variable(model, Fhat[1:n, 1:s])
            tmp = hcat(tmp, Fhat)
        else
            @variable(model, Hhat[1:n, 1:v])
            tmp = hcat(tmp, Hhat)
        end
    end

    if options.is_bilin
        @variable(model, Nhat[1:n, 1:w])
        tmp = hcat(tmp, Nhat)
    end

    if options.has_const
        @variable(model, Khat[1:n, 1])
        tmp = hcat(tmp, Khat)
    end

    Ot = @expression(model, tmp')
    @objective(model, Min, sum((D * Ot .- Rt).^2) + options.λ_tik*sum(Ot.^2))
    @info "Done."

    @info "Optimize model."
    JuMP.optimize!(model)
    @info """\n
    Constraint           = Non-Constrained 
    Tikhonov             = $(options.λ_tik)
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
    Bhat = options.has_control ? value.(Bhat)[:, :] : 0
    if options.which_quad_term == "F"
        Fhat = options.is_quad ? value.(Fhat) : 0
        Hhat = F2Hs(Fhat)
    else
        Hhat = options.is_quad ? value.(Hhat) : 0
        Fhat = H2F(Hhat)
    end
    Nhat = options.is_bilin ? value.(Nhat) : 0
    Khat = options.has_const ? value.(Khat) : 0
    @info "Done."
    return Ahat, Bhat, Fhat, Hhat, Nhat, Khat
end


"""
Non-constrainted Tikhonov Optimization of Operator Inference (NCT)

# Arguments
- `D`: data matrix
- `Rt`: transpose of the derivative matrix
- `λ`: tikhonov constant
- `Xt_hat`: state data
- `dims`: important dimensions
- `options`: options for the operator inference set by the user
- `IG`: Initial Guesses

# Returns
- Inferred operators

"""
function NCT_Optimize(D::Matrix, Rt::Union{Matrix, Transpose},
        dims::Dict, options::OpInf_options, IG::operators)
    # Some dimensions to unpack for convenience
    n = dims[:n]
    p = dims[:p]
    s = dims[:s]
    v = dims[:v]
    w = dims[:w]

    @info "Initialize optimization model."
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", options.max_iter)
    if !options.opt_verbose
        set_silent(model)
    end
    @variable(model, Ahat[1:n, 1:n])
    set_start_value.(Ahat, IG.A)
    tmp = Ahat

    if options.has_control
        @variable(model, Bhat[1:n, 1:p])
        set_start_value.(Bhat, IG.B)
        tmp = hcat(tmp, Bhat)
    end

    if options.is_quad
        if options.which_quad_term == "F"
            @variable(model, Fhat[1:n, 1:s])
            set_start_value.(Fhat, IG.F)
            tmp = hcat(tmp, Fhat)
        else
            @variable(model, Hhat[1:n, 1:v])
            set_start_value.(Hhat, IG.H)
            tmp = hcat(tmp, Hhat)
        end
    end

    if options.is_bilin
        @variable(model, Nhat[1:n, 1:w])
        set_start_value.(Nhat, IG.N)
        tmp = hcat(tmp, Nhat)
    end

    if options.has_const
        @variable(model, Khat[1:n, 1])
        set_start_value.(Khat, IG.K)
        tmp = hcat(tmp, Khat)
    end

    Ot = @expression(model, tmp')
    @objective(model, Min, sum((D * Ot .- Rt).^2) + options.λ_tik*sum(Ot.^2))
    @info "Done."

    @info "Optimize model."
    JuMP.optimize!(model)
    @info """\n
    Constraint           = Non-Constrained 
    Tikhonov             = $(options.λ_tik)
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
    Bhat = options.has_control ? value.(Bhat)[:, :] : 0
    if options.which_quad_term == "F"
        Fhat = options.is_quad ? value.(Fhat) : 0
        Hhat = F2Hs(Fhat)
    else
        Hhat = options.is_quad ? value.(Hhat) : 0
        Fhat = H2F(Hhat)
    end
    Nhat = options.is_bilin ? value.(Nhat) : 0
    Khat = options.has_const ? value.(Khat) : 0
    @info "Done."
    return Ahat, Bhat, Fhat, Hhat, Nhat, Khat
end
