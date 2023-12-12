export NC_Optimize, NC_Optimize_output

"""
    NC_Optimize(D::Matrix, Rt::Union{Matrix, Transpose}, 
        dims::Dict, options::Abstract_Options, IG::operators)

Optimization version of Standard Operator Inference (NC)

## Arguments
- `D`: data matrix
- `Rt`: transpose of the derivative matrix
- `dims`: important dimensions
- `options`: options for the operator inference set by the user
- `IG`: Initial Guesses

## Returns
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

    # Linear A matrix
    if options.system.is_lin
        @variable(model, Ahat[1:n, 1:n])
        if options.optim.initial_guess
            set_start_value.(Ahat, IG.A)  # set initial value of Ahat
        end
        tmp = Ahat  # temporary variable: which is the objective matrix holding linear matrix, Ahat
    end

    # Input B matrix
    if options.system.has_control
        @variable(model, Bhat[1:n, 1:p])
        if options.optim.initial_guess
            set_start_value.(Bhat, IG.B) # set initial value of Bhat
        end
        tmp = (@isdefined tmp) ? hcat(tmp, Bhat) : Bhat  # concatenate Bhat to objective matrix
    end

    # Quadratic F or H matrix
    if options.system.is_quad
        if options.optim.which_quad_term == "F"
            @variable(model, Fhat[1:n, 1:s])
            if options.optim.initial_guess
                set_start_value.(Fhat, IG.F)  # set initial value of Fhat
            end
            tmp = (@isdefined tmp) ? hcat(tmp, Fhat) : Fhat  # concatenate Fhat to objective matrix
        else
            @variable(model, Hhat[1:n, 1:v])
            if options.optim.initial_guess
                set_start_value.(Hhat, IG.H)  # set initial value of Hhat
            end
            tmp = (@isdefined tmp) ? hcat(tmp, Hhat) : Hhat # concatenate Hhat to objective matrix
        end
    end

    # Bilinear N matrix
    if options.system.is_bilin
        @variable(model, Nhat[1:n, 1:w])
        if options.optim.initial_guess
            set_start_value.(Nhat, IG.N)  # set initial value of Nhat
        end
        tmp = (@isdefined tmp) ? hcat(tmp, Nhat) : Nhat # concatenate Nhat to objective matrix
    end

    # Constant K matrix
    if options.system.has_const
        @variable(model, Khat[1:n, 1])
        if options.optim.initial_guess
            set_start_value.(Khat, IG.K)  # set initial value of Khat
        end
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
    Ahat = options.system.is_lin ? value.(Ahat) : 0
    Bhat = options.system.has_control ? value.(Bhat)[:, :] : 0
    Fhat = options.system.is_quad ? options.optim.which_quad_term=="F" ? value.(Fhat) : H2F(value.(Hhat)) : 0
    Hhat = options.system.is_quad ? options.optim.which_quad_term=="H" ? value.(Hhat) : F2Hs(value.(Fhat)) : 0
    Nhat = options.system.is_bilin ? value.(Nhat) : 0
    Khat = options.system.has_const ? value.(Khat) : 0
    @info "Done."
    return Ahat, Bhat, Fhat, Hhat, Nhat, Khat
end


"""
    NC_Optimize_output(Y::Matrix, Xhat_t::Union{Matrix, Transpose}, 
        dims::Dict, options::Abstract_Options)

Output optimization for the standard operator inference (for operator `C`)

## Arguments
- `Y`: the output matrix 
- `Xt_hat`: the state matrix

## Return
- the output state matrix `C`
"""
function NC_Optimize_output(Y::Matrix, Xhat_t::Union{Matrix, Transpose}, 
        dims::Dict, options::Abstract_Options)
    # Some dimensions to unpack for convenience
    n = dims[:n]
    q = dims[:q]

    Yt = transpose(Y)

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
